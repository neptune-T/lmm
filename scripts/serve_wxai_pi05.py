#!/usr/bin/env python3
"""Serve pi0.5 with the WXAI dual-arm differential-drive contract.

Uses :func:`lmm.policy.wxai_transforms.wxai_data_transform_group` -- the same
function any future fine-tuning config must use -- so the transforms applied at
inference are the ones applied at training by construction, not by review.

Replaces the previous approach of substituting OpenPI's ``AlohaOutputs``. That
only fixed the 14-D truncation; it left ALOHA's joint-flip mask, angular
gripper conversion and vx/vy base semantics in the path.

Normalization statistics are NOT calibrated for this embodiment. The server
refuses to start on ALOHA statistics unless explicitly overridden, and stamps
``norm_stats_calibrated`` into every health response either way.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import pathlib
import pickle
import sys
import threading
from typing import Any

import numpy as np


def checkpoint_identity(path: pathlib.Path) -> str:
    """Stable identity over checkpoint inventory and small metadata files."""
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        rel = item.relative_to(path).as_posix()
        size = item.stat().st_size
        digest.update(f"{rel}\0{size}\0".encode())
        if size <= 1_000_000:
            digest.update(item.read_bytes())
    return digest.hexdigest()


ALOHA_ASSET_IDS = frozenset({"trossen", "trossen_mobile"})


def build_policy(
    checkpoint: pathlib.Path,
    *,
    num_steps: int,
    asset_id: str,
    allow_uncalibrated_norm_stats: bool,
):
    import jax.numpy as jnp
    from openpi import transforms
    from openpi.models import model as openpi_model
    from openpi.policies import policy as policy_module
    from openpi.training import config as training_config

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from lmm.policy.wxai_transforms import (
        ACTION_DIM,
        STATE_DIM,
        assert_base_dims_absolute,
        wxai_data_transform_group,
    )

    assert_base_dims_absolute()

    if asset_id in ALOHA_ASSET_IDS and not allow_uncalibrated_norm_stats:
        raise RuntimeError(
            f"asset_id {asset_id!r} holds Mobile ALOHA statistics: dual ViperX "
            "arms and a Slate omnidirectional base. They are dimensionally "
            "compatible with this contract but semantically wrong for WXAI, and "
            "action dims 14-15 have been redefined from vx/vy to forward/yaw, so "
            "their base statistics describe a different quantity. Compute WXAI "
            "statistics before fine-tuning. To proceed anyway for smoke testing "
            "only, pass --allow-uncalibrated-norm-stats; health will report "
            "norm_stats_calibrated=false."
        )

    train = training_config.get_config("pi05_aloha")
    assets = dataclasses.replace(
        train.data.assets, assets_dir=str(checkpoint / "assets"), asset_id=asset_id
    )
    train = dataclasses.replace(
        train, data=dataclasses.replace(train.data, assets=assets)
    )
    data = train.data.create(train.assets_dirs, train.model)
    if data.norm_stats is None:
        raise RuntimeError(f"no normalization statistics for asset_id {asset_id!r}")
    if np.asarray(data.norm_stats["state"].mean).shape != (STATE_DIM,):
        raise RuntimeError(f"state statistics are not {STATE_DIM}-D")
    if np.asarray(data.norm_stats["actions"].mean).shape != (ACTION_DIM,):
        raise RuntimeError(f"action statistics are not {ACTION_DIM}-D")

    params = openpi_model.restore_params(checkpoint / "params", dtype=jnp.bfloat16)
    model = train.model.load(params)

    # The WXAI group fully replaces data_transforms; only model_transforms
    # (tokenizer, resize, padding) are reused from the ALOHA config.
    wxai = wxai_data_transform_group()

    return policy_module.Policy(
        model,
        transforms=[
            transforms.InjectDefaultPrompt(None),
            *wxai.inputs,
            transforms.Normalize(data.norm_stats, use_quantiles=data.use_quantile_norm),
            *data.model_transforms.inputs,
        ],
        output_transforms=[
            *data.model_transforms.outputs,
            transforms.Unnormalize(data.norm_stats, use_quantiles=data.use_quantile_norm),
            *wxai.outputs,
        ],
        sample_kwargs={"num_steps": num_steps},
        metadata=train.policy_metadata,
    )


class ServedPolicy:
    def __init__(
        self,
        checkpoint: pathlib.Path,
        num_steps: int,
        *,
        asset_id: str,
        allow_uncalibrated_norm_stats: bool,
    ) -> None:
        self.checkpoint = checkpoint
        self.num_steps = num_steps
        self.asset_id = asset_id
        self.norm_stats_calibrated = asset_id not in ALOHA_ASSET_IDS
        self.policy = build_policy(
            checkpoint,
            num_steps=num_steps,
            asset_id=asset_id,
            allow_uncalibrated_norm_stats=allow_uncalibrated_norm_stats,
        )
        self.identity = checkpoint_identity(checkpoint)
        self.calls = 0
        self.lock = threading.Lock()

    def health(self) -> dict[str, Any]:
        return {
            "policy_version": f"pi05_base:wxai:{self.asset_id}",
            "precision_backend": "bf16",
            "checkpoint_sha256": self.identity,
            "contract": "lmm_wxai_contract_v1",
            "state_dim": 14,
            "action_dim": 16,
            "base_semantics": "forward_yaw",
            "action_labels_14_15": [
                "base_forward_velocity_mps",
                "base_yaw_velocity_radps",
            ],
            "arm_action_semantics": "absolute_joint_targets_rad",
            "gripper_convention": "0=open,1=closed",
            "norm_stats_asset_id": self.asset_id,
            "norm_stats_calibrated": self.norm_stats_calibrated,
            "control_hz_nominal": 50,
            "control_hz_verified": False,
            "calls": self.calls,
        }

    def reset(self, seed: int | None) -> dict[str, Any]:
        if seed is not None:
            import jax

            self.policy._rng = jax.random.key(int(seed))
        return {"ok": True, "seed": seed}

    def infer(self, observation: dict[str, Any]) -> np.ndarray:
        with self.lock:
            result = self.policy.infer(observation)
            actions = np.asarray(result["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != 16:
                raise RuntimeError(
                    f"WXAI output contract produced {actions.shape}, expected [H,16]"
                )
            if not np.isfinite(actions).all():
                raise RuntimeError("policy produced non-finite action")
            self.calls += 1
            return actions[0]


def handler_for(served: ServedPolicy):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            if self.path != "/healthz":
                self.send_error(404)
                return
            body = json.dumps(served.health(), sort_keys=True).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            try:
                size = int(self.headers.get("Content-Length", "0"))
                request = pickle.loads(self.rfile.read(size))
                op = request.get("op")
                if op == "health":
                    response = served.health()
                elif op == "reset":
                    response = served.reset(request.get("seed"))
                elif op == "infer":
                    response = served.infer(request["observation"])
                else:
                    raise ValueError(f"unknown operation {op!r}")
            except Exception as exc:
                self.send_error(500, explain=repr(exc))
                return
            body = pickle.dumps(response, protocol=pickle.HIGHEST_PROTOCOL)
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18116)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument(
        "--norm-stats-asset-id",
        required=True,
        help="asset_id under <checkpoint>/assets holding the normalization "
             "statistics. ALOHA ids (trossen, trossen_mobile) are refused "
             "unless --allow-uncalibrated-norm-stats is also passed.",
    )
    parser.add_argument(
        "--allow-uncalibrated-norm-stats",
        action="store_true",
        help="SMOKE TESTING ONLY: serve on statistics from a different "
             "embodiment. Health reports norm_stats_calibrated=false.",
    )
    args = parser.parse_args()
    served = ServedPolicy(
        args.checkpoint.resolve(),
        args.num_steps,
        asset_id=args.norm_stats_asset_id,
        allow_uncalibrated_norm_stats=args.allow_uncalibrated_norm_stats,
    )
    print(json.dumps({"ready": True, **served.health()}, sort_keys=True), flush=True)
    ThreadingHTTPServer((args.host, args.port), handler_for(served)).serve_forever()


if __name__ == "__main__":
    main()
