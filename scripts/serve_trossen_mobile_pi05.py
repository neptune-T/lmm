#!/usr/bin/env python3
"""Serve pi0.5 with the full Mobile ALOHA 14-D/16-D contract.

The upstream ALOHA output adapter intentionally truncates actions to 14
dimensions.  This server substitutes only that final adapter so the two
Mobile ALOHA base-velocity channels survive normalization and inference.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import pathlib
import pickle
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


class MobileAlohaOutputs:
    """Standard ALOHA arm/gripper encoding while preserving base vx/vy."""

    def __init__(self, *, adapt_to_pi: bool) -> None:
        self.adapt_to_pi = adapt_to_pi

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        from openpi.policies import aloha_policy

        actions = np.asarray(data["actions"])
        if actions.ndim != 2 or actions.shape[1] < 16:
            raise ValueError(f"expected [horizon, >=16] actions, got {actions.shape}")
        aloha = aloha_policy._encode_actions(
            actions[:, :14].copy(), adapt_to_pi=self.adapt_to_pi
        )
        return {"actions": np.concatenate((aloha, actions[:, 14:16]), axis=1)}


def build_policy(checkpoint: pathlib.Path, *, num_steps: int):
    import jax.numpy as jnp
    from openpi import transforms
    from openpi.models import model as openpi_model
    from openpi.policies import aloha_policy
    from openpi.policies import policy as policy_module
    from openpi.training import config as training_config

    train = training_config.get_config("pi05_aloha")
    mobile_assets = dataclasses.replace(
        train.data.assets,
        assets_dir=str(checkpoint / "assets"),
        asset_id="trossen_mobile",
    )
    train = dataclasses.replace(train, data=dataclasses.replace(train.data, assets=mobile_assets))
    data = train.data.create(train.assets_dirs, train.model)
    if data.norm_stats is None:
        raise RuntimeError("trossen_mobile normalization statistics were not loaded")
    if np.asarray(data.norm_stats["state"].mean).shape != (14,):
        raise RuntimeError("trossen_mobile state statistics are not 14-D")
    if np.asarray(data.norm_stats["actions"].mean).shape != (16,):
        raise RuntimeError("trossen_mobile action statistics are not 16-D")

    params = openpi_model.restore_params(checkpoint / "params", dtype=jnp.bfloat16)
    model = train.model.load(params)
    outputs = []
    replaced = 0
    for transform in data.data_transforms.outputs:
        if isinstance(transform, aloha_policy.AlohaOutputs):
            outputs.append(MobileAlohaOutputs(adapt_to_pi=transform.adapt_to_pi))
            replaced += 1
        else:
            outputs.append(transform)
    if replaced != 1:
        raise RuntimeError(f"expected one AlohaOutputs transform, replaced {replaced}")

    return policy_module.Policy(
        model,
        transforms=[
            transforms.InjectDefaultPrompt(None),
            *data.data_transforms.inputs,
            transforms.Normalize(data.norm_stats, use_quantiles=data.use_quantile_norm),
            *data.model_transforms.inputs,
        ],
        output_transforms=[
            *data.model_transforms.outputs,
            transforms.Unnormalize(data.norm_stats, use_quantiles=data.use_quantile_norm),
            *outputs,
        ],
        sample_kwargs={"num_steps": num_steps},
        metadata=train.policy_metadata,
    )


class ServedPolicy:
    def __init__(self, checkpoint: pathlib.Path, num_steps: int) -> None:
        self.checkpoint = checkpoint
        self.num_steps = num_steps
        self.policy = build_policy(checkpoint, num_steps=num_steps)
        self.identity = checkpoint_identity(checkpoint)
        self.calls = 0
        self.lock = threading.Lock()

    def health(self) -> dict[str, Any]:
        return {
            "policy_version": "pi05_base:trossen_mobile",
            "precision_backend": "bf16",
            "checkpoint_sha256": self.identity,
            "state_dim": 14,
            "action_dim": 16,
            "base_semantics": "velocity_xy",
            "control_hz": 50,
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
                raise RuntimeError(f"mobile output contract produced {actions.shape}, expected [H,16]")
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
    args = parser.parse_args()
    served = ServedPolicy(args.checkpoint.resolve(), args.num_steps)
    print(json.dumps({"ready": True, **served.health()}, sort_keys=True), flush=True)
    ThreadingHTTPServer((args.host, args.port), handler_for(served)).serve_forever()


if __name__ == "__main__":
    main()
