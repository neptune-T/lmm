#!/usr/bin/env python3
"""Stage 1 — run the end-to-end VLA on the full task from episode init.

No semantic skill sequencer. The policy decides when navigation ends and
manipulation begins.

    python scripts/rollout_full_task.py \
        --experiment-name e2e_bf16_sanity_v1 \
        --output-dir runs/e2e_sanity \
        --policy-checkpoint /path/to/bf16 --policy-version q0 \
        --env-config configs/env/behavior_ovmm_v1.json \
        --rollout-config configs/rollout/full_task_v1.json
"""

from __future__ import annotations

import argparse

from _common import add_common_args, load_config, require_simulator, resolve_manifest

from lmm.env.behavior_env import BehaviorEnv, BehaviorEnvConfig
from lmm.env.simulator_adapter import AdapterConfig
from lmm.instrumentation.event_detector import EventDetector, EventDetectorConfig
from lmm.policy.base import ActionSpec
from lmm.policy.pi05_policy import OVMM_ACTION_SPEC, Pi05Policy, Pi05PolicyConfig
from lmm.rollout.client import PolicyClient, PolicyClientConfig
from lmm.rollout.runner import RolloutConfig, RolloutRunner
from lmm.rollout.server import validate_health
from lmm.schemas import write_jsonl


def main() -> None:
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--env-config", required=True)
    p.add_argument("--rollout-config", required=True)
    p.add_argument("--server-url", default="http://127.0.0.1:8080")
    args = p.parse_args()

    env_cfg = load_config(args.env_config)
    roll_cfg = load_config(args.rollout_config)
    action_spec = ActionSpec(**env_cfg.get("action_spec", {
        "base_dim": OVMM_ACTION_SPEC.base_dim,
        "arm_dim": OVMM_ACTION_SPEC.arm_dim,
        "gripper_dim": OVMM_ACTION_SPEC.gripper_dim,
    }))

    manifest = resolve_manifest(
        args,
        task=roll_cfg.get("task", ""),
        precision_backend=roll_cfg.get("precision_backend", "bf16"),
        rollout_mode="full_task",
        random_seeds=roll_cfg.get("seeds", [0]),
    )
    if args.dry_run:
        print("[dry-run] manifest resolved; stopping before simulator work")
        return

    require_simulator()

    client = PolicyClient(PolicyClientConfig(url=args.server_url))
    health = validate_health(client.health())
    print(f"[policy] {health}")

    policy = Pi05Policy(
        Pi05PolicyConfig(
            server_url=args.server_url,
            policy_version=args.policy_version,
            precision_backend=manifest.precision_backend,
            action_spec=action_spec,
        ),
        client,
    )
    env = BehaviorEnv(
        BehaviorEnvConfig(
            benchmark_root=env_cfg["benchmark_root"],
            task=env_cfg.get("task", ""),
            max_steps=env_cfg.get("max_steps", 1500),
            adapter=AdapterConfig(**env_cfg.get("adapter", {})),
        ),
        action_spec,
        detector=EventDetector(EventDetectorConfig(**env_cfg.get("event_detector", {}))),
    )
    runner = RolloutRunner(
        env, policy,
        RolloutConfig(max_steps=env_cfg.get("max_steps", 1500),
                      precision_backend=manifest.precision_backend,
                      task=roll_cfg.get("task", "")),
    )

    out = args.output_dir / "full_rollouts.jsonl"
    for ep_i in range(roll_cfg.get("n_episodes", 1)):
        for seed in roll_cfg.get("seeds", [0]):
            rec = runner.run_full_task(
                rollout_id=f"full::ep{ep_i}::seed{seed}",
                episode_id=f"ep{ep_i}", scene_id="", seed=seed,
            )
            write_jsonl(out, [rec])
            print(f"[rollout] ep{ep_i} seed{seed} "
                  f"success={rec.terminal_success} steps={rec.trajectory_length}")
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
