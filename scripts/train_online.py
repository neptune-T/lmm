#!/usr/bin/env python3
"""Stage 7 — one guarded online RL update from verified continuations.

Consumes generic verified trajectory groups. Refuses to update when every
sibling group is degenerate (the BRANCH_SUPPORT_NOT_OBSERVED case) or when the
trust-region guard trips.
"""

from __future__ import annotations

import argparse
import json

from _common import add_common_args, load_config, resolve_manifest

from lmm.rl.advantages import AdvantageConfig, degenerate_fraction, normalize_groups
from lmm.rl.dataset_builder import DatasetConfig, audit_grouping, build_dataset
from lmm.rl.grpo import GRPOConfig
from lmm.rl.reward import RewardConfig
from lmm.schemas import read_jsonl


def main() -> None:
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--verifications", required=True, help="verifications.jsonl")
    p.add_argument("--rl-config", required=True)
    p.add_argument("--optimizer-step", type=int, default=1)
    args = p.parse_args()

    cfg = load_config(args.rl_config)
    reward_cfg = RewardConfig(**cfg["reward"])
    ds_cfg = DatasetConfig(**cfg["dataset"])
    adv_cfg = AdvantageConfig(**cfg["advantage"])
    grpo_cfg = GRPOConfig(**cfg["grpo"])

    manifest = resolve_manifest(
        args,
        precision_backend="bf16",
        rollout_mode="verify",
        rl_algorithm="grpo",
        learning_rate=grpo_cfg.learning_rate,
        update_epochs=grpo_cfg.update_epochs,
    )

    verifications = list(read_jsonl(args.verifications))
    samples = build_dataset(verifications, reward_cfg, ds_cfg)
    audit = audit_grouping(samples)
    print(f"[dataset] {json.dumps(audit, indent=2, default=str)}")
    if not audit["all_groups_single_root"]:
        raise SystemExit("ERROR: a group spans multiple roots; refusing to train")

    normalized, stats = normalize_groups(samples, adv_cfg)
    frac = degenerate_fraction(stats)
    print(f"[groups] {len(stats)} groups, degenerate fraction {frac:.3f}")
    for s in stats:
        print(f"  {s.group_id} root={s.root_id} n={s.size} "
              f"mean_r={s.mean_reward:.3f} std={s.std_reward:.3f} "
              f"degenerate={s.degenerate}")

    if not normalized:
        raise SystemExit(
            "TERMINAL: all sibling groups are degenerate. This reproduces "
            "BRANCH_SUPPORT_NOT_OBSERVED and does not authorize an online "
            "update. Re-run frontier discovery with a wider candidate scan."
        )

    (args.output_dir / "training_samples.json").write_text(
        json.dumps([s.to_dict() for s in normalized], indent=2), encoding="utf-8")
    print(f"[ready] {len(normalized)} samples across "
          f"{len({s.group_id for s in normalized})} groups")

    if args.dry_run:
        print("[dry-run] dataset built and audited; no optimizer step taken")
        return

    raise SystemExit(
        "Applying a gradient step requires the trainable pi0.5 backend. "
        "Implement PolicyBackend (lmm/rl/trainer.py) against the JAX/OpenPI "
        "checkpoint and pass it to OnlineTrainer. Do not launch this without "
        "the checks in docs/NEXT_EXPERIMENT.md."
    )


if __name__ == "__main__":
    main()
