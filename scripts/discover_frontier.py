#!/usr/bin/env python3
"""Stages 2-4 — snapshot candidates, probe them, locate the frontier.

    rollout -> candidate snapshots -> K_probe continuations -> V_hat -> select

Supports an optional coarse-to-fine second pass around uncertain regions.
"""

from __future__ import annotations

import argparse
import json

from _common import add_common_args, load_config, require_simulator, resolve_manifest

from lmm.frontier.candidate_proposer import CandidateProposerConfig, propose, refine_around
from lmm.frontier.continuation_probe import ContinuationProbeRunner, ProbeConfig
from lmm.frontier.frontier_estimator import estimate_all
from lmm.frontier.frontier_selector import FrontierSelectorConfig, select
from lmm.schemas import read_jsonl, write_jsonl


def main() -> None:
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--rollouts", required=True, help="full_rollouts.jsonl from stage 1")
    p.add_argument("--frontier-config", required=True)
    p.add_argument("--refine", action="store_true", help="run the coarse-to-fine pass")
    args = p.parse_args()

    cfg = load_config(args.frontier_config)
    prop_cfg = CandidateProposerConfig(**cfg["candidate_proposer"])
    probe_cfg = ProbeConfig(**cfg["probe"])
    sel_cfg = FrontierSelectorConfig(**cfg["selector"])

    manifest = resolve_manifest(
        args,
        precision_backend=probe_cfg.probe_backend,
        rollout_mode="probe",
        candidate_strategy=prop_cfg.strategy,
        k_probe=probe_cfg.k_probe,
        k_cheap=probe_cfg.k_cheap,
    )

    rollouts = list(read_jsonl(args.rollouts))
    print(f"[input] {len(rollouts)} full rollouts")

    all_candidates = []
    for r in rollouts:
        cands = propose(r.trajectory_length, prop_cfg,
                        step_events=[s.events for s in r.steps])
        all_candidates.append((r, cands))
        print(f"[candidates] {r.rollout_id}: {[c.step for c in cands]}")

    if args.dry_run:
        print("[dry-run] candidates proposed; stopping before simulator work")
        (args.output_dir / "candidates_preview.json").write_text(
            json.dumps(
                {r.rollout_id: [{"step": c.step, "reason": c.reason} for c in cs]
                 for r, cs in all_candidates},
                indent=2),
            encoding="utf-8",
        )
        return

    require_simulator()
    raise SystemExit(
        "Snapshot capture requires the simulator backend. Bind BehaviorEnv.make "
        "and a SnapshotStore here; see lmm/env/snapshot.py and "
        "docs/NEXT_EXPERIMENT.md."
    )


if __name__ == "__main__":
    main()
