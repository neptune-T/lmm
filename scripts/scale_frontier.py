#!/usr/bin/env python3
"""Stage 5 — scale cheap exploration from selected frontier roots.

Draws ``K_cheap`` stochastic continuations from each selected root using the
same rollout backend as discovery, in ``explore`` mode. Emits
CheapExplorationRecord with explicit root and group identity.
"""

from __future__ import annotations

import argparse

from _common import add_common_args, load_config, require_simulator, resolve_manifest

from lmm.frontier.continuation_probe import ProbeConfig
from lmm.rl.dataset_builder import group_id_for
from lmm.schemas import read_jsonl


def main() -> None:
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--frontier-summaries", required=True,
                   help="frontier_summaries.jsonl from discover_frontier.py")
    p.add_argument("--snapshots", required=True, help="snapshots.jsonl")
    p.add_argument("--frontier-config", required=True)
    args = p.parse_args()

    cfg = load_config(args.frontier_config)
    probe_cfg = ProbeConfig(**cfg["probe"])

    summaries = [s for s in read_jsonl(args.frontier_summaries)
                 if getattr(s, "selected_as_frontier", False)]
    snapshots = {s.root_id: s for s in read_jsonl(args.snapshots)}
    print(f"[input] {len(summaries)} selected frontier roots")

    resolve_manifest(
        args,
        precision_backend=probe_cfg.probe_backend,
        rollout_mode="explore",
        k_cheap=probe_cfg.k_cheap,
        number_of_roots=len(summaries),
    )
    for s in summaries:
        print(f"[plan] root={s.root_id} V_hat={s.success_rate:.3f} "
              f"K_cheap={probe_cfg.k_cheap} group={group_id_for(s.root_id, args.policy_version)}")
        if s.root_id not in snapshots:
            raise SystemExit(f"no snapshot record for selected root {s.root_id!r}")

    if args.dry_run:
        print("[dry-run] exploration plan resolved; stopping before simulator work")
        return

    require_simulator()
    raise SystemExit(
        "Cheap exploration requires a running cheap policy server and the "
        "simulator. Wire RolloutRunner.run_continuation through "
        "ContinuationProbeRunner.explore here; see docs/PIPELINE.md."
    )


if __name__ == "__main__":
    main()
