#!/usr/bin/env python3
"""Stage 6 — re-run top cheap candidates at high fidelity (BF16).

Cheap outcomes select; they never score. Every verification is an independent
execution and is recorded with full lineage:

    source_root -> cheap_rollout_id -> cheap_score -> selection_rank
                -> verification_rollout_id -> verified_success
"""

from __future__ import annotations

import argparse

from _common import add_common_args, load_config, require_simulator, resolve_manifest

from lmm.schemas import read_jsonl
from lmm.verification.bf16_verifier import VerifierConfig, rank_candidates


def main() -> None:
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--cheap-records", required=True, help="cheap_exploration.jsonl")
    p.add_argument("--snapshots", required=True)
    p.add_argument("--k-verify", type=int, default=8)
    p.add_argument("--eval-config", default=None)
    args = p.parse_args()

    backend = "bf16"
    if args.eval_config:
        backend = load_config(args.eval_config)["fidelity_eval"]["verify_backend"]

    # Raises if a cheap backend was passed: verification must be high fidelity.
    verifier_cfg = VerifierConfig(k_verify=args.k_verify, backend=backend)

    candidates = list(read_jsonl(args.cheap_records))
    by_root: dict[str, list] = {}
    for c in candidates:
        by_root.setdefault(c.root_id, []).append(c)

    resolve_manifest(
        args,
        precision_backend=verifier_cfg.backend,
        rollout_mode="verify",
        k_verify=args.k_verify,
        number_of_roots=len(by_root),
    )
    for root_id, cands in sorted(by_root.items()):
        top = rank_candidates(cands, verifier_cfg)
        print(f"[plan] root={root_id} cheap={len(cands)} -> verify {len(top)}: "
              f"{[round(c.cheap_score, 3) for c in top]}")

    if args.dry_run:
        print("[dry-run] verification plan resolved; stopping before simulator work")
        return

    require_simulator()
    raise SystemExit(
        "Verification requires the BF16 policy server and the simulator. Wire "
        "BF16Verifier against RolloutRunner here; see docs/PIPELINE.md."
    )


if __name__ == "__main__":
    main()
