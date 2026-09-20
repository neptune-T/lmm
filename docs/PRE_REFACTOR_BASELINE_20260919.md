# Pre-Refactor Reproducibility Baseline — 2026-09-19

> **This is a baseline of the companion repository**
> [neptune-T/llm_rollout](https://github.com/neptune-T/llm_rollout), captured
> before the refactor that produced *this* repository. Every path, branch and
> commit below refers to that repo. Kept here because it is the provenance
> record for the canonical code.

Captured automatically before any cleanup action. Nothing was modified before this file was written.

## Git state (repo: lmm_rollout_project)

```
remote:  git@github.com-443:neptune-T/llm_rollout.git
branch:  diagnosis/event-reference-four-arm-v1
commit:  43aedb0ed6ab6997ad1af211f7392a565160d180
subject: fix: anchor four-arm runner cwd to HomeRobot root
date:    2026-09-03T06:11:18-04:00
```

## Worktrees

```
/home/plote/longhorizon/lmm_rollout_project                     43aedb0 [diagnosis/event-reference-four-arm-v1]
/home/plote/longhorizon/lmm_frontier_attribution_diagnostic_v1  7aa5964 [experiment/frontier-coverage-preserving-update-v1]
/home/plote/longhorizon/lmm_frontier_grpo_v1                    d95b667 [diagnosis/frontier-conditioning-v1]
/home/plote/longhorizon/lmm_shared_legacy_replication_v1        edfbc02 [diagnostic/shared-legacy-replication-v1]
```

## Local branches

```
analysis/frontier-attribution-diagnostic-v1  b367f12  results: record frontier attribution diagnostic v1
diagnosis/bplace-negative-support-v1  7217991  ops: log overnight stage heartbeats
diagnosis/event-aligned-root-bank-v2  c5a93dc  results: close event root v2 with policy qualification fail
diagnosis/event-reference-four-arm-v1  43aedb0  fix: anchor four-arm runner cwd to HomeRobot root
diagnosis/frontier-conditioning-v1  d95b667  results: record completed TRAIN conditioning diagnostic
diagnostic/shared-legacy-replication-v1  edfbc02  results: record completed shared legacy replication
experiment/frontier-coverage-preserving-update-v1  7aa5964  perf: reuse frozen runtime guard during coverage evaluation
fast/frontier-grpo-v1  e560792  results: record fast-track support no-signal and C1 curve
fast/frontier-grpo-v2-expanded-support  6c879a4  report expanded support pass and online pre-update blocker
fast/frontier-grpo-v3-online-eligibility  249fbe2  add detached Frontier GRPO v3 execution chain
main  845b8e3  Record unavailable F1-v4 rescue checkpoints
recovery/action-contract-bplace-v1  ea40032  Report exact-restore blocked branch-support pilot
recovery/action-representation-alignment-v1  f4ddcc3  Record action representation alignment decision
recovery/closed-loop-structured-executor-v1  2ed4c93  Record closed-loop BF16 asset blocker
recovery/phase-a-v2-formal-sft-dev-prefix  f9e9b2a  Add independent recovery SFT backup verifier
recovery/pi05-fresh-process-determinism-v1  8ada141  Aggregate fresh-process pi05 determinism runs
recovery/servable-vla-live-closed-loop-smoke-v1  64eb344  Live runner: mask NAV inactive joint channels per mode contract
recovery/server-20260821  760debf  Record final Physical Activation decision
recovery/sft-closeout-branch-support-handoff  408de22  Point handoff at complete recovery backup
recovery/shared-server-determinism-branch-v2  1a9c4f4  Record negative TRAIN branch-support result
recovery/structured-cheap-high-ranking-v1  9091b79  Record structured quantized runtime contract failure
recovery/structured-proposer-learnability-v1  2f063ff  Record structured proposer learnability failure
recovery/task-matched-vla-lineage-rebuild-v1  d3239f2  Stamp final decision: TINY_SFT_OVERFIT_PASS
recovery/task-quality-vla-sft-prefix-b0-v1  338e3eb  Record DEV-v2 prereg freeze hash (2c41fb13)
recovery/vla-substrate-qualification-v1  50a1d43  Record final decision: VLA_CANONICAL_ASSET_BLOCKED
repair/pi05-event-aligned-place-suffix-v1  a2ca020  results: close pi05 repair at reference replay gate
```

## Uncommitted changes at baseline (all untracked; no tracked modifications, no stashes)

```
?? configs/mobile_bimanual_pi05_contract_v1.json
?? configs/ovmm_mobile_bimanual_assets_v1.json
?? configs/r1pro_pi05_contract_v1.json
?? configs/rl/
?? docs/GRPO_V6_TRANSACTIONAL_GUARD_INTEGRATION.md
?? docs/PRE_REFACTOR_BASELINE_20260919.md
?? habitat_patch/
?? results/grpo_q20_trust_region_audit_20260915.json
?? scripts/analysis/audit_grpo_q20_trust_region.py
?? scripts/rollout/mobile_bimanual_pi05_adapter.py
?? scripts/rollout/ovmm_trossen_pi05_adapter.py
?? scripts/rollout/r1pro_pi05_adapter.py
?? scripts/rollout/run_ovmm_trossen_mobile_ai.sh
?? scripts/train/
?? tests/test_grpo_q20_trust_region_audit.py
?? tests/test_mobile_bimanual_pi05_adapter.py
?? tests/test_ovmm_trossen_pi05_adapter.py
?? tests/test_post_update_trust_region_guard.py
?? tests/test_r1pro_pi05_adapter.py
```

stash entries: 0

## Filesystem inventory (workspace root `/home/plote/longhorizon`)

Captured 2026-09-19T05:42:53-04:00. Top-level directory sizes and file counts.

| Path | Size | Files | Notes |
|---|---:|---:|---|
| `audit_exports` | 4.0K | 0 |  |
| `basecode` | 29M | 298 |  |
| `benchmark` | 26G | 118934 |  |
| `experiments` | 5.9M | 18 |  |
| `fp8_prefilter_v4` | 24K | 4 |  |
| `fp8_prospective_v5` | 208K | 16 |  |
| `icra2027` | 276K | 5 |  |
| `icra2027-template` | 1.3M | 18 |  |
| `lmm_frontier_attribution_diagnostic_v1` | 280M | 2798 |  |
| `lmm_frontier_grpo_v1` | 279M | 2793 |  |
| `lmm_rollout_project` | 2.1G | 12748 |  |
| `lmm_shared_legacy_replication_v1` | 281M | 2858 |  |
| `ovm_hf_upload` | 5.6G | 6 |  |
| `paper` | 104M | 284 |  |
| `paper_data` | 718M | 12304 |  |
| `Planning` | 1.1M | 132 |  |
| `__pycache__` | 24K | 1 |  |
| `remote_results` | 16M | 112 |  |
| `results` | 76K | 1 |  |
| `rltask` | 121M | 538 |  |
| `scripts` | 304K | 21 |  |

### Loose files at workspace root

```
      8765  ./build_online_v3_audit.py
      7117  ./FP8_PTQ_TAIL_LOCALIZATION_V2_REPORT.md
      9675  ./fp8_q4_handoff_builder.py
        51  ./package.json
       445  ./package-lock.json
```

### Host / hardware at baseline

```
hostname: plote
gpu:      NVIDIA GeForce RTX 3070 Laptop GPU, 8192 MiB
disk:     916G total, 4.9G available (100% used)
python:   Python 3.13.9
torch/jax/habitat/openpi: not installed in the default interpreter
```
