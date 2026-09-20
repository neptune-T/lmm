# Legacy

**Nothing was deleted.** This directory is a *pointer*, not a dumping ground.

The legacy pipeline is **not in this repository**. It lives in the companion
repo, at its original paths:

> **https://github.com/neptune-T/llm_rollout**

See [`../docs/LEGACY_CODE_LOCATION.md`](../docs/LEGACY_CODE_LOCATION.md) for the
path convention, and [`../docs/LEGACY_EXPERIMENTS.md`](../docs/LEGACY_EXPERIMENTS.md)
for what each historical experiment concluded and whether it still holds.

## Why the legacy code was not moved

Historical experiment scripts must keep reproducing their original results.
They are referenced by exact path from frozen protocol documents, `a800_*.sh`
launchers, `scripts/ops/run_task_quality_vla_pipeline_v1.py` and existing
tests. Relocating them would silently break reproduction of published results.

So they stayed where they were, and their key modules carry a
`LEGACY EXPERIMENT PIPELINE` banner comment. The banners are comment-only —
`ast.dump()` is byte-identical to the prior revision for all nine marked files,
so behaviour is provably unchanged.

## Map (all paths in `neptune-T/llm_rollout`)

### OVMM semantic skill sequencer — demoted from canonical

```
scripts/rollout/semantic_mode_event_controller_v1.py
scripts/rollout/semantic_mode_event_controller_v2.py
configs/semantic_mode_event_controller_v1.json
tests/test_semantic_mode_event_controller_v{1,2}.py
```

Canonical replacement here: [`lmm/instrumentation/event_detector.py`](../lmm/instrumentation/event_detector.py)
— the same observable-predicate idea, but **observation only**. It exposes no
action-producing API and is proven non-interfering by test.

### Semantic root pipeline — predefined pre-place / E5 roots

```
scripts/diagnostics/capture_event_aligned_root_bank_v2.py
scripts/diagnostics/qualify_event_aligned_root_bank_v2.py
scripts/diagnostics/finalize_event_aligned_root_bank_v2_gate.py
scripts/rollout/ovmm_place_entry_root_collector_v2_1_1.py
scripts/rollout/pi0_ovmm_pre_place_release_boundary_probe.py
configs/branch_support/
```

Canonical replacement here: [`lmm/frontier/`](../lmm/frontier/) — roots are
proposed by trajectory position and selected by **measured `V_hat`**, not by
semantic label.

### Frontier GRPO / online v1–v3 — branch-only

Not on `main`. Only on these branches of `llm_rollout`:

| Branch | `scripts/frontier_rl/` |
|---|---|
| `diagnostic/shared-legacy-replication-v1` | 36 files — most complete GRPO math, **start here** |
| `experiment/frontier-coverage-preserving-update-v1` | 39 files |
| `diagnosis/frontier-conditioning-v1` | 29 files |

Canonical replacement here: [`lmm/rl/`](../lmm/rl/).

### FP8 / PTQ quantization

On the workstation only, outside every git repository:
`scripts/fp8_rollout_scaling/`, `fp8_prospective_v5/`, `fp8_prefilter_v4/`,
`experiments/fp8_*`.

Canonical replacement here: [`lmm/rollout/precision_backend.py`](../lmm/rollout/precision_backend.py)
for *naming*. The quantisation implementation has no replacement and must be
ported from those scripts when needed.

## Do not

- Re-run the 20-round online GRPO hoping for a different sign. It returned
  `NO_RETAINED_IMPROVEMENT` with monotone degradation in shaping metrics.
- Cite the V3 Spearman 0.9107 as evidence that cheap rollout preserves
  **success** ordering — there were zero successes on either side of that
  comparison. It is proxy-metric evidence only.
- Substitute the public `Neptune-T/LMM-GRPO-Handoff` q0 for the frozen q0 of
  the 20-round run. Different LFS hashes.
