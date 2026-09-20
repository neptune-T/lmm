# Where the legacy code lives

This repository holds **only the canonical implementation** of the current
method. The historical codebase is not duplicated here — it stays in the
companion repository:

> **https://github.com/neptune-T/llm_rollout**
> branch `refactor/canonical-lmm-v1` (and the diagnosis/recovery branches)

## Path convention used throughout this repo's docs

Any path of the following shapes refers to **`neptune-T/llm_rollout`**, not to
this repository:

| Path shape | Example | What it is |
|---|---|---|
| `scripts/rollout/...` | `scripts/rollout/semantic_mode_event_controller_v2.py` | 405 historical rollout drivers and probes |
| `scripts/diagnostics/...` | `scripts/diagnostics/capture_event_aligned_root_bank_v2.py` | 53 root-bank / gate diagnostics |
| `scripts/ops/...`, `scripts/analysis/...` | `scripts/ops/run_task_quality_vla_pipeline_v1.py` | pipeline orchestration, analysis |
| `scripts/frontier_rl/...` | `scripts/frontier_rl/frontier_grpo_update_v2.py` | frontier GRPO — **only on non-default branches** |
| `configs/branch_support/...`, `configs/phase_a_stage_configs/...` | `configs/branch_support/train_branch_support_pilot_v2.json` | frozen experiment configs |
| Historical `docs/<SHOUTY_NAME>.md` | `docs/FRESH_POLICY_PROCESS_FIRST_DIVERGENCE_AUDIT_V1.md` | 229 frozen protocols and closeouts |

Paths that exist **in this repository** are: `lmm/`, `scripts/` (the 8
canonical entry points), `tests/`, `configs/{env,policy,rollout,frontier,rl,eval}/`,
`docs/` (the 9 documents written for the refactor), `legacy/README.md`,
`experiments/`.

## Why it is split this way

The historical tree could not be moved. Frozen protocol documents, `a800_*.sh`
launchers and ops pipelines reference those files by exact path, so relocating
them would break reproduction of published results. They were therefore marked
in place — with comment-only `LEGACY EXPERIMENT PIPELINE` banners whose ASTs
are byte-identical to the prior revision — and left in `llm_rollout`.

This repository is the clean canonical implementation, with no legacy mass and
its own history.

## Artifacts outside both repositories

These live on the research workstation and are **not** in any git repository:

| Path | Size | Note |
|---|---|---|
| `paper_data/server_pull_20260913/` | 718 MB | **Irreplaceable.** Server evidence pull; the only copies of `online_driver_v4.py` / `v5.py`, the 20-round training audit and 64 held-out execution records |
| `scripts/fp8_rollout_scaling/` | 292 KB | FP8/Q4 quantisation code — **no version control** |
| `experiments/fp8_*` | 5.9 MB | Hardware audit, native-FP8 deferral manifest, V3 ranking records |
| `benchmark/` | 26 GB | BEHAVIOR-1K + home-robot upstream |
| `basecode/` | 29 MB | openpi + openvla upstream |

See [`CODEBASE_INVENTORY_20260919.md`](CODEBASE_INVENTORY_20260919.md) for the
full inventory and [`LEGACY_EXPERIMENTS.md`](LEGACY_EXPERIMENTS.md) for what
each historical experiment concluded.
