# LMM Codebase Inventory — 2026-09-19

> **Path note.** Paths such as `scripts/rollout/...`, `scripts/diagnostics/...`,
> `configs/branch_support/...` and historical `docs/<NAME>.md` refer to the
> companion repository **[neptune-T/llm_rollout](https://github.com/neptune-T/llm_rollout)**,
> not to this one. See [`LEGACY_CODE_LOCATION.md`](LEGACY_CODE_LOCATION.md).

Written **before** any restructuring, per the cleanup mandate. Companion to
[`PRE_REFACTOR_BASELINE_20260919.md`](PRE_REFACTOR_BASELINE_20260919.md), which
records the exact git and filesystem state at entry.

## 0. Executive summary of the confusion

Five findings explain why the folder structure stopped reflecting the method:

1. **The workspace root is not a git repository.** `/home/plote/longhorizon/.git`
   is an *empty directory*. Everything at the workspace root — including all FP8 /
   Q4 quantization code and the FP8 experiment artifacts — has **no version
   control at all**. The only real repo is `lmm_rollout_project/`.
2. **The canonical code is split across git branches, not directories.** The
   online-GRPO / frontier RL implementation (`scripts/frontier_rl/`, 36 files)
   does **not exist** on the currently checked-out branch. It lives only on
   three other branches, materialised through three sibling git *worktrees*
   that look like independent project copies but are not.
3. **The pipeline that produced the headline results is not in git anywhere.**
   `online_driver_v4.py` / `online_driver_v5.py` — the drivers behind the 20-round
   online update — exist only inside `paper_data/server_pull_20260913/`, an
   un-versioned pull from a server that has since been lost.
4. **The A800 server was restored on 2026-09-20.** The recovered q20 model,
   optimizer state and q4 model remain available, and the host now has a
   working HomeRobot/HSSD/OpenPI stack. Exact q0 / q5 / q10 weights remain
   unrecoverable; q20 is Stretch-only and is not a dual-arm checkpoint.
5. **`Planning/` is an unimplemented scaffold.** It already contains the
   conceptual package layout now being requested (`src/longhorizon/{envs,
   policies,rollout,scoring,selection,optimization,evaluation,quantization}`)
   but holds **zero `.py` files** — only READMEs, from June.

## 1. Environment reality check

| Fact | Value | Consequence |
|---|---|---|
| GPU | 1× RTX 3070 Laptop, 8 GB | Cannot serve pi0.5 BF16; cannot run Habitat/BEHAVIOR |
| Disk | 916 G, **4.9 G free (100 % used)** | No large copies; moves must stay on-filesystem |
| Interpreter | Python 3.13.9 | `torch`, `jax`, `habitat`, `openpi` all absent |
| A800 cluster | **restored** (verified 2026-09-20) | 4×A800 80GB; canonical paths are documented in `A800_SERVER.md` |

The end-to-end BF16 sanity check (milestone §21) **cannot be executed on this
machine**. It is specified in [`NEXT_EXPERIMENT.md`](NEXT_EXPERIMENT.md) for a
GPU host instead.

## 2. Inventory table

Paths are relative to `/home/plote/longhorizon`. "Canonical" means the new
end-to-end method depends on it; "Legacy" means preserved for historical
reproduction only.

| Path | Purpose | Still relevant? | Status | Important artifacts | Dependencies | Recommended action |
|---|---|---|---|---|---|---|
| `lmm_rollout_project/` | The only real git repo (`neptune-T/llm_rollout`); 344 `.py`, 229 docs, 38 tests | Yes | **Canonical host** | git history, configs, tests | git | Host the new `lmm/` package here |
| `lmm_rollout_project/scripts/rollout/` (405 files) | Historical rollout drivers, probes, root collectors | Partly | **Legacy (in place)** | suffix-rank, release-boundary, snapshot probes | habitat, openpi | Keep path stable; mark legacy, mine for reusable logic |
| `.../scripts/rollout/semantic_mode_event_controller_v{1,2}.py` | OVMM semantic skill/mode controller | No (demoted) | **Legacy (in place)** | frozen contracts in `docs/` | — | Mark legacy; re-implement as *observer* in `lmm/instrumentation/` |
| `.../scripts/rollout/pi0_http_policy_server*.py` | BF16 pi0.5 HTTP serving (3 variants) | Yes | **Canonical source material** | serving contract | jax, openpi | Basis for `lmm/rollout/server.py` contract |
| `.../scripts/rollout/ovmm_snapshot_restore_probe.py` | Habitat sim snapshot save/restore | Yes | **Canonical source material** | `MEASURE_STATE_PREFIXES`, `TASK_STATE_FIELDS` | habitat | Basis for `lmm/env/snapshot.py` |
| `.../scripts/diagnostics/` (53 files) | Event-aligned root bank v2 capture/qualify/finalize | No (superseded) | **Legacy (in place)** | root-bank gates | habitat | Mark legacy; frontier is now empirical |
| `.../scripts/frontier_rl/` **(other branches only)** | Frontier GRPO update v1/v2, online v1/v3 protocol | Partly | **Legacy (branch-only)** | GRPO math, root grouping | numpy | Document branch pointers; do not duplicate |
| `.../scripts/train/`, `configs/rl/` (untracked) | GRPO v6 post-update trust-region guard | Yes | **Canonical source material** | guard config | numpy | Fold concept into `lmm/rl/` |
| `.../docs/` (229 files) | Frozen protocols, prereg, closeouts | Yes (as history) | **Legacy record** | negative-result freezes | — | Preserve; index in `LEGACY_EXPERIMENTS.md` |
| `lmm_frontier_grpo_v1/` | **Worktree** → `diagnosis/frontier-conditioning-v1` | Partly | Legacy worktree | 29 `frontier_rl` files | git | Keep as worktree; do not treat as a copy |
| `lmm_shared_legacy_replication_v1/` | **Worktree** → `diagnostic/shared-legacy-replication-v1` | Partly | Legacy worktree | 36 `frontier_rl` files | git | Keep; authoritative for GRPO math |
| `lmm_frontier_attribution_diagnostic_v1/` | **Worktree** → `experiment/frontier-coverage-preserving-update-v1` | Partly | Legacy worktree | 39 `frontier_rl` files | git | Keep |
| `scripts/fp8_rollout_scaling/` (root, **no git**) | JAX FP8 PTQ quantize/evaluate, FP8+Q4 policy server, LoRA convert | Yes | **Canonical source material** | quantize/serve path | jax, openpi | Document in `QUANTIZATION_BACKENDS.md`; **bring under git** |
| `fp8_prospective_v5/` (root, no git) | v5 prospective capture/qualify/analysis | Partly | Legacy | v5 bank scripts | — | Preserve; document |
| `fp8_prefilter_v4/` (root, no git) | v4 prefilter + BF16 verification report | No | Legacy artifact | report, bank manifest | — | Preserve as artifact |
| `experiments/fp8_rollout_scaling_v1/` | **Hardware audit + native-FP8 deferral** | Yes | **Authoritative** | `hardware_audit.json`, `native_fp8_deferred_manifest.json` | — | Cite verbatim in quantization doc |
| `experiments/fp8_ptq_deployment_aligned_v3/` | Ranking preservation, paired physics, Q4 freeze | Yes | Legacy result | `ranking_records.jsonl`, `q4_candidate_freeze.json` | — | Preserve; cite for Q3 |
| `paper_data/server_pull_20260913/` (718 M) | Server evidence pull + HF recovery | Yes | **Irreplaceable** | 20-round audit, 64 held-out records, cascade timing | — | **Never move or delete**; back up off-machine |
| `.../hf_recovery_20260914/expanded/code/` | `online_driver_v5.py`, `canonical_train_success_v5.py`, … | Yes | **Legacy, un-versioned** | the actual online RL driver | — | Read-only reference; the real GRPO loop |
| `Planning/` (separate git repo) | Design docs + empty package scaffold | Partly | Reference | `Doc/` research notes | git | Keep as design reference only |
| `benchmark/` (26 G) | BEHAVIOR-1K + home-robot upstream | Yes | External dep | simulator + assets | — | Leave untouched |
| `basecode/` (29 M) | openpi + openvla upstream | Yes | External dep | policy code | — | Leave untouched |
| `rltask/Sana` (121 M) | Sana / Sol-RL upstream | Unknown | Unknown | — | — | Leave; not on canonical path |
| `remote_results/`, `results/`, `audit_exports/` | Older result pulls (`audit_exports/` is empty) | As history | Legacy artifacts | release-boundary replications | — | Preserve in place |
| `paper/`, `icra2027/`, `icra2027-template/` | Reference papers + LaTeX | Yes | Writing | `main.tex`, refs | — | Leave |
| `build_online_v3_audit.py`, `fp8_q4_handoff_builder.py` (root) | One-off audit/handoff builders | No | Legacy one-offs | — | — | Preserve; document as one-offs |
| `node_modules/`, `__pycache__/`, `.tmp-build/` | Build noise | No | Noise | none | — | Safe to ignore (not deleted) |

## 3. What the historical results actually say

Three frozen results define the scientific starting point. All are **negative**,
and all are preserved.

| Result | Where | Verdict |
|---|---|---|
| TRAIN K=8 branch-support pilot | `LMM_CRITICAL_PATH_STATUS.md` | `BRANCH_SUPPORT_NOT_OBSERVED` — 64/64 siblings terminated `released_outside`; 0 successes, 0 mixed roots |
| Fresh-process determinism | `docs/FRESH_POLICY_PROCESS_FIRST_DIVERGENCE_AUDIT_V1.md` | `MODEL_FORWARD_BACKEND_DIVERGENCE` — 274/320 float32 outputs differed across identical cold processes |
| 20-round online GRPO | `paper_closure/single_execution_eval/` | `NO_RETAINED_IMPROVEMENT` — q0 1/16, q5 0/16, q10 0/16, q20 0/16 on frozen held-out |

**These three directly motivate the new canonical method.** A hand-picked
semantic `PRE_PLACE` root yielded a root bank with *zero outcome variance*, so
GRPO had no signal to learn from — which is exactly what an empirical,
`V_hat`-based frontier definition is designed to avoid. The determinism finding
constrains how probes must be executed (one persistent policy server, fresh
simulator process per branch).

## 4. Ambiguous naming found (resolved in `QUANTIZATION_BACKENDS.md`)

| Old term | What it actually meant |
|---|---|
| "FP8" | FP8 **storage** with dequantisation to BF16 compute. Never native FP8 tensor-core execution on SM80 |
| "Q4" | Overloaded: (a) a 4-bit weight-only preview backend, (b) **training round 4** in `q0/q5/q10/q20` checkpoint naming |
| "cheap" | Any non-BF16 rollout backend, unqualified |
| "champion" | The BEHAVIOR-challenge winning checkpoint |
| "mixed" | Roots with both success and failure siblings |

`q0…q20` are **checkpoint round indices**, not precisions. The collision between
"Q4 = round 4" and "Q4 = 4-bit" is the single worst naming hazard in the repo.

## 5. Recommended action summary

1. Build the canonical `lmm/` package inside `lmm_rollout_project` (the only
   versioned tree). ✔
2. Leave all 405 historical `scripts/rollout/` files at their existing paths —
   frozen protocols and shell launchers reference them by exact path. Mark, do
   not move. ✔
3. Never physically relocate `paper_data/` or `benchmark/`. ✔
4. Bring the root-level FP8 code under version control (**open item**).
5. Mirror the three HF large objects to a second store (**open item**, blocked
   by 4.9 G free disk).
