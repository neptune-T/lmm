# Legacy Experiments

> **Path note.** Paths such as `scripts/rollout/...`, `scripts/diagnostics/...`,
> `configs/branch_support/...` and historical `docs/<NAME>.md` refer to the
> companion repository **[neptune-T/llm_rollout](https://github.com/neptune-T/llm_rollout)**,
> not to this one. See [`LEGACY_CODE_LOCATION.md`](LEGACY_CODE_LOCATION.md).

Every historical experiment that produced a real result, where its code and
artifacts live now, and whether its conclusions still hold.

**Nothing here was deleted. Negative results are kept in full — they are the
main reason the method changed.**

## How legacy code is preserved

Three mechanisms, chosen per case:

1. **In place, marked.** The 405 files in `scripts/rollout/` and 53 in
   `scripts/diagnostics/` keep their exact paths. Frozen protocols, `a800_*.sh`
   launchers and reproduction commands in `docs/` reference them by path;
   moving them would break reproduction, which the cleanup mandate forbids.
   Key modules carry a `LEGACY EXPERIMENT PIPELINE` banner comment (comments
   only — no behavioural change).
2. **In git branches / worktrees.** The frontier-RL implementation exists only
   on non-default branches, materialised as sibling worktrees. Pointers below.
3. **In un-versioned artifact trees.** The FP8 work and the server pull live at
   the workspace root, outside any repository. Flagged as an open risk.

## A. OVMM semantic skill sequencer

**What it was.** A handcrafted phase machine
`NAV_TO_OBJ -> GAZE -> PICK -> NAV_TO_REC -> GAZE_AT_REC -> PLACE -> FALL_WAIT`
driving VLA execution mode and discrete pick/release events.

| | |
|---|---|
| Location | `scripts/rollout/semantic_mode_event_controller_v1.py`, `_v2.py` |
| Config | `configs/semantic_mode_event_controller_v1.json` |
| Tests | `tests/test_semantic_mode_event_controller_v1.py`, `_v2.py` |
| Contracts | `docs/semantic_mode_event_controller_contract_v1.md`, `docs/semantic_controller_independence_audit_v1.md`, `docs/b0_predicate_contract_v1.md` |
| Status | **Legacy. Not the canonical policy architecture.** |

v1 read `curr_skill` from the teacher skill machine. v2 removed that dependency
and derived mode/events from online observable predicates only, with runtime
counters making teacher-independence auditable.

**Are its conclusions still valid?** The independence audit is valid and
useful: v2 genuinely does not call a teacher policy. But
`docs/claim_and_evidence_matrix_v2.md` already records the remaining
limitation — v2's B0 predicates are simulator-oracle-dependent. Either way, v2
still *controlled* NAV/MANIP mode, which is precisely what the end-to-end
method removes.

**Canonical replacement.** `lmm/instrumentation/event_detector.py` keeps the
observable-predicate idea as pure observation. It has no action-producing API
and is proven non-interfering by test.

## B. Pre-place release boundary / suffix ranking

| | |
|---|---|
| Code | `scripts/rollout/pi0_ovmm_pre_place_release_boundary_probe.py` (117 KB), `pi0_ovmm_suffix_rank_probe.py`, `pi0_ovmm_task_suffix_rank_probe.py` |
| Artifacts | `../remote_results/pi05_pre_place_release_boundary/` |
| Date | 2026-07-29 / 07-30 |
| Root definition | Predefined semantic `PRE_PLACE` boundary |
| Status | Legacy |

Tested whether ranking candidate place suffixes from a semantic pre-place root
selects better outcomes. Superseded as a *frontier definition*: the root was
chosen by semantic position, not by measured continuation success.

## C. Event-aligned root bank v1 / v2

| | |
|---|---|
| Code | `scripts/diagnostics/capture_event_aligned_root_bank_v2.py`, `qualify_event_aligned_root_bank_v2.py`, `finalize_event_aligned_root_bank_v2_gate.py` |
| Config | `configs/branch_support/event_aligned_root_bank_*.json` |
| Docs | `docs/ROOT_BANK_CONFIRMATORY_RULE_V2.md`, `EVENT_ALIGNED_ROOT_BANK_V2_START_HERE.md` |
| Metadata | `../paper_data/server_pull_20260913/event_aligned_root_bank_v2_metadata/` |
| Result | `EVENT_ROOT_V2_POLICY_QUALIFICATION_FAIL` |
| Status | Legacy |

Built root banks aligned to semantic events (`E5`, pre-place). The `event-e5-*`
root ids that appear throughout the paper-closure data come from here.

## D. TRAIN branch-support pilot (the pivotal negative result)

| | |
|---|---|
| Config | `configs/branch_support/train_branch_support_pilot_v2.json` |
| Prereg | `docs/TRAIN_BRANCH_SUPPORT_PILOT_PREREG_V2.md` (frozen at commit `2d22bc7` **before** outcomes) |
| Summary | `LMM_CRITICAL_PATH_STATUS.md` |
| Result | **`BRANCH_SUPPORT_NOT_OBSERVED`** |

8 TRAIN roots across 5 scenes, 8 siblings each, Sobol seed 410021, root-only
noise. All 64 were valid physical runs: 64 real VLA inferences, 64 replans,
zero fallback, zero clipping, zero engineering errors, zero server restarts.

**All 64 terminated `released_outside`.** Successes 0, failures 64, mixed roots
0, rescue roots 0, pass@1/2/4/8 all 0.0.

Timing: env init 1261.8 s, snapshot restore 0.437 s, VLA inference 621.5 s,
suffix physics 23.8 s, terminal verification 0.006 s, total 1951.6 s.

**Still valid — and it is the strongest argument for the new method.** A root
bank with zero outcome variance gives group-relative RL nothing to learn from.
Semantic root selection cannot detect this in advance. `V_hat` estimation can,
and `lmm/rl/advantages.py` now refuses to train on such a bank rather than
silently producing zero advantages.

## E. Fresh-process determinism audit

| | |
|---|---|
| Docs | `docs/FRESH_POLICY_PROCESS_FIRST_DIVERGENCE_AUDIT_V1.md`, `PI05_NONDETERMINISM_SOURCE_AUDIT_V1.md`, `SHARED_PERSISTENT_POLICY_FRESH_SIM_PROTOCOL_V2.md` |
| Result | **`MODEL_FORWARD_BACKEND_DIVERGENCE`** |

TRAIN root `train-2001-held10-275` in three cold policy processes. Canonical
request, prompt/token ids, raw and preprocessed observation tensors, explicit
noise tensor, sampler key, config/norm/checkpoint identity and all parameter
leaves were exact; parameter-manifest SHA-256 identical in all processes.
First difference: **274/320 float32 values of the raw `(1,10,32)` model
output** in process 2.

Protocol V2 (one persistent stateless server, explicit root-only noise bytes,
serial requests, fresh simulator process per execution) then passed Matrix D
**40/40** across 8 exact root groups, including order-independence `A,B,C,A`.

**Still valid, and binding on the canonical pipeline.** It is why `Pi05Policy`
holds a client to an already-running server and never spawns one per
continuation. See `docs/PIPELINE.md` §Determinism contract.

Two retained pre-outcome engineering failures: running from the project cwd
raised `FileNotFoundError: data/default.physics_config.json`; using
`CUDA_VISIBLE_DEVICES=3` with `--gpu-id 3` raised an EGL device error. Valid
runs use home-robot as cwd and remapped GPU index 0.

## F. Frontier GRPO / online v1-v3 — **branch-only**

The frontier-RL implementation is **not on the default branch** and not on the
currently checked-out branch.

| Branch | Worktree | `scripts/frontier_rl/` |
|---|---|---|
| `diagnosis/frontier-conditioning-v1` | `../lmm_frontier_grpo_v1/` | 29 files |
| `diagnostic/shared-legacy-replication-v1` | `../lmm_shared_legacy_replication_v1/` | 36 files |
| `experiment/frontier-coverage-preserving-update-v1` | `../lmm_frontier_attribution_diagnostic_v1/` | 39 files |
| `main`, `diagnosis/event-reference-four-arm-v1` | — | **0 files** |

Key modules: `frontier_grpo_update_v1.py`, `frontier_grpo_update_v2.py`,
`run_frontier_grpo_online_v1.py`, `finalize_online_v3_protocol.py`,
`frontier_replication_core_v1.py`, `audit_frontier_grpo_chain_v1.py`.

Tests: `tests/test_frontier_grpo_math_v2.py`, `test_frontier_grpo_v1.py`,
`test_frontier_shared_legacy_replication_v1.py` (in those worktrees).

**Deliberately not duplicated into `legacy/`.** Copying 36 files out of a
branch would create a second, ambiguous authority. Git already preserves them,
and the worktrees keep them on disk and runnable. `diagnostic/shared-legacy-
replication-v1` is the most complete GRPO-math version — start there.

## G. FP8 / PTQ quantization (V1-V5)

Code and artifacts live at the **workspace root**, outside version control.

| Version | Verdict | Where |
|---|---|---|
| V1 | `FP8_PTQ_OFFLINE_FIDELITY_GATE_FAIL` (secondary `FP8_PTQ_TAIL_ERROR_NOT_LOCALIZED`) | `../experiments/fp8_rollout_scaling_v1/` |
| V2 | `FP8_PTQ_SELECTIVE_TAIL_REDUCED_BUT_GATE_FAIL` | `../FP8_PTQ_TAIL_LOCALIZATION_V2_REPORT.md` |
| V3 | `FP8_PTQ_RANKING_NOT_PRESERVED` | `../experiments/fp8_ptq_deployment_aligned_v3/` |
| V4 | `FP8_PREFILTER_ENGINEERING_BLOCKED` | `../fp8_prefilter_v4/` |
| V5 | prospective capture (see scripts) | `../fp8_prospective_v5/` |

Code: `../scripts/fp8_rollout_scaling/` (quantize, evaluate, servers, parity,
LoRA conversion).

**V1 numbers.** BF16 parameters 6,751,104,544 bytes; FP8 checkpoint
4,079,445,264 bytes; storage reduction **39.5737%**; coverage **80.2044%**
(2,707,342,848 / 3,375,552,272 params); symmetric E4M3FN, per output channel,
scale `max_abs/448`; full-load dequantisation 5.368 s. Matched offline metrics:
relative L2 0.07929, MAE 0.003760, cosine 0.99685, max abs error 1.76653,
clipping disagreement 1.25%.

Note the correction already recorded in the V2 report: V1 was **re-classified**
from `FP8_PTQ_UTILITY_DEGRADED` to `FP8_PTQ_OFFLINE_FIDELITY_GATE_FAIL`,
because rollout utility was never actually evaluated. That correction stands.

**V3 ranking numbers (directly relevant to Q3).** 4 roots x K=8 = 64 rollouts.
Mean Spearman rho **0.9107**, mean Kendall tau-b **0.7857**, mean top-4 overlap
**1.0**, top-1 exact agreement **50%**.

> **Important caveat.** In the V3 root summaries both `bf16_successful_candidates`
> and `q4_successful_candidates` are **empty** — there were no successes on
> either side. The correlations are therefore over a continuous proxy metric,
> not over task success. V3 is evidence about *proxy* ranking preservation
> only, and must not be cited as evidence that cheap rollout preserves
> success ordering. Q3 remains genuinely open.

**Hardware, still valid and binding.** A800 SM80,
`NATIVE_FP8_COMPUTE_UNSUPPORTED_ON_CURRENT_A800`. Runtime was
`FP8_STORAGE_BF16_COMPUTE` / `STORAGE_SAVING_ONLY`: eligible weights serialised
as E4M3FN plus scales, fully dequantised to BF16 at load. No native-FP8
compute, acceleration or runtime-VRAM claim was made — see
`docs/QUANTIZATION_BACKENDS.md`.

Also recorded: the official OpenPI JAX-to-PyTorch path **failed** the BF16
parity gate (relative L2 0.576, cosine 0.827), so the PyTorch backend was
rejected as a PTQ reference and an explicit-LoRA conversion did not fix it.

## H. Completion-aware online GRPO, 20 rounds (the headline negative result)

| | |
|---|---|
| Code | `../paper_data/server_pull_20260913/hf_recovery_20260914/expanded/code/online_driver_v5.py`, `canonical_train_success_v5.py`, `fp8_round_v5.py` |
| Earlier drivers | `.../solrl_fp8_frontier_v1_20260907/overnight/online_driver_v4.py`, `cont_train.py`, `canonical_train_rank_v2.py` |
| Training audit | `.../expanded/training_audit/` — rounds 001-020 |
| Evaluation | `.../expanded/paper_closure/single_execution_eval/` |
| Result | **`NO_RETAINED_IMPROVEMENT`** |

Frozen held-out single-execution evaluation, 4 roots x 4 seeds, no best-of-N,
engineering failures counted as binary failures:

| Checkpoint | Success | Rate | Mean utility | Mean progress | Mean terminal IoU |
|---|---:|---:|---:|---:|---:|
| q0 | 1/16 | 0.0625 | 0.2488 | -0.0062 | 0.0488 |
| q5 | 0/16 | 0.0000 | 0.0050 | 0.0251 | 0.0493 |
| q10 | 0/16 | 0.0000 | -0.0496 | -0.2479 | 0.0498 |
| q20_final | 0/16 | 0.0000 | -0.0863 | -0.4314 | 0.0514 |

Paired q20 vs q0, clustered on `root_id`: success root-mean delta `-0.0625`,
cluster bootstrap 95% CI `[-0.1875, 0.0]`, exact McNemar p `1.0`, contingency
`both=0, neither=15, q0_only=1, q20_only=0`. Utility delta `-0.3350`, CI
`[-0.7786, -0.0012]`.

**Still valid.** 20 rounds of online GRPO from semantic roots did not retain
improvement, and on shaping metrics degraded monotonically with round count.
Total execution: `46 h 43 min` for 20 rounds (median ~1.99 h/round, one 7.80 h
stall between rounds 5 and 6), plus ~2 h 50 min for held-out evaluation,
temporal accounting and the cascade benchmark.

Do not re-run this hoping for a different sign. The recommended path is the
method change, not a repeat.

## I. Four-arm event-reference diagnostic (most recent)

| | |
|---|---|
| Branch | `diagnosis/event-reference-four-arm-v1` (currently checked out) |
| Code | `scripts/diagnostics/` four-arm runner + aggregation |
| Config | `configs/diagnostics/event_reference_four_arm_v1.json` |
| Docs | `docs/FOUR_ARM_PROTOCOL.md` |
| Status | In progress at cleanup time; frozen protocol, read-only objective audit |

## J. Infrastructure loss and recovery (2026-09-14)

The A800 server was lost. Recovery queried the Hugging Face API directly rather
than trusting local manifests.

**Recovered:** q20 BF16 model, q20 optimizer master state, q4 final model
(sizes and LFS SHA-256 verified against the final backup manifest); a
40,571,668-byte metadata/code archive, SHA-256
`b4a4017178ddf21a51bf5cde238695d7e6428710f6e049a7f42f2407ca930be5`, 5,667
members (16 run-local code files, 496 paper-closure files, 5,154
training-audit files).

**Not recovered:** exact q0 / q5 / q10 weights; round 002-020 intermediate
checkpoints; raw `.npz` files; videos; the full container/conda environment;
proof that simulation datasets, asset mounts and root-state binaries were
backed up.

**Consequence:** training can resume from the q20 boundary, but the original
20-update checkpoint trajectory **cannot be replayed exactly**, and the frozen
q0/q5/q10 matrix cannot be re-executed. The public
`Neptune-T/LMM-GRPO-Handoff` q0 (LFS hash `1caf9813...`) is **not** the frozen
q0 of this run (`64bbbeaf...`) and must not be substituted.

**Two open risks carried forward:**

1. The HF repo was found to return `private=false` on both authenticated and
   anonymous API calls on 2026-09-14, despite `Private` in its name and a
   backup manifest recording it as private. If that was not intentional it
   should be changed in the HF settings.
2. The three large objects (~18.5 GB) exist as a single online copy. The local
   volume has **4.9 GB free**, so a second physical mirror cannot be made here.

## Not core contributions

Preserved where implemented, but not part of the canonical method: adaptive
bidirectional curriculum, moving frontier curriculum, continual-learning
replay, landing-zone regularization, adversarial skill chaining, automatic
semantic task decomposition, SFT consolidation, complex root archive
replacement, full multi-stage reverse curriculum.
