# Manipulation frontier feasibility pilot (written 2026-10-19, before any pilot rollout)

Frozen: v2-B (sha256 405f2d5d…) and the PA-OG + base-lock default contract (pa_eval.py: MANIP1_MAX 600, MANIP2_MAX 500,
settle 120, HOLD_GAP 0.05, TUCK_RAD 0.30, HOLD_K 3, H_exec 5, render schedule FRAME_EVERY 2). No SFT, RL, FP4.
Code: `scripts/alphabot2_habitat/frontier_pa.py` (source + rollout), `frontier_plan.py`, `frontier_analysis.py`,
`frontier_chain.sh`; outputs `/workspace/runtime/runs/v2/frontier_pilot/`.

**Data boundary.** Only v2 TRAIN-manifest units on the OVMM train split (the frozen train eval units, units_train20)
are used for sources, roots and frontier selection; frontier_pa.py refuses any other unit. Held-out (v2 SFT-val)
episodes and OVMM-val never enter.

**Determinism.** Policy servers run with `--max-batch 1`; every response must report batch_size 1 (else the job
errors). Sources log every side-effecting simulator call in order (step / render / contact query) plus a controller
snapshot at every policy query (phase, phase step, condition count, executor IK seed + quaternion memory, commanded
grippers, grasp rotation, history-frame events, seed base). Restore = fresh reset + fix_start + exact replay of that log
up to the root's query; history frames are re-rendered at their original events.

**Sources and roots.** Sources: PA episodes on units_train20 × noise seeds 100 and 101 (policy-visited states).
Roots: policy query boundaries in a policy phase, labelled (privileged per-step logs, selection only):
grasp (MANIP1, not held, nearest pad within 12 cm), lift (held, < 5 cm up), tuck (held, lifted, an arm > 0.3 rad from
the tuck), place (MANIP2, held, last query ≥ 10 steps before the source's release; or 15 steps into MANIP2).
Quota grasp 8 / lift 6 / tuck 6 / place 10 (fill from other subtasks if short), at most one per (source episode,
subtask), at most 3 per unit, round-robin over units. Selection is deterministic from the source traces only.

**Gates (in order).**
1. Restore check: per root, noise −1 (the source's own seeds) must reproduce the source episode bitwise (event
   sequence and every executed joint action, whole episode) and its OVMM funnel. Only passing roots are swept; fewer
   than 24 passing roots stops the pilot.
2. Matched determinism: noise 0 re-run for 6 roots (one per subtask first) must equal the sweep's noise-0 rollout bitwise.

**Sweep.** 8 fixed noise indices per passing root (query seed base shifted by 1,000,003·7,919·(j+1)), continuing the
frozen controller from the root to the episode end. Outcome = OVMM `ovmm_place_success` at the end; also carry-ready
reached, MANIP2 reached, release in MANIP2, place phase.

**Reported.** Per root successes / 8; mixed roots (0 < k < 8), all-success, all-fail; by subtask and by source outcome;
full-success rate; effective positives = successes inside mixed roots (non-zero advantage) and their yield per rollout
and per allocated GPU-hour (allocated GPUs × wall-clock span per stage; client time and server inference GPU-seconds too).

**"Frontier exists"** (pre-registered): at least 6 mixed roots spread over at least 4 units, with at least 2 subtasks
having a mixed root. "Weak": 3–5 mixed roots or a single subtask. "Absent": ≤ 2 mixed roots (all-success / all-fail
dominate). Reported with the numbers; no RL, SFT or FP4 is started from this result.
