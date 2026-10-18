# v2 NAV decision rules (written 2026-10-17 ~19:30, before B's closed-loop eval and before any B/A paired result)

Analyses: `scripts/alphabot2_pi05/nav_causal.py` (paired NAV, unified onset, drift order), `rescore_trigger.py`,
`stall_diag.py` (15 cm reach-like arm threshold; 10/15/20 cm sensitivity). Inputs are the frozen v2 eval
(A, B, sanity_v1; H_exec 5, fp32) and the teacher-forced probe (`runs/v2/trigger`). No model / config changes.

## 1. B vs A NAV (preset effect-size targets, NOT significance tests)
Same units and noise seeds (train set + OVMM-val seed 0 traces; find from all eval rows). B "clearly better" only if
it beats A on >= 3 of 4 and is not worse on the 4th:
- closed-loop early-reach onset rate (far NAV, 15 cm) <= 1/2 of A's
- median stall fraction >= 0.15 lower
- median closest distance >= 1.0 m smaller
- reached-within-1 m or find: strictly more episodes, and >= 3
Always reported with the paired per-episode statistics (n, better/worse/tie, median diff, bootstrap 95 % CI of the
mean diff, exact sign-test p). A target met with a CI spanning 0 / sign test p > 0.05 is reported as "target met,
not statistically supported"; small counts (e.g. 3/30 approach) are never sufficient on their own.

## 2. Task-progress representation vs recovery data (only if B is not clearly better)
Unified onset statistic: first early-reach per eligible far-NAV query, teacher-forced (TF, expert observations)
and closed loop (CL) identically; CL exposure split by path state at each query; drift ordering per episode.
- Task-progress first: TF rate (training units) >= 1/2 of the CL on-path rate, AND >= half of the CL first triggers
  have no drift before them (no >= 1 m path deviation before the trigger).
- Recovery data first: CL off-path rate >= 3x both the TF rate and the CL on-path rate, AND >= 2/3 of CL first
  triggers are preceded by >= 1 m drift at least 50 steps earlier.
- Must hold at 15 cm and not flip at 10 or 20 cm. Otherwise: INCONCLUSIVE, reported with the numbers; no
  attribution forced, and never both interventions at once.

## 3. Matched-state perturbation (v2-B), written 2026-10-18 before any perturbation rollout
Design: `scripts/alphabot2_habitat/perturb_plan.py` (20 units: 12 training-set eval units + 8 held-out v2 SFT-val
episodes, at most one replica per base episode (amended before any rollout: the first plan had 5 replicas of one
held-out episode); 2 expert navigate_to_object states each, >= 3.5 m from the expert stop point; levels none / pos 0.25,
0.5, 1.0 m lateral / yaw 15, 30, 60 deg; seeds 0, 1; matched noise across levels), `perturb_eval.py` (validated
expert replay -> safe perturbation -> 400-step B rollout, fp32, H_exec 5), `perturb_analysis.py`. A replay error
> 2 cm / 1 deg or an unsafe level is skipped and reported, never replaced.
Primary outcome: false reach (far-NAV stop + arm > 15 cm) within the 400-step rollout; 10 / 20 cm sensitivity.
- DEVIATION IS A MAIN CAUSE -> expert-corrected recovery data / DAgger pilot, if on at least one axis:
  (a) Cochran-Armitage trend p < 0.05 with false reach increasing with the perturbation, AND
  (b) largest level vs none, matched on (state, seed): more perturbed-only than none-only false reaches, sign p < 0.05, AND
  (c) the none-level false-reach rate <= 1/2 of the largest level's;
  holding at 15 cm and not reversing at 10 / 20 cm.
- DEVIATION IS NOT THE MAIN CAUSE -> reconsider task-state representation, if: the none-level false-reach rate is
  >= 2/3 of every perturbed level's rate on both axes AND no axis has a trend p < 0.05.
- Otherwise INCONCLUSIVE (reported with the numbers; no forced attribution, never both interventions at once).
Secondary (reported, not decisive): early false reach (< 100 steps), unified hazard, base-only stall, NAV progress,
reached-within-1 m of the stop point, recovery rate, share of stall time explained by false reach.
Amendment (2026-10-18, after the 14-job engineering pilot, before the 560-job run): conditions (a)-(c) and the
"not the main cause" rule are evaluated PER AXIS on the COMMON SUBSET of (state, seed) whose four levels of that
axis all have a valid rollout (pos and yaw analysed separately; no feasibility-selection bias between levels);
an axis with < 10 common (state, seed) pairs is INCONCLUSIVE. Effective / feasible n per level, the side used for
lateral pushes, and the within-pair matched slopes (false reach, stall, base-only stall, progress) are always
reported; the all-available analysis is secondary. Infeasible levels are skipped, never replaced.

## 4. DAgger recovery SFT pilot (user-approved 2026-10-18), written before any pilot training
Data: expert-recovery demonstrations re-recorded with full observations from the 18 TRAIN-unit roots only
(`recovery_collect.py`; held-out roots refused in code), each with B's real 15-step pre-root prefix (history
context, sampling weight 0); anchors = recovery_navigate + recovery_reach_onset (first 15 frames of the arrival
switch); tuck / post-arrival manipulation weight 0. Normalization frozen to v2's exact stats (B's).
Arms, both initialised from v2-B best (`/dev/shm/lmm_runtime/sft/v2B/best`, sha256 recorded before/after, never
modified), identical config, seed 1000, budget 2000 steps x effective batch 16, lr 5e-6 peak (100 warmup) cosine to
2.5e-6, B's loss (w=1) and history:
  CONTROL     v2 train data only, B's sampler
  RECOVERY    v2 train + recovery episodes; v2 frames keep B's relative weights; recovery anchors get 20 % mass
Evaluated checkpoint: `last` of each arm (fixed budget; v2-val loss contains no recovery states, so best-by-val
selection would be biased). fp32 serving, H_exec 5.
PRIMARY: 18 HELD-OUT recovery roots x 2 noise seeds (36 paired continuations per arm, 600 steps, same roots / seeds as
the B baseline): approach (target <= 1 m), false reach, recovery success (approach without a prior false reach).
"Recovery data improves closed-loop recovery" if RECOVERY beats CONTROL on approach in the (root, seed)-paired
comparison with exact sign test p < 0.05, and false reach is not higher; otherwise "not shown" (an improvement with
p >= 0.05 is reported as a trend only).
SECONDARY: training-unit roots (in-distribution of the recovery data; reported separately, never pooled with held-out);
frozen normal-init OVMM-val (30 units x 2 seeds) + training set (20) funnel incl. full success_count/eval_count, run
after the root eval, to check that a local recovery gain does not leave full success at 0 unnoticed or regress find.
Expansion of collection, task-progress heads, RL or FP4 are out of scope until the results are reviewed.
