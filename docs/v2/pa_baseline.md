# Planner-assisted baseline (started 2026-10-19; written before any M-SFT or PA result)

Goal: reproducible non-zero OVMM full-task success. NAV is no longer required to be learned from RGB.

## 1. Manipulation from expert arrival (diagnostic, v2-B unchanged)
`scripts/alphabot2_habitat/manip_plan.py` / `manip_from_arrival.py` / `manip_run.sh`, outputs
`/workspace/runtime/runs/v2/manip_arrival/`. 20 v2-TRAIN units + 20 v2 SFT-VAL (held-out episode) units, one replica per
base episode. State = validated fresh-reset replay of the accepted expert episode to
- `obj`: first frame of `reach` (expert NAV to the pick standoff finished, nothing held), 600 policy steps;
- `recep`: first frame of `reach_place` (expert NAV to the place standoff finished, object held), 500 policy steps.
Per (unit, entry): `expert_rest` (replay of the expert's remaining actions) + v2-B policy seeds 0, 1 (fp32, H_exec 5,
history = the expert episode's own head frames at t-15/-10/-5). Restore check 2 cm / 1 deg.
Outcomes: obj -> grasp (target snapped), lift (held, >= 5 cm above its start height), carry-ready (held and both
arms within 0.3 rad of the NAV tuck, rescored from traces), held at end; recep -> release, OVMM place phase (ever),
ovmm_place_success at end / ever.

First rows (11:23-11:45): restore error 0.0 cm / 0.0 deg; expert_rest completes full success from both entries;
policy grasps from `obj` in 4/6, places from `recep` 0/7 (hovers over the receptacle without opening, then turns away
and drops on the floor; or releases at the receptacle edge -> object falls). v2-B's sampler gave the place segment
(reach_place..retract) ~0.23 effective epochs.

## 2. Minimal manipulation fix: M-SFT v1 (`configs/alphabot2_pi05/sft_msft_v1.yaml`)
From v2-B best (never modified), same data / contract / loss / history; navigation, tuck and settle frames weight 0;
place stages up-weighted (reach_place 1.5, lower 1.5, release 2, grasp 1.5, others 1); 3000 steps x 16 (48k samples,
30 % of v2-B's budget), lr 1e-5 peak -> 2.5e-6, seed 1001, single GPU (GPU 2). Evaluated checkpoint: `last` (v2-VAL loss
is dominated by navigation frames, so best-by-val would not select for manipulation). Its NAV quality is irrelevant
here: the planner navigates.
Evaluation: the same 240 manip_from_arrival jobs (paired with v2-B per unit / entry / seed).
"M-SFT fixes manipulation" if recep place success (ever) rises from v2-B's rate to >= 25 % of valid policy jobs and obj
grasp does not drop by more than 10 points. No further SFT round before the PA results are reviewed.

## 3. Planner-assisted full task: PA-OG (`scripts/alphabot2_habitat/pa_eval.py`, `pa_run.sh`)
LABEL: planner-assisted, ORACLE GOAL. Not an end-to-end result. Privileged inputs are confined to the planner's
goals (target object pose / goal receptacle surfaces + navmesh + IK feasibility: the expert's standoff planner). The
policy's inputs (3 RGB, 16-D state, instruction, head history) and the 18-D CartesianMobileAction18 interface are
unchanged; every phase switch uses observable signals only:
NAV1 planner -> (planner arrival) -> MANIP1 pi0.5 -> (holding: gripper commanded closed and blocked > 0.05 short of
the command; both arms within 0.3 rad of the NAV tuck; 3 consecutive steps; timeout 600) -> NAV2 planner (arms to the
exact tuck, place standoff planned from the object as held) -> (arrival) -> MANIP2 pi0.5 -> (not holding and arms
tucked, 3 steps; timeout 500) -> hold still 120 steps -> episode end. Funnel from OVMM's own measures at episode end:
find, grasp (pick phase), carry (find-receptacle phase), place (place phase), full = ovmm_place_success.
Sets: the frozen v2 eval sets (OVMM-val 30 units, split val; train set 20 units), seeds 0 and 1 (seed = noise
seed of the policy; the planner is deterministic). Success is reported as success_count / eval_count per seed
pass, with videos and traces of every success. A success is "replayable" when the recorded episode is re-run from the
same unit and seed and succeeds again.
