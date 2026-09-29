# AlphaBot 2 canonical camera posture (Stage 1)

2026-09-29, Isaac Sim 4.5, zero-delay app (see
[`ALPHABOT2_OBSERVATION_TIMING_CONTRACT.md`](ALPHABOT2_OBSERVATION_TIMING_CONTRACT.md)).

## Decision

| Joint | Canonical value | Note |
|---|---|---|
| `head_joint1` (neck yaw) | **0.0 rad** | symmetric for the two arms |
| `head_joint2` (neck pitch, + = down) | **0.20 rad** | head camera optical axis 41.4° below horizontal (29.9° at 0) |
| `body_joint1..4` (torso) | 0.0 (unchanged) | torso fixed, as required |

This is `simworld.CANONICAL_HEAD = (0.0, 0.20)`. It is set by teleport and
held by the joint drives at every reset. The neck is **not** part of the policy
action (CartesianMobileAction18 has no neck dims), and the same pose is used
for demonstrations, SFT data and evaluation. The measured `head_joint2` range is
[−0.401, +0.384] rad; 0.20 leaves 0.18 rad of margin.

## Method (`scripts/alphabot2_isaac/stage1/camera_posture.py`)

A bounded sweep over the neck only, with `head_joint1 = 0` and `head_joint2 ∈ {0, 0.1, 0.2, 0.25, 0.3, 0.33, 0.36, 0.38}`.
The head camera (policy key `camera_top`) is scored. A sampled 3-D point counts as visible if
it projects inside the 320×240 image and the rendered depth at that pixel is
not more than 3 cm in front of it, so occlusion by the robot's own body counts.
The depth comes straight from the annotator. A first version read the
callback-cached `get_current_frame()` depth, which can hold an earlier state; it
gave contradictory numbers and was discarded.

| Category | Robot state | Points |
|---|---|---|
| navigation floor | 1.25 m from table A, facing it | floor corridor 0.6–3.0 m ahead, ±0.8 m |
| navigation table | same | table-A top grid |
| approach (0.6 m, 0.3 m before the grasp standoff) | facing the table | near half of the table top, object top face, receptacle rim/floor |
| manipulation workspace | at the grasp standoff | table-top grid over the arms' reach (chassis x 0.30–0.65, y ±0.45) |
| object / receptacle / both arms | at the standoff, both arms at pre-grasp and at grasp height | object top, box points, pad midpoints + gripper bases of BOTH arms |

Rule: among poses at least 0.02 rad inside the joint limit, maximize the worst
scored category, then the mean, then prefer the smaller pitch. A category no
admissible pose can see (max < 0.1) cannot discriminate between poses; it is
reported instead of scored.

## Results (fraction of points visible)

| `head_joint2` | pitch | nav floor | nav table | appr. 0.6 table / object / box | appr. 0.3 table / object / box | manip. workspace | worst | mean |
|---:|---:|---:|---:|---|---|---:|---:|---:|
| 0.00 | 29.9° | 0.30 | 0.91 | 0.89 / 1.00 / 0.94 | 0.00 / 0.00 / 0.17 | 0.00 | 0.00 | 0.47 |
| 0.10 | 35.6° | 0.41 | 0.89 | 0.89 / 1.00 / 0.89 | 0.39 / 1.00 / 0.78 | 0.00 | 0.00 | 0.70 |
| **0.20** | **41.4°** | **0.51** | **0.89** | **0.89 / 1.00 / 0.89** | **0.86 / 1.00 / 0.94** | **0.14** | **0.14** | **0.79** |
| 0.25 | 44.2° | 0.48 | 0.89 | 0.89 / 1.00 / 0.89 | 0.71 / 1.00 / 1.00 | 0.14 | 0.14 | 0.78 |
| 0.30 | 47.1° | 0.48 | 0.94 | 0.89 / 1.00 / 0.89 | 0.71 / 1.00 / 1.00 | 0.11 | 0.11 | 0.78 |
| 0.33 | 48.8° | 0.48 | 0.94 | 0.89 / 1.00 / 0.89 | 0.64 / 1.00 / 1.00 | 0.06 | 0.06 | 0.77 |
| 0.36 | 50.5° | 0.44 | 0.94 | 0.89 / 1.00 / 0.89 | 0.68 / 1.00 / 0.89 | 0.06 | 0.06 | 0.76 |
| 0.38 | 51.7° | 0.41 | 0.89 | 0.89 / 1.00 / 0.89 | 0.61 / 1.00 / 0.83 | 0.06 | 0.06 | 0.73 |

At zero neck (the bootstrap posture), the head loses the table, the object and
the box during the last 0.3 m of the approach and never sees the workspace. At
0.20 the head keeps the table, object and receptacle in view through navigation
and the whole approach, and still sees the table and both grippers at the
standoff.

**Structurally invisible at the grasp standoff (every admissible pose):** the
object, the receptacle and the grippers at grasp height. The points are in the
frustum but behind the robot's own torso/shoulders (in frustum 1.0, unoccluded
≈ 0.0 at high pitch), or just below the frame at lower pitch. With the torso
fixed, no neck pose shows the grasp itself. **The wrist cameras carry the final
centimetres of manipulation**, as on the real robot (its wrist D405s look along
the gripper, fingers at the image bottom).

## Other cameras (not adjustable; recorded for the record)

* **Chest camera** (`camera_front`, `body_link4/front_camera`, 22° down): torso
  fixed, so it is constant. It sees the table from a distance and the floor;
  manipulation-workspace visibility is 0.0 at the standoff. It is kept as a
  policy input only because the public checkpoint was trained with four views.
* **Wrist cameras** (`camera_left` = right wrist, `camera_right` = left wrist):
  at the home pose ~47–52 % of each wrist image is black. Instance
  segmentation shows these pixels have **no prim** (id 0): they are the **sky**.
  With the arm at home the D405 looks almost horizontally (optical axis
  z ≈ 0.01), and it is mounted beside the gripper, so the horizon is a vertical
  line through the image centre, and the scene has no dome or sky, so
  everything above it renders black. This is not an occlusion or a mount bug.
  The near clip (2, 5, 8 cm) makes no difference. When the gripper points down
  for manipulation, the table and object fill the view (see the pilot frame).

## Render noise (measured, kept)

Every camera shows speckle and streak noise from the RTX real-time renderer at
320×240. Two consecutive renders of a static scene differ by 10–11 / 255 on
average. None of these changed it: DLSS mode (perf / quality / DLAA / TAA /
FXAA / none), hard sun, sampled lighting off, GI/AO/reflections off, three
extra renders per observation, auto-exposure off, or a dome light (worse:
14.9). Path tracing with the OptiX denoiser is clean (speckle 0.002 vs 0.019),
but it costs 1.32 s per action vs 0.25 s, i.e. 5×
(`scripts/alphabot2_isaac/stage1/render_quality.py`,
`artifacts/alphabot2_stage1/render_quality*.json`). The real-time renderer is
kept, exactly as in the bootstrap. The noise is identical in demonstrations and
in evaluation, so it acts as sensor noise, not as a train/test mismatch; it is
a sim-fidelity limitation, recorded here.
