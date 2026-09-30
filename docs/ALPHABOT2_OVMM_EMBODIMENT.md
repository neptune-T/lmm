# OVMM tasks/scenes with an AlphaBot 2 dual-arm mobile embodiment

Status: in progress (2026-09-30). This document records what was built, what
was measured, and every deviation from standard OVMM caused by the embodiment.
Isaac Sim remains the embodiment reference ([`ALPHABOT2_PI05_EXECUTION_CONTRACT.md`](ALPHABOT2_PI05_EXECUTION_CONTRACT.md),
[`ALPHABOT2_OBSERVATION_TIMING_CONTRACT.md`](ALPHABOT2_OBSERVATION_TIMING_CONTRACT.md)).
Habitat / HomeRobot OVMM is the task simulator.

## 1. Robot description (`lmm/embodiments/alphabot2/habitat/build_urdf.py`)

| Item | Value |
|---|---|
| Sources (pinned) | `fiveages-sim/robot_descriptions@bd491829` (ai2_bot2, aidal, rm75), `fiveages-sim/robot-descriptions-common@92fa1ebf` (inspire EG2-4C2, sensor_models) |
| Expansion | `robot.xacro type:=inspire`; the xacro has the true mirrored right arm; the Isaac USD reuses left-arm joint definitions |
| Articulated | 14 arm joints; 2 grippers x (1 actuated + 5 mimic joints) = 26 |
| Fixed (v1) | torso `body_joint1..4` at the canonical crouch (§4); neck at (0, 0.20); wheels/casters (the base moves kinematically, OVMM convention) |
| Geometry | 42 collision + 59 visual elements, all from the xacro; nothing invented |
| Habitat orientation | root link `habitat_base` + fixed joint rpy (-pi/2, 0, 0) |

**Orientation, measured live.** `habitat_sim` does not convert a ROS z-up URDF: with an identity
root the robot lies on its side (head at world y = 0), which was the WXAI failure. habitat-lab's
`base_rot` setter overwrites the root rotation with a pure yaw, so a runtime correction would
be undone. The correction is therefore part of the model. Result: upright, head camera at
y = 1.582 m (torso 0), forward +x, left -z.

**Link frames.** Habitat puts a link's scene node at the link's *inertial* frame. Link frames are
recovered through the URDF inertial origins; the error is 0.0 on all probed links, versus 0.144 m
on `base_link` without it.

## 2. Robot class and action (`alphabot2_robot.py`, `alphabot2_actions.py`)

* All joints and links are resolved **by name**, and the joint-position offsets are checked
  against the table measured live. Any reordering raises.
* 16-D proprio `[left 7, left closed fraction, right 7, right closed fraction]`. The URDF joints
  *are* the vendor/xacro joints, so no sign/offset map is needed here; Isaac needs one because
  the USD differs.
* Gripper: joint 0 = closed, 0.82 = open, fraction f = (0.82 - q)/0.82; the Isaac convention.
  The five mimic joints are driven kinematically, because Bullet ignores `<mimic>`.
* Cameras: head = the xacro `head_camera` optical frame; wrists = the xacro D405 origin on
  `link8`, aimed as in Isaac. habitat-lab binds sensors to camera prefixes in `__init__`, before
  name resolution is possible. They are rebound in `reconfigure` (WXAI's cameras were very likely
  never attached to their links).
* **One policy action = one `env.step`.** `AlphaBot2JointMobileAction` stages, in a single
  step: both arms (kinematic joint targets, clamped to the xacro limits and reported), the base
  (v, omega), then both grippers. Measured: **0.1000 s of simulation = 20 x 0.005 s physics
  substeps per action**. `EmbodiedTask.step` always adds `step_physics(1/physics_target_sps)`
  (3-4 extra substeps at habitat's default); `ac_freq_ratio = 19` with `physics_target_sps = 200`
  makes the total exact.
* Base: policy units m/s and rad/s, integrated over 0.1 s in 5 pieces. Each piece is filtered by
  the navmesh with sliding (`pathfinder.try_step`); there is no [-1, 1] normalization.
* The joint-space interface (`JointMobileAction18`) is the Habitat action. π0.5's
  `CartesianMobileAction18` is lowered by the Isaac-qualified `CartesianExecutor` before it.

## 3. Isaac <-> Habitat semantic consistency

| Check | Result |
|---|---|
| Arm FK (URDF vs `kinematics.fk`, 200 random q per arm, CPU) | <= 2.5e-16 m / 1.5e-16 rad |
| Arm FK live in Habitat (14 joints moved one at a time) | 2.7e-6 m / 5.9e-7 rad |
| Joint limits | identical to the vendor limits |
| Gripper link separations at the 5 Isaac-measured joint values | identical (0.0362 ... 0.1016 m) |
| Wrist D405 / gripper_base at home (vs Isaac) | 0.5 mm / 3.5 mm (the known USD↔xacro offset) |
| Base: +v forward, -v back, +omega turns toward the robot's own left | +-0.15000 m, +-0.2500 rad over 0.5 s |

**Discrepancy recorded.** The EG2-4C2 *pad-surface* gap from the xacro meshes is 1.4 cm (closed)
to 8.0 cm (open); the Isaac "pad separations" are link-origin distances. In Isaac a 5 cm cube was
held at a link separation of 0.047 m, i.e. the USD finger geometry differs from the xacro one.
Habitat uses the xacro geometry.

## 4. Torso posture (decision; reversible)

With the torso at 0 the shoulders are at 1.44 m. Both arms then reach grasp points only 0.65-1.2 m
above the floor, which covers **17 % (val) / 37 % (train)** of OVMM episodes by pick+place height.
The AlphaBot 2 torso is a knee-style lift. A fixed, upright crouch `(-a, 2a, -a, 0)`:

| a | shoulder height | reach band | val | train |
|---|---|---|---|---|
| 0 | 1.44 m | 0.65-1.2 m | 17 % | 37 % |
| 0.6 | 1.33 m | 0.55-1.2 m | 25 % | 59 % |
| 1.0 | 1.14 m | 0.40-1.2 m | 51 % | 94 % |
| **1.1 (canonical)** | ~1.09 m | ~0.35-1.2 m | | |

The torso is fixed at **(-1.1, 2.2, -1.1, 0)**, 0.087 rad inside the body_joint1 limit. It is
never a policy input or action. The arm kinematics, frame B and the CartesianExecutor are
unchanged: they are defined in the torso-attached **arm frame A** (`kinematics.arm_frame_in_chassis`),
which equals `base_link` at torso 0. At this posture A is a pure translation of (-0.104, 0, -0.359) m.

## 5. Navigation

* Footprint (URDF collision geometry): chassis turning radius **0.378 m**. With the crouch the
  arms at home reach 0.411 m; the navigation tuck, 0.10 rad from home, stays at 0.369 m.
* **Navmesh radius 0.38 m** (Stretch 0.30), height 1.5 m (the robot is 1.43 m tall when
  crouched). The navmesh is recomputed and cached per robot class, so Stretch's is untouched.
* Radius audit (24 val episodes, 10 scenes, arms tucked, torso 0):

  | radius | goals unreachable at reset | both legs arrived | both legs collision-free | worst penetration |
  |---|---|---|---|---|
  | 0.38 | 7 | 12 | 11 | 1.1 cm |
  | 0.42 | 13 | 8 | 8 | 0 |
  | 0.46 | 17 | 7 | 7 | 0 |

* Follower: Detour shortest paths hug the navmesh edge. OVMM's `allow_sliding: False` filter
  stopped the base dead when it grazed that edge (measured: blocked after about 1 m). The
  AlphaBot base slides along the edge instead, which already keeps the chassis radius from real
  geometry. A differential-drive segment follower turns in place at path vertices. Every leg that
  had a path arrived (arrival ≈ 4.6 cm median), path efficiency ≈ 1.00, and OVMM's own
  `nav_to_pick` / `nav_to_place` success held on every arrived leg.

## 6. Grasping (dual gripper)

* Two `RearrangeGraspManager`s, index 0 = left hand and 1 = right hand; the active index follows
  the hand that grasps, so OVMM's single-object measures follow it. A hand cannot take an object
  the other hand holds.
* Kinematic magic snap: simulator infrastructure. It is logged as an event and never enters
  the observation. It fires only if the pad midpoint is inside the object's box (+5 mm) and the
  object's width across the fingers lies within the measured pad gap. The fingers stop at the
  object width.
* Fix: in kinematic mode habitat never sets `keep_T`, so `update_object_to_grasp` teleported the
  object's origin onto the pad midpoint, with the hand's orientation, every step. The object's
  pose in the hand is now recorded at grasp time.
* Fingertips reach 2.85 cm past the pad midpoint, so a top grasp keeps the pad midpoint
  2.85·cos(tilt) + 0.6 cm above the support. Objects lower than about 3 cm are infeasible.
  The rim pinch of flat objects is disabled: the lower finger would pass through the support
  (22-33 contact steps measured).

## 7. Observation timing (Habitat)

`scripts/alphabot2_habitat/timing_sync.py`. Robot motion (base turning, both arms swinging) was
re-rendered from the recorded states: head and both wrists matched exactly at offset 0 in 28/28
observable steps, and proprio equals the state. A physics-falling object with OVMM's
`concur_render: True`: 0/6 frames in sync (0.1 s stale). With `concur_render: False`: every
frame re-renders exactly (error 0.000) at offset 0. AlphaBot uses `concur_render: False`.

## 8. OVMM compatibility

**Unchanged:** HSSD scenes, OVMM episode files and object/receptacle semantics, task language
(object category, start/goal receptacle categories), start-pose sampling, target objects and goal
receptacles, and every OVMM measurement (phase successes, `ovmm_place_success` with its stability
check, navigation measures, hold-constraint drop rule).

**Embodiment-specific, documented:**
1. Robot: AlphaBot 2 URDF, 16-D proprio, one 18-D atomic action (replaces Stretch's `arm_action`,
   `base_velocity` waypoint teleport and the 10-D joint sensors).
2. Navmesh radius 0.38 (not 0.30), height 1.5; OVMM start poses that are not navigable for AlphaBot
   are moved to the nearest navigable point (distance recorded); episodes whose goals are
   unreachable on this navmesh are infeasible for this embodiment.
3. Base filter slides (`try_step`), where OVMM uses no sliding for the discrete agent.
4. 10 Hz continuous control: max episode steps 3000 (300 s), because Stretch's 1250 counted coarse
   teleport steps.
5. `concur_render: False`; 20 physics substeps of 0.005 s per step.
6. Magic grasp restricted to what the EG2-4C2 can physically close on (width 1.6-7.2 cm, height >= 3 cm).
7. Fixed crouched torso (§4).

**Geometric feasibility screen** (`feasibility_screen.py`; gripper + reach only, never outcomes):
**6,331 / 37,477 train episodes feasible (38/38 scenes)**; 133 / 1,199 val. Aperture is the
binding constraint (stuffed toys, ramekins, tape, shoes, butter dishes are too wide).

## 9. Qualification status

| Item | Status |
|---|---|
| Embodiment smoke (15 checks + FK) | **16/16** (torso 0 and crouched) |
| Navigation-only qualification | passes on feasible episodes (§5) |
| Manipulation (left / right / auto) | in progress: planner fixed after train diagnostics |
| Snapshot/restore | implemented (`snapshot.py`, `snapshot_qual.py`); not yet run |
| Head posture in OVMM scenes | sweep implemented (`camera_sweep.py`); running when the host dropped |
| Pilot (10-20 trajectories) | not started |
