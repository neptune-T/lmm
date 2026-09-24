# Ai2 Bot2 asset ↔ AlphaBot 2 kinematic audit

Static audit, extracted from the xacro at the pinned revisions in
[`ALPHABOT2_SIM_ASSET_PROVENANCE.md`](ALPHABOT2_SIM_ASSET_PROVENANCE.md).

> **Static only.** Nothing here was verified in a running simulator — Isaac Sim
> is unreachable (see [`ALPHABOT2_ISAAC_BLOCKER.md`](ALPHABOT2_ISAAC_BLOCKER.md)).
> Every row is "what the description file declares", not "what the physics did".

## Summary

| Component | Asset | Official AlphaBot 2 | Verdict |
|---|---|---|---|
| Left arm DoF | 7 revolute (`left_joint1..7`) | 7 | **MATCH** |
| Right arm DoF | 7 revolute (`right_joint1..7`) | 7 | **MATCH** |
| Two-finger gripper per side | Inspire EG2-4C2, 1 actuated joint | two-finger | **PLAUSIBLE** |
| Mobile base | 2 driving + 4 caster wheels, differential | six-wheeled, differential | **MATCH** |
| Torso / waist | 4 revolute (`body_joint1..4`) | lifting waist, 0–240 cm | **PLAUSIBLE** |
| Neck / head | 2 revolute (`head_joint1..2`) | head present | **MATCH** |
| Cameras | head, front(chest), back, chassis, 2× wrist D405 | head/chest/back RGB + 2 wrist RGB-D | **MATCH** |
| Robot height | not directly declared | 180 cm | **UNKNOWN** |
| Arm reach | link3 0.290 m + link4 0.309 m ≈ 0.60 m + EE | 700 mm excl. EE | **PLAUSIBLE** |
| Total mass | chassis 64.8 kg + arms ~7.7 kg + body | 135 kg | **PLAUSIBLE** |

**No `MISMATCH` found.** Two items need the real robot to resolve, flagged below.

## Kinematic tree

```
base_footprint
└── base_link                      (chassis, 64.788 kg, collision box 0.6 × 0.46 × 0.15)
    ├── driving_left_link          revolute/continuous, y=+0.208, r=0.085
    ├── driving_right_link         revolute/continuous, y=-0.208, r=0.085
    ├── driven_{fl,rl,fr,rr}_link_v → _link_h    4 passive casters (swivel + roll)
    ├── lidar_link (Orbbec MS500), chassis_camera (Orbbec Dabai DW), charge_port
    └── body_link1 → body_link2 → body_link3 → body_link4     4 revolute
        ├── front_camera  (fixed)      ← chest view
        ├── back_camera   (fixed)
        ├── ego           (fixed)
        ├── head_link1 → head_link2    2 revolute
        │   └── head_camera (fixed)    ← head view, ON A MOVABLE NECK
        ├── left_base_link  → left_link1..7 → left_link8 → left_ee
        └── right_base_link → right_link1..7 → right_link8 → right_ee
             (with type:=inspire)      → {side}_gripper_base → … → {side}_eef
```

## Arm joints — per side, `direction = +1` (left) / `-1` (right)

| Policy dim | Joint | Type | Axis (left) | Axis (right) | Lower | Upper | Effort | Vel |
|---:|---|---|---|---|---:|---:|---:|---:|
| — | `{s}_joint0` | fixed | — | — | — | — | — | — |
| 0 / 8 | `{s}_joint1` | revolute | `0 0 1` | `0 0 1` | −3.10 | 3.10 | 60 | 3.14 |
| 1 / 9 | `{s}_joint2` | revolute | `0 0 1` | `0 0 1` | **−0.44** | 3.10 | 60 | 3.14 |
| 2 / 10 | `{s}_joint3` | revolute | `0 0 1` | **`0 0 -1`** | −3.10 | 3.10 | 30 | 3.14 |
| 3 / 11 | `{s}_joint4` | revolute | `0 0 1` | **`0 0 -1`** | **−2.27** | **0.00** | 30 | 3.14 |
| 4 / 12 | `{s}_joint5` | revolute | `0 0 1` | **`0 0 -1`** | −3.10 | 3.10 | 10 | 3.14 |
| 5 / 13 | `{s}_joint6` | revolute | `0 0 1` | **`0 0 -1`** | −1.22 | 1.22 | 10 | 3.14 |
| 6 / 14 | `{s}_joint7` | revolute | `0 0 1` | `0 0 1` | **−2.09** | **0.49** | 10 | 3.14 |
| — | `{s}_joint8` | fixed | — | — | — | — | — | — (D405 mount) |

Parent/child: `{s}_base_link → link1 → … → link7 → link8 → {s}_ee`.
Units: radians. Zero convention: URDF zero = the pose the meshes are authored in.

### ⚠ Left/right axis asymmetry — highest-risk finding

**Joints 3, 4, 5, 6 have opposite axis signs between left and right**
(`0 0 1` vs `0 0 -1`); joints 1, 2, 7 share the same axis. So a left/right
mirrored posture is **not** obtained by negating all seven joint values.

This is corroborated by the shipped home poses, where only `joint1` is negated:

```
home_2 left  = [ 0.8333, 0, 0, -1.7800, 0, -0.6410, 0]
home_2 right = [-0.8333, 0, 0, -1.7800, 0, -0.6410, 0]
```

`joint4 = -1.78` and `joint6 = -0.641` are **identical, not negated**, which is
only consistent with the axes already being mirrored in the model.

Any left/right mirroring utility, and any reuse of the public
`mirrored_left_arm` / `mirrored_right_arm` checkpoints, must respect this.

### Asymmetric joint limits worth noting

`joint2 ∈ [−0.44, 3.10]`, `joint4 ∈ [−2.27, 0]` and `joint7 ∈ [−2.09, 0.49]`
are strongly one-sided. A naive "small positive perturbation" calibration
sweep will hit a limit immediately on `joint4` (upper bound is exactly 0) and
must perturb negative there.

## Gripper — Inspire EG2-4C2

| Joint | Type | Axis | Limits | Mimic |
|---|---|---|---|---|
| `{s}_gripper_joint` | revolute | `0 -1 0` | **[0, 0.82]** | — **the only actuated joint** |
| `{s}_left_inner_joint` | revolute | `0 -1 0` | [0, 0.82] | mimics `gripper_joint` |
| `{s}_right_inner_joint` | revolute | `0 1 0` | [0, 0.82] | mimics, ×1, +0 |
| `{s}_right_outer_joint` | revolute | `0 1 0` | [0, 0.82] | mimics, ×1, +0 |
| `{s}_left_pad_joint` | revolute | `0 1 0` | [0, 0.82] | mimics, ×1, +0 |
| `{s}_right_pad_joint` | revolute | `0 -1 0` | [0, 0.82] | mimics, ×1, +0 |

**1 actuated + 5 mimic joints per side.** `ros2_controllers.yaml` confirms it:
`left_gripper_controller: joint: left_gripper_joint` via
`adaptive_gripper_controller/AdaptiveGripperController`.

✅ **One scalar reliably represents this gripper** — the contract's single
gripper dimension per side is structurally correct for this asset.

⚠ **Aperture direction is NOT established.** Limits are `[0, 0.82]` rad, but
which end is open is not stated anywhere in the description, and mimic
multipliers are all `+1` with mixed axis signs. Whether `0 = open` or
`0 = closed` **must be measured in simulation**. The policy contract fixes
`0.0 = fully open, 1.0 = fully closed`; the Isaac adapter owns that mapping and
currently declares it `UNVERIFIED`.

⚠ `2FEG` (public dataset) vs `EG2-4C2` (asset): plausibly the same
"2-finger electric gripper" family, **not proven identical**. Verdict
**PLAUSIBLE**, not MATCH.

## Body and head

| Joint | Type | Axis | Lower | Upper | Effort | Vel |
|---|---|---|---:|---:|---:|---:|
| `body_joint1` | revolute | `0 0 1` | −1.1868 | 0.0 | 1000 | 10 |
| `body_joint2` | revolute | `0 0 1` | −0.38397 | 2.7576 | 1000 | 10 |
| `body_joint3` | revolute | `0 0 1` | −1.5708 | 0.5236 | 1000 | 10 |
| `body_joint4` | revolute | `0 0 1` | −1.5708 | 1.5708 | 1000 | 10 |
| `head_joint1` | revolute | `0 0 1` | −1.5708 | 1.5708 | 1000 | 10 |
| `head_joint2` | revolute | `0 0 1` | −0.4014 | 0.38397 | 1000 | 10 |

The collision meshes name these `waist_yaw_torso_link`, `waist_pitch_waist_link`,
`knee_pitch_thigh_link`, `ankle_pitch_calf_link`, `head_yaw_neck_link`,
`head_pitch_head_link` — i.e. a **waist + knee + ankle lifting chain**,
consistent with the advertised 0–240 cm vertical range.

This **matches the public RoboCOIN dataset layout exactly**: 4 `torso_joint_*`
and 2 `neck_joint_*`. Strong independent corroboration.

v1 holds all six fixed (`home_1 = [0,0,0,0]` body, `[0,0]` head).

## Mobile base

| Element | Value |
|---|---|
| Driving wheels | `driving_left_joint`, `driving_right_joint`, axis `-1 0 0` |
| Wheel positions | y = ±0.208 → **track width 0.416 m** |
| Wheel radius | 0.085 m (collision cylinder, length 0.034) |
| Casters | 4 × (`_v` swivel + `_h` roll), spheres r=0.0375, at x=±0.215, y=±0.15 |
| Total wheels | **6** ✅ matches vendor "six-wheeled" |
| Base collision | box 0.6 × 0.46 × 0.15 at z=0.1 |
| nav2 footprint | `[[0.31,0.2],[0.31,-0.2],[-0.35,-0.2],[-0.35,0.2]]` → 0.66 × 0.40 m |
| Base frame | `base_footprint` |
| Odometry | `/odom` (Isaac ground truth) |
| Command topic | `/cmd_vel` (`cmd_vel_out_topic`) |

### Drive model: differential — confirmed by construction AND by config

Two independently driven wheels on a shared lateral axis plus passive casters
is a textbook differential drive. The nav2 velocity limits settle it:

```
max_velocity: [1.0, 0.0, 2.8]
min_velocity: [0.0, 0.0, -2.8]
             [ vx,  vy,  ω ]
```

**`vy` is pinned to 0.0 in both bounds** — the planner is forbidden from
commanding lateral motion. That is a differential (nonholonomic) base, not an
omnidirectional one.

So the asset's base is `(v, ω)` on `/cmd_vel`, with `v ∈ [0, 1.0] m/s`
(forward-only under nav2) and `ω ∈ [−2.8, 2.8] rad/s`.

✅ **Directly compatible with policy action dims 16 (forward) and 17 (yaw).**

⚠ nav2's `min_velocity[0] = 0.0` forbids *reverse* under the shipped
navigation config. The policy contract permits negative forward velocity, so
either the contract clamps to ≥0 for nav-driven motion, or the nav2 config is
adjusted. Recorded, not silently resolved.

**Wheel velocities are never exposed to the policy** — the adapter converts
`(v, ω)` and the diff-drive plugin handles wheels.

## ros2_control interfaces

| Controller | Joints | Command | State |
|---|---|---|---|
| `ocs2_wbc_controller` (full body) | 4 body + 14 arm | position | position |
| `ocs2_arm_controller` (split) | 14 arm | position | position |
| `body_joint_controller` | `body_joint1..4` | position | position, velocity |
| `head_joint_controller` | `head_joint1..2` | position | position, velocity |
| `{side}_gripper_controller` | `{side}_gripper_joint` | — | — (adaptive) |

**Command interface is `position`** ✅ — the asset natively supports the
joint-position arm targets the v1 contract specifies. No IK needed for the
smoke path.

Isaac hardware interface: `ros2_control_hardware_type:=isaac` selects
`topic_based_ros2_control/TopicBasedSystem` with
`/isaac/joint_command` and `/isaac/joint_states`. **Isaac integration is over
ROS 2 topics, not a direct Python API** — this materially raises the dependency
footprint (Isaac Sim + ROS 2 + ros2_control + three custom fiveages
controllers).

## Naming: asset ↔ public dataset

| Asset | Public dataset | Verdict |
|---|---|---|
| `left_joint1..7` | `ZPF_left_Joint1..7` | **PLAUSIBLE** — 1:1 by index, not proven |
| `left_gripper_joint` | `2FEG_l_Joint1` | **PLAUSIBLE** |
| `body_joint1..4` | `torso_joint_1..4` | **PLAUSIBLE** |
| `head_joint1..2` | `neck_joint_1..2` | **PLAUSIBLE** |

The asset's **collision meshes are named `ZPF_left_Link1..8`**, using the same
`ZPF_` (智平方) prefix as the datasets, even though the xacro joints are
generically named. That is meaningful evidence the asset derives from vendor
CAD, and it is the strongest single indicator that this third-party asset
tracks the real robot.

**It is still not proof of index alignment.** Confirm against the real SDK
before training on public data.

## Items requiring the real robot

1. **Gripper aperture direction** — is `gripper_joint = 0` open or closed?
2. **Joint index alignment** — is asset `left_joint3` the same physical axis as
   dataset `ZPF_left_Joint3`?
