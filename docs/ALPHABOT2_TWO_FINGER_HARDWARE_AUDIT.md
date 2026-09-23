# AlphaBot 2 two-finger hardware audit

Target: **AlphaBot 2 + dual 7-DoF arms + two-finger grippers + mobile base**
(configuration confirmed by the project owner, 2026-09-21).

## Audit sources, and what they could not give

There is **no AlphaBot 2 URDF, SDK or robot description available publicly**
(see [`ALPHABOT2_PUBLIC_DATA.md`](ALPHABOT2_PUBLIC_DATA.md)). This audit is
therefore assembled from **joint names recorded in public LeRobot datasets**,
which establish existence, count and ordering but carry no kinematic data.

**Axes, limits, zero conventions, parent/child links and velocity limits
cannot be reported, because no kinematic description of this robot exists in
any public source.** They are listed as `UNAVAILABLE` rather than guessed. The
brief asked for them per joint; inventing plausible values would be worse than
reporting the gap, because every downstream collision, reachability and IK
result would silently inherit the invention.

## Controllable joints

### Arms — 7 DoF per side, high confidence

Two independent dataset families agree on count and ordering:
`left_arm_joint_1..7_rad` (RoboCOIN) and `ZPF_left_Joint1..7.pos`
(`ZPF` = 智平方).

| Policy dim | RoboCOIN name | ZPF name | Axis | Lower | Upper | Zero | Vel limit | VLA v1 |
|---:|---|---|---|---|---|---|---|:--:|
| 0–6 | `left_arm_joint_1..7_rad` | `ZPF_left_Joint1..7` | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | yes |
| 8–14 | `right_arm_joint_1..7_rad` | `ZPF_right_Joint1..7` | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | yes |

Unit: **radians** — RoboCOIN names carry an explicit `_rad` suffix.
Parent/child links: `UNAVAILABLE` (requires a URDF).

### Grippers — two-finger, 1 controllable dimension per side

| Policy dim | RoboCOIN name | ZPF name | Controllable dims | VLA v1 |
|---:|---|---|---:|:--:|
| 7 | `left_gripper_open` | `2FEG_l_Joint1.pos` | 1 | yes |
| 15 | `right_gripper_open` | `2FEG_r_Joint1.pos` | 1 | yes |

`2FEG` = **2-Finger Electric Gripper**, independently corroborating the
confirmed two-finger configuration. Both sources expose exactly **one scalar
per side**.

Physical unit, open limit, closed limit, sign, and continuous-vs-thresholded
execution: **all UNAVAILABLE** without the SDK. The policy-facing convention is
defined independently so the learned policy does not depend on these unknowns.

> The vendor separately advertises 16-DoF five-fingered hands. Our unit uses
> two-finger grippers, so that configuration is out of scope — but it confirms
> the end effector is swappable, so a contract hard-coding one scalar per side
> is specific to this configuration and not a property of the product.

### Torso and head/neck — present, excluded from the VLA in v1

| Joints | Source | VLA v1 |
|---|---|:--:|
| `torso_joint_1..4_rad` (4) | RoboCOIN | excluded |
| `neck_joint_1..2_rad` (2) | RoboCOIN | excluded |

**Flagged tension**: the vendor advertises a lifting waist with 0–240 cm
vertical work range, central to mobile-manipulation reach. Excluding the torso
means holding it fixed or driving it outside the policy, which constrains the
reachable workspace. Defensible for v1; revisit once the SDK exists.

### Mobile base — semantics entirely unknown

| Property | Status |
|---|---|
| Kinematics (differential vs omnidirectional) | **UNKNOWN**. "Six-wheeled" is published; six wheels disambiguates nothing. |
| Command form (`v,ω` / `vx,vy,ω` / pose) | **UNKNOWN** — no public SDK |
| Units (normalized vs physical) | **UNKNOWN** |
| Positive forward / positive yaw | **UNVERIFIED** |
| Limits | 6 km/h ~ 1.67 m/s max published; per-axis unknown |
| Command rate | **UNKNOWN** |

**No public AlphaBot 2 dataset contains a base action** — including
`alphabot2/2026_07_30_move_base_pick_combined`, whose name refers to the *task*
(the robot repositions to pick) while its recorded 16-D action stays arm +
gripper only. The base convention cannot be recovered from data either.

Dims 16–17 (forward, yaw) are therefore a **project decision, not an observed
fact**. `assert_ready_for_training()` refuses until `base_verified=True`.

## Camera frames

| Camera | Vendor | RoboCOIN | Canonical v1 |
|---|---|---|:--:|
| Head RGB | 1920x1080 | `cam_front_head_rgb` @ 480x640 | base view |
| Chest RGB | 1920x1080 | `cam_front_chest_rgb` @ 480x640 | fallback |
| Back RGB | 1920x1080 | — | no |
| Left wrist | 1280x720 RGB-D | `cam_left_wrist_rgb` @ 480x640 | yes, RGB only |
| Right wrist | 1280x720 RGB-D | `cam_right_wrist_rgb` @ 480x640 | yes, RGB only |

Exact simulator/SDK frame names and extrinsics: **UNAVAILABLE**.

## Summary

| Item | Status |
|---|---|
| Left arm = 7 DoF | confirmed, two sources |
| Right arm = 7 DoF | confirmed, two sources |
| Left gripper controllable dims = 1 | confirmed, two sources |
| Right gripper controllable dims = 1 | confirmed, two sources |
| Torso joints | exist (4); excluded from v1 |
| Head/neck joints | exist (2); excluded from v1 |
| Camera frames | names known from data; SDK frames unavailable |
| Joint axes / limits / zeros / velocity limits | **UNAVAILABLE — no URDF exists** |
| Mobile-base semantics | **UNKNOWN — no SDK, no data** |
