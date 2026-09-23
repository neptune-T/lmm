# AlphaBot 2 — specification audit

Audited 2026-09-21. **Official facts and third-party facts are kept in separate
tables and are never merged.**

> **Name collision warning.** Waveshare sells an unrelated "AlphaBot2"
> Raspberry Pi robot car kit. Public search results and GitHub repositories for
> "AlphaBot2" frequently refer to that product. Anything cited here is checked
> to be the AI² Robotics / 智平方 wheeled humanoid.

## Tier 1 — vendor-published

Source: AI² Robotics (ai2robotics.com), product announcements, April 2025.

| Item | Value |
|---|---|
| Vendor | AI² Robotics / 智平方, Shenzhen |
| Product | AlphaBot 2 (2nd generation), announced 2025-04 |
| Form factor | Wheeled humanoid, dual-arm |
| Height / weight | 180 cm / 135 kg |
| Total DoF | **34+**, incl. **16 in the hands** |
| End effector | Two **five-fingered** dexterous hands (10 fingers) |
| Arm reach | 700 mm per arm, excluding end effector |
| Arm payload | 5 kg rated, 10 kg max per arm |
| Waist/legs | Lifting + tilting structure, **vertical work range 0–240 cm** |
| Base | **Six-wheeled**, max speed 6 km/h |
| Runtime | 6 h+, auto-recharge |
| Cameras | head / chest / back **1920×1080 RGB**; both wrists **1280×720 RGB-D** |
| Perception | 360°×360° coverage claimed |
| Compute | NVIDIA AGX Orin 64 GB |
| OS / stack | Ubuntu 22.04 LTS, "AlphaBrain" VLA platform |

**Not published anywhere public**: per-arm DoF, joint names, joint limits,
zero positions, joint/Cartesian control APIs, FK/IK availability, base control
semantics and units, command rates, camera extrinsics/intrinsics, URDF or any
robot description. The company site exposes an "AlphaBrain Platform" and a
"Documents" area but no downloadable specification; the stated route is
`business@ai2robotics.com`.

## Tier 2 — inferred from public datasets (NOT vendor-confirmed)

Derived from LeRobot dataset schemas (see
[`ALPHABOT2_PUBLIC_DATA.md`](ALPHABOT2_PUBLIC_DATA.md)). Two independent
sources agree on arm DoF, which is why it is recorded as high-confidence.

| Item | Value | Confidence |
|---|---|---|
| Arm DoF | **7 per arm** | High — two independent datasets |
| Torso DoF | **4** (`torso_joint_1..4_rad`) | Medium — one dataset |
| Neck DoF | **2** (`neck_joint_1..2_rad`) | Medium — one dataset |
| Gripper (as shipped in these datasets) | **1 scalar "open" value per side** | High — both datasets |
| Gripper hardware in one dataset | `2FEG_l/r_Joint1` = 2-finger electric gripper | Medium |
| Joint unit | radians | High — names carry `_rad` |
| EE pose convention | position + **euler xyz** (RoboCOIN) or position + **quaternion xyzw** (alphabot2) | High, but the two disagree |

### The two "34"s are not the same number

The vendor's **34 DoF** counts 16 hand DoF. The RoboCOIN **34-D state vector**
is `7 arm + 6 EE pose + 7 arm + 6 EE pose + 2 gripper + 2 neck + 4 torso`, which
includes a *redundant Cartesian copy* of the arms and only a scalar per gripper.
They coincide numerically and mean different things. Do not treat the dataset
width as a DoF count.

### Contradiction: five-fingered hands vs. parallel grippers

The vendor advertises 16-DoF five-fingered hands. Both public datasets expose a
**single scalar open value per side**, and one names the joints `2FEG_*`
(2-finger electric gripper). The consistent reading is that AlphaBot 2 ships
with **swappable end effectors**, and all public data was collected with
parallel grippers rather than dexterous hands.

**Consequence for this project: the end-effector DoF of the specific unit we
will use is unknown and must be confirmed against the physical robot.** A
contract that hard-codes one scalar per gripper is correct for the public data
and possibly wrong for our hardware.

## Open items — blocking a final contract

| Question | Status |
|---|---|
| Base kinematics: differential, omnidirectional, or other? | **UNKNOWN.** "Six-wheeled" is published; the wheel configuration is not. |
| Base command form: `(v, ω)` / `(vx, vy, ω)` / pose target? | **UNKNOWN.** No public SDK, no example code. |
| Base units, limits, sign convention, command rate | **UNKNOWN** |
| Whether the SDK's primary arm interface is joint or Cartesian | **UNKNOWN** — this decides whether on-robot IK is required |
| Joint limits / zero positions | **UNKNOWN** — no URDF |
| Which end effector our unit carries | **UNKNOWN** |

No public dataset contains a mobile-base action of any kind, so the base
convention cannot be inferred from data either. This is the single largest gap,
and it is directly on the critical path because the method is *mobile*
manipulation.
