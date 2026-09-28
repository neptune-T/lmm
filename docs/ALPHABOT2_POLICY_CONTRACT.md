# AlphaBot 2 v1 policy contract

> ## ✅ UPDATED 2026-09-28 — live Isaac qualification (simulator layer only)
>
> The 16-D / 18-D shape was executed end-to-end in Isaac Sim 4.5 on the public
> third-party Ai2 Bot2 asset ([`ALPHABOT2_ISAAC_LIVE_SMOKE.md`](ALPHABOT2_ISAAC_LIVE_SMOKE.md)).
> Measured **in the sim asset**: arm joint conventions per side (vendor/xacro
> convention ↔ USD maps), gripper polarity (EG2-4C2 joint 0.0 = closed; policy
> 0 = open → joint 0.82), base signs (+v along chassis +x, +ω counter-clockwise),
> 200 Hz physics, 20 physics steps per action (10 Hz), cameras (240×320 RGB,
> unmirrored). None of this establishes the **real robot's** SDK conventions;
> the real-API layer stays as documented below.

> ## ⚠ UPDATED 2026-09-24
>
> The 16-D / 18-D shape is **unchanged and confirmed appropriate**: the Ai2
> Bot2 asset exposes exactly 7 revolute joints per arm, one actuated gripper
> joint per side, and a differential base commanded as `(v, ω)`. Its
> ros2_control command interface is **`position`**, so joint-space arm targets
> need no IK.
>
> Corrected: this document described base semantics as unverified with
> "no public dataset contains a base action" as the reason. The base
> **kinematics** are now established on both the real robot and in the asset.
> What remains unverified is the **live sign and unit mapping**, which is
> gated in `AlphaBot2IsaacAdapter.base_signs_verified`.

Frozen 2026-09-21. Executable source of truth:
[`lmm/embodiments/alphabot2/contract.py`](../lmm/embodiments/alphabot2/contract.py).

Target: **AlphaBot 2 + dual 7-DoF arms + two-finger grippers + mobile base**.

## Proprio — 16-D

```
 0:6   left arm joints    (7)   rad
 7     left gripper       (1)   closed fraction [0,1]
 8:14  right arm joints   (7)   rad
 15    right gripper      (1)   closed fraction [0,1]
```

## Physical action — 18-D

```
 0:6   left arm absolute joint targets   rad
 7     left gripper                      closed fraction [0,1]
 8:14  right arm absolute joint targets  rad
 15    right gripper                     closed fraction [0,1]
 16    base forward velocity             m/s
 17    base yaw velocity                 rad/s
```

Torso and head are **deliberately excluded**. `assert_contract_invariants()`
fails if any label containing `torso`, `neck`, `head`, `waist` enters the
action space.

## Joint-space is canonical for v1

Arm targets are absolute joint positions. Reasons: clean correspondence to
proprio, straightforward π0.5 delta transformation, simpler snapshot
semantics, and no IK introduced as an extra experimental variable.

This matters more than usual here: **we have no kinematic model of this robot**,
so an IK-based Cartesian contract could not even be implemented. Official
Cartesian/FK/IK APIs may still be used later by scripted oracles; they do not
enter the learned contract.

## Delta-action transform

Arms are deltas w.r.t. current state; **grippers and base stay absolute**.

The mask is **16 elements, not 18**. OpenPI's `DeltaActions` computes
`dims = len(mask)` then evaluates `np.where(mask, state[..., :dims], 0)`. The
state is 16-D, so an 18-element mask broadcasts `(18,)` against `(16,)` and
raises. The prefix stopping at 16 is exactly what leaves the base absolute.

This is the same failure mode found and fixed in the WXAI contract, where the
numbers were 14 and 16. The reasoning does not transfer by memory, so it is
restated and pinned by `test_18_element_mask_would_raise`.

## Model interface

Standard OpenPI padded interface: physical 16-D state and 18-D action, padded
to **32-D**. Confirmed against a real public AlphaBot π0.5 checkpoint
(`alphabot2/aibot2_grab_bag_combined_over_under_pi05`): `max_state_dim = 32`,
`max_action_dim = 32`, `chunk_size = 50`.

`alphabot2_transform_group()` returns input and output transforms together, so
training and inference share one definition by construction.

`decode_actions()` rejects anything narrower than 18 — a truncated head can
never be silently read as a valid action.

## Gripper semantics

Policy-facing, unit-free by design:

```
0.0 = fully open
1.0 = fully closed
```

Adapters translate this to whatever the simulator or SDK wants (width,
position, current). Physical unit, open/closed limits, sign, and
continuous-vs-thresholded execution are **UNKNOWN** pending the SDK; grasp
events are simulator-specific and are not part of the contract.

**Not inherited from WXAI.** WXAI used a carriage position written
kinematically with a 0.5 snap threshold; none of that is known to apply here.

## Base semantics

Policy-facing values are **physical** (m/s, rad/s) and stay independent of any
simulator normalization — adapters own the conversion.

Real robot (operator-confirmed official facts): differential base with a
linear/angular velocity API. Unverified on hardware: positive forward and yaw
directions, limits, command rate, and whether the API expects normalized or
physical units.

Sim asset (measured live 2026-09-28): differential; +v moves along chassis +x,
+ω turns counter-clockwise about world +z; achieved/commanded 1.007 forward,
0.986 reverse, 0.974 / 0.976 yaw (caster scrub).

**No public dataset contains a base action**, so dims 16–17 are a project
decision. `assert_ready_for_training()` refuses while `base_verified=False`.

## Camera contract

```
cam_front       -> base_0_rgb          (head; chest is the documented fallback)
cam_left_wrist  -> left_wrist_0_rgb
cam_right_wrist -> right_wrist_0_rgb
```

Head over chest: it sits on a 2-DoF neck atop a 180 cm robot, giving the widest
forward view and the best sight-line over receptacles during navigation. Chest
is rigidly mounted and therefore the *more stable* view if the neck turns out
to move during manipulation — a trade-off that cannot be settled without the
robot, so both are named and the choice is revisitable.

Depth excluded: π0.5 is RGB, and depth has no RoboCOIN counterpart.

## Verification status

| Element | Verified |
|---|:--:|
| 7-DoF arms, ordering | yes — two independent datasets |
| 1 gripper dim per side | yes — two independent datasets |
| Torso/head exist and are excluded | yes |
| 32-D model padding | yes — real π0.5 checkpoint config |
| Gripper physical units/limits | real: **no** · sim asset: **yes** (live, EG2-4C2 joint 0.0 = closed) |
| Base semantics, units, signs, rate | real: kinematics yes (official), signs/units/rate **no** · sim asset: **yes** (live) |
| Joint axes/limits/zeros | real: **no** · public xacro ↔ sim USD: **yes** (maps identified live; see kinematic audit) |
