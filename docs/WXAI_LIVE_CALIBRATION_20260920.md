# WXAI Live Calibration — A800, 2026-09-20

All values measured on `aibox-r8966c444da7-676d647d8b-dpddc` (4×A800-80GB),
GPU 1, OVMM episode 743. Nothing below is inferred from configuration alone.

## URDF actually loaded

`/nix/yanjun/zts/lmm/home-robot/data/robots/trossen_mobile_ai/urdf/mobile_ai.urdf`
sha256 `c4264e59d4097e29…`, 55,119 bytes, 0 remaining `package://` URIs.

Upstream source (`assets/robots/trossen_mobile_upstream/…` and
`codebase/bimanual_assets/…`) are byte-identical to each other
(`a136bac19f424e5d…`, 57,413 bytes). The installed copy differs only by the
installer's `package://` → `../meshes/` rewrite.

`scripts/audit_wxai_urdf.py` → **PASS all cross-checks**.

## Per-joint table (complete)

| dim | URDF joint | Habitat link | axis | sign | zero | lower | upper |
|---:|---|---|---|---:|---:|---:|---:|
| 0 | `follower_left_joint_0` | `follower_left_link_1` | z | +1 | 0 | −3.0543 | +3.0543 |
| 1 | `follower_left_joint_1` | `follower_left_link_2` | y | +1 | 0 | **0.0** | +3.1416 |
| 2 | `follower_left_joint_2` | `follower_left_link_3` | y | −1 | 0 | **0.0** | +2.3562 |
| 3 | `follower_left_joint_3` | `follower_left_link_4` | y | −1 | 0 | −1.5708 | +1.5708 |
| 4 | `follower_left_joint_4` | `follower_left_link_5` | z | −1 | 0 | −1.5708 | +1.5708 |
| 5 | `follower_left_joint_5` | `follower_left_link_6` | x | +1 | 0 | −3.1416 | +3.1416 |
| 6 | `follower_left_{right,left}_carriage_joint` | carriages | y | −1 | 0.044 m = open | 0.0 | 0.044 |
| 7–12 | `follower_right_joint_0..5` | `follower_right_link_1..6` | same as left | same | 0 | same | same |
| 13 | `follower_right_{right,left}_carriage_joint` | carriages | y | −1 | 0.044 m = open | 0.0 | 0.044 |
| 14 | — | differential base | — | +x local | 0 | −0.25 | +0.25 m/s |
| 15 | — | differential base | — | see below | 0 | −0.5 | +0.5 rad/s |

**Joints 1 and 2 have lower limit exactly 0.0.** Negative targets are invalid.
Any oracle, dataset or normalization must respect this; a symmetric
zero-centred prior is wrong for those two dimensions on both arms.

## Joint indexing — verified one-to-one

Link ids are **not** joint-position indices.

```
arm link ids : 20 21 22 23 24 25 | 43 44 45 46 47 48
jp indices   :  8  9 10 11 12 13 | 16 17 18 19 20 21
carriages    : left jp[15,14]    | right jp[23,22]
n_joint_positions = 26
```

Commanding each of the 12 arm slots in isolation changed **exactly one**
joint-position index, with no cross-talk. The previously flagged risk is
resolved: Habitat's translation is correct.

## Temporal semantics — 120 Hz, not 50 Hz

`ctrl_freq = 120.0`, `ac_freq_ratio = 4`. One `env.step` advances exactly
**1/120 s** of simulated time, confirmed by two independent measurements:

| Measurement | Per step | Expected at 120 Hz |
|---|---|---|
| Yaw, 20 steps | 0.0041667 rad | 0.5/120 = 0.0041667 |
| Forward, 6 headings × 60 steps | 2.0833 mm | 0.25/120 = 2.0833 mm |

Wall-clock throughput was 42.5 env.step/s — simulation speed, not control
rate. **The 50 Hz figure previously carried in the contract was wrong** and any
dataset fps must be derived from 120 Hz.

## Base signs

* **Forward**: positive `base_vel[0]` moves along the robot's local **+x** in
  `base_transformation`. 2.0833 mm/step at all six tested headings.
* **Yaw**: positive `base_vel[1]` **decreases** `arctan2(heading.z, heading.x)`
  by `0.5/120` rad per step — i.e. the opposite handedness to that expression.

An earlier measurement at episode 743's start pose gave 1.12 mm/step at 44° off
axis. That was the robot blocked against scene geometry, not a contract
problem: the 10-step and 20-step runs returned byte-identical distance. Free
space resolves it cleanly.

## Canonical physics configuration (frozen)

```
allow_dyn_slide : True     <- the ONLY runnable value
allow_sliding   : False
physics_config  : ./data/default.physics_config.json
ctrl_freq       : 120.0
ac_freq_ratio   : 4
```

`allow_dyn_slide=False` raises
`AttributeError: 'OVMMSim' object has no attribute 'get_collisions'`, because
`BaseVelAction.update_base` calls it only when dynamic sliding is disabled.
This is not a tuning choice — the overlay yaml declares `false` and is
**unrunnable as written**. The yaml should be corrected to match.

## Gripper — bug found and fixed

`set_gripper_target_state_for_ee` updates the joint **motor** target correctly
(0.044 → 0.0 confirmed), but the carriages never move: after 2 s of
`step_physics` they read 0.044 m unchanged, through both the direct call and
the action path. The articulation is driven kinematically by
`ArmAbsPosKinematicAction`, so motor targets are never integrated.

**Every gripper command in every run before this date was a no-op.** The
earlier smoke test did not catch it because it only performed zero-velocity
steps and renders.

Fix: write carriage joint positions directly. Verified after the fix:

| requested | observed left | observed right |
|---:|---:|---:|
| 0.00 | 0.00 | 0.00 |
| 0.25 | 0.25 | 0.00 |
| 0.50 | 0.50 | 0.00 |
| 0.75 | 0.75 | 0.00 |
| 1.00 | 1.00 | 0.00 |

Continuous, exact, and the two hands are independent. Note this makes the
gripper **continuous** at the actuation layer; `BimanualArmAction._apply_grip`
still binarizes its own grasp trigger at `g = 0.5`, which remains the
snap/desnap threshold.

## Atomic 16-D transition — verified

`env.step({"action": ("arm_action", "base_velocity"), …})` applies both within
one transition. Measured for a single call:

| channel | commanded | observed |
|---|---:|---:|
| left j0 | 0.21905257 | 0.21905257 |
| right j0 | 0.02991096 | 0.02991096 |
| left gripper | 0.75 | 0.75 |
| right gripper | 0.25 | 0.25 |
| yaw | −0.0020833 rad | 0.0020833 rad (sign per convention above) |

`sim_steps = 1`. The previous split path spent two simulator steps per policy
action; that is now removed.
