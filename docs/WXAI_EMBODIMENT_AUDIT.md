# WXAI Embodiment Calibration Audit — 2026-09-20

Scope: embodiment calibration only. No fine-tuning, data collection, frontier
probing or RL was run, and `lmm/frontier/` and `lmm/rl/` were not modified.

## 0. Blocking fact

**The WXAI URDF is not present on this machine.**

```
benchmark/home-robot/data/robots/   ->  hab_stretch  (only)
find / -name 'mobile_ai.urdf'       ->  no results
find / -type d -name 'trossen_arm_description'  ->  no results
```

The Habitat overlay declares
`articulated_agent_urdf: data/robots/trossen_mobile_ai/urdf/mobile_ai.urdf`,
and `install.sh` copies it from an external `trossen_arm_description` package
that is also absent. The installer has therefore not been run here.

Consequence: **axis, sign, zero position and joint limits cannot be
determined.** They exist only in the URDF, and no other artifact in the
repository encodes them. Those four columns are unfilled below, and no
substitute value was invented.

`artifacts/trossen_semantic_smoke_20260920/` does contain a real episode
(`habitat_episode_id: "743"`) with plausible joint values, so the URDF existed
on whichever host ran that smoke. Note its health block reports
`"base_semantics": "velocity_xy"` and `"policy_version": "pi05_base:trossen_mobile"`
— that run predates the forward/yaw correction and is not evidence for the
current contract.

## 1. Per-joint mapping table

Columns marked **URDF** are unavailable. Everything else is derived from
`habitat_patch/trossen_mobile_ai/{trossen_mobile_ai_robot.py,validate_overlay.py}`
and is authoritative for ordering.

| policy dim | label | Habitat link (resolved by name) | URDF joint | axis | sign | zero | lower | upper | unit |
|---:|---|---|---|---|---|---|---|---|---|
| 0 | `left_joint_0_absolute_rad` | `follower_left_link_1` | `follower_left_joint_0` | **URDF** | **URDF** | **URDF** | **URDF** | **URDF** | rad |
| 1 | `left_joint_1_absolute_rad` | `follower_left_link_2` | `follower_left_joint_1` | **URDF** | **URDF** | **URDF** | **URDF** | **URDF** | rad |
| 2 | `left_joint_2_absolute_rad` | `follower_left_link_3` | `follower_left_joint_2` | **URDF** | **URDF** | **URDF** | **URDF** | **URDF** | rad |
| 3 | `left_joint_3_absolute_rad` | `follower_left_link_4` | `follower_left_joint_3` | **URDF** | **URDF** | **URDF** | **URDF** | **URDF** | rad |
| 4 | `left_joint_4_absolute_rad` | `follower_left_link_5` | `follower_left_joint_4` | **URDF** | **URDF** | **URDF** | **URDF** | **URDF** | rad |
| 5 | `left_joint_5_absolute_rad` | `follower_left_link_6` | `follower_left_joint_5` | **URDF** | **URDF** | **URDF** | **URDF** | **URDF** | rad |
| 6 | `left_gripper_closed_fraction` | `follower_left_carriage_right` + `_left` | `follower_left_{right,left}_carriage_joint` | prismatic | — | 0.044 m = open | 0.0 m | 0.044 m | unitless fraction |
| 7 | `right_joint_0_absolute_rad` | `follower_right_link_1` | `follower_right_joint_0` | **URDF** | **URDF** | **URDF** | **URDF** | **URDF** | rad |
| 8 | `right_joint_1_absolute_rad` | `follower_right_link_2` | `follower_right_joint_1` | **URDF** | **URDF** | **URDF** | **URDF** | **URDF** | rad |
| 9 | `right_joint_2_absolute_rad` | `follower_right_link_3` | `follower_right_joint_2` | **URDF** | **URDF** | **URDF** | **URDF** | **URDF** | rad |
| 10 | `right_joint_3_absolute_rad` | `follower_right_link_4` | `follower_right_joint_3` | **URDF** | **URDF** | **URDF** | **URDF** | **URDF** | rad |
| 11 | `right_joint_4_absolute_rad` | `follower_right_link_5` | `follower_right_joint_4` | **URDF** | **URDF** | **URDF** | **URDF** | **URDF** | rad |
| 12 | `right_joint_5_absolute_rad` | `follower_right_link_6` | `follower_right_joint_5` | **URDF** | **URDF** | **URDF** | **URDF** | **URDF** | rad |
| 13 | `right_gripper_closed_fraction` | `follower_right_carriage_right` + `_left` | `follower_right_{right,left}_carriage_joint` | prismatic | — | 0.044 m = open | 0.0 m | 0.044 m | unitless fraction |
| 14 | `base_forward_velocity_mps` | differential base | `left_wheel` / `right_wheel` | — | + = forward | 0 | -0.25 | +0.25 | m/s |
| 15 | `base_yaw_velocity_radps` | differential base | `left_wheel` / `right_wheel` | — | + = CCW (unverified) | 0 | -0.5 | +0.5 | rad/s |

**Naming offset, verified and safe:** the validator asserts joints
`follower_*_joint_0..5` (0-indexed) while the robot class resolves links
`follower_*_link_1..6` (1-indexed). Standard URDF chaining means joint *i*
drives link *i+1*, so the two are consistent. `scripts/audit_wxai_urdf.py`
checks this per joint rather than assuming it.

**State ordering is confirmed.** `pi05_state()` returns
`concatenate([arm[:6], [left_grip], arm[6:], [right_grip]])` where `arm`
follows `LEFT_ARM_LINKS + RIGHT_ARM_LINKS`. That is exactly
`[left 0-5, left grip, right 0-5, right grip]`, matching `STATE_LABELS`.

**Base sign convention is unverified.** Positive yaw = counter-clockwise is the
Habitat convention but was not measured, and the robot class applies a
`-π/2` x-rotation in `base_transformation` to convert ROS z-up to Habitat y-up.

## 2. Gripper audit — passes, with one asymmetry

Constants from `trossen_mobile_ai_robot.py`: `gripper_open_state = 0.044` m
(per carriage, four carriages), `gripper_closed_state = 0.0`,
`gripper_state_eps = 0.003`.

**Bidirectional consistency verified exactly.** Action path
`target = open_state × (1 − fraction)`; observation path
`fraction = 1 − clip(mean(carriage)/0.044, 0, 1)`. These are exact inverses at
0.00 / 0.25 / 0.50 / 0.75 / 1.00. `0 = open, 1 = closed` holds in both
directions. Pinned by `TestGripperEmbodimentCalibration`.

**Asymmetry: the action is binary, the observation is continuous.**
`BimanualArmAction._apply_grip` thresholds the `[-1, 1]` grip command against
`grasp_threshold = 0.0`, then calls `set_gripper_target_state_for_ee` with
exactly `1.0` or `0.0`. The bridge maps closed fraction `g` to `2g − 1`, so the
effective policy-space threshold is exactly `g = 0.5`. Every intermediate
gripper target the policy emits is collapsed at execution.

The proprioceptive state, by contrast, reports a continuous carriage average.
So a demonstration dataset will contain continuous gripper *observations* and
gripper *actions* that only ever took effect as 0 or 1. **Record gripper
actions as the binarized value actually executed**, or the model will be
trained to predict a precision the simulator discards.

Also note `_apply_grip` has a second effect: crossing the threshold triggers
magic-grasp snap/desnap within `grasp_thresh_dist = 0.15` m. The gripper
dimension is therefore a grasp *trigger*, not only a carriage target.

## 3. Remaining embodiment mismatch

| # | Issue | Severity |
|---|---|---|
| 1 | URDF absent — axes, signs, zeros, limits unknown | **blocking** |
| 2 | Normalization statistics describe dual ViperX + Slate, not WXAI + diff-drive | **blocking** |
| 3 | No dual-arm OVMM demonstration source exists (see below) | **blocking** |
| 4 | Gripper action binarized at 0.5 while observation is continuous | high |
| 5 | `link_id == joint_position_index` assumed under `maintain_link_order=False` | high |
| 6 | Base yaw sign and the ROS-z-up → Habitat-y-up rotation unverified | medium |
| 7 | `allow_dyn_slide`: overlay yaml says `false`, `TrossenHabitatBackend.make` overrides to `True` | medium |
| 8 | Arms and base are separate task actions — one policy step costs two sim steps | medium |
| 9 | 50 Hz is a label; achieved rate never measured | medium |

On #5: `params.arm_joints` is populated with **link ids** resolved by name, then
used to index joint positions. Name resolution correctly defends against
Bullet's reordering under `maintain_link_order=False`, but whether link id
equals joint-position index is a separate assumption that only a live Habitat
run can confirm.

On #7: this override was added because `OVMMSim` in the restored Habitat
revision lacks `get_collisions()`, which `BaseVelAction` calls only when
dynamic sliding is disabled. Navmesh filtering still applies, but contact
behaviour differs from the overlay's declared configuration. The two should be
reconciled rather than silently diverging.

## 4. The audit tool

`scripts/audit_wxai_urdf.py` produces the missing columns in one command once
the URDF is installed:

```bash
python scripts/audit_wxai_urdf.py \
  --urdf <home-robot>/data/robots/trossen_mobile_ai/urdf/mobile_ai.urdf \
  --output docs/wxai_joint_table.md --json docs/wxai_joint_table.json
```

It cross-checks that all 12 arm joints exist, that joint *i* drives link *i+1*
as the robot class expects, that arm joints are revolute with finite limits,
that each carriage pair is prismatic with symmetric travel, and that
`gripper_open_state = 0.044` lies inside the URDF carriage range. It exits
non-zero on any failure, so it is usable as a gate.

Validated against a synthetic URDF built to `validate_overlay.py`'s structure:
passes clean, and catches all three injected faults (permuted link, asymmetric
carriage travel, open state outside range).
