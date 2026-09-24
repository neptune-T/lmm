# AlphaBot 2 — public datasets, checkpoints and simulator assets

> ## ⚠ CORRECTED 2026-09-24 — section A of this document is WRONG
>
> This audit concluded that **no public AlphaBot 2 simulator asset existed at
> any fidelity**. That conclusion is **false**. A usable public third-party
> stack was subsequently located:
>
> * `github.com/fiveages-sim/robot_descriptions` — `ai2_bot2_description`
>   (xacro, meshes, ros2_control, nav2, OCS2), Apache-2.0
> * `github.com/fiveages-sim/robot_usds` — `humanoid/Ai2/Bot2/Bot2.usda`
>   Isaac Sim assets, Apache-2.0
>
> The search missed them because it queried vendor and Hugging Face channels
> and searched for "AlphaBot"; the asset is published by a third party under
> the name **"Ai2 Bot2"**, which neither query matched.
>
> A second claim is also corrected: this document reported that **no π0.5
> AlphaBot adaptation existed**. Two do — see
> [`ALPHABOT2_PI05_CHECKPOINT_AUDIT.md`](ALPHABOT2_PI05_CHECKPOINT_AUDIT.md).
>
> What still holds: **no public dataset contains a mobile-base action**,
> including the one named `move_base_pick`.
>
> Current facts: [`ALPHABOT2_SIM_ASSET_PROVENANCE.md`](ALPHABOT2_SIM_ASSET_PROVENANCE.md),
> [`ALPHABOT2_SIM_KINEMATIC_AUDIT.md`](ALPHABOT2_SIM_KINEMATIC_AUDIT.md).

Audited 2026-09-21 against the Hugging Face Hub API and public repositories.
Schemas below are read from each dataset's `meta/info.json`, not from READMEs.

## A. Simulator / robot-description assets

| Asset | Result |
|---|---|
| Official URDF | **NOT FOUND** |
| Official meshes | **NOT FOUND** |
| ROS `robot_description` package | **NOT FOUND** |
| MJCF / MuJoCo | **NOT FOUND** |
| USD / Isaac Sim | **NOT FOUND** |
| Gazebo | **NOT FOUND** |
| Third-party full-body description | **NOT FOUND** |
| Official SDK / API docs | **NOT FOUND** (vendor site routes to `business@ai2robotics.com`) |

There is **no public simulator model of AlphaBot 2 at any fidelity**, official
or third-party. Searches that appear to find "AlphaBot2" descriptions resolve to
the unrelated Waveshare Raspberry Pi car kit.

This is decisive: the LMM method needs a simulator for frontier discovery and
cheap rollout scaling, and no AlphaBot 2 simulator body exists to put in it.

## B. Public datasets

### B.1 RoboCOIN — `RoboCOIN/AI2_Alphabot_2_*`, `RoboCOIN/alpha_bot_2_*`

Third-party aggregation. arXiv:2511.17441, "RoboCOIN: An Open-Sourced Bimanual
Robotic Data Collection for Integrated Manipulation" — 83 authors, 180,000+
demonstrations, 15 robot platforms, 421 tasks, CC-BY-4.0 (repos tagged
apache-2.0). AlphaBot 2 is one covered platform; the paper does not state
whether that subset is a first-party AI² contribution.

~40 task repos. Schema (`AI2_Alphabot_2_pick_test_tube`, representative):

| Field | Value |
|---|---|
| `codebase_version` | v2.1 |
| `robot_type` | `AI2_Alphabot_2` |
| `fps` | **30** |
| Example scale | `stamp_document`: 987 episodes / 369,702 frames |
| Cameras | `cam_front_head_rgb`, `cam_front_chest_rgb`, `cam_left_wrist_rgb`, `cam_right_wrist_rgb`, all `[480, 640, 3]` |
| Language | `language: en, zh` |

`observation.state` and `action`, both `float32[34]`, identical layout:

```
 0- 6  left_arm_joint_1..7_rad
 7-12  left_eef_pos_x/y/z_m, left_eef_rot_euler_x/y/z_rad
13-19  right_arm_joint_1..7_rad
20-25  right_eef_pos_x/y/z_m, right_eef_rot_euler_x/y/z_rad
26     left_gripper_open
27     right_gripper_open
28-29  neck_joint_1..2_rad
30-33  torso_joint_1..4_rad
```

Auxiliary channels: `gripper_open_scale_{state,action}[2]`,
`eef_sim_pose_{state,action}[12]`, and int32 `eef_direction`, `eef_velocity`,
`eef_acc_mag`, `gripper_mode`, `gripper_activity`, each `[2]`.

Notable properties:

* The 34-D vector is **not pure joint space** — it carries a redundant
  Cartesian copy of both arms. Joint and EE channels can disagree.
* **No base dimension.** No name contains base, wheel or chassis.
* Action is absolute, same layout as state.

### B.2 `alphabot2/Aibot2_*`

**Individual Hugging Face account, no stated affiliation.** It does not claim to
be AI² Robotics. 115 datasets, 85 models. The `ZPF_` joint prefix (智平方
initials) suggests genuine hardware access, but this is **not attributable** and
must be treated as third-party.

Schema (`Aibot2_combined_pick_object_datasets_updatad_2026_06_07_15fps`):

| Field | Value |
|---|---|
| `codebase_version` | v3.0 |
| `robot_type` | `aibot2` |
| `fps` | **15** |
| Scale | 684 episodes / 193,530 frames |
| Cameras | `camera_front`, `camera_left`, `camera_right`, `camera_top`, all `[240, 320, 3]` |

**State and action are in different spaces:**

```
observation.state  float32[16]  JOINT
  0- 6  ZPF_left_Joint1..7.pos
  7-13  ZPF_right_Joint1..7.pos
 14     2FEG_l_Joint1.pos          (2-finger electric gripper)
 15     2FEG_r_Joint1.pos

action             float32[16]  CARTESIAN
  0- 6  left_arm.x/y/z/qx/qy/qz/qw
  7-13  right_arm.x/y/z/qx/qy/qz/qw
 14     left_gripper.pos
 15     right_gripper.pos
```

This **is** the "16-D Cartesian dual-arm action" referenced in the migration
brief, and its exact semantics are now verified: **position + quaternion
(x, y, z, qx, qy, qz, qw) per arm, plus one gripper scalar per arm.** It is
7-D per arm, not a 6-D euler pose. It contains **no base dimension**.

## C. Public policy checkpoints

From the same unattributed `alphabot2` account: ~85 repos, naming indicates
**ACT** (`act_chunk50_20000`) and **SmolVLA** (`_smolvla`) checkpoints, with
`mirrored_left_arm` / `mirrored_right_arm` variants and `10fps`/`15fps` tags.

No π0 / π0.5 AlphaBot adaptation was found. No official AI² Robotics checkpoint
was found.

## D. What the public data cannot give us

| Need | Status |
|---|---|
| Mobile base actions | **ABSENT from every public dataset.** Both sources are stationary bimanual manipulation. |
| Navigation / long-horizon mobile episodes | **ABSENT** |
| Simulator body | **ABSENT** |
| Joint limits / zeros | **ABSENT** (no URDF) |
| Camera calibration | **ABSENT** |
| Base control semantics | **ABSENT** from both docs and data |

The two dataset families also disagree on fps (30 vs 15), camera naming and
resolution, EE rotation convention (euler vs quaternion), and whether the
action is joint or Cartesian. **They cannot be pooled without an explicit
conversion**, and neither supports the mobile part of mobile manipulation.
