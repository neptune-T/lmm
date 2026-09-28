# Public AlphaBot 2 π0.5 checkpoint audit

Offline audit, 2026-09-24. **Not used for simulator control.** Its purpose is
to inform the joint-vs-Cartesian decision that comes *after* the embodiment
smoke.

## Checkpoint

| Field | Value |
|---|---|
| Repository | `alphabot2/aibot2_grab_bag_combined_over_under_pi05` |
| Publisher | `alphabot2` — **individual HF account, no stated AI² affiliation** |
| Policy type | `pi05` |
| Base model | **`lerobot/pi05_base`** |
| Weights | `model.safetensors`, 7,473,096,344 B (~7.4 GB) |
| Files | `config.json`, `train_config.json`, `policy_{pre,post}processor.json` + normalizer safetensors |
| Training dataset | `alphabot2/aibot2_grab_bag_combined_over_under` |
| Optimizer LR | 2.5e-05 |
| Training steps | 20,000 |

A second π0.5 checkpoint exists:
`alphabot2/lerobot_Aibot2_combined_pick_object_JC_intern_2026_06_08_15fps_pi05_20260608_2232`.

## Model interface

| Field | Value |
|---|---|
| `n_obs_steps` | 1 |
| `chunk_size` | 50 |
| `n_action_steps` | 50 |
| `max_state_dim` | **32** |
| `max_action_dim` | **32** |
| State input | `observation.state` [16] |
| Action output | `action` [16] |
| Images | `camera_front`, `camera_left`, `camera_right`, `camera_top`, each `[3, 240, 320]`, resized to **224×224** |
| Normalization | STATE `MEAN_STD`, ACTION `MEAN_STD`, VISUAL `IDENTITY` |
| Image augmentation | ColorJitter / SharpnessJitter / RandomAffine configured but **disabled** |

✅ Confirms the **32-D OpenPI padding** the LMM contract assumes.

## Dataset schema

`alphabot2/aibot2_grab_bag_combined_over_under`: LeRobot v3.0, `robot_type`
`aibot2`, **10 fps**, 351 episodes, 21,050 frames, video AV1.

### State — joint space, 16-D

```
 0- 6  ZPF_left_Joint1..7.pos
 7-13  ZPF_right_Joint1..7.pos
14     2FEG_l_Joint1.pos
15     2FEG_r_Joint1.pos
```

### Action — Cartesian, 16-D

```
 0- 6  left_arm.x, left_arm.y, left_arm.z, left_arm.qx, left_arm.qy, left_arm.qz, left_arm.qw
 7-13  right_arm.x, right_arm.y, right_arm.z, right_arm.qx, right_arm.qy, right_arm.qz, right_arm.qw
14     left_gripper.pos
15     right_gripper.pos
```

### ⚠ The layout is NOT "xyz + quat + gripper per side"

The migration brief asked whether the 16-D action is
`left xyz+quat+gripper / right xyz+quat+gripper`. **It is not.** Both grippers
are appended at the **end**, after both arm poses — not interleaved per side:

```
[left pose 7][right pose 7][left grip 1][right grip 1]     ← actual
[left pose 7][left grip 1][right pose 7][right grip 1]     ← NOT this
```

This also differs from the LMM v1 contract, which **does** interleave
(`0:7` left arm, `7` left gripper, `8:15` right arm, `15` right gripper).
Any future reuse of this checkpoint or its data needs an explicit reindex.

### Confirmed semantics

| Property | Finding |
|---|---|
| Space | **State is joint; action is Cartesian.** Not the same space. |
| Quaternion ordering | **`qx, qy, qz, qw`** — x-first, w-last, explicit in the names |
| Position units | metres (**assumed**; not declared in metadata) |
| Absolute vs delta | **Not determinable from metadata.** No delta transform is declared; the names carry no `delta` marker. Presumed absolute EE pose, **unproven**. |
| Quaternion hemisphere | **No sign/hemisphere handling is declared anywhere.** See below. |
| Gripper convention | scalar `.pos` per side; open/closed direction **not declared** |
| Language conditioning | `tasks.parquet` present; this dataset family carries **no `language` tag** (the RoboCOIN family does, en+zh) |
| Normalization | MEAN_STD over state and action, from the dataset's own `stats.json` |

### ⚠ Quaternion double-cover is unhandled

`q` and `−q` represent the same rotation. The config applies plain `MEAN_STD`
normalization to all 16 action dims, including the 8 quaternion components,
and declares no canonicalisation (e.g. forcing `qw ≥ 0`).

If the recorded data ever crosses hemispheres, the mean/std statistics and the
regression target are both corrupted in a way that is invisible in the loss.
**This must be checked before any Cartesian policy decision** — it is a
concrete argument in favour of keeping the joint-space contract.

## Relevance to the LMM project

| Question | Answer |
|---|---|
| Usable as a starting checkpoint for our 18-D joint action head? | **No** — different action space (Cartesian), different width (16 vs 18), different ordering, no base dimensions. |
| Useful as evidence? | **Yes** — confirms `lerobot/pi05_base` works on this embodiment, confirms 32-D padding, and confirms a real 16-D Cartesian convention exists. |
| Does it contain base actions? | **No.** No public AlphaBot dataset does, including the one named `move_base_pick`. |

## Deferred decision

Per the migration order of operations, the joint-vs-Cartesian choice is
**not** made here. Inputs recorded for that decision:

1. The asset's ros2_control command interface is **`position`** (joint-space),
   so joint control needs no IK.
2. The public π0.5 checkpoint uses **Cartesian**, so its weights and data do
   not transfer directly to a joint head.
3. Cartesian brings the unhandled **quaternion double-cover** risk above.
4. Cartesian would require IK in the loop; the asset provides OCS2 whole-body
   control that could supply it, but that adds a large dependency.

## Addendum 2026-09-28 — relation to the live-qualified simulator

Offline only. The checkpoint was **not** loaded or run, and it did **not**
shape the smoke contract (16-D joint state / 18-D joint + base action).
Relevant facts from the live Isaac qualification
([`ALPHABOT2_ISAAC_LIVE_SMOKE.md`](ALPHABOT2_ISAAC_LIVE_SMOKE.md)):

| Checkpoint property | Simulator side | Consequence |
|---|---|---|
| Dataset fps **10** | policy rate **10 Hz** (200 Hz physics / 20) | rates line up; chosen for this reason, not to fit the weights |
| State `ZPF_{left,right}_Joint1..7.pos` + `2FEG_{l,r}_Joint1.pos`, grippers last | contract interleaves grippers (`0:7`, `7`, `8:15`, `15`) | explicit reindex needed for any reuse |
| Joint zero/sign convention undeclared | sim state is in the public xacro convention (USD↔xacro maps identified live) | whether `ZPF_*` equals the xacro convention is **unknown**; comparing the checkpoint's `stats.json` state means with the xacro home pose would be a cheap first check |
| Gripper `.pos` direction undeclared | sim EG2-4C2: joint 0 = closed; policy 0 = open | the checkpoint's gripper polarity must be established before reuse |
| Cameras `camera_front/left/right/top`, 240×320 | sim renders `cam_front` + two wrist views at 240×320 | resolution matches; the wrist-vs-third-person identity of `camera_left/right` is still unverified; no `camera_top` in the sim contract |
| Cartesian EE action, quaternion `qx..qw` | sim gripper_base pose available from PhysX / xacro FK (≤ 4 mm) | a Cartesian head is technically supportable later; not adopted |

The joint-vs-Cartesian decision stays deferred.
