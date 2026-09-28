# Public AlphaBot π0.5 checkpoint — execution contract

Resolved 2026-09-28 from the **actual** checkpoint files, the dataset metadata
and a quantitative analysis of all 21,050 recorded frames. Code:
[`lmm/embodiments/alphabot2/pi05_contract.py`](../lmm/embodiments/alphabot2/pi05_contract.py).
Analysis scripts: `scripts/alphabot2_pi05/`.

| | |
|---|---|
| Checkpoint | `alphabot2/aibot2_grab_bag_combined_over_under_pi05` @ `cf43edfb58116c241b298a26342f800bf436d1d6` (7,473,096,344 B `model.safetensors`, Apache-2.0, not gated) |
| Dataset | `alphabot2/aibot2_grab_bag_combined_over_under` @ `7f4eaaffb520757efd5144a3285698578f681bbc` (LeRobot v3.0, `robot_type` `aibot2`) |
| Publisher | individual HF account `alphabot2`; no stated AI² affiliation |
| Loader | lerobot **0.6.1** `PI05Policy` + the checkpoint's own pre/post-processor JSON and normalizer safetensors; "All keys loaded successfully" |

## 1. Model / OpenPI configuration

| Field | Value | Source |
|---|---|---|
| Policy | `pi05`, base `lerobot/pi05_base`, PaliGemma `gemma_2b` + action expert `gemma_300m`, bf16 | `config.json` |
| Chunk | `chunk_size = 50`, `n_action_steps = 50` (whole chunk executed before re-inference) | `config.json` |
| Sampling | flow matching, 10 inference steps | `config.json` |
| Padding | state and action padded to 32 | `config.json` |
| Images | 4 cameras, 240×320 RGB → resize-with-pad to 224×224 → [−1, 1] inside the model; no image normalization (`VISUAL: IDENTITY`); augmentation disabled in training | `config.json`, `train_config.json`, `modeling_pi05.py` |
| Training | 20,000 steps, batch 16, lr 2.5e-5 cosine | `train_config.json` |

## 2. State path

`observation.state` (16) → MEAN_STD normalize (checkpoint stats) → digitized
into 256 bins over [−1, 1] (values beyond ±1 clip) → written into the prompt.
There is no separate state projection.

```
Task: {task.strip(), '_'→' '}, State: {16 bin ids};\nAction:
```

Tokenized with the PaliGemma SentencePiece model with BOS, max length 200, right
padding. The checkpoint config names the **gated** `google/paligemma-3b-pt-224`.
We use openpi's public `gs://big_vision/paligemma_tokenizer.model` (sha256
`8986bb4f…`), encoded exactly as openpi does. Without BOS, predictions get
measurably worse (§8).

**State ordering** (names from `meta/info.json`):
`0:7 ZPF_left_Joint1..7`, `7:14 ZPF_right_Joint1..7`, `14 2FEG_l_Joint1`,
`15 2FEG_r_Joint1`. The grippers come **last**, unlike the LMM proprio, which
interleaves them.

**Joint convention** (fitted, §7): left = xacro convention; right =
`(q1, q2, −q3, q4, −q5, q6, −q7)` in xacro terms, offsets 0.

**Gripper state**: the real 2FEG reading uses the sim EG2 joint's polarity and
range (0 = closed). The dataset "open" reading is 0.8122 on the right and 0.7959
on the left; the left gripper never moved (std 0.0008, so a sim value of 0.82
would sit at +30σ). Sim → dataset: `g_joint × open_reading / 0.82`.

## 3. Action path

Model output (normalized) → MEAN_STD unnormalize → out. **No relative/delta
transform**: `relative_actions_processor.enabled = false` and
`absolute_actions_processor.enabled = false`.

**Ordering** (verified from `meta/info.json` names *and* the value ranges):

```
0:3    left xyz            3:7    left  qx qy qz qw
7:10   right xyz          10:14   right qx qy qz qw
14     left gripper       15      right gripper
```

The expected layout was correct.

| Property | Finding | Evidence |
|---|---|---|
| Absolute vs delta | **absolute EE pose targets** | processors disabled. Positions sit around 0.6–0.9 m and match the arm's forward kinematics under one fixed frame (§7); deltas would be near 0 |
| Position units | metres | ranges x 0.46–0.96, y ±0.09–0.68, z 0.61–0.97; FK fit to mm |
| Reference frame **B** | **one frame for both arms**; `p_B = R·p_chassis + t` with R ≈ 4.4° pitch and t = (−0.128, +0.023, −0.397) m, relative to our zero-torso chassis | shared-frame fit: left 5.8 mm / 0.70°, right 12.7 mm / 1.86° median — no worse than separate per-arm fits |
| EE frame **E** | +x along the gripper approach, origin at the TCP ≈ 0.133–0.143 m beyond `gripper_base`; within ~1.5° of an axis permutation of `gripper_base` | same fit, nearly identical on both arms |
| Quaternion order | x, y, z, w | names; the w column is the one near 1 |
| Quaternion norm | unit (0.99999994–1.0000003) | data |
| Hemisphere | canonical: **qw ≥ 0 in 100 % of frames, 0 sign flips between consecutive frames**; no canonicalization step in the processors, and plain MEAN_STD per component | data, processors |
| Gripper action | 0 = open; close commands are recorded as values in [0.5, 1]; left gripper action is **always 0** | data: action ≥ 0.5 ↔ state 0.14, action 0 ↔ state 0.70 |
| Time alignment | actions are **leading teleoperation targets**: the recorded joint state lags them by ~0.4–0.8 s, and the arm often saturates at joint limits (e.g. right j4 ≈ 0) while the target keeps moving | lag scan (§7); right target vs current pose median 60 mm |
| Base action | none | 16-D only |

## 4. Timing and data

10 fps (all timestamp deltas 0.1000 s), 351 episodes, median 33 frames (5–410).
Tasks (4):

| task_index | task | frames |
|---:|---|---:|
| 0 | Grab bag from top | 5,663 |
| 1 | Grab bag from bottom | 4,114 |
| 2 | Close gripper and do nothing | 1,818 |
| 3 | Pick object from table and put it inside the box | 9,455 |

Essentially right-arm manipulation: the left arm is nearly static and its gripper
never actuates.

## 5. Cameras (from decoded training frames)

| Checkpoint key | What it shows | Simulator camera |
|---|---|---|
| `camera_front` | chest camera: fixed forward view, no robot parts | `cam_chest` (`body_link4/front_camera`) |
| `camera_top` | head camera: forward/down, sees the right arm on the image right | `cam_front` (`head_link2/head_camera`) |
| `camera_left` | **wrist of the arm the state/action call "right"** (the only one that grasps) | `cam_right_wrist` |
| `camera_right` | wrist of the "left" arm | `cam_left_wrist` |

The wrist names are crossed relative to the arm names. Feeding each wrist view
into the key named after its own arm was measurably worse (§8).

## 6. What this means for our stack

* The checkpoint is an **EE-pose policy**. `CartesianMobileAction18` keeps its
  16 dims verbatim and appends `(v, ω)`. `CartesianExecutor` lowers that to
  the live-qualified `JointMobileAction18` through: frame B → tool E →
  `gripper_base` → xacro IK (limits, seeded by the previous solution) →
  vendor joints → `AlphaBot2IsaacAdapter`.
* Its state input is joint-space in the dataset convention.
  `dataset_state_from_proprio()` converts it: reindexing grippers, flipping
  right q3/q5/q7, calibrating gripper readings.
* The torso is not in the state. Frame B is defined relative to the arms at our
  zero torso, which is all the executor needs. Camera viewpoints still depend on
  the real torso height, which is unknown.

## 7. How the frames were resolved (quantitative)

1. Tried a naive global fit of action vs xacro FK(state): left 15.6 mm / 1.6°;
   right **208 mm / 42°**. The right arm is inconsistent.
2. Sign/offset search with a frame-free invariant (the rotation angle between
   two samples is independent of base/tool frames, even mirrored): no fit using
   all frames. The cause was tracking failures and the target lead.
3. Joint-limit coincidences in the raw ranges: j2 and j4 hit their xacro limits
   unflipped (j4 min −2.269); right j6 max 1.212 ≈ 1.22; right j7 spans
   [−0.489, 0.914], which fits [−2.09, 0.49] only when flipped.
4. A limit-constrained discrete search, followed by a robust (soft-L1) full fit,
   picked right signs (+,+,−,+,−,+,−), offsets 0.
5. Lag scan: the right-arm fit improves until the state lags by 0.8 s (median
   24 mm / 2.9° on all tasks); pick-and-place frames fit best (12 mm / 1.8°).
6. A shared base frame for both arms loses nothing versus per-arm frames, so it
   is adopted.

Residual discrepancies come from teleop tracking lag, joint-limit saturation,
an unobserved torso, and differences between the public xacro and the real arm.
They are **not** fully explained. The left arm moves little, so its convention
rests on less evidence.

## 8. Checkpoint executed offline on training frames

12 frames × 3 tasks, 4 decoded camera views, same seed. Predicted chunk vs
recorded action targets:

| Variant | first step, right arm (median) | 20-step mean, right arm |
|---|---:|---:|
| **As contracted** | 97 mm | 88 mm |
| black images | 107 mm | 135 mm |
| wrist cameras swapped back to "own-arm" names | 97 mm | 105 mm |
| wrong task string | 86 mm | 101 mm |
| state from another episode | 220 mm | 152 mm |
| tokenizer without BOS | 105 mm | 112 mm |

The contracted pipeline is the best variant, and the model visibly uses images,
wrist assignment, task and state. The ~9 cm first-step error is in line with
the targets' inherent lead over the observed arm (right target vs current pose:
60 mm median, 73–92 mm in the bag tasks). Gripper open/close predictions follow
the recorded commands (0.96–0.99 vs 1; 0.01 vs 0), and predicted quaternions
stay unit-norm with qw > 0.

Inference: 0.38 s per 50-step chunk (bf16, one A800, no `torch.compile`); the
first call takes 0.81 s.
