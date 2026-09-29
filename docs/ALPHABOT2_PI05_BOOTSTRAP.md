# AlphaBot 2 π0.5 policy bootstrap (public checkpoint → Cartesian sim control → pilot data)

2026-09-28/29, Isaac Sim 4.5 on the A800 (GPU 3), public third-party Ai2 Bot2
asset, 200 Hz physics, 10 Hz policy. Starts from canonical commit `cdf88d0`
(`ALPHABOT2_ISAAC_EMBODIMENT_SMOKE_PASS`). Execution contract:
[`ALPHABOT2_PI05_EXECUTION_CONTRACT.md`](ALPHABOT2_PI05_EXECUTION_CONTRACT.md).
No SFT, frontier discovery, rollout scaling or online RL was run.

## 1. Interfaces

| Interface | Role | Status |
|---|---|---|
| `JointMobileAction18` (= `contract.ACTION_LABELS`) | **low-level embodiment/debug interface**: vendor-convention joint targets + closed-fraction grippers + (v, ω) | frozen, live-qualified at `cdf88d0`, unchanged |
| `CartesianMobileAction18` (`cartesian.py`) | **policy-facing**: the public checkpoint's 16 dims verbatim (left xyz, left qxyzw, right xyz, right qxyzw, left grip, right grip) + (v, ω) | new, live-qualified here |
| 16-D proprio (canonical) | vendor joints, grippers interleaved | unchanged |
| checkpoint state | 16-D, dataset convention, grippers last | `pi05_contract.dataset_state_from_proprio()` |

`CartesianExecutor` lowers Cartesian to joint actions:

    frame B → tool E → gripper_base → xacro IK (limits, seeded by the previous solution)
    → JointMobileAction18 → AlphaBot2IsaacAdapter

It never bypasses the calibrated joint map. Each action is reported with
IK validity, pose error, joint step and limit contact.

## 2. Cartesian IK/FK qualification

CPU: `kinematics.fk` equals the xacro parser to 2e-16. FK→IK→FK round trips,
limit handling, continuity, and quaternion sign and scale invariance are covered
by 20 tests in `tests/test_alphabot2_cartesian.py`.

Live (`scripts/alphabot2_isaac/cartesian_live.py`, `artifacts/alphabot2_cartesian_live.json`):

| Test (each arm) | IK valid | Discontinuities / elbow flips | Tracking while moving | Settled |
|---|---:|---|---|---|
| +x_B 0.12 m line | 100 % | 0 / 0 | 1.5–1.8 mm, 0.05° | 3.9 mm, 0.05° |
| +z_B 0.08 m line | 100 % | 0 / 0 | 4.0–4.4 mm, 0.03° | 3.8 mm |
| circle r = 5 cm | 100 % | 0 / 0 | 5.9 mm, 0.04° | 3.8 mm |
| rotate 20° about z_B | 100 % | 0 / 0 | 3.7–3.9 mm, 0.4° | 0.03–0.07° |

The ~3.8 mm settled offset is the known constant USD↔xacro frame offset.
Commanded displacement directions reproduce in B (+x 0.116 m → +0.119 m), so
the frames and the quaternion convention are correct on both arms.

**Replay of recorded dataset targets**: 10 episodes, base still. Recorded
Cartesian targets pass through our IK, and the resulting sim joints are compared
with the joints the real robot recorded:

| Task | IK valid | Sim vs real joints (median max-abs, best lag) |
|---|---:|---|
| Pick object → box (5 ep.) | 100 % | **0.05–0.28 rad** at 0.7–0.8 s lag |
| Grab bag (5 ep.) | 88–100 % | 0.17–1.87 rad; shoulder j1/j3 differ, the redundant 7th DoF |

The pick-and-place agreement is an independent confirmation of the fitted
frames and conventions. Workspace reachability at the dataset mean orientation:
left 79 %, right 58 % of the min–max box; failures sit at joint 4/7 limits, as on
the real robot.

## 3. Public checkpoint zero-shot smoke

The real checkpoint was served (lerobot 0.6.1, bf16, GPU 3) to the Isaac client
(`scripts/alphabot2_pi05/pi05_server.py`, `scripts/alphabot2_isaac/pi05_zero_shot.py`).

- **Setup:** manipulation only, with **v = ω = 0**. The scene is placed in frame B at the training workspace: table, 5 cm cube, flat "bag", open box.
- **Cameras:** mapped per the contract, crossed wrists included.
- **Start pose:** the dataset's median episode-start pose.
- **Tasks:** the dataset's own task strings.

| Episode | Task | Chunking | IK valid | Min TCP–object | Gripper cmds | Outcome |
|---|---|---|---:|---:|---:|---|
| pick_s0_c50 | Pick object … box | 50 | 100 % | 0.220 m | 0 | no contact |
| pick_s1_c50 | Pick object … box | 50 | 100 % | 0.211 m | 10 (in air) | no contact |
| pick_s0_r10 | Pick object … box | replan 10 | 100 % | 0.223 m | 0 | no contact |
| bagtop_s0_c50 | Grab bag from top | 50 | 100 % | 0.295 m | 54 (in air) | no contact |
| bagtop_s1_r10 | Grab bag from top | replan 10 | 96 % | 0.328 m | 3 | no contact |

**Task success: 0/5.** Objects were never touched, and TCP–object distance grew
over each episode. Logs, chunks and the camera inputs are in
`artifacts/alphabot2_pi05_zero_shot/`.

**Failure classification: `POLICY_DISTRIBUTION_SHIFT`.** The interface
categories are excluded by measurement:

| Category | Excluded because |
|---|---|
| OBSERVATION_MAPPING | offline on training frames the contracted inputs beat every corrupted variant (contract §8); in sim the cameras show the intended views |
| ACTION_SEMANTICS / FRAME_MISMATCH | in sim the first predicted target lies **9–82 mm (median ≈ 40 mm)** from the arm's current pose, matching the training data's lead (60 mm). The policy reads our state and emits targets in the right frame. Recorded-target replay reproduces the real joints (§2) |
| GRIPPER_SEMANTICS | predicted gripper values are binary-like {≈0, ≈0.96} as in training; the executor maps them with the verified polarity |
| IK_EXECUTION | IK valid 96–100 %; no discontinuities |

What remains is the scene: real shelves, bags and office floor in training
versus flat-shaded sim primitives. The checkpoint has **no zero-shot
manipulation prior usable in this sim**. It does carry embodiment-consistent
state→pose knowledge, which makes it a reasonable SFT initialization.

## 4. Privileged demonstration expert (offline only)

`scripts/alphabot2_isaac/expert_pilot.py`. Uses privileged poses; never part
of a policy.

- **Navigation**: differential go-to-pose on (v, ω). Uses a pre-point 0.35 m
  before the goal to align the heading away from tables, then drives straight;
  stops on longitudinal error, and the arm absorbs the ≤ 1.5 cm lateral residual.
  Backs off 0.2–0.3 m before turning away from a table.
- **Grasp**: top-down, IK inside the vendor limits, several seeds. Joint-space
  reach to a pre-grasp, then a guarded straight descent (stops on finger contact
  with the table/box, separation ≤ 1 mm). Close until the gripper converges on
  the object, with a hard failure otherwise. Lift 10 cm.
- **Place**: navigate to the box, lower from 6 cm above with a guarded
  descent, release, retract, return home.
- Every action is a `CartesianMobileAction18` executed through
  `CartesianExecutor` → adapter; **the executed action is the label**.
  Joint-space segments are labelled with the FK of the commanded joints, so
  every label is reachable.

Bugs found and fixed on the way (all reported, none hidden):
1. Fingertips were driven into the table, which jammed closing until the lift.
   Fixed with the guarded descent and a target 1 cm above the cube centre.
2. `isaac_backend.contact_pairs()` counted CONTACT_LOST events and
   **speculative contacts**: pairs within the contact offset, reported 8–19 cm
   apart. It now counts only points with separation ≤ 1 mm. This could only have
   over-reported, so earlier "no contact" results stand (see the correction in
   `ALPHABOT2_ISAAC_LIVE_SMOKE.md`).
3. Place-above pose infeasible for the top-down wrist (joint 6 limit). Lowered
   to 6 cm above the rim.
4. Images lagged the proprio by one action: the annotators return the previous
   render. Fixed by reading images after the action (§7); the first pilot
   recording was discarded and re-recorded.

## 5. Pilot dataset

Schema (per episode, `/nix/yanjun/zts/alphabot2/pilot/epNNN/`):

| Field | Content | Policy input? |
|---|---|:--:|
| `images.npz` | `cam_front` (head, canonical front), `cam_left_wrist`, `cam_right_wrist`, `cam_chest` (extra), 240×320 RGB | yes (chest optional) |
| `lowdim.npz: proprio` | canonical 16-D | yes |
| `meta.json: instruction` | e.g. "Use the right arm to pick up the red cube and put it in the blue box." | yes |
| `lowdim.npz: action` | **CartesianMobileAction18**, including real (v, ω) | label |
| `lowdim.npz: obj_pose, chassis, grasp, stage_id, sim_time` | privileged | **no** (metadata) |
| `meta.json: events, init, success, expert_ik, arrival` | privileged | **no** |

Timing: proprio at step t is read **before** action t. Images are read right
**after** action t; because the annotators deliver the previous render, they
show that same state (verified, §7).

| | Count |
|---|---:|
| Episodes (all successful) | **10 / 10** |
| Full navigate → pick → navigate → place | 5 (left 2, right 3) |
| Navigate → pick → lift → back off | 5 (left 3, right 2) |
| Initial base/object configurations | 10 distinct (base ±0.25 m, yaw ±26°, object ±12 cm) |
| Steps (10 Hz) | 2,772 (4.6 min robot time) |
| Base motion | real (v, ω) labels on every navigation step |

Per-episode metadata: `artifacts/alphabot2_pilot/` (plus one synchronized
frame, `pilot_ep001_t120.png`: head, left wrist, right wrist holding the cube,
chest).

**Framing observation.** At zero neck and torso angles, the head camera (30°
down) and the chest camera (22° down) look past a 0.67 m table. During
tabletop manipulation the canonical front view mostly shows the floor, and the
wrist cameras carry the manipulation. Pitching the neck (`head_joint2`, up to
0.40 rad) for manipulation phases is a cheap improvement for the next data
round. It is left unchanged here to keep the qualified head frame. A LeRobot v3 copy of the
successful episodes (checkpoint state convention, 18-D action, checkpoint
camera keys) sits at `/nix/yanjun/zts/alphabot2/pilot_lerobot`
(`scripts/alphabot2_pi05/pilot_to_lerobot.py`).

## 6. Replay validation

Every saved episode was re-run from its recorded initial configuration: world
reset, base pose, object pose, home arms. The recorded action sequence was
executed through the identical executor→adapter path.

| | Result |
|---|---|
| Replays succeeding | **10 / 10** |
| Proprio vs recording | **bitwise identical** (max abs diff 0.0) in all 10 |
| Object trajectory vs recording | 0.0 m max difference in all 10 |

## 7. Synchronization

`scripts/alphabot2_isaac/sync_check.py` uses two stimuli: a block teleported
into view with the arm still, and the right hand commanded 10 cm sideways. It
finds when the head and right-wrist images and the proprio first reflect them.
Images are 16×16 block-averaged to rise above RTX denoiser noise.

| Pairing | Block: image changes at | Arm: proprio / image change at | Synchronized |
|---|---|---|:--:|
| images read *before* action t (original loop) | K+2 | K+1 / **K+2** | ✗ |
| images read *after* action t (recorder now) | **K+1** | K+1 / **K+1** | ✅ |

**Finding:** the camera annotators return the **previous** render. An image read
before action t shows the state one action old, although its reported
`rendering_time` says otherwise (the metadata is off by one frame). Extra
`world.render()` calls, or rendering on the last two substeps, do not change
this. The recorder therefore reads proprio before action t and images right
after action t, which puts both on the same state (all cameras share one
render; the timestamps sit a constant 5 ms, one substep, before the proprio read).
The first pilot recording was paired the old way and was **discarded and
re-recorded** with the fix.

Closed-loop consequence: a policy deciding action t can only see images of the
state before action t−1, i.e. 0.1 s of camera latency. That was the case in the
zero-shot runs; it does not change their conclusion.

## 8. Snapshot implications

The pilot does not branch. Replays start from a world reset plus the recorded
initial configuration, which is bitwise deterministic. No snapshot branching was
used for data generation. The snapshot audit's rule stands: restored siblings
are deterministic, but a contact-loaded restored state is not treated as
identical to the uninterrupted run.

## 9. Throughput and parallelization

| Measurement | Value |
|---|---|
| Sim only, 1 robot, CPU PhysX, no render | 100.8 physics steps/s → 0.20 s per action (**0.50× real time**) |
| + 3 cameras (320×240, RTX) | 0.26 s per action |
| π0.5 inference (bf16, 1 A800, 50-step chunk) | **0.38–0.41 s** per chunk (first call 0.81 s) |
| Closed loop, 4 cameras + π0.5 (zero-shot runs) | 267 s / 750 actions = **0.36 s per action (0.28× RT)** |
| Expert recording, 4 cameras + compressed storage | 1,032 s / 2,772 steps = 0.37 s per step |
| Replay, 4 cameras, no storage | 911 s / 2,772 steps = 0.33 s per step |

Parallelization probe (`scripts/alphabot2_isaac/parallel_probe.py`,
`artifacts/alphabot2_parallel_probe.json`), N robots in one scene, 400 steps, no render:

| PhysX | N | Scene steps/s | Robot-steps/s | Aggregate RT factor |
|---|---:|---:|---:|---:|
| CPU | 1 | 100.8 | 100.8 | 0.50 |
| CPU | 2 | 54.3 | 108.5 | 0.54 |
| CPU | 4 | 28.7 | 114.7 | 0.57 |
| CPU | 8 | 13.9 | 111.0 | 0.56 |
| GPU | 1 | 8.9 | 8.9 | 0.04 |
| GPU | 4 | 9.3 | 37.3 | 0.19 |

* A CPU-PhysX scene is effectively single-threaded: extra robots in one scene
  buy nothing. Scaling means **more processes** (host: 64 cores with a 60-core
  quota, 4 shared A800s).
* GPU PhysX runs this articulation, including mimic joints, but has a ~110 ms
  fixed per-step cost in this unengineered setup. It scales linearly in N and
  overtakes CPU only beyond ~12 robots per scene. A real GPU pipeline (cloner,
  tensor API only, no USD sync, tiled cameras) is engineering not done here.

**K-way rollout-scaling estimate.** One closed-loop continuation costs ~0.36 s
per action per process. A continuation of H = 300 actions (30 s) is ~108 s. K = 8
continuations per snapshot is ~14 min per process, plus 2.8 s per branch for
canonical restore if exact branching is wanted. With P processes (one core plus
~4 GB GPU each; the policy served batched on shared GPUs), 1,000 snapshots × K = 8
≈ 8,000 × 108 s ≈ 240 process-hours: **≈ 15 h at P = 16, ≈ 4 h at P = 60**.
Rendering and inference dominate; the physics alone is ~0.2 s per action.

## 10. SFT readiness

| Precondition (task spec) | Status |
|---|---|
| Cartesian action semantics resolved | ✅ execution contract; live Cartesian qualification (§2) |
| π0.5 checkpoint executes correctly | ✅ as an *interface*: offline input sensitivity; in-sim first-target consistency (§3). Not a successful zero-shot controller (distribution shift) |
| Mobile 18-D trajectories recorded | ✅ 10 episodes, real (v, ω) labels |
| Saved episodes replay | ✅ 10/10, bitwise identical |
| Camera/state/action synchronization verified | ✅ after the image-latency fix (§7) |

Prepared, **not launched**:

* `configs/alphabot2_pi05_sft_pilot.json`: fine-tunes from the public checkpoint
  with an 18-D action output (the 32-D padding already has room), 2,000 steps,
  batch 8, W&B off, no credentials in the file.
* `scripts/alphabot2_pi05/sft_train.py`: `lerobot-train` plus one shim that
  gives the tokenizer step openpi's public PaliGemma SentencePiece model (the
  saved processor names the gated HF repo). lerobot-train itself recomputes the
  normalizer stats from the dataset.
* `scripts/alphabot2_pi05/pilot_to_lerobot.py`: LeRobot v3 dataset at
  `/nix/yanjun/zts/alphabot2/pilot_lerobot`. It uses the checkpoint's state
  convention and camera keys; stage labels stay out of the features.
* `scripts/alphabot2_pi05/sft_dryrun.py` (`artifacts/alphabot2_sft_dryrun.json`):
  dataset + policy + processors built exactly as for training, with **one
  forward pass, no backward, no optimizer**. 10 episodes / 2,772 frames,
  action chunk (50, 18), 18-D stats with real base variance (std v 0.135 m/s,
  ω 0.228 rad/s), loss 1.53 (finite), 9.8 GB.

Before a real SFT: the pilot is a *schema* pilot (2,772 frames, one scene
layout, one object) and would only overfit. Scale collection and diversify
objects, layouts and lighting first, and decide the head-camera pitch.
