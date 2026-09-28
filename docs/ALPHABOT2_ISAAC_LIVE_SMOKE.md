# AlphaBot 2 — live Isaac embodiment smoke

**Asset:** public third-party Ai2 Bot2 (`fiveages-sim`), classification
`PUBLIC_THIRD_PARTY_ASSET`. It has not been verified as the internal AI²
Robotics digital twin ([provenance](ALPHABOT2_SIM_ASSET_PROVENANCE.md)).

**Run:** 2026-09-28, Isaac Sim 4.5.0-rc.36 headless on one A800, CPU PhysX at
the asset-declared **200 Hz**, 20 physics steps per action (**10 Hz** policy).
Driven exclusively through the committed code:
`lmm.embodiments.alphabot2.isaac_adapter.AlphaBot2IsaacAdapter` →
`isaac_backend.IsaacArticulationBackend`.

**Reproduce** (on the host, Isaac's `python.sh`):
`scripts/alphabot2_isaac/live_smoke.py`. Raw output:
[`artifacts/alphabot2_live_smoke.json`](../artifacts/alphabot2_live_smoke.json),
[`artifacts/alphabot2_joint_calibration.json`](../artifacts/alphabot2_joint_calibration.json),
renders in [`artifacts/alphabot2_cameras/`](../artifacts/alphabot2_cameras/).
All 12 sections reported `ok`.

## Milestone criteria

| # | Criterion | Result | Evidence |
|---|---|---|---|
| 1 | Loads in Isaac | ✅ | 42 DOFs, 48 links, floating base (`fixed_base = False`), 112.1 kg; three asset fixes applied and reported (provenance doc) |
| 2 | Two 7-DoF arms identified | ✅ | 14 arm DOFs by path; per-side joint maps identified and verified against xacro FK |
| 3 | Two Inspire grippers controllable | ✅ | both sides monotonic 0.102 → 0.036 m, identical, mimics track exactly |
| 4 | Base motion works | ✅ | forward/reverse/±yaw signs correct, ratios 0.97–1.01 |
| 5 | 16-D state read correctly | ✅ | vendor home read back within 1.2e-4 rad, grippers 0.0 (open) |
| 6 | 18-D action executes correctly | ✅ | 28/28 per-joint moves, gripper sweep, base sweep, all through `apply_action` |
| 7 | RGB cameras render | ✅ | head + 2 wrist, 240×320×3 uint8 RGB, synchronised, head view unmirrored and upright |
| 8 | Arm/gripper/base in one transition | ✅ | one `apply_action` = 20 physics steps, all six channels moved |
| 9 | Snapshot/restore feasible | ✅ | exact readback; bitwise-identical branches ([audit](ALPHABOT2_SNAPSHOT_RESTORE_AUDIT.md)) |
| 10 | No stale WXAI/ALOHA semantics | ✅ | base is (v, ω) — no (vx, vy); gripper is closed-fraction — no carriage/snap threshold; physical units, no Habitat normalisation |

## 1. Census and load

* DOFs: 14 arm + 2 actuated gripper + 10 gripper mimic + 4 body + 2 head + 2
  drive wheels + 8 caster = **42**; wheels resolve to
  `Ai2_Bot2_Chassis/joints/driving_{left,right}_joint`.
* Physics dt 0.005 s, rendering dt 0.005 s, exactly 1 physics step per
  `world.step()` with and without rendering.
* Live arm limits equal the vendor limits pushed through the joint maps on all
  14 joints (right `joint3`/`joint5` after the limit patch).
* At the vendor home: **no self-contacts**; only the 2 drive wheels and 4
  casters touch the ground.

## 2. 16-D state

Vendor home `home_2` commanded through `apply_action`, read back through
`observe_proprio()` (vendor/xacro convention):

```
[ 0.8334, -0.0, 0.0, -1.7799, 0.0, -0.641, 0.0,  0.0001,
 -0.8334,  0.0, 0.0, -1.78,   0.0, -0.641, 0.0,  0.0   ]
```

Max error vs home 1.2e-4 rad. Gripper bases at chassis (0.2957, ±0.2270,
1.0534) — mirror residual 0.0 mm — approaching forward. Isaac vs xacro FK:
3.7 mm, 0.08° per side. No base pose is included in the state.

## 3. Per-joint arm calibration (all 14 joints)

Each joint moved ±0.15 rad from home in vendor convention through the adapter
(10 actions each). Checked: tracking, crosstalk onto the other 13 joints, the
gripper_base rotation axis vs the xacro FK prediction (catches sign inversions
and joint swaps), displacement vs FK, and motion of the *other* gripper
(catches left/right swaps).

| Side | Joint | Policy idx | USD DOF | sign | offset | track err | crosstalk | axis cos | disp err vs FK | other arm | pass |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|:--:|
| left | 1 | 0 | `left_joint1` | +1 | 0 | 0.00036 | 0.00005 | 1.00000 | 0.68 mm | 38 µm | ✅ |
| left | 2 | 1 | `left_joint2` | +1 | −π/2 | 0.00034 | 0.00038 | 1.00000 | 0.45 mm | 8 µm | ✅ |
| left | 3 | 2 | `left_joint3` | +1 | −π/2 | 0.00034 | 0.00043 | 1.00000 | 0.40 mm | 4 µm | ✅ |
| left | 4 | 3 | `left_joint4` | +1 | 0 | 0.00037 | 0.00036 | 1.00000 | 0.64 mm | 11 µm | ✅ |
| left | 5 | 4 | `left_joint5` | +1 | +π/2 | 0.00038 | 0.00039 | 1.00000 | 0.36 mm | 3 µm | ✅ |
| left | 6 | 5 | `left_joint6` | +1 | 0 | 0.00037 | 0.00038 | 1.00000 | 0.60 mm | 4 µm | ✅ |
| left | 7 | 6 | `left_joint7` | +1 | 0 | 0.00036 | 0.00037 | 1.00000 | 0.60 mm | 2 µm | ✅ |
| right | 1 | 8 | `right_joint1` | +1 | 0 | 0.00037 | 0.00037 | 1.00000 | 0.68 mm | 55 µm | ✅ |
| right | 2 | 9 | `right_joint2` | +1 | −π/2 | 0.00034 | 0.00039 | 1.00000 | 0.45 mm | 3 µm | ✅ |
| right | 3 | 10 | `right_joint3` | **−1** | +π/2 | 0.00037 | 0.00043 | 1.00000 | 0.40 mm | 4 µm | ✅ |
| right | 4 | 11 | `right_joint4` | +1 | 0 | 0.00036 | 0.00039 | 1.00000 | 0.64 mm | 11 µm | ✅ |
| right | 5 | 12 | `right_joint5` | **−1** | −π/2 | 0.00039 | 0.00038 | 1.00000 | 0.36 mm | 2 µm | ✅ |
| right | 6 | 13 | `right_joint6` | **−1** | 0 | 0.00036 | 0.00037 | 1.00000 | 0.60 mm | 3 µm | ✅ |
| right | 7 | 14 | `right_joint7` | +1 | 0 | 0.00037 | 0.00037 | 1.00000 | 0.60 mm | 1 µm | ✅ |

(track/crosstalk in rad, worst of the ± moves.) **28/28 pass.** No swaps, no
sign inversions *once the per-side map is applied*, no left/right swap, no
mimic coupling between arm joints. Home is safe (no self-contact). The
unmapped asset *does* have sign differences on right joints 3, 5, 6 and
offsets on joints 2, 3, 5 — that is exactly what the map encodes; see the
[kinematic audit](ALPHABOT2_SIM_KINEMATIC_AUDIT.md).

## 4. Gripper calibration

Policy command `g` (0 = open, 1 = closed) through `apply_action`, 1 s settle,
both sides:

| g | readback L / R | joint (rad) | pad-link separation L / R (m) |
|---:|---:|---:|---:|
| 0.00 | 0.0000 / 0.0000 | 0.8200 | 0.10162 / 0.10162 |
| 0.25 | 0.2375 / 0.2375 | 0.6252 | 0.08550 / 0.08550 |
| 0.50 | 0.4869 / 0.4869 | 0.4207 | 0.06836 / 0.06836 |
| 0.75 | 0.7369 / 0.7369 | 0.2157 | 0.05167 / 0.05167 |
| 1.00 | 0.9869 / 0.9869 | 0.0107 | 0.03617 / 0.03617 |

Monotonic, left = right exactly, all 5 mimic joints equal the actuated joint.
Readback lags the command by ≤ 0.013 (the position drive's steady-state
error), consistently in the closing direction.

## 5. Atomic 18-D action

One action changing every channel (left/right joint 1, both grippers, v = 0.15
m/s, ω = 0.3 rad/s): `world_steps_consumed = 20`, backend 20, adapter
`sim_steps_before/after` 10680 → 10700, sim time +0.100 s. All six channels
moved within that single transition. The adapter stages joint targets, then
the base twist, then advances physics once (also pinned by CPU tests).

## 6. Base calibration (2 s per test, then 0.5 s stop)

| Command | Along chassis +x | Lateral | Δyaw | Ratio |
|---|---:|---:|---:|---:|
| v = +0.2 m/s | +0.4029 m | 2.0 mm | 0.0004 | 1.007 |
| v = −0.2 m/s | −0.3946 m | 2.5 mm | −0.0091 | 0.986 |
| ω = +0.5 rad/s | 0.9 mm | 0.5 mm | **+0.9735** | 0.974 |
| ω = −0.5 rad/s | 1.1 mm | 1.3 mm | **−0.9756** | 0.976 |

Convention: +v along the chassis +x, +ω counter-clockwise about world +z.
Arms drift ≤ 8e-5 rad while driving. Wheel velocities stay inside the backend.

## 7. Cameras

Mounts: the asset ships camera mount frames but no camera prims. `cam_front`
is attached to `head_link2/head_camera` (an optical frame; 30.0° pitch down vs
37.0° in the xacro), 69° HFOV. The wrist cameras sit at each bracket's `d405`
prim, aimed at the pad midpoint + 0.10 m, with fingers at the image bottom,
87° HFOV. These are **spec/geometry-derived extrinsics, not vendor
calibration**. No depth.

| Camera | Shape | Content check |
|---|---|---|
| `cam_front` | 240×320×3 uint8 | red marker (dead ahead) at x = 0.499; green (robot-left, near) x = 0.264, y = 0.984; blue (robot-right, far) x = 0.648, y = 0.623 → **not mirrored, upright** |
| `cam_left_wrist` | 240×320×3 uint8 | grasp cube centred (x = 0.49, y = 0.52) between the fingers |
| `cam_right_wrist` | 240×320×3 uint8 | yellow target centred (x = 0.51, y = 0.42); fingers at the image bottom |

RGB order was confirmed with pure red/green/blue markers. All three cameras
report the same `rendering_frame`/`rendering_time` (synchronised; one frame per
action). The RTX frames show residual denoiser streaking; the content is
legible. With 3 cameras a policy step costs 0.26 s wall-clock.

The wrist views appear rolled about 90° at the vendor home pose because the
D405 bracket sits beside the gripper; that is the asset's geometry, not
mirroring (marker order checked in the right wrist view).

## 8. Grasp and snapshot/restore under contact

Top-down left-hand pick, planned with xacro IK inside the vendor limits and
rehearsed live without the object (no self-contacts; pad midpoint 7 mm from
the planned point, fingers separating horizontally, approach straight down). A 5 cm, 50 g cube on a pedestal: descend 8 cm, close, lift
8 cm, drive the base backwards while turning.
* Closed pad separation 0.0617 m (vs 0.036 m empty), so the fingers were
  stopped by the cube; 7 gripper bodies in contact with it.
* The cube rose 0.0824 m with the 0.08 m lift. It was still in contact after
  the base travelled 0.146 m.
* Snapshot/restore: see [`ALPHABOT2_SNAPSHOT_RESTORE_AUDIT.md`](ALPHABOT2_SNAPSHOT_RESTORE_AUDIT.md).

## 9. Bounded navigation

Tucked vendor home pose. Route: reverse 0.6 m, turn +90°, forward 1.0 m, turn
−90°, forward 0.5 m (2.1 m commanded, open loop). Four 0.3 × 0.3 × 0.5 m
obstacles sit off the route, plus the manipulation props.
* **Collision-free**: no robot ↔ obstacle contact over 156 actions (contact
  reports on every robot body, including the chassis box collider).
* Arms held within 1e-4 rad, chassis tilt 0.0°.
* Final pose (−0.171, +1.035) m, yaw +0.05 rad, against the ideal (−0.1, +1.0)
  m, 0 rad. The error is open-loop dead reckoning with the 0.974 yaw ratio;
  there is no planner.

**Earlier attempt, reported rather than hidden:** a previous run placed an
obstacle inside the final leg's swept path. The robot drove into it and the
contact report flagged it (chassis `base_footprint`, a caster and
`body_link1`). The route was fine; my obstacle layout was wrong and has been
moved.

## 10. Reset reproducibility and timing

Two `world.reset()` + home cycles: state difference 0.0 (bitwise), chassis
0.0 m. Timing: see [`ALPHABOT2_TEMPORAL_AUDIT.md`](ALPHABOT2_TEMPORAL_AUDIT.md).
It runs at 0.56× real time without rendering; boot takes ~70 s.

## Not covered by this smoke

* The real robot: every convention here is between two *public*
  descriptions. The real SDK's joint zeros/signs, gripper units and base
  signs are not established.
* Wrist/head extrinsics are not vendor-calibrated.
* Torso/neck stay fixed (v1 contract).
* No nav stack, no OVMM, no policy in the loop.
