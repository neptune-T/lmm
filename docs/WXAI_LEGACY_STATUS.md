# WXAI: historical / feasibility embodiment

**Status: FROZEN 2026-09-21.** Superseded as the canonical target by AI² Robotics
AlphaBot 2. Nothing was deleted; all commits, reports, tests and runtime notes
remain in this repository and on the A800.

Debugging of WXAI OVMM navigation is **stopped**. It should resume only if
needed to reproduce a historical result.

## Last canonical commits

| Commit | What it established |
|---|---|
| `844053e` | Aligned canonical LMM with the restored A800 dual-arm OVMM runtime |
| `75d76db` | Audited and gated Trossen Mobile pi0.5 semantics |
| `d1fb0f7` | Replaced the ALOHA transform stack with the symmetric WXAI 14D/16D contract |
| `545ce8c` | Audited WXAI embodiment calibration, specified the data path |
| `413e34d` | Calibrated the WXAI embodiment on the live A800 Habitat runtime |
| `4999312` | Froze the WXAI temporal contract, added the OVMM demonstration oracle |
| `b288d13` | Diagnosed navigation: the robot is loaded lying on its side |

## What was verified on real hardware-in-the-loop simulation

* **14-D state / 16-D action contract**, with dims 14–15 as forward/yaw
  differential-drive velocity. Per-dimension round-trip tested.
* **Joint indexing**: policy dims 0–5 / 7–12 map to the 12 arm joints; the
  measured `joint_positions` offsets are stable and cross-checked at load.
* **Gripper actuation**: the original overlay set a *motor target* that physics
  never integrated, so every gripper command was silently a no-op. Fixed by
  writing carriage joint positions directly; exact for all five tested
  fractions, on both hands, verified through the overlay itself rather than an
  adapter workaround.
* **Base signs**: forward and yaw signs confirmed correct.
* **Temporal contract**: simulator/control rate measured at **120 Hz** (not the
  assumed 50 Hz). Dataset rate chosen explicitly rather than inherited.
* **Atomic action**: one 16-D policy action maps to one logical control
  transition; arms + grippers + base no longer consume two env transitions.
* **`allow_dyn_slide`**: `false` is unrunnable on this Habitat revision; `true`
  frozen as the canonical setting in both the installed and source overlays.

## What remained blocked

**The URDF is ROS z-up and the articulation is loaded into Habitat's y-up world
without the correction ever being applied to the physical body.** The
`base_transformation` property post-multiplies `rotation(-pi/2, x)`, but that
only changes the *reported* transform used for pose maths.

Consequences measured with the base at a navigable free-space point:
`cam_high_link` at y=0.096 instead of ~1.4; left EE +0.357 above the base while
the right EE is −0.288 *below* it (underground); `base_link` buried to −0.31 m
and penetrating at every pose including open free space. This produced 11,817
robot-scene collisions and is the reason the oracle never reached manipulation.

Ruled out by measurement: navmesh agent radius, arm safe pose, and the navmesh
path itself. Full analysis in
[`WXAI_NAVIGATION_ROOT_CAUSE.md`](WXAI_NAVIGATION_ROOT_CAUSE.md).

The fix is known (rotate the articulation at load, re-derive
`base_transformation`, re-verify every world-frame quantity) but is **not being
pursued**.

## Why the project moved to AlphaBot 2

WXAI was a *feasibility* embodiment: a dual-arm robot that happened to be
available inside Habitat/OVMM, used to prove the end-to-end contract, the
frontier machinery and the data path. It was never the platform the final
experiments would run on.

AlphaBot 2 is the actual real-robot platform available for final experiments,
so it becomes the canonical target. Everything WXAI established that is
*embodiment-independent* carries over unchanged — the schemas, the frontier
estimator, the RL grouping, the evaluation split, the precision-backend
registry. What does not carry over is the WXAI-specific 14D/16D contract, its
joint indexing, its gripper convention and its Habitat adapter.

**Do not copy WXAI conventions into AlphaBot 2 without independent
verification.** In particular the 6-DoF-arm, 1-DoF-gripper, differential-drive
forward/yaw shape is specific to WXAI and does not describe AlphaBot 2.

## Preserved artifacts

* [`WXAI_EMBODIMENT_AUDIT.md`](WXAI_EMBODIMENT_AUDIT.md)
* [`WXAI_LIVE_CALIBRATION_20260920.md`](WXAI_LIVE_CALIBRATION_20260920.md)
* [`WXAI_TEMPORAL_CONTRACT.md`](WXAI_TEMPORAL_CONTRACT.md)
* [`WXAI_DATA_PREPARATION.md`](WXAI_DATA_PREPARATION.md)
* [`WXAI_NAVIGATION_ROOT_CAUSE.md`](WXAI_NAVIGATION_ROOT_CAUSE.md)
* `lmm/policy/wxai_transforms.py`, `lmm/env/wxai_habitat_bridge.py`,
  `lmm/env/trossen_habitat_backend.py`, `tests/test_wxai_contract.py`
* A800 snapshot `/nix/yanjun/zts/lmm/canonical_lmm` plus the overlay patches in
  `lmm_rollout_project/habitat_patch/trossen_mobile_ai/`

Frontier and RL logic were never WXAI-specific and are untouched by this
freeze.
