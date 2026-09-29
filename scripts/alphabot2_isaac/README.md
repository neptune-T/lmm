# AlphaBot 2 on Isaac Sim — live scripts

Run under Isaac Sim 4.5's `python.sh` on a GPU host. They are not part of the
CPU test suite.

| Script | Purpose |
|---|---|
| `live_smoke.py` | Full embodiment smoke through `AlphaBot2IsaacAdapter` + `IsaacArticulationBackend`. Writes `live_smoke.json`, `joint_calibration.json` and camera PNGs to `$SMOKE_OUT`. The committed `artifacts/alphabot2_*` files are its 2026-09-28 output. |
| `xacro_fk.py` | numpy FK / Jacobian / IK of the public Ai2 Bot2 xacro (vendor joint convention). Used by the smoke for per-joint sign checks and grasp planning. |
| `cartesian_live.py` | Live qualification of `CartesianExecutor`: reachability, line/circle/rotation paths per arm, replay of recorded dataset targets vs the real robot's joints. Output: `artifacts/alphabot2_cartesian_live.json`. |
| `pi05_zero_shot.py` | Zero-shot smoke of the public pi0.5 checkpoint (client of `scripts/alphabot2_pi05/pi05_server.py`), v = omega = 0. Output: `artifacts/alphabot2_pi05_zero_shot/`. |
| `expert_pilot.py` | OFFLINE privileged expert: navigate -> pick -> (navigate -> place), recorded at 10 Hz as CartesianMobileAction18, then every episode replayed. Output: `artifacts/alphabot2_pilot/` (metadata), data on the host. |
| `sync_check.py` | Camera/state/action alignment (block + arm stimuli): images must be read after the action. Output: `artifacts/alphabot2_sync_check.json`. |
| `gripper_geometry_probe.py` | EG2-4C2 finger collision extent along the approach (open/closed). |
| `parallel_probe.py` | N robots per scene, CPU vs GPU PhysX throughput. Output: `artifacts/alphabot2_parallel_probe.json`. |
| `identification/` | The scripts that identified the per-side USD↔xacro joint maps, **kept exactly as run** on 2026-09-28 (they import the pre-repository builder `bot2_build.py` and use host paths). Output: `artifacts/alphabot2_arm_map_identification.json`. |

```bash
export CUDA_VISIBLE_DEVICES=1 LMM_REPO=/path/to/lmm SMOKE_OUT=/tmp/alphabot2_smoke
export ROBOT_USDS=/path/to/robot_usds           # fiveages-sim/robot_usds @ faeb3c29
export BOT2_XACRO=/path/to/robot_descriptions/humanoid/Ai2Robotics/ai2_bot2_description/xacro/components
setsid nohup /path/to/isaacsim-4.5/python.sh scripts/alphabot2_isaac/live_smoke.py > smoke.log 2>&1 < /dev/null &
```

Host notes (RTX driver-check bypass, detached launch over the relay) are in
`docs/ALPHABOT2_SIM_ASSET_PROVENANCE.md`.
