# AlphaBot 2 temporal audit

**Status: MEASURED live** on 2026-09-28 (Isaac Sim 4.5.0-rc.36, headless, one
A800, CPU PhysX), on the public third-party Ai2 Bot2 asset. Evidence:
`artifacts/alphabot2_live_smoke.json` (sections `census`, `atomic_18d_action`,
`timing`, `cameras`), produced by `scripts/alphabot2_isaac/live_smoke.py`.

## Chosen and measured rates

| Quantity | Value | How it was established |
|---|---|---|
| Physics rate | **200 Hz** (dt = 0.005 s) | Declared by the asset's own scene, `humanoid/Ai2/Bot2/env/empty.usda` (`physxScene:timeStepsPerSecond = 200`, CPU dynamics, SAP broadphase). Applied through `World(physics_dt=0.005)` and read back live: `world.get_physics_dt() = 0.005`. |
| Rendering dt | 0.005 s | `rendering_dt` set equal to `physics_dt`; read back live. |
| Physics steps per `world.step()` | **1**, with and without render | `current_time_step_index` before/after one call, both modes. |
| Physics steps per policy action | **20** | `apply_action` measured: `world_steps_consumed = 20`, backend counter 20, adapter reports `sim_steps_before/after` 10680 → 10700; sim time advanced exactly 0.100 s. |
| Policy (action) rate | **10 Hz** | 200 Hz / 20. Chosen to match the fps of the dataset behind the audited public AlphaBot π0.5 checkpoint (`alphabot2/aibot2_grab_bag_combined_over_under`, 10 fps). The other public family uses 15 fps, which is not an integer divisor of 200 Hz. |
| Camera update | 1 frame per action | Cameras render on the last physics substep of each action only; all three cameras report the same `rendering_frame` and `rendering_time` (synchronised). |
| Wall clock, no render | 0.18 s per action → **0.56× real time** | 20 actions timed. |
| Wall clock, 3 cameras 320×240 | 0.26 s per action | 4 actions timed after warm-up. |
| Boot (SimulationApp headless) | ~70 s | |

**The 60 Hz trap.** Isaac's `World()` defaults to 60 Hz. The first live runs
on 2026-09-28 used that default without noticing; it is *not* the asset's
rate. Every number above is from runs at the asset-declared 200 Hz.

## Declared by the asset but not used on this route

| Quantity | Value | Source | Status |
|---|---|---|---|
| `controller_manager.update_rate` | 50 Hz | `config/ros2_control/ros2_controllers.yaml`, `inspire.yaml` | Not used: no ROS 2 on the host; the backend drives PhysX position drives directly. |
| `ocs2_*.mpc_frequency` | 25 Hz | same | Not used. |
| nav2 `odom_duration` | 0.1 s | `config/nav2/nav2_params_isaac_gt.yaml` | Not used. |

## Precedent that motivated measuring

The WXAI embodiment carried a nominal 50 Hz that turned out to be 120 Hz on
the live runtime. Here the same class of error was caught in the other
direction: the simulator's *default* (60 Hz) differed from the asset's
declared scene rate (200 Hz).

## Consequences recorded for later work

* A recorded dataset at 10 fps holds one frame per action; decimation factor
  = 20 physics ticks per recorded frame.
* Re-targeting to 15 fps needs a physics rate that 15 divides (e.g. 240 Hz);
  that is a deliberate change of the physics rate and must be re-measured, not
  assumed.
