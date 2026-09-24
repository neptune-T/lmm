# AlphaBot 2 temporal audit

**Status: DECLARED values only. Nothing measured.**

A live measurement requires a running Isaac Sim, which is currently
unreachable ([`ALPHABOT2_ISAAC_BLOCKER.md`](ALPHABOT2_ISAAC_BLOCKER.md)).
Per instruction, no rate is invented — the table below separates what the
configuration *declares* from what remains unmeasured.

## Declared by the asset

| Quantity | Value | Source |
|---|---|---|
| `controller_manager.update_rate` | **50 Hz** | `config/ros2_control/ros2_controllers.yaml`, `inspire.yaml` |
| `ocs2_wbc_controller.update_rate` | 50 Hz | same |
| `ocs2_*.mpc_frequency` | **25 Hz** | same |
| `body_joint_controller.update_rate` | 50 Hz | same |
| `head_joint_controller.update_rate` | 50 Hz | same |
| nav2 `odom_duration` | 0.1 s | `config/nav2/nav2_params_isaac_gt.yaml` |
| Arm `home_duration` | 2.0 s | ros2_controllers.yaml |
| Body `home_duration` / `movej_duration` | 3.0 s | ros2_controllers.yaml |

## NOT measured — required before any data collection

| Quantity | Status | Why it matters |
|---|---|---|
| Isaac **physics** frequency | **UNMEASURED** | The controller's declared 50 Hz is the ros2_control rate, *not* the physics rate. They are different numbers and are routinely confused. |
| Isaac **rendering** / camera update rate | **UNMEASURED** | Determines whether images are synchronised with proprio at record time. |
| ROS 2 bridge round-trip rate | **UNMEASURED** | `topic_based_ros2_control` adds a topic hop between Isaac and the controller. |
| Physics steps per policy action | **UNMEASURED** | `AlphaBot2IsaacAdapter.physics_steps_per_action` is `None` and `apply_action` refuses until set. |
| Action hold duration | **UNMEASURED** | |
| Wall-clock throughput | **UNMEASURED** | Must be reported separately from simulation time. |
| Dataset recording FPS | **UNDECIDED** | Public AlphaBot datasets use 10, 15 and 30 fps; the right value depends on the measured physics rate. |

## Precedent that motivates this caution

The previous WXAI embodiment carried a nominal 50 Hz through configs, docs and
the policy-server health response. When finally measured on the live runtime it
was **120 Hz** — every derived quantity, including the intended dataset rate,
had been wrong. The contract now refuses to run rather than inherit a number.

## Measurement procedure, once Isaac is reachable

1. Read the physics dt from the Isaac stage; do not infer it from the
   controller rate.
2. Advance N physics steps, measure elapsed simulation time and wall-clock
   time separately; report both.
3. Measure camera update rate independently of the physics rate.
4. Apply one 18-D action via `apply_action` and record
   `sim_steps_before` / `sim_steps_after` to fix
   `physics_steps_per_action` empirically.
5. Only then choose the dataset FPS and the decimation factor, and record all
   three numbers (physics Hz, dataset FPS, ticks per recorded action) explicitly.
