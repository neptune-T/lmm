# AlphaBot 2 simulator asset provenance

> ## The simulation asset is a public third-party Ai2 Bot2 asset and has not yet been verified as the exact internal AI² Robotics digital twin.
>
> It is **not** an official AI² Robotics release. It is published by the
> `fiveages-sim` organisation. Classification: **`PUBLIC_THIRD_PARTY_ASSET`**.
> Do not describe it as official in any paper, report or commit message.

Audited 2026-09-24.

## Repositories and pinned revisions

| Repository | Commit (HEAD at audit) | License | Size | Last push |
|---|---|---|---|---|
| `github.com/fiveages-sim/robot_descriptions` | `bd491829961a8562306515001103af3b3737fbc7` | Apache-2.0 | ~31 MB (sparse) | — |
| `github.com/fiveages-sim/robot_usds` | `faeb3c29436b8d2421e73687e5f90df17cc90655` | Apache-2.0 | 433 MB | 2026-09-23 |
| `github.com/fiveages-sim/robot-descriptions-common` (submodule `common`) | `ed4d46a359d3672aa72b06d18e8aa529e5558f28` | Apache-2.0 | ~112 MB (sparse) | — |

Apache-2.0 permits use, modification and redistribution with attribution — no
license blocker for research use.

**Not vendored into the LMM source tree.** Both are external dependencies,
checked out under a scratch/workspace path. A vendor/submodule decision is
deferred until the stack is proven to launch.

## Ai2 Bot2 files used

Package `humanoid/Ai2Robotics/ai2_bot2_description`:

```
xacro/robot.xacro                         full robot composition root
xacro/components/arm.xacro                Ai2Bot2Arm macro (both arms)
xacro/components/body.xacro               waist + head chain
xacro/components/chassis.xacro            Ai2Bot2Chassis macro (base + wheels)
xacro/ros2_control/{robot,arm,body}.xacro ros2_control hardware interfaces
config/ros2_control/ros2_controllers.yaml controller set, no gripper
config/ros2_control/inspire.yaml          controller set WITH grippers
config/nav2/nav2_params_isaac_gt.yaml     nav2 + Isaac ground-truth odometry
config/ocs2/{task,split,gripper_inspire,split_inspire}.info  OCS2 tasks
meshes/Visual/arm/*.glb                   arm visual meshes
meshes/Collision/arm/ZPF_{left,right}_Link*.STL   arm collision meshes
meshes/Visual/chassis/*.glb               chassis visual meshes
meshes/Collision/body/*.STL               body collision meshes
```

## Inspire gripper files used

Package `gripper/inspire_description` (inside `common`):

```
xacro/EG2-4C2.xacro                       Gripper macro, 1 actuated + 5 mimic joints
xacro/ros2_control/EG2-4C2.xacro          gripper hardware interface
meshes/EG2-4C2/{baselink,inner_link,outer_link,pad}.obj
```

Selected by `type:=inspire`, which sets `selected_gripper_type = inspire_eg2`
and mounts `{side}_gripper_base` to `{side}_ee` with a +90° z rotation.

## Camera / sensor assets referenced

| Asset | Package | Used for |
|---|---|---|
| `realsense/d405.glb` | `sensor_models` | both wrist cameras (on `{side}_link8`) |
| `orbbec/ms500.glb` | `sensor_models` | chassis lidar (`lidar_link`) |
| `orbbec/Dabai_DW.glb` | `sensor_models` | chassis camera |
| `ee/camera_stand_d405.glb`, `ee/ee_fb_camera.glb` | `rm75_description` | wrist camera mount |
| `woosh_agv/wheel.glb`, `woosh_agv/panel.glb` | `aidal_description` | drive wheels, chassis panel |

## Isaac Sim USD assets

Repository `robot_usds`:

```
humanoid/Ai2/Bot2/Bot2.usda        7,282 B  composition root
humanoid/Ai2/Bot2/payloads/        USD payloads
humanoid/Ai2/Bot2/components/      component USDs
humanoid/Ai2/Bot2/Textures/        textures
humanoid/Ai2/Bot2/env/             environment
grippers/Inspire/                  Inspire gripper USDs
```

`Bot2.usda` is a small composition root (7 KB); the geometry lives in
`payloads/`. **The USD contents have not been opened or validated** — that
requires Isaac Sim, which is currently unreachable (see
`ALPHABOT2_ISAAC_BLOCKER.md`). Collision/physics fidelity of the USD is
therefore **UNKNOWN**, not confirmed.

## Cross-package dependencies

The Ai2 Bot2 description does **not** stand alone. It references
`inspire_description`, `sensor_models`, `rm75_description` and
`aidal_description`. A working checkout needs `robot_descriptions` **plus** its
`common` submodule **plus** the `manipulator/Realman` package. All submodule
URLs are `git@github.com:` (SSH), so an HTTPS mirror or SSH credentials are
required.

## Notable provenance observation

The arm end-effector meshes come from `rm75_description` — the **Realman RM75**,
a 7-DoF arm. The Ai2 Bot2 arm shares its wrist-camera mount and EE feedback
camera with that package. This is consistent with a real robot built around a
7-DoF Realman-class arm, but it also means part of the modelled EE geometry is
inherited from a different vendor's package rather than measured from AlphaBot 2.
