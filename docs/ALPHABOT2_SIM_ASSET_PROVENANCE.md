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
`payloads/`. The USD was opened and simulated live on 2026-09-28 — see
"Live qualification route" below. It carries full PhysX articulation physics
(drives, limits, colliders, masses) once the `Physics` variant is `physx`.

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

## Live qualification route (2026-09-28)

| Item | Value |
|---|---|
| Host | restored A800 host (`aibox-…-dpddc`), GPU 1 (`CUDA_VISIBLE_DEVICES=1`); GPU 0 belongs to another user |
| Simulator | Isaac Sim **4.5.0-rc.36**, headless `SimulationApp`, CPU PhysX, RTX renderer |
| Isaac install | private copy at `/nix/yanjun/zts/alphabot2/isaacsim-4.5` of `/share/cys/isaacsim-4.5` (read-only use of the original). `/workspace/issaacsim-4.5` is on bosfs2, which drops exec bits — unrunnable. |
| `robot_usds` | `faeb3c29436b8d2421e73687e5f90df17cc90655`, clean tree, cloned on the host |
| `robot_descriptions` | `bd491829961a8562306515001103af3b3737fbc7`, sparse (`humanoid/Ai2Robotics/ai2_bot2_description/xacro`), used only as the kinematic reference for FK checks |
| `robot-descriptions-common` | not needed on this route: the Inspire EG2-4C2 gripper is taken from `robot_usds/grippers/Inspire/EG2_4C2/Inspire_EG2_4C2.usda` |
| Control route | Isaac Python articulation API (`isaacsim.core.prims.SingleArticulation` + PhysX tensor view). **No ROS 2** exists on the host, so the asset's `topic_based_ros2_control` path is not used. |
| Code | `lmm/embodiments/alphabot2/isaac_backend.py` (stage build + live backend), `scripts/alphabot2_isaac/live_smoke.py` |

### Launch quirk: RTX driver-version false negative

Driver 535.261.03 has a minor version (261) that overflows the 8-bit Vulkan
`driverVersion` field; RTX reads it as 535.5, declares the driver too old and
then hangs. The backend appends `--/rtx/verifyDriverVersion/enabled=false`
(`isaac_backend.RTX_DRIVER_CHECK_BYPASS`) before `SimulationApp` starts.
Rendering was verified afterwards (RGB frames from three cameras).

### Asset fixes applied at load time (all in `isaac_backend.build_stage`, all reported)

1. **Gripper re-attachment.** With `EE=EG2_4C2` the gripper's rigid bodies are
   nested under the arm's `link7`; PhysX folds nested bodies into their
   ancestor, so the fingers never became DOFs. The EE variant is set to `None`
   and the gripper is referenced as a sibling `Gripper_{Left,Right}` at the
   identical pose (re-attachment error 0.0 m), welded to `link7` by a
   FixedJoint.
2. **World anchor removed.** The standalone gripper USD ships `root_joint`, a
   FixedJoint with an empty `body0` — a weld to the world. Left active it made
   the whole robot fixed-base through its hands. Deactivated; the robot is
   floating-base on its wheels (`fixed_base = False`, root link `body_link4`).
3. **Right-arm joint limits.** The USD right arm reuses the left arm's joint
   definitions (`Side/right.usda` differs from `left.usda` only in joint names
   and a y-mirrored link1 *visual*). Its joints 3 and 5 therefore ship the left
   arm's ranges, which — through the identified right-arm joint map — exclude
   a 0.083 rad band containing the vendor home pose. Widened to the mapped
   vendor range: `right_joint3` [−267.6°, 87.6°] → [−87.6°, 267.6°],
   `right_joint5` [−87.6°, 267.6°] → [−267.6°, 87.6°].

Also set: `ROS` variant `none` (no ROS bridge graphs).

### Still not established

Nothing above shows that this asset is the internal AI² Robotics digital twin.
The live runs establish that the *public asset* is internally consistent and
controllable, and how its joint conventions relate to the public xacro — not
that either matches the real robot's SDK conventions.
