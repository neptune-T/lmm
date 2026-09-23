"""AlphaBot 2 simulator adapter — blocked, deliberately non-functional.

Asset search performed 2026-09-21 across every channel requested:

1. AI² Robotics official documentation / downloads — no specification page and
   no developer download; the site routes to ``business@ai2robotics.com``.
2. Official SDK packages — none public.
3. Official GitHub / Gitee organisations — none found.
4. Robot packages bundled with public AlphaBot 2 tooling — the Hugging Face
   repositories contain only LeRobot data and policy weights.
5. Datasets / checkpoints referencing a robot-description package — the
   dataset ``meta/`` directories hold only ``info.json``, ``stats.json``,
   ``tasks.parquet`` and ``episodes/``. No URDF, calibration or kinematics.
6. Third-party assets — none. Public "AlphaBot2 URDF" hits resolve to the
   unrelated Waveshare Raspberry Pi car kit.

Result: **no credible full-body AlphaBot 2 simulator asset exists publicly**,
in any format (URDF/Xacro, meshes, USD/Isaac, MJCF/MuJoCo, SDF/Gazebo, ROS or
ROS2 ``robot_description``, collision models, camera frames, controller
configs).

Per the stop conditions, integration halts here. This module does not
fabricate geometry, does not stub joint states, and does not substitute a
different robot.
"""

from __future__ import annotations

from .contract import AlphaBot2Contract


class SimulatorAssetUnavailable(RuntimeError):
    """No AlphaBot 2 simulator body exists to load."""


MISSING_ASSETS = {
    "urdf_xacro": "NOT_FOUND",
    "meshes": "NOT_FOUND",
    "collision_models": "NOT_FOUND",
    "usd_isaac": "NOT_FOUND",
    "mjcf_mujoco": "NOT_FOUND",
    "sdf_gazebo": "NOT_FOUND",
    "ros_robot_description": "NOT_FOUND",
    "camera_frames": "NOT_FOUND",
    "controller_configs": "NOT_FOUND",
    "official_sdk": "NOT_FOUND",
}

_MESSAGE = (
    "AlphaBot 2 simulator integration is BLOCKED: no full-body robot "
    "description exists publicly in any format.\n\n"
    "Missing: " + ", ".join(sorted(MISSING_ASSETS)) + ".\n\n"
    "Frontier discovery and cheap rollout scaling both require a simulator, so "
    "this blocks the middle of the method for AlphaBot 2. The smallest "
    "unblocking asset is an official URDF/Xacro plus collision meshes and "
    "camera frames; an Isaac USD or MuJoCo MJCF would also work but would "
    "raise the simulator-backend question.\n\n"
    "Do not work around this by converting a different robot or by authoring "
    "approximate geometry: a fabricated body invalidates every collision, "
    "reachability and snapshot result computed on top of it.\n\n"
    "See docs/ALPHABOT2_PUBLIC_DATA.md."
)


class AlphaBot2SimAdapter:
    """Refuses to construct. Present so the gap cannot be overlooked."""

    def __init__(self, contract: AlphaBot2Contract) -> None:
        raise SimulatorAssetUnavailable(_MESSAGE)

    @staticmethod
    def availability() -> dict[str, object]:
        return {
            "available": False,
            "reason": "no public full-body robot description at any fidelity",
            "searched_channels": 6,
            **MISSING_ASSETS,
        }
