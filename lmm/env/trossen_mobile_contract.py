"""Exact OpenPI Mobile ALOHA 14-D state / 16-D action contract.

This module is deliberately dependency-light.  It is shared by the Habitat
process (Python 3.9), the OpenPI policy process (Python 3.11), and the unit
tests, so index or unit changes cannot be hidden in either runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


STATE_LABELS = (
    *(f"left_joint_{i}" for i in range(6)),
    "left_gripper_closed_fraction",
    *(f"right_joint_{i}" for i in range(6)),
    "right_gripper_closed_fraction",
)
ACTION_LABELS = (*STATE_LABELS, "base_velocity_x", "base_velocity_y")


@dataclass(frozen=True)
class TrossenMobileCommand:
    """One command in physical (not normalized) units."""

    arm_joint_targets_rad: np.ndarray
    left_gripper_closed_fraction: float
    right_gripper_closed_fraction: float
    base_velocity_xy: np.ndarray

    def habitat_arm_args(self) -> dict[str, np.ndarray]:
        """Arguments expected by ``BimanualArmAction``.

        Habitat's overlay uses [-1, 1] grip commands with negative=open and
        non-negative=closed, while OpenPI uses a continuous closed fraction.
        """
        return {
            "arm_action": self.arm_joint_targets_rad.astype(np.float32),
            "left_grip_action": np.asarray(
                [2.0 * self.left_gripper_closed_fraction - 1.0], dtype=np.float32
            ),
            "right_grip_action": np.asarray(
                [2.0 * self.right_gripper_closed_fraction - 1.0], dtype=np.float32
            ),
        }


def decode_openpi_action(action: Sequence[float]) -> TrossenMobileCommand:
    """Decode the native OpenPI ordering without clipping or rescaling."""
    value = np.asarray(action, dtype=np.float32).reshape(-1)
    if value.shape != (16,) or not np.isfinite(value).all():
        raise ValueError(f"expected one finite 16-D action, got {value.shape}")
    grips = value[[6, 13]]
    if not np.logical_and(grips >= 0.0, grips <= 1.0).all():
        raise ValueError(
            "OpenPI action dims 6 and 13 are closed fractions and must be in [0, 1]"
        )
    return TrossenMobileCommand(
        arm_joint_targets_rad=np.concatenate((value[:6], value[7:13])),
        left_gripper_closed_fraction=float(value[6]),
        right_gripper_closed_fraction=float(value[13]),
        base_velocity_xy=value[14:16].copy(),
    )


def encode_openpi_observation(
    habitat_observation: Mapping[str, Any], *, prompt: str
) -> dict[str, Any]:
    """Convert Habitat camera/state keys to the public ALOHA policy schema."""
    state = np.asarray(habitat_observation["trossen_bimanual_joint"], dtype=np.float32)
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError(f"expected finite 14-D Trossen state, got {state.shape}")
    if not np.logical_and(state[[6, 13]] >= 0.0, state[[6, 13]] <= 1.0).all():
        raise ValueError("state gripper closed fractions must be in [0, 1]")

    def chw(name: str) -> np.ndarray:
        image = np.asarray(habitat_observation[name])
        if image.ndim != 3 or image.shape[-1] not in (3, 4):
            raise ValueError(f"{name} must be HWC RGB/RGBA, got {image.shape}")
        return np.moveaxis(image[..., :3], -1, 0).astype(np.uint8, copy=False)

    return {
        "state": state,
        "images": {
            "cam_high": chw("head_rgb"),
            "cam_left_wrist": chw("left_wrist_rgb"),
            "cam_right_wrist": chw("right_wrist_rgb"),
        },
        "prompt": str(prompt),
    }


def assert_differential_base_executable(
    command: TrossenMobileCommand, *, lateral_tolerance_mps: float = 1e-4
) -> np.ndarray:
    """Map the executable x-only subset to Habitat ``[forward, yaw]``.

    The OpenPI contract's second base channel is lateral velocity, not yaw.
    The installed Trossen Mobile AI URDF has a differential base.  Silently
    sending y velocity to Habitat's yaw channel would therefore be wrong; this
    function fails closed unless lateral velocity is effectively zero.
    """
    vx, vy = map(float, command.base_velocity_xy)
    if abs(vy) > lateral_tolerance_mps:
        raise RuntimeError(
            "OpenPI requested lateral base velocity "
            f"vy={vy:.6f} m/s, but the installed Trossen Mobile AI backend is "
            "differential-drive. It must not be reinterpreted as yaw velocity."
        )
    return np.asarray([vx, 0.0], dtype=np.float32)

