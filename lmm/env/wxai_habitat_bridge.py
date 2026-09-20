"""Habitat boundary for the WXAI differential-drive embodiment.

Complements :mod:`lmm.policy.wxai_transforms`, which owns the OpenPI boundary.
Dimension labels and indices are imported from there so the two boundaries
cannot drift apart.

This module is deliberately dependency-light (numpy only). It is shared by the
Habitat process (Python 3.9), the policy process (Python 3.11) and the unit
tests, so an index or unit change cannot hide in either runtime.

Units
-----
The policy boundary is **physical**: arm targets in radians, gripper closed
fractions in ``[0, 1]``, base velocities in m/s and rad/s.

Habitat's ``BaseVelAction`` is **normalized** ``[-1, 1]``, scaled by the
overlay's ``lin_speed`` / ``ang_speed``. That conversion happens in exactly one
function, :func:`to_habitat_base`. The previous implementation returned raw
m/s from a function documented as producing Habitat ``[forward, yaw]``, which
would have executed ``vx = 0.09 m/s`` as ``0.0227 m/s``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..policy.wxai_transforms import (
    ACTION_DIM,
    ACTION_LABELS,
    ARM_INDICES,
    BASE_INDICES,
    GRIPPER_INDICES,
    STATE_DIM,
    STATE_LABELS,
    ContractError,
    validate_gripper_range,
)

__all__ = [
    "ACTION_LABELS", "STATE_LABELS", "BaseLimits", "ContractError", "WXAICommand",
    "decode_policy_action", "encode_habitat_observation", "from_habitat_base",
    "to_habitat_base",
]


#: joint_positions indices owned by each gripper's two carriages, measured on
#: the live A800 runtime (2026-09-20). Link ids are NOT joint indices: the arm
#: occupies links 20-25 / 43-48 but jp indices 8-13 / 16-21.
CARRIAGE_JP_INDICES: dict[int, tuple[int, int]] = {0: (15, 14), 1: (23, 22)}

#: Measured control period. One env.step advances exactly 1/120 s of sim time,
#: confirmed independently by yaw (0.5/120 rad) and forward (0.25/120 m).
CONTROL_HZ = 120.0

#: Carriage travel at full open, from the URDF and MobileManipulatorParams.
GRIPPER_OPEN_M = 0.044


@dataclass(frozen=True)
class BaseLimits:
    """Differential-drive scaling from the Habitat overlay.

    Defaults mirror ``habitat_patch/trossen_mobile_ai/ovmm_trossen_mobile_ai.yaml``
    (``lin_speed: 0.25``, ``ang_speed: 0.5``). Pass explicitly rather than
    relying on the defaults whenever the overlay is reconfigured.
    """

    lin_speed_mps: float = 0.25
    ang_speed_radps: float = 0.5

    def __post_init__(self) -> None:
        if self.lin_speed_mps <= 0 or self.ang_speed_radps <= 0:
            raise ContractError("base speed limits must be positive")


@dataclass(frozen=True)
class WXAICommand:
    """One decoded policy command in physical units."""

    arm_joint_targets_rad: np.ndarray          # [12]
    left_gripper_closed_fraction: float
    right_gripper_closed_fraction: float
    base_forward_mps: float
    base_yaw_radps: float

    def habitat_arm_args(self) -> dict[str, np.ndarray]:
        """Arguments for the overlay's ``BimanualArmAction``.

        The overlay takes ``[-1, 1]`` grip inputs with negative meaning open.
        A closed fraction ``g`` maps to ``2g - 1``, preserving ``0 = open`` and
        ``1 = closed``.
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

    @property
    def base_is_stationary(self) -> bool:
        return self.base_forward_mps == 0.0 and self.base_yaw_radps == 0.0


def decode_policy_action(action: Sequence[float]) -> WXAICommand:
    """Split one physical 16-D action into its actuator groups.

    Performs no clipping or rescaling: range enforcement belongs to
    :func:`to_habitat_base` and the actuator call, where it is reported.
    """
    value = np.asarray(action, dtype=np.float64).reshape(-1)
    if value.shape != (ACTION_DIM,):
        raise ContractError(f"expected one {ACTION_DIM}-D action, got {value.shape}")
    if not np.isfinite(value).all():
        raise ContractError("action contains non-finite values")
    validate_gripper_range(value, "action")
    return WXAICommand(
        arm_joint_targets_rad=value[list(ARM_INDICES)].copy(),
        left_gripper_closed_fraction=float(value[GRIPPER_INDICES[0]]),
        right_gripper_closed_fraction=float(value[GRIPPER_INDICES[1]]),
        base_forward_mps=float(value[BASE_INDICES[0]]),
        base_yaw_radps=float(value[BASE_INDICES[1]]),
    )


def to_habitat_base(
    command: WXAICommand, limits: BaseLimits = BaseLimits(), *, strict: bool = True
) -> np.ndarray:
    """Convert physical base velocity to Habitat's normalized ``[forward, yaw]``.

    The one and only unit boundary for the base. ``strict=True`` refuses a
    command the configured actuator cannot reach, rather than silently
    saturating it — a saturated command would make the executed trajectory
    diverge from the commanded one without any record of it.
    """
    forward = command.base_forward_mps / limits.lin_speed_mps
    yaw = command.base_yaw_radps / limits.ang_speed_radps
    normalized = np.asarray([forward, yaw], dtype=np.float32)
    if strict and np.max(np.abs(normalized)) > 1.0 + 1e-6:
        raise ContractError(
            f"base command forward={command.base_forward_mps:.4f} m/s, "
            f"yaw={command.base_yaw_radps:.4f} rad/s exceeds the configured "
            f"actuator limits (+-{limits.lin_speed_mps} m/s, "
            f"+-{limits.ang_speed_radps} rad/s); normalized={normalized.tolist()}"
        )
    return np.clip(normalized, -1.0, 1.0)


def from_habitat_base(
    normalized: Sequence[float], limits: BaseLimits = BaseLimits()
) -> tuple[float, float]:
    """Inverse of :func:`to_habitat_base`, for tests and logging."""
    value = np.asarray(normalized, dtype=np.float64).reshape(-1)
    if value.shape != (2,):
        raise ContractError(f"expected 2-D normalized base command, got {value.shape}")
    return (
        float(value[0] * limits.lin_speed_mps),
        float(value[1] * limits.ang_speed_radps),
    )


def encode_habitat_observation(
    habitat_observation: Mapping[str, Any], *, prompt: str
) -> dict[str, Any]:
    """Convert Habitat keys to the policy input schema.

    Emits raw HWC images; :class:`lmm.policy.wxai_transforms.WXAIInputs`
    normalizes layout and dtype, so image handling lives in one place.
    """
    state = np.asarray(habitat_observation["trossen_bimanual_joint"], dtype=np.float64)
    if state.shape != (STATE_DIM,):
        raise ContractError(f"expected {STATE_DIM}-D WXAI state, got {state.shape}")
    if not np.isfinite(state).all():
        raise ContractError("state contains non-finite values")
    validate_gripper_range(state, "state")

    images = {}
    for source, key in (
        ("cam_high", "head_rgb"),
        ("cam_left_wrist", "left_wrist_rgb"),
        ("cam_right_wrist", "right_wrist_rgb"),
    ):
        if key in habitat_observation:
            images[source] = np.asarray(habitat_observation[key])
    if "cam_high" not in images:
        raise ContractError("head_rgb is required; it is the policy's base view")

    return {"state": state, "images": images, "prompt": str(prompt)}
