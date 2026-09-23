"""AlphaBot 2 real-robot adapter — interface only.

The SDK is not public: no API reference, no example code, no package. Every
method raises. The interface *shape* is committed now so simulation and real
hardware are forced to share one policy-facing contract:

    canonical contract -> sim adapter  -> AlphaBot 2 simulator
    canonical contract -> real adapter -> AlphaBot 2 SDK

Robot-specific detail stays behind this boundary and must not reach the VLA
policy interface.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence

import numpy as np

from .contract import (
    ACTION_DIM,
    BASE_FORWARD_INDEX,
    BASE_YAW_INDEX,
    GRIPPER_INDICES,
    LEFT_ARM_INDICES,
    RIGHT_ARM_INDICES,
    AlphaBot2Contract,
    ContractError,
)


class AlphaBot2SDK(Protocol):
    """The vendor surface this adapter would drive. Not publicly documented."""

    def read_arm_joint_positions(self) -> Sequence[float]: ...
    def command_arm_joint_targets(self, targets: Sequence[float]) -> None: ...
    def read_gripper(self) -> Sequence[float]: ...
    def command_gripper(self, closed_fractions: Sequence[float]) -> None: ...
    def command_base(self, forward_mps: float, yaw_radps: float) -> None: ...
    def read_cameras(self) -> dict[str, Any]: ...


_NO_SDK = (
    "The AlphaBot 2 SDK is not publicly documented (audited 2026-09-21): no "
    "API reference, example code or package was found. Unresolved and "
    "required before this adapter can be written: base command form, units, "
    "limits, sign convention and rate; whether the base is differential or "
    "omnidirectional; the gripper's physical unit and open/closed limits; and "
    "whether the primary arm interface is joint-space or Cartesian."
)


def split_action(action: Sequence[float]) -> dict[str, Any]:
    """Split a physical 18-D action into actuator groups.

    Pure and SDK-independent, so the contract's dimension ordering is testable
    without any robot. Performs no unit conversion — that belongs to whichever
    adapter owns the hardware.
    """
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    if a.shape != (ACTION_DIM,):
        raise ContractError(f"expected {ACTION_DIM}-D action, got {a.shape}")
    if not np.isfinite(a).all():
        raise ContractError("action contains non-finite values")
    g = a[list(GRIPPER_INDICES)]
    if g.min() < -1e-6 or g.max() > 1.0 + 1e-6:
        raise ContractError("gripper values must be closed fractions in [0, 1]")
    return {
        "left_arm_joint_targets_rad": a[list(LEFT_ARM_INDICES)].copy(),
        "right_arm_joint_targets_rad": a[list(RIGHT_ARM_INDICES)].copy(),
        "left_gripper_closed_fraction": float(a[GRIPPER_INDICES[0]]),
        "right_gripper_closed_fraction": float(a[GRIPPER_INDICES[1]]),
        "base_forward_mps": float(a[BASE_FORWARD_INDEX]),
        "base_yaw_radps": float(a[BASE_YAW_INDEX]),
    }


class AlphaBot2RealAdapter:
    """Policy-facing real-robot boundary. Unimplemented by design."""

    def __init__(self, contract: AlphaBot2Contract, sdk: AlphaBot2SDK | None = None) -> None:
        self.contract = contract
        self._sdk = sdk

    def read_proprio(self):
        raise NotImplementedError(_NO_SDK)

    def send_action(self, action: Sequence[float]) -> None:
        split_action(action)          # validate before failing, so shape bugs surface first
        raise NotImplementedError(_NO_SDK)

    def read_cameras(self):
        raise NotImplementedError(_NO_SDK)

    @staticmethod
    def availability() -> dict[str, object]:
        return {
            "available": False,
            "sdk_public": False,
            "base_semantics": "UNKNOWN",
            "base_kinematics": "UNKNOWN",
            "gripper_physical_units": "UNKNOWN",
            "arm_interface_joint_or_cartesian": "UNKNOWN",
        }
