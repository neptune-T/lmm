"""Policy-facing Cartesian mobile action and its execution on the qualified joint interface.

``CartesianMobileAction18`` keeps the public AlphaBot pi0.5 checkpoint's 16
manipulation dims exactly as trained and appends the base::

    0:3    left EE xyz          (m, checkpoint frame B)
    3:7    left EE quaternion   (x, y, z, w)
    7:10   right EE xyz
    10:14  right EE quaternion
    14     left gripper         (0 = open, 1 = closed)
    15     right gripper
    16     base forward velocity v      (m/s)
    17     base yaw velocity omega      (rad/s, counter-clockwise)

Execution never bypasses the calibrated joint path::

    CartesianMobileAction18 -> (frame B, tool E) -> gripper_base target
        -> xacro IK (limits, seeded by the previous solution)
        -> JointMobileAction18 (vendor joints) -> AlphaBot2IsaacAdapter
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import kinematics as K
from . import pi05_contract as P
from .contract import ACTION_DIM, ContractError

CARTESIAN_MOBILE_ACTION18_LABELS: tuple[str, ...] = (*P.ACTION_NAMES, "base.v", "base.omega")
CART_LEFT_XYZ, CART_LEFT_QUAT = slice(0, 3), slice(3, 7)
CART_RIGHT_XYZ, CART_RIGHT_QUAT = slice(7, 10), slice(10, 14)
CART_LEFT_GRIPPER, CART_RIGHT_GRIPPER, CART_BASE_V, CART_BASE_OMEGA = 14, 15, 16, 17
_SLOTS = {"left": (CART_LEFT_XYZ, CART_LEFT_QUAT, CART_LEFT_GRIPPER, slice(0, 7), 7),
          "right": (CART_RIGHT_XYZ, CART_RIGHT_QUAT, CART_RIGHT_GRIPPER, slice(8, 15), 15)}

#: IK validity thresholds for one action.
IK_POS_TOL_M = 0.005
IK_ROT_TOL_RAD = np.deg2rad(2.0)
#: Largest joint change one 0.1 s action may ask for (xacro velocity limit x dt).
MAX_JOINT_STEP_RAD = K.JOINT_VEL_LIMIT * 0.1


@dataclass
class CartesianExecutor:
    """Lowers CartesianMobileAction18 to JointMobileAction18, one arm at a time.

    Seeds each IK with the previous joint solution, so a smooth Cartesian
    target stream gives a smooth joint stream without elbow flips. Every step
    is reported; nothing is silently clamped except by the joint limits.
    """

    q: dict[str, np.ndarray]                     # current vendor joints per arm (the IK seed)
    prev_quat: dict[str, np.ndarray | None] = field(default_factory=lambda: {"left": None, "right": None})

    @classmethod
    def from_proprio(cls, proprio16) -> "CartesianExecutor":
        p = np.asarray(proprio16, float)
        return cls(q={"left": p[0:7].copy(), "right": p[8:15].copy()})

    def lower(self, cart18) -> tuple[np.ndarray, dict[str, Any]]:
        a = np.asarray(cart18, float).reshape(-1)
        if a.shape != (ACTION_DIM,):
            raise ContractError(f"expected a {ACTION_DIM}-D CartesianMobileAction18, got {a.shape}")
        if not np.isfinite(a).all():
            raise ContractError("Cartesian action contains non-finite values")
        joint = np.zeros(ACTION_DIM)
        report: dict[str, Any] = {}
        for side, (sx, sq, sg, jsl, jg) in _SLOTS.items():
            raw_norm = float(np.linalg.norm(a[sq]))
            quat = P.canonical_quat(a[sq], self.prev_quat[side])
            target = P.gripper_base_from_action(side, a[sx], quat)
            res = K.ik(side, target, self.q[side])
            valid = bool(res.pos_err_m <= IK_POS_TOL_M and res.rot_err_rad <= IK_ROT_TOL_RAD)
            report[side] = {"ik_valid": valid, "pos_err_m": res.pos_err_m, "rot_err_rad": res.rot_err_rad,
                            "iterations": res.iterations, "at_limit": res.at_limit,
                            "max_joint_step_rad": res.max_step_rad,
                            "discontinuous": bool(res.max_step_rad > MAX_JOINT_STEP_RAD),
                            "quat_norm_in": raw_norm}
            self.q[side] = res.q
            self.prev_quat[side] = quat
            joint[jsl] = res.q
            joint[jg] = P.gripper_closed_fraction_from_action(a[sg])
        joint[16], joint[17] = a[CART_BASE_V], a[CART_BASE_OMEGA]
        return joint, report


def cartesian_from_joint_action(joint18, prev_quat: dict | None = None) -> np.ndarray:
    """JointMobileAction18 -> CartesianMobileAction18 (FK + frames). Used to label demonstrations."""
    j = np.asarray(joint18, float)
    prev_quat = prev_quat or {}
    out = np.zeros(ACTION_DIM)
    for side, (sx, sq, sg, jsl, jg) in _SLOTS.items():
        xyz, quat = P.action_from_vendor_joints(side, j[jsl], prev_quat.get(side))
        out[sx], out[sq], out[sg] = xyz, quat, j[jg]
    out[16], out[17] = j[16], j[17]
    return out
