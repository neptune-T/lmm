"""AlphaBot 2 arm kinematics (vendor/xacro joint convention), numpy only.

Parameters are copied from the public Ai2 Bot2 xacro
(``fiveages-sim/robot_descriptions`` @ ``bd491829961a8562306515001103af3b3737fbc7``,
``humanoid/Ai2Robotics/ai2_bot2_description/xacro/components/{arm,body}.xacro``,
Apache-2.0), resolved for ``direction = +1`` (left) and ``-1`` (right), torso at
zero. The same chain was live-verified against the Isaac USD (per-joint axes
cos = 1.000, gripper_base <= 4 mm; docs/ALPHABOT2_ISAAC_LIVE_SMOKE.md).

Frames: every pose is ``gripper_base`` of the Inspire EG2-4C2 in the chassis
``base_link`` frame (+x forward, +z up); ``gripper_base`` +z is the approach.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_P2 = 1.5708  # the xacro writes pi/2 as 1.5708; kept verbatim

#: body_joint1..4 origins (xyz, rpy); torso joints held at zero.
BODY_CHAIN = (
    ((0.19155, 0.0, 0.294), (-_P2, 0.0, 0.0)),
    ((0.0, -0.38704, 0.0), (0.0, 0.0, 0.0)),
    ((0.0, -0.27, 0.0), (0.0, 0.0, 0.0)),
    ((0.0, -0.1655, 0.0), (_P2, 0.0, 0.0)),
)

#: (joint0 mount, [joint1..7 (xyz, rpy, axis)], joint8) per side.
ARM_CHAIN = {
    "left": (
        ((-0.061, 0.0485, 0.3255), (-_P2, -_P2, 0.0)),
        (
            ((0.0, 0.0, 0.1785), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            ((0.0, 0.0, 0.0), (-_P2, -_P2, 0.0), (0.0, 0.0, 1.0)),
            ((0.0, -0.28998, 0.0), (_P2, -_P2, 0.0), (0.0, 0.0, 1.0)),
            ((0.01, 0.0, 0.0), (_P2, 0.0, 0.0), (0.0, 0.0, 1.0)),
            ((-0.01, 0.3088, 0.0), (-_P2, 0.0, 0.0), (0.0, 0.0, 1.0)),
            ((0.0, 0.0, 0.0), (_P2, -_P2, 0.0), (0.0, 0.0, 1.0)),
            ((0.01, 0.0, 0.0), (_P2, -_P2, 0.0), (0.0, 0.0, 1.0)),
        ),
        ((0.0, -0.04512, 0.0), (_P2, 3.1415926, 0.0)),
    ),
    "right": (
        ((-0.061, -0.0485, 0.3255), (_P2, -_P2, 0.0)),
        (
            ((0.0, 0.0, 0.1785), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            ((0.0, 0.0, 0.0), (-_P2, -_P2, 0.0), (0.0, 0.0, 1.0)),
            ((0.0, -0.28998, 0.0), (_P2, -_P2, 0.0), (0.0, 0.0, -1.0)),
            ((-0.01, 0.0, 0.0), (_P2, 0.0, 0.0), (0.0, 0.0, -1.0)),
            ((0.01, 0.3088, 0.0), (-_P2, 0.0, 0.0), (0.0, 0.0, -1.0)),
            ((0.0, 0.0, 0.0), (_P2, -_P2, 0.0), (0.0, 0.0, -1.0)),
            ((0.01, 0.0, 0.0), (_P2, -_P2, 0.0), (0.0, 0.0, 1.0)),
        ),
        ((0.0, -0.04512, 0.0), (_P2, 3.1415926, 0.0)),
    ),
}

#: ee (+0.0695 z) then the gripper mount (+pi/2 yaw), both sides.
EE_OFFSET = ((0.0, 0.0, 0.0695), (0.0, 0.0, 0.0))
GRIPPER_MOUNT = ((0.0, 0.0, 0.0), (0.0, 0.0, _P2))

#: Joint limits (vendor convention), identical for both sides.
JOINT_LIMITS = np.array([(-3.1, 3.1), (-0.44, 3.1), (-3.1, 3.1), (-2.27, 0.0),
                         (-3.1, 3.1), (-1.22, 1.22), (-2.09, 0.49)])
#: xacro velocity limit, rad/s, all joints.
JOINT_VEL_LIMIT = 3.14


def _rpy(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _tf(xyz, rpy) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = _rpy(*rpy)
    m[:3, 3] = xyz
    return m


def _axis_rot(axis, q: float) -> np.ndarray:
    a = np.asarray(axis, float)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    m = np.eye(4)
    m[:3, :3] = np.eye(3) + math.sin(q) * k + (1 - math.cos(q)) * k @ k
    return m


def rotvec(r: np.ndarray) -> np.ndarray:
    """Axis * angle of a rotation matrix (robust near 0 and pi)."""
    c = float(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))
    ang = math.acos(c)
    if ang < 1e-9:
        return np.zeros(3)
    if ang > math.pi - 1e-6:
        # axis from the symmetric part
        m = (r + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(m)))
        v = m[:, i] / math.sqrt(max(m[i, i], 1e-12))
        return v * ang
    v = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    return v / (2.0 * math.sin(ang)) * ang


_PRE = {}
for _s, (_mount, _joints, _j8) in ARM_CHAIN.items():
    _m = np.eye(4)
    for _xyz, _rpy_ in BODY_CHAIN:
        _m = _m @ _tf(_xyz, _rpy_)
    _PRE[_s] = _m @ _tf(*_mount)
_POST = {s: _tf(*ARM_CHAIN[s][2]) @ _tf(*EE_OFFSET) @ _tf(*GRIPPER_MOUNT) for s in ARM_CHAIN}
_ORIG = {s: [_tf(xyz, rpy) for xyz, rpy, _ in ARM_CHAIN[s][1]] for s in ARM_CHAIN}
_AXES = {s: [np.asarray(ax) for _, _, ax in ARM_CHAIN[s][1]] for s in ARM_CHAIN}


def fk(side: str, q) -> np.ndarray:
    """gripper_base pose (4x4) in the chassis frame for vendor-convention joints ``q``."""
    m = _PRE[side].copy()
    for k in range(7):
        m = m @ _ORIG[side][k] @ _axis_rot(_AXES[side][k], float(q[k]))
    return m @ _POST[side]


def jacobian(side: str, q, eps: float = 1e-6) -> np.ndarray:
    """6x7: rows (dx, dy, dz, rotvec) of gripper_base in the chassis frame."""
    q = np.asarray(q, float)
    m0 = fk(side, q)
    jac = np.zeros((6, 7))
    for k in range(7):
        dq = q.copy()
        dq[k] += eps
        m1 = fk(side, dq)
        jac[:3, k] = (m1[:3, 3] - m0[:3, 3]) / eps
        jac[3:, k] = rotvec(m1[:3, :3] @ m0[:3, :3].T) / eps
    return jac


def pose_error(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(position error m, rotation error rad) between two 4x4 poses."""
    return (float(np.linalg.norm(a[:3, 3] - b[:3, 3])),
            float(np.linalg.norm(rotvec(a[:3, :3].T @ b[:3, :3]))))


@dataclass(frozen=True)
class IKResult:
    q: np.ndarray
    converged: bool
    pos_err_m: float
    rot_err_rad: float
    iterations: int
    at_limit: tuple[int, ...]       # joints (1-based) within 1e-3 of a limit
    max_step_rad: float             # max |q - seed|


def ik(side: str, target: np.ndarray, seed, *, margin: float = 0.02, max_iters: int = 150,
       pos_tol: float = 1e-4, rot_tol: float = 1e-3, rot_weight: float = 0.3,
       damping: float = 1e-4, step_cap: float = 0.2) -> IKResult:
    """Damped least squares from ``seed``, projected into the limits (shrunk by ``margin``).

    Stays on the branch of the seed (capped steps, no restarts), so feeding the
    previous solution as the seed gives continuous joint trajectories and no
    elbow flips; the caller checks ``max_step_rad`` for discontinuities.
    """
    lo, hi = JOINT_LIMITS[:, 0] + margin, JOINT_LIMITS[:, 1] - margin
    seed = np.clip(np.asarray(seed, float), lo, hi)
    q = seed.copy()
    w = np.array([1, 1, 1, rot_weight, rot_weight, rot_weight])
    it = 0
    for it in range(1, max_iters + 1):
        m = fk(side, q)
        err = np.concatenate([target[:3, 3] - m[:3, 3], rotvec(target[:3, :3] @ m[:3, :3].T)])
        if np.linalg.norm(err[:3]) < pos_tol and np.linalg.norm(err[3:]) < rot_tol:
            break
        jw = jacobian(side, q) * w[:, None]
        dq = jw.T @ np.linalg.solve(jw @ jw.T + damping * np.eye(6), err * w)
        n = float(np.linalg.norm(dq))
        if n > step_cap:
            dq *= step_cap / n
        q = np.clip(q + dq, lo, hi)
    pe, re = pose_error(fk(side, q), target)
    at = tuple(k + 1 for k in range(7) if q[k] <= lo[k] + 1e-3 or q[k] >= hi[k] - 1e-3)
    return IKResult(q=q, converged=bool(pe < max(pos_tol * 10, 1e-3) and re < max(rot_tol * 10, 0.01)),
                    pos_err_m=pe, rot_err_rad=re, iterations=it, at_limit=at,
                    max_step_rad=float(np.max(np.abs(q - seed))))
