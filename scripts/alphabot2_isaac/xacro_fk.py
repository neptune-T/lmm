"""Forward kinematics of the Ai2 Bot2 arms, parsed from the public xacro.

numpy + re only, so it runs both under Isaac's python.sh and on a plain host.
The xacro (``fiveages-sim/robot_descriptions``,
``humanoid/Ai2Robotics/ai2_bot2_description/xacro/components``) is the only
public kinematic description of the robot; it defines the *vendor* joint
convention the policy contract uses.

Frames: poses are returned in the xacro ``base_link`` (chassis) frame. The
gripper frame is ``gripper_base`` of the Inspire EG2-4C2 as mounted by the
xacro (``joint8`` -> ``ee`` (+0.0695 z) -> yaw +pi/2). Its +z is the approach.
"""

from __future__ import annotations

import math
import re

import numpy as np


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


def _rot(axis, q: float) -> np.ndarray:
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    m = np.eye(4)
    m[:3, :3] = np.eye(3) + math.sin(q) * k + (1 - math.cos(q)) * k @ k
    return m


def _nums(s: str) -> list[float]:
    return [float(v) for v in s.split()]


def _resolve(block: str, direction: int | None) -> str:
    if direction is None:
        return block
    block = re.sub(r'<xacro:if value="\$\{direction == (-?1)\}">(.*?)</xacro:if>',
                   lambda m: m.group(2) if int(m.group(1)) == direction else "", block, flags=re.S)
    return block.replace("${direction}", str(direction))


def _joints(xml: str, direction: int | None = None) -> dict:
    out = {}
    for m in re.finditer(r'<joint name="([^"]+)" type="([^"]+)">(.*?)</joint>', xml, re.S):
        b = _resolve(m.group(3), direction)
        o = re.search(r'<origin([^>]*)/>', b)
        xyz = _nums(re.search(r'xyz="([^"]+)"', o.group(1)).group(1)) if o and 'xyz=' in o.group(1) else [0, 0, 0]
        rpy = _nums(re.search(r'rpy="([^"]+)"', o.group(1)).group(1)) if o and 'rpy=' in o.group(1) else [0, 0, 0]
        ax = re.search(r'<axis xyz="([^"]+)"', b)
        lim = re.search(r'<limit lower="([^"]+)" upper="([^"]+)"', b)
        out[m.group(1)] = {"xyz": xyz, "rpy": rpy, "axis": _nums(ax.group(1)) if ax else [0, 0, 1],
                           "limit": (float(lim.group(1)), float(lim.group(2))) if lim else None}
    return out


class XacroArms:
    """Both arms of the Ai2 Bot2 xacro. ``side`` is ``"left"`` or ``"right"``."""

    def __init__(self, components_dir: str) -> None:
        arm = open(f"{components_dir}/arm.xacro").read()
        self.body = _joints(open(f"{components_dir}/body.xacro").read())
        self.arm = {"left": _joints(arm, 1), "right": _joints(arm, -1)}

    def limits(self, side: str) -> list[tuple[float, float]]:
        return [self.arm[side][f"${{prefix}}joint{k}"]["limit"] for k in range(1, 8)]

    def axes(self, side: str) -> list[list[float]]:
        return [self.arm[side][f"${{prefix}}joint{k}"]["axis"] for k in range(1, 8)]

    def fk(self, side: str, q7) -> np.ndarray:
        """gripper_base pose (4x4) in the chassis base_link frame. Torso at zero."""
        j = self.arm[side]
        m = np.eye(4)
        for n in ("body_joint1", "body_joint2", "body_joint3", "body_joint4"):
            m = m @ _tf(self.body[n]["xyz"], self.body[n]["rpy"])
        m = m @ _tf(j["${prefix}joint0"]["xyz"], j["${prefix}joint0"]["rpy"])
        for k in range(1, 8):
            jk = j[f"${{prefix}}joint{k}"]
            m = m @ _tf(jk["xyz"], jk["rpy"]) @ _rot(jk["axis"], float(q7[k - 1]))
        m = m @ _tf(j["${prefix}joint8"]["xyz"], j["${prefix}joint8"]["rpy"])
        return m @ _tf([0, 0, 0.0695], [0, 0, 0]) @ _tf([0, 0, 0], [0, 0, 1.5708])

    def jacobian(self, side: str, q7, eps: float = 1e-6) -> np.ndarray:
        """6x7 numerical Jacobian: rows = (dx, dy, dz, rotvec) in the chassis frame."""
        q7 = np.asarray(q7, float)
        m0 = self.fk(side, q7)
        jac = np.zeros((6, 7))
        for k in range(7):
            dq = q7.copy()
            dq[k] += eps
            m1 = self.fk(side, dq)
            jac[:3, k] = (m1[:3, 3] - m0[:3, 3]) / eps
            jac[3:, k] = rotvec(m1[:3, :3] @ m0[:3, :3].T) / eps
        return jac

    def ik_pose(self, side: str, q7, goal: np.ndarray, iters: int = 300,
                margin: float | None = None) -> np.ndarray:
        """Damped-least-squares IK from ``q7`` to a 4x4 gripper_base goal
        (chassis frame). Steps are capped so the solution stays on the branch
        nearest the seed; with ``margin`` the iterate is projected into the
        vendor limits shrunk by that margin. Check the residual with
        :func:`pose_error`."""
        q = np.asarray(q7, float).copy()
        lim = np.array(self.limits(side)) if margin is not None else None
        for _ in range(iters):
            m = self.fk(side, q)
            err = np.concatenate([goal[:3, 3] - m[:3, 3], rotvec(goal[:3, :3] @ m[:3, :3].T)])
            if np.linalg.norm(err) < 1e-7:
                break
            jac = self.jacobian(side, q)
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), err)
            n = np.linalg.norm(dq)
            q += dq if n < 0.2 else dq * (0.2 / n)
            if lim is not None:
                q = np.clip(q, lim[:, 0] + margin, lim[:, 1] - margin)
        return q

    def ik_translate(self, side: str, q7, dp, margin: float | None = 0.02) -> np.ndarray:
        """Joint vector that moves the gripper by ``dp`` (chassis frame) and keeps
        its orientation."""
        goal = self.fk(side, q7)
        goal[:3, 3] += np.asarray(dp, float)
        return self.ik_pose(side, q7, goal, margin=margin)


def rotvec(r: np.ndarray) -> np.ndarray:
    """Axis * angle of a rotation matrix."""
    c = np.clip((np.trace(r) - 1) / 2, -1.0, 1.0)
    ang = math.acos(c)
    if ang < 1e-9:
        return np.zeros(3)
    v = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    return v / (2 * math.sin(ang)) * ang


def pose_error(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(position error m, rotation error deg) between two 4x4 poses."""
    return (float(np.linalg.norm(a[:3, 3] - b[:3, 3])),
            math.degrees(float(np.linalg.norm(rotvec(a[:3, :3].T @ b[:3, :3])))))
