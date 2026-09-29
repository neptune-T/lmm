"""Minimal URDF forward kinematics (numpy only) for cross-backend checks.

Parses joints (origin xyz/rpy, axis, type, mimic) of a URDF and computes link
poses in any ancestor frame for given joint values. Used to verify that the
Habitat AlphaBot 2 URDF (built from the xacro) has exactly the kinematics of
the Isaac-qualified reference ``lmm.embodiments.alphabot2.kinematics``.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_URDF = Path(__file__).with_name("alphabot2.urdf")


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis_rot(a, q):
    a = np.asarray(a, float) / np.linalg.norm(a)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(q) * k + (1 - math.cos(q)) * k @ k


@dataclass
class Joint:
    name: str
    type: str
    parent: str
    child: str
    T: np.ndarray                   # origin transform parent -> joint frame
    axis: np.ndarray
    lower: float | None
    upper: float | None
    mimic: tuple | None             # (joint, multiplier, offset)


class URDFKin:
    def __init__(self, path=DEFAULT_URDF):
        root = ET.parse(str(path)).getroot()
        self.links = [l.get("name") for l in root.findall("link")]
        self._inertial = {}
        for l in root.findall("link"):
            o = l.find("inertial/origin") if l.find("inertial") is not None else None
            T = np.eye(4)
            if o is not None:
                T[:3, :3] = _rpy(*[float(v) for v in o.get("rpy", "0 0 0").split()])
                T[:3, 3] = [float(v) for v in o.get("xyz", "0 0 0").split()]
            self._inertial[l.get("name")] = T
        self.joints: dict[str, Joint] = {}
        self.by_child: dict[str, Joint] = {}
        for j in root.findall("joint"):
            o = j.find("origin")
            xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = _rpy(*rpy), xyz
            ax = j.find("axis")
            lim = j.find("limit")
            mm = j.find("mimic")
            jt = Joint(name=j.get("name"), type=j.get("type"), parent=j.find("parent").get("link"),
                       child=j.find("child").get("link"), T=T,
                       axis=np.array([float(v) for v in (ax.get("xyz") if ax is not None else "1 0 0").split()]),
                       lower=float(lim.get("lower")) if lim is not None and lim.get("lower") else None,
                       upper=float(lim.get("upper")) if lim is not None and lim.get("upper") else None,
                       mimic=(mm.get("joint"), float(mm.get("multiplier", 1)), float(mm.get("offset", 0)))
                       if mm is not None else None)
            self.joints[jt.name] = jt
            self.by_child[jt.child] = jt
        roots = [l for l in self.links if l not in self.by_child]
        if len(roots) != 1:
            raise ValueError(f"URDF must have exactly one root link, found {roots}")
        self.root = roots[0]

    def inertial(self, link: str) -> np.ndarray:
        """Link frame -> inertial (COM) frame; Habitat's link scene nodes sit at the latter."""
        return self._inertial[link]

    def movable(self):
        return [n for n, j in self.joints.items() if j.type != "fixed"]

    def chain(self, link: str, base: str | None = None):
        out = []
        while link != (base or self.root):
            if link not in self.by_child:
                raise ValueError(f"{base} is not an ancestor")
            j = self.by_child[link]
            out.append(j)
            link = j.parent
        return out[::-1]

    def pose(self, link: str, q: dict | None = None, base: str | None = None) -> np.ndarray:
        """4x4 pose of ``link`` in ``base`` (default: root) for joint values ``q`` (name -> rad)."""
        q = q or {}
        T = np.eye(4)
        for j in self.chain(link, base):
            T = T @ j.T
            if j.type in ("revolute", "continuous", "prismatic"):
                v = q.get(j.name, 0.0)
                if j.mimic is not None:
                    v = q.get(j.mimic[0], 0.0) * j.mimic[1] + j.mimic[2]
                M = np.eye(4)
                if j.type == "prismatic":
                    M[:3, 3] = j.axis / np.linalg.norm(j.axis) * v
                else:
                    M[:3, :3] = _axis_rot(j.axis, v)
                T = T @ M
        return T
