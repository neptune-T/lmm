"""Search a navigation ("tuck") posture that keeps both arms inside the chassis turning circle.

Bounded random search + local refinement over the left arm's 7 vendor joints
(the right arm is the vendor mirror, joint1 negated, as for the xacro home).
Objective: the largest horizontal distance of any arm/gripper collision vertex
from the drive-wheel centre. Constraint: every arm/gripper vertex stays
``margin`` outside the axis-aligned boxes of the torso/head/chassis collision
meshes and above ``min_z`` (no floor contact); left and right arms stay
``margin`` apart. Reported with the home posture for comparison.

    venv_tools/bin/python -m lmm.embodiments.alphabot2.habitat.tuck_search URDF_DIR
"""
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from lmm.embodiments.alphabot2 import kinematics as K

from .urdf_kin import URDFKin, _rpy

HOME_L = np.array((0.8333, 0.0, 0.0, -1.7800, 0.0, -0.6410, 0.0))
MIRROR = np.array([-1, 1, 1, 1, 1, 1, 1.0])


def load_vertices(d: Path, kin: URDFKin, n_per_mesh=200):
    import trimesh
    root = ET.parse(str(d / "alphabot2.urdf")).getroot()
    out = {}
    rng = np.random.default_rng(0)
    for link in root.findall("link"):
        for c in link.findall("collision"):
            m = c.find("geometry/mesh")
            if m is None:
                continue
            o = c.find("origin")
            T0 = np.eye(4)
            if o is not None:
                T0[:3, :3] = _rpy(*[float(v) for v in o.get("rpy", "0 0 0").split()])
                T0[:3, 3] = [float(v) for v in o.get("xyz", "0 0 0").split()]
            v = trimesh.load(str(d / m.get("filename")), force="mesh").vertices
            if len(v) > n_per_mesh:
                v = v[rng.choice(len(v), n_per_mesh, replace=False)]
            v = (T0[:3, :3] @ v.T).T + T0[:3, 3]
            out.setdefault(link.get("name"), []).append(v)
    return {k: np.concatenate(v) for k, v in out.items()}


def arm_links(kin, side):
    """Links whose pose depends on the side's arm joints (arm + gripper)."""
    return [l for l in kin.links if any(j.name.startswith(f"{side}_joint") for j in kin.chain(l) if j.type != "fixed")]


def main(urdf_dir):
    d = Path(urdf_dir)
    kin = URDFKin(d / "alphabot2.urdf")
    V = load_vertices(d, kin)
    wl = kin.pose("driving_left_link", base="base_footprint")[:3, 3]
    wr = kin.pose("driving_right_link", base="base_footprint")[:3, 3]
    c = (wl + wr) / 2
    L = {s: [l for l in V if l in set(arm_links(kin, s))] for s in ("left", "right")}
    moving = set(L["left"]) | set(L["right"])
    # torso, head and chassis SURFACES (dense samples, KD-tree): an axis-aligned box per link is far
    # too conservative for the diagonal knee links of a crouched torso
    import trimesh
    from scipy.spatial import cKDTree
    root_el = ET.parse(str(d / "alphabot2.urdf")).getroot()
    surf, static = [], []
    for link in root_el.findall("link"):
        name = link.get("name")
        if name in moving or not name.startswith(("body_link", "head_link", "base_link")):
            continue
        for col in link.findall("collision"):
            m = col.find("geometry/mesh")
            if m is None:
                continue
            o = col.find("origin")
            T0 = np.eye(4)
            if o is not None:
                T0[:3, :3] = _rpy(*[float(v) for v in o.get("rpy", "0 0 0").split()])
                T0[:3, 3] = [float(v) for v in o.get("xyz", "0 0 0").split()]
            mesh = trimesh.load(str(d / m.get("filename")), force="mesh")
            pts_s, _ = trimesh.sample.sample_surface(mesh, 20000, seed=0)
            w = kin.pose(name, base="base_footprint") @ T0
            surf.append((w[:3, :3] @ pts_s.T).T + w[:3, 3])
            static.append(name)
    tree = cKDTree(np.concatenate(surf))

    CHECK = {s: [l for l in L[s] if not l.startswith((f"{s}_link1", f"{s}_link2", f"{s}_base_link"))] for s in L}

    def pts(side, q, links=None):
        qq = {f"{side}_joint{i + 1}": v for i, v in enumerate(q)}
        qq[f"{side}_gripper_joint"] = 0.82
        out = []
        for l in (links or L[side]):
            T = kin.pose(l, qq, base="base_footprint")
            out.append((T[:3, :3] @ V[l].T).T + T[:3, 3])
        return np.concatenate(out)

    MARGIN, MIN_Z = 0.02, 0.30

    def score(qL):
        pl, pr = pts("left", qL), pts("right", MIRROR * qL)
        P = np.concatenate([pl, pr])
        radius = float(np.max(np.linalg.norm(P[:, :2] - c[:2], axis=1)))
        P = np.concatenate([pts("left", qL, CHECK["left"]), pts("right", MIRROR * qL, CHECK["right"])])
        dist, _ = tree.query(P, k=1)
        pen = float(np.sum(dist < MARGIN))
        pen += float(np.sum(P[:, 2] < MIN_Z))
        # arms apart
        if np.min(pl[:, 1]) - np.max(pr[:, 1]) < MARGIN:
            pen += 50
        return radius, pen

    lo, hi = K.JOINT_LIMITS[:, 0] + 0.05, K.JOINT_LIMITS[:, 1] - 0.05
    rng = np.random.default_rng(1)
    home_r, home_pen = score(HOME_L)
    R_MAX = 0.37                       # inside the chassis turning circle (0.378 m)
    # objective: the feasible posture CLOSEST TO HOME (short, natural nav<->manipulation transitions)
    def cost(qq):
        r_, pen_ = score(qq)
        return (np.inf if pen_ > 0 or r_ > R_MAX else float(np.linalg.norm(qq - HOME_L))), r_
    best = (np.inf, None, None)
    for _ in range(6000):
        # sample around home first, then globally
        qs = np.clip(HOME_L + rng.normal(0, 0.8, 7), lo, hi) if _ % 2 else rng.uniform(lo, hi)
        cst, rr = cost(qs)
        if cst < best[0]:
            best = (cst, qs, rr)
    if best[1] is None:
        raise SystemExit(json.dumps({"no_feasible_sample": True, "home": [home_r, home_pen], "static": sorted(static)}))
    cst, q, r = best
    for step in (0.2, 0.1, 0.05, 0.02):
        for _ in range(400):
            cand = np.clip(q + rng.normal(0, step, 7), lo, hi)
            cc, rc = cost(cand)
            if cc < cst:
                q, r, cst = cand, rc, cc
    gl = kin.pose("left_gripper_base", {f"left_joint{i + 1}": v for i, v in enumerate(q)}, base="base_footprint")[:3, 3]
    if q is None:
        raise SystemExit("no collision-free tuck found")
    out = {"wheel_centre_xy": c[:2].tolist(), "home": {"radius_m": home_r, "penalty": home_pen},
           "tuck": {"left_q": q.round(4).tolist(), "right_q": (MIRROR * q).round(4).tolist(), "radius_m": r,
                    "joint_distance_from_home_rad": float(np.linalg.norm(q - HOME_L)), "radius_bound_m": R_MAX,
                    "left_gripper_base_xyz": gl.round(3).tolist()},
           "static_links_checked": sorted(set(static)),
           "constraints": {"margin_m": MARGIN, "min_z_m": MIN_Z, "search": "6000 random + 1600 local"}}
    print(json.dumps(out, indent=1))
    (d / "tuck.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
