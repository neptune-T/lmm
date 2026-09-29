"""AlphaBot 2 floor footprint in the fixed safe posture, from the URDF collision geometry.

Every <collision> mesh of the Habitat URDF is placed by FK at the home posture
(arms at the xacro home_2, grippers open), its vertices projected on the floor
plane of the ROS base frame. Reported: convex hull, extents, and the radius of
the smallest circle about the differential-drive rotation centre (midpoint of
the two drive wheels) that contains the hull -- the clearance a base turning in
place actually needs. The whole body is reported, and separately the part
below 0.40 m (chassis only), since the arms at home are high and forward.

    venv_tools/bin/python footprint.py URDF_DIR
"""
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .urdf_kin import URDFKin, _rpy

HOME = {"left": (0.8333, 0.0, 0.0, -1.7800, 0.0, -0.6410, 0.0), "right": (-0.8333, 0.0, 0.0, -1.7800, 0.0, -0.6410, 0.0)}


def _cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def hull2d(p):
    p = np.unique(np.round(p, 5), axis=0)
    p = p[np.lexsort((p[:, 1], p[:, 0]))]
    def half(pts):
        h = []
        for q in pts:
            while len(h) >= 2 and _cross2(h[-1] - h[-2], q - h[-2]) <= 0:
                h.pop()
            h.append(q)
        return h
    lo, up = half(p), half(p[::-1])
    return np.array(lo[:-1] + up[:-1])


def main(urdf_dir: str):
    import trimesh
    d = Path(urdf_dir)
    urdf = d / "alphabot2.urdf"
    kin = URDFKin(urdf)
    q = {f"{s}_joint{i + 1}": v for s in HOME for i, v in enumerate(HOME[s])}
    q.update({"left_gripper_joint": 0.82, "right_gripper_joint": 0.82})
    root = ET.parse(str(urdf)).getroot()
    pts, per_link = [], {}
    for link in root.findall("link"):
        for c in link.findall("collision"):
            g = c.find("geometry")
            o = c.find("origin")
            T0 = np.eye(4)
            if o is not None:
                T0[:3, :3] = _rpy(*[float(v) for v in o.get("rpy", "0 0 0").split()])
                T0[:3, 3] = [float(v) for v in o.get("xyz", "0 0 0").split()]
            if g.find("mesh") is not None:
                m = g.find("mesh")
                mesh = trimesh.load(str(d / m.get("filename")), force="mesh")
                s = [float(v) for v in m.get("scale", "1 1 1").split()]
                v = mesh.vertices * s
            elif g.find("box") is not None:
                sx, sy, sz = [float(x) / 2 for x in g.find("box").get("size").split()]
                v = np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)])
            elif g.find("cylinder") is not None:
                r, l = float(g.find("cylinder").get("radius")), float(g.find("cylinder").get("length"))
                a = np.linspace(0, 2 * np.pi, 24)
                v = np.array([[r * np.cos(t), r * np.sin(t), z] for t in a for z in (-l / 2, l / 2)])
            else:
                continue
            T = kin.pose(link.get("name"), q, base="base_footprint") @ T0
            w = (T[:3, :3] @ v.T).T + T[:3, 3]
            pts.append(w)
            per_link[link.get("name")] = {"x": [float(w[:, 0].min()), float(w[:, 0].max())],
                                          "y": [float(w[:, 1].min()), float(w[:, 1].max())],
                                          "z": [float(w[:, 2].min()), float(w[:, 2].max())]}
    P = np.concatenate(pts)
    wl = kin.pose("driving_left_link", base="base_footprint")[:3, 3]
    wr = kin.pose("driving_right_link", base="base_footprint")[:3, 3]
    c = (wl + wr) / 2
    out = {"rotation_centre_xy": c[:2].round(4).tolist(), "wheel_track_m": float(np.linalg.norm(wl - wr)),
           "base_link_in_footprint": kin.pose("base_link", base="base_footprint")[:3, 3].round(4).tolist()}
    for name, sel in (("whole_body", np.ones(len(P), bool)), ("below_0.40m", P[:, 2] < 0.40)):
        H = hull2d(P[sel][:, :2])
        r = np.linalg.norm(P[sel][:, :2] - c[:2], axis=1)
        out[name] = {"x_range_m": [float(P[sel][:, 0].min()), float(P[sel][:, 0].max())],
                     "y_range_m": [float(P[sel][:, 1].min()), float(P[sel][:, 1].max())],
                     "turning_radius_about_wheel_centre_m": float(r.max()), "hull_vertices": len(H)}
    out["tallest_m"] = float(P[:, 2].max())
    out["per_link"] = per_link
    print(json.dumps({k: v for k, v in out.items() if k != "per_link"}, indent=1))
    (d / "footprint.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
