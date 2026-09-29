"""EG2-4C2 free finger gap vs gripper joint, from the URDF collision meshes.

For each joint value the left-finger links (outer, inner, pad) and the
right-finger links are posed by FK (mimic joints included) in the
gripper_base frame, and the minimum distance between the two finger
assemblies' surfaces is measured (dense surface samples + KD-tree). That gap
is the width of the widest object the fingers can close on at that joint
value; it replaces the pad-LINK-origin separation, which is not the contact
surface (Isaac: a 5 cm cube was held at a link separation of 0.047 m).

    venv_tools/bin/python -m lmm.embodiments.alphabot2.habitat.gripper_gap URDF_DIR
"""
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .urdf_kin import URDFKin, _rpy


def main(urdf_dir, side="left"):
    import trimesh
    from scipy.spatial import cKDTree
    d = Path(urdf_dir)
    kin = URDFKin(d / "alphabot2.urdf")
    root = ET.parse(str(d / "alphabot2.urdf")).getroot()
    fingers = {"L": [f"{side}_left_outer_link", f"{side}_left_inner_link", f"{side}_left_pad"],
               "R": [f"{side}_right_outer_link", f"{side}_right_inner_link", f"{side}_right_pad"]}
    meshes = {}
    for link in root.findall("link"):
        if link.get("name") not in fingers["L"] + fingers["R"]:
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
            pts, _ = trimesh.sample.sample_surface(mesh, 15000, seed=0)
            meshes.setdefault(link.get("name"), []).append((T0, pts))
    rows = []
    for q in np.linspace(0.0, 0.82, 42):
        side_pts = {}
        for k, links in fingers.items():
            P = []
            for l in links:
                T = kin.pose(l, {f"{side}_gripper_joint": float(q)}, base=f"{side}_gripper_base")
                for T0, pts in meshes.get(l, []):
                    M = T @ T0
                    P.append((M[:3, :3] @ pts.T).T + M[:3, 3])
            side_pts[k] = np.concatenate(P)
        dist, _ = cKDTree(side_pts["R"]).query(side_pts["L"], k=1)
        padL = [((kin.pose(f"{side}_left_pad", {f"{side}_gripper_joint": float(q)}, base=f"{side}_gripper_base") @ T0)[:3, :3] @ pts.T).T
                + (kin.pose(f"{side}_left_pad", {f"{side}_gripper_joint": float(q)}, base=f"{side}_gripper_base") @ T0)[:3, 3]
                for T0, pts in meshes.get(f"{side}_left_pad", [])]
        padR = [((kin.pose(f"{side}_right_pad", {f"{side}_gripper_joint": float(q)}, base=f"{side}_gripper_base") @ T0)[:3, :3] @ pts.T).T
                + (kin.pose(f"{side}_right_pad", {f"{side}_gripper_joint": float(q)}, base=f"{side}_gripper_base") @ T0)[:3, 3]
                for T0, pts in meshes.get(f"{side}_right_pad", [])]
        pad_gap = float(cKDTree(np.concatenate(padR)).query(np.concatenate(padL), k=1)[0].min()) if padL and padR else None
        a = kin.pose(f"{side}_left_pad", {f"{side}_gripper_joint": float(q)}, base=f"{side}_gripper_base")[:3, 3]
        b = kin.pose(f"{side}_right_pad", {f"{side}_gripper_joint": float(q)}, base=f"{side}_gripper_base")[:3, 3]
        rows.append({"q": round(float(q), 4), "free_gap_m": round(float(dist.min()), 4), "pad_gap_m": pad_gap,
                     "pad_link_sep_m": round(float(np.linalg.norm(a - b)), 4)})
    out = {"side": side, "rows": rows, "max_pad_gap_m": max(r["pad_gap_m"] for r in rows), "max_free_gap_m": max(r["free_gap_m"] for r in rows),
           "min_free_gap_m": min(r["free_gap_m"] for r in rows)}
    print(json.dumps(out, indent=1))
    (d / "gripper_gap.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
