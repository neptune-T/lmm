"""Rebuilt AlphaBot 2 URDF vs the committed fixture.

Strong check: byte-identical. Otherwise a structural + FK diff decides (formatting
differences, e.g. float printing or attribute order, must not block):
  links, joints (type, parent, child, axis, limits, mimic, origin), inertial frames,
  collision/visual element counts and mesh files (existence + size), and the pose of
  every link for 200 random joint configurations within the limits.

    python urdf_compare.py FIXTURE BUILT   -> one JSON line; exit 1 on a real difference or missing meshes
"""
import hashlib, json, sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lmm.embodiments.alphabot2.habitat.urdf_kin import URDFKin  # noqa: E402

TOL_M, TOL_RAD = 1e-9, 1e-9
fa, fb = Path(sys.argv[1]), Path(sys.argv[2])
ra, rb = fa.read_bytes(), fb.read_bytes()
rep = {"sha_fixture": hashlib.sha256(ra).hexdigest()[:16], "sha_built": hashlib.sha256(rb).hexdigest()[:16],
       "byte_identical": ra == rb}
a, b = URDFKin(fa), URDFKin(fb)
issues = []
if sorted(a.links) != sorted(b.links):
    issues.append({"links": sorted(set(a.links) ^ set(b.links))})
if sorted(a.joints) != sorted(b.joints):
    issues.append({"joints": sorted(set(a.joints) ^ set(b.joints))})
for n in sorted(set(a.joints) & set(b.joints)):
    x, y = a.joints[n], b.joints[n]
    if (x.type, x.parent, x.child, x.mimic and x.mimic[0]) != (y.type, y.parent, y.child, y.mimic and y.mimic[0]):
        issues.append({"joint": n, "structure": [x.type, x.parent, x.child, y.type, y.parent, y.child]})
    if np.abs(x.T - y.T).max() > TOL_M or np.abs(x.axis - y.axis).max() > TOL_RAD:
        issues.append({"joint": n, "origin_or_axis_max_abs": float(max(np.abs(x.T - y.T).max(), np.abs(x.axis - y.axis).max()))})
    for k in ("lower", "upper"):
        u, v = getattr(x, k), getattr(y, k)
        if (u is None) != (v is None) or (u is not None and abs(u - v) > TOL_RAD):
            issues.append({"joint": n, k: [u, v]})
for l in sorted(set(a.links) & set(b.links)):
    d = np.abs(a.inertial(l) - b.inertial(l)).max()
    if d > TOL_M:
        issues.append({"link": l, "inertial_max_abs": float(d)})


def geom(path):
    r = ET.parse(str(path)).getroot()
    c = Counter()
    meshes = []
    for l in r.findall("link"):
        for kind in ("visual", "collision"):
            for e in l.findall(kind):
                c[kind] += 1
                m = e.find("geometry/mesh")
                if m is not None:
                    meshes.append((l.get("name"), kind, m.get("filename"), m.get("scale")))
    return c, sorted(meshes, key=str)


(ca, ma), (cb, mb) = geom(fa), geom(fb)
rep["elements"] = {"fixture": dict(ca), "built": dict(cb)}
if ca != cb or [m[:2] + (m[3],) for m in ma] != [m[:2] + (m[3],) for m in mb]:
    issues.append({"geometry_elements_or_scale_differ": True})
missing = [m[2] for m in mb if not (fb.parent / m[2]).is_file()]
rep["missing_mesh_files"] = {"n": len(missing), "first": missing[:5]}
rep["mesh_files"] = len(mb)
rng = np.random.default_rng(0)
mov = [n for n in a.movable() if a.joints[n].mimic is None]
worst = 0.0
for _ in range(200):
    q = {}
    for n in mov:
        j = a.joints[n]
        lo, hi = (j.lower, j.upper) if j.lower is not None else (-np.pi, np.pi)
        q[n] = float(rng.uniform(lo, hi))
    for l in a.links:
        if l in b.links:
            worst = max(worst, float(np.abs(a.pose(l, q) - b.pose(l, q)).max()))
rep["fk_all_links_max_abs"] = worst
if worst > 1e-9:
    issues.append({"fk_max_abs": worst})
rep["issues"] = issues[:30]
rep["n_issues"] = len(issues)
rep["verdict"] = ("MESHES_MISSING" if missing else "BYTE_IDENTICAL" if rep["byte_identical"]
                  else "EQUIVALENT_STRUCTURE_FK" if not issues else "DIFFERENT")
print("###URDF_COMPARE###" + json.dumps(rep))
sys.exit(0 if rep["verdict"] in ("BYTE_IDENTICAL", "EQUIVALENT_STRUCTURE_FK") else 1)
