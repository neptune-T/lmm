"""Build the Habitat AlphaBot 2 URDF from the public Ai2 Bot2 xacro (pinned).

Source of truth for kinematics is the xacro (it carries the true mirrored right
arm; the Isaac USD reuses the left-arm definitions). Sources, pinned:

* fiveages-sim/robot_descriptions @ bd491829961a8562306515001103af3b3737fbc7
  (ai2_bot2_description, aidal_description, rm75_description)
* fiveages-sim/robot-descriptions-common @ 92fa1ebf7b7b6a36f8e277dda48706903e57101d
  (inspire_description = EG2-4C2 gripper, sensor_models)

Expansion: ``robot.xacro`` with ``type:=inspire`` (both Inspire EG2-4C2
grippers), ``$(find pkg)`` resolved to the pinned package directories.

Post-processing, each step reported in ``build_report.json``:

1. meshes referenced by the URDF are copied next to it and paths made
   relative (Habitat does not resolve ``file://``/``package://``);
2. ROS-only tags (ros2_control, gazebo, transmission) removed;
3. v1 fixed joints: torso ``body_joint1..4`` at the canonical crouch (``--torso``)
   and the neck at the canonical posture (``--head-pitch``) become ``fixed`` with the angle composed into the
   joint origin; drive wheels and casters become ``fixed`` because Habitat
   moves the base kinematically (OVMM convention) -- their geometry is kept;
4. ``<mimic>`` tags are kept for the record (Bullet ignores them); the robot
   class drives each gripper's mimic joints from the parsed multiplier/offset.

No geometry is invented: every visual/collision element comes from the xacro.

    venv_tools/bin/python build_urdf.py OUT_DIR [--head-pitch 0.2] [--head-yaw 0.0]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

DEPS = Path(os.environ.get("ALPHABOT2_DEPS", "/workspace/runtime/deps"))
PACKAGES = {
    "ai2_bot2_description": DEPS / "robot_descriptions/humanoid/Ai2Robotics/ai2_bot2_description",
    "aidal_description": DEPS / "robot_descriptions/humanoid/Realman/aidal_description",
    "rm75_description": DEPS / "robot_descriptions/manipulator/Realman/rm75_description",
    "inspire_description": DEPS / "rd_common/gripper/inspire_description",
    "sensor_models": DEPS / "rd_common/sensor_models",
}
REVISIONS = {"robot_descriptions": "bd491829961a8562306515001103af3b3737fbc7",
             "robot-descriptions-common": "92fa1ebf7b7b6a36f8e277dda48706903e57101d"}
#: Canonical fixed v1 torso posture for OVMM (knee-style lift, chest upright: (-a, 2a, -a, 0), a = 1.1).
#: With the torso at 0 the arms reach only grasp points 0.65-1.2 m above the floor, i.e. 17 % (val) /
#: 37 % (train) of OVMM episodes; this crouch reaches 0.35-1.2 m (see docs/ALPHABOT2_OVMM_EMBODIMENT.md).
CANONICAL_TORSO = (-1.1, 2.2, -1.1, 0.0)
ROS_ONLY = ("ros2_control", "gazebo", "transmission")


def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]]


def R_to_rpy(R):
    sp = -R[2][0]
    if abs(sp) < 1 - 1e-9:
        p = math.asin(sp)
        r = math.atan2(R[2][1], R[2][2])
        y = math.atan2(R[1][0], R[0][0])
    else:                                   # gimbal lock
        p = math.copysign(math.pi / 2, sp)
        r = 0.0
        y = math.atan2(-R[0][1], R[1][1])
    return r, p, y


def axis_R(a, q):
    x, y, z = a
    n = math.sqrt(x * x + y * y + z * z)
    x, y, z = x / n, y / n, z / n
    c, s, C = math.cos(q), math.sin(q), 1 - math.cos(q)
    return [[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C]]


def matmul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def expand(head_pitch: float) -> str:
    import xacro
    from xacro import substitution_args as sa

    def _eval_find(pkg):                            # $(find pkg) -> pinned directory, nothing else
        if pkg not in PACKAGES:
            raise KeyError(f"package {pkg!r} is not one of the pinned sources")
        return str(PACKAGES[pkg])
    sa._eval_find = _eval_find
    doc = xacro.process_file(str(PACKAGES["ai2_bot2_description"] / "xacro/robot.xacro"), mappings={"type": "inspire"})
    return doc.toprettyxml(indent="  ")


def freeze(joint: ET.Element, q: float) -> dict:
    """revolute/continuous joint at angle q -> fixed joint with the rotation in its origin."""
    o = joint.find("origin")
    if o is None:
        o = ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")
    rpy = [float(v) for v in o.get("rpy", "0 0 0").split()]
    ax = joint.find("axis")
    a = [float(v) for v in (ax.get("xyz") if ax is not None else "1 0 0").split()]
    R = matmul(rpy_to_R(*rpy), axis_R(a, q))
    new = R_to_rpy(R)
    before = {"type": joint.get("type"), "rpy": rpy, "axis": a, "q": q}
    o.set("rpy", " ".join(f"{v:.9f}" for v in new))
    joint.set("type", "fixed")
    for tag in ("axis", "limit", "dynamics", "mimic", "safety_controller", "calibration"):
        for e in joint.findall(tag):
            joint.remove(e)
    return before


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--head-pitch", type=float, default=0.2)
    ap.add_argument("--head-yaw", type=float, default=0.0)
    ap.add_argument("--torso", default=",".join(str(v) for v in CANONICAL_TORSO))
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "meshes").mkdir(parents=True)
    xml = expand(args.head_pitch)
    root = ET.fromstring(xml)
    rep = {"sources": REVISIONS, "xacro": "ai2_bot2_description/xacro/robot.xacro type:=inspire",
           "packages": {k: str(v) for k, v in PACKAGES.items()}, "removed_tags": {}, "frozen_joints": {},
           "meshes": {}, "head_posture": {"head_joint1": args.head_yaw, "head_joint2": args.head_pitch}}
    for tag in ROS_ONLY:
        found = root.findall(tag)
        for e in found:
            root.remove(e)
        rep["removed_tags"][tag] = len(found)
    # --- fixed v1 joints
    joints = {j.get("name"): j for j in root.findall("joint")}
    head = {"head_joint1": args.head_yaw, "head_joint2": args.head_pitch}
    torso = {f"body_joint{i + 1}": float(v) for i, v in enumerate(args.torso.split(","))}
    rep["torso_posture"] = torso
    for name, q in {**torso, **head}.items():
        if name not in joints:
            raise SystemExit(f"xacro has no joint {name!r}; refusing to guess")
        rep["frozen_joints"][name] = freeze(joints[name], q)
    base_like = [n for n, j in joints.items() if j.get("type") in ("continuous", "revolute")
                 and re.search(r"(driving|driven|caster|wheel)", n)]
    for n in base_like:
        rep["frozen_joints"][n] = freeze(joints[n], 0.0)
    # --- meshes: copy, relative paths, keep the package-relative layout
    for m in root.iter("mesh"):
        fn = m.get("filename")
        src = fn.replace("file://", "")
        pkg = next((k for k, v in PACKAGES.items() if src.startswith(str(v))), None)
        if pkg is None or not os.path.exists(src):
            raise SystemExit(f"mesh not found in the pinned packages: {fn}")
        rel = Path("meshes") / pkg / Path(src).relative_to(PACKAGES[pkg])
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
            rep["meshes"][str(rel)] = hashlib.sha256(dst.read_bytes()).hexdigest()[:16]
        m.set("filename", str(rel))
    # --- Habitat is y-up and habitat_sim does NOT convert a ROS z-up URDF (measured: identity root ->
    # robot lying on its side, the WXAI failure). habitat-lab's base_rot setter overwrites the root
    # rotation with a pure yaw, so the correction must live in the model: a new root link whose
    # fixed joint rotates the ROS base -90 deg about x. Root frame: +x forward, +y up, +z right.
    old_root = next(l.get("name") for l in root.findall("link")
                    if l.get("name") not in {j.find("child").get("link") for j in root.findall("joint")})
    hb = ET.Element("link", name="habitat_base")
    hj = ET.Element("joint", name="habitat_yup_joint", type="fixed")
    ET.SubElement(hj, "origin", xyz="0 0 0", rpy=f"{-math.pi / 2:.12f} 0 0")
    ET.SubElement(hj, "parent", link="habitat_base")
    ET.SubElement(hj, "child", link=old_root)
    root.insert(0, hj)
    root.insert(0, hb)
    rep["habitat_root"] = {"new_root": "habitat_base", "ros_root": old_root, "joint": "habitat_yup_joint",
                           "rpy": [-math.pi / 2, 0, 0]}
    rep["counts"] = {"links": len(root.findall("link")),
                     "joints": {t: sum(1 for j in root.findall("joint") if j.get("type") == t)
                                for t in sorted({j.get("type") for j in root.findall("joint")})},
                     "collision_elements": sum(1 for _ in root.iter("collision")),
                     "visual_elements": sum(1 for _ in root.iter("visual"))}
    rep["movable_joints"] = [j.get("name") for j in root.findall("joint") if j.get("type") != "fixed"]
    rep["mimic"] = {j.get("name"): {"joint": j.find("mimic").get("joint"),
                                    "multiplier": float(j.find("mimic").get("multiplier", 1)),
                                    "offset": float(j.find("mimic").get("offset", 0))}
                    for j in root.findall("joint") if j.find("mimic") is not None}
    ET.indent(root, space="  ")
    urdf = out / "alphabot2.urdf"
    urdf.write_text('<?xml version="1.0"?>\n' + ET.tostring(root, encoding="unicode"))
    rep["urdf_sha256"] = hashlib.sha256(urdf.read_bytes()).hexdigest()
    (out / "build_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("counts", "movable_joints", "removed_tags", "urdf_sha256")}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
