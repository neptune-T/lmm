#!/usr/bin/env python3
"""Emit the per-joint WXAI calibration table from the installed URDF.

The URDF is the only authority for axes, signs, zero positions and limits.
This script reads it and produces the table the embodiment audit requires, so
the numbers are extracted rather than transcribed by hand.

    python scripts/audit_wxai_urdf.py \
        --urdf <home-robot>/data/robots/trossen_mobile_ai/urdf/mobile_ai.urdf \
        --output docs/wxai_joint_table.md

Cross-checks performed, each of which has silently broken a robot before:

* the 12 arm joints named by ``validate_overlay.py`` all exist;
* joint i drives link i+1, matching the link names the Habitat robot class
  resolves (``follower_*_joint_0..5`` vs ``follower_*_link_1..6``);
* every arm joint is revolute with finite limits;
* both gripper carriage pairs are prismatic and share a travel range;
* the gripper open state used by the robot class lies inside that range.

Exit status is non-zero if any check fails, so this is usable as a gate.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from lmm.policy.wxai_transforms import ACTION_LABELS, ARM_INDICES, GRIPPER_INDICES

# Joint names asserted by habitat_patch/trossen_mobile_ai/validate_overlay.py
ARM_JOINTS = [f"follower_left_joint_{i}" for i in range(6)] + [
    f"follower_right_joint_{i}" for i in range(6)
]
GRIPPER_JOINT_PAIRS = {
    "left": ("follower_left_right_carriage_joint", "follower_left_left_carriage_joint"),
    "right": ("follower_right_right_carriage_joint", "follower_right_left_carriage_joint"),
}
# Link names resolved by TrossenMobileAIRobot.reconfigure()
ARM_LINKS = [f"follower_left_link_{i}" for i in range(1, 7)] + [
    f"follower_right_link_{i}" for i in range(1, 7)
]
#: MobileManipulatorParams.gripper_open_state in trossen_mobile_ai_robot.py
GRIPPER_OPEN_STATE_M = 0.044


def parse_joints(urdf: pathlib.Path) -> dict[str, dict]:
    root = ET.parse(urdf).getroot()
    out: dict[str, dict] = {}
    for node in root.findall("joint"):
        axis = node.find("axis")
        limit = node.find("limit")
        origin = node.find("origin")
        child = node.find("child")
        out[node.attrib["name"]] = {
            "type": node.attrib.get("type"),
            "axis": [float(v) for v in axis.attrib["xyz"].split()] if axis is not None else None,
            "lower": float(limit.attrib["lower"]) if limit is not None and "lower" in limit.attrib else None,
            "upper": float(limit.attrib["upper"]) if limit is not None and "upper" in limit.attrib else None,
            "effort": float(limit.attrib["effort"]) if limit is not None and "effort" in limit.attrib else None,
            "velocity": float(limit.attrib["velocity"]) if limit is not None and "velocity" in limit.attrib else None,
            "origin_rpy": [float(v) for v in origin.attrib.get("rpy", "0 0 0").split()] if origin is not None else [0, 0, 0],
            "child_link": child.attrib["link"] if child is not None else None,
        }
    return out


def dominant_axis(axis: list[float] | None) -> tuple[str, int]:
    """Return ``(axis_letter, sign)`` for a unit-ish URDF axis vector."""
    if axis is None:
        return ("?", 0)
    idx = max(range(3), key=lambda i: abs(axis[i]))
    return ("xyz"[idx], 1 if axis[idx] >= 0 else -1)


def audit(urdf: pathlib.Path) -> tuple[list[dict], list[str]]:
    joints = parse_joints(urdf)
    problems: list[str] = []

    missing = [j for j in ARM_JOINTS if j not in joints]
    if missing:
        problems.append(f"missing arm joints: {missing}")
    for pair in GRIPPER_JOINT_PAIRS.values():
        for j in pair:
            if j not in joints:
                problems.append(f"missing gripper joint: {j}")

    rows: list[dict] = []
    for slot, joint_name in enumerate(ARM_JOINTS):
        policy_dim = ARM_INDICES[slot]
        info = joints.get(joint_name)
        link = ARM_LINKS[slot]
        if info is None:
            rows.append({"policy_dim": policy_dim, "joint": joint_name,
                         "link": link, "status": "MISSING"})
            continue
        letter, sign = dominant_axis(info["axis"])
        if info["type"] != "revolute":
            problems.append(f"{joint_name} is {info['type']}, expected revolute")
        if info["lower"] is None or info["upper"] is None:
            problems.append(f"{joint_name} has no finite limits")
        if info["child_link"] != link:
            problems.append(
                f"{joint_name} drives {info['child_link']!r}, but the Habitat "
                f"robot class resolves {link!r} for policy dim {policy_dim}"
            )
        rows.append({
            "policy_dim": policy_dim,
            "label": ACTION_LABELS[policy_dim],
            "joint": joint_name,
            "link": link,
            "type": info["type"],
            "axis": letter,
            "sign": sign,
            "zero_rad": 0.0,
            "lower_rad": info["lower"],
            "upper_rad": info["upper"],
            "unit": "rad",
        })

    for side, pair in GRIPPER_JOINT_PAIRS.items():
        policy_dim = GRIPPER_INDICES[0 if side == "left" else 1]
        infos = [joints.get(j) for j in pair]
        if any(i is None for i in infos):
            continue
        for name, info in zip(pair, infos):
            if info["type"] != "prismatic":
                problems.append(f"{name} is {info['type']}, expected prismatic")
        lowers = {i["lower"] for i in infos}
        uppers = {i["upper"] for i in infos}
        if len(lowers) != 1 or len(uppers) != 1:
            problems.append(
                f"{side} carriage pair has asymmetric travel: lower={lowers} upper={uppers}"
            )
        lo, hi = infos[0]["lower"], infos[0]["upper"]
        if lo is not None and hi is not None and not (lo <= GRIPPER_OPEN_STATE_M <= hi):
            problems.append(
                f"{side} gripper_open_state {GRIPPER_OPEN_STATE_M} m is outside "
                f"the URDF carriage range [{lo}, {hi}]"
            )
        rows.append({
            "policy_dim": policy_dim,
            "label": ACTION_LABELS[policy_dim],
            "joint": " + ".join(pair),
            "link": f"follower_{side}_carriage_right/left",
            "type": infos[0]["type"],
            "axis": dominant_axis(infos[0]["axis"])[0],
            "sign": dominant_axis(infos[0]["axis"])[1],
            "zero_rad": 0.0,
            "lower_rad": lo,
            "upper_rad": hi,
            "unit": "m (carriage travel); policy dim is a unitless closed fraction",
        })

    return rows, problems


def render(rows: list[dict], problems: list[str], urdf: pathlib.Path) -> str:
    lines = [
        "# WXAI per-joint calibration table",
        "",
        f"Generated by `scripts/audit_wxai_urdf.py` from `{urdf}`.",
        "",
        "| policy dim | label | URDF joint | Habitat link | type | axis | sign | zero | lower | upper | unit |",
        "|---:|---|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        if r.get("status") == "MISSING":
            lines.append(f"| {r['policy_dim']} | — | `{r['joint']}` | `{r['link']}` | **MISSING** | | | | | | |")
            continue
        lines.append(
            f"| {r['policy_dim']} | `{r['label']}` | `{r['joint']}` | `{r['link']}` | "
            f"{r['type']} | {r['axis']} | {r['sign']:+d} | {r['zero_rad']} | "
            f"{r['lower_rad']} | {r['upper_rad']} | {r['unit']} |"
        )
    lines += ["", "## Cross-check result", ""]
    lines += ["All checks passed."] if not problems else [f"- **{p}**" for p in problems]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--urdf", type=pathlib.Path, required=True)
    ap.add_argument("--output", type=pathlib.Path)
    ap.add_argument("--json", type=pathlib.Path)
    args = ap.parse_args()

    if not args.urdf.is_file():
        sys.exit(
            f"URDF not found: {args.urdf}\n"
            "Install it with habitat_patch/trossen_mobile_ai/install.sh, which "
            "copies <trossen_arm_description>/urdf/generated/mobile_ai.urdf into "
            "<home-robot>/data/robots/trossen_mobile_ai/urdf/."
        )

    rows, problems = audit(args.urdf)
    text = render(rows, problems, args.urdf)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"[written] {args.output}")
    else:
        print(text)
    if args.json:
        args.json.write_text(json.dumps({"rows": rows, "problems": problems}, indent=2), encoding="utf-8")
    if problems:
        print(f"\nFAILED {len(problems)} cross-check(s)", file=sys.stderr)
        sys.exit(1)
    print("PASS all cross-checks", file=sys.stderr)


if __name__ == "__main__":
    main()
