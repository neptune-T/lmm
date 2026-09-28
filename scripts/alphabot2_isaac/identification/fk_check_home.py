"""Independent FK from the vendor xacro, to test the USD<->xacro joint-offset mapping.

The arm chain is PARSED from arm.xacro for each side (resolving the ${direction}
branches), not hand-transcribed, so a transcription slip can't masquerade as a
result. Compares the gripper approach axis at xacro home_2 with Isaac's measurement.
"""
import re, math, json
import numpy as np
X = "/nix/yanjun/zts/alphabot2/deps/robot_descriptions/humanoid/Ai2Robotics/ai2_bot2_description/xacro/components"

def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                     [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                     [-sp,   cp*sr,            cp*cr]])
def T(xyz, rpy):
    M = np.eye(4); M[:3, :3] = rpy_to_R(*rpy); M[:3, 3] = xyz; return M
def axis_rot(axis, q):
    a = np.asarray(axis, float); a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    M = np.eye(4); M[:3, :3] = np.eye(3) + math.sin(q) * K + (1 - math.cos(q)) * K @ K; return M
def nums(s): return [float(v) for v in s.split()]

def resolve(block, direction):
    """Keep the <xacro:if value="${direction == D}"> branch matching this side."""
    def keep(m):
        return m.group(2) if float(m.group(1)) == direction else ""
    block = re.sub(r'<xacro:if value="\$\{direction == (-?1)\}">(.*?)</xacro:if>', keep, block, flags=re.S)
    return block.replace("${direction}", str(direction))

def parse_joints(xml, direction=None):
    out = {}
    for m in re.finditer(r'<joint name="([^"]+)" type="([^"]+)">(.*?)</joint>', xml, re.S):
        body = resolve(m.group(3), direction) if direction is not None else m.group(3)
        o = re.search(r'<origin([^>]*)/>', body)
        xyz = nums(re.search(r'xyz="([^"]+)"', o.group(1)).group(1)) if o and 'xyz=' in o.group(1) else [0, 0, 0]
        rpy = nums(re.search(r'rpy="([^"]+)"', o.group(1)).group(1)) if o and 'rpy=' in o.group(1) else [0, 0, 0]
        ax = re.search(r'<axis xyz="([^"]+)"', body)
        out[m.group(1)] = {"type": m.group(2), "xyz": xyz, "rpy": rpy,
                           "axis": nums(ax.group(1)) if ax else [0, 0, 1]}
    return out

arm_src = open(f"{X}/arm.xacro").read()
body = parse_joints(open(f"{X}/body.xacro").read())

def fk(side, q7):
    direction = 1 if side == "left" else -1
    J = parse_joints(arm_src, direction)
    M = np.eye(4)
    for n in ("body_joint1", "body_joint2", "body_joint3", "body_joint4"):   # body at 0
        M = M @ T(body[n]["xyz"], body[n]["rpy"])
    M = M @ T(J["${prefix}joint0"]["xyz"], J["${prefix}joint0"]["rpy"])
    for k in range(1, 8):
        j = J[f"${{prefix}}joint{k}"]
        M = M @ T(j["xyz"], j["rpy"]) @ axis_rot(j["axis"], q7[k - 1])
    j8 = J["${prefix}joint8"]; M = M @ T(j8["xyz"], j8["rpy"])
    M = M @ T([0, 0, 0.0695], [0, 0, 0])            # ee_joint
    M = M @ T([0, 0, 0], [0, 0, 1.5708])            # inspire gripper mount
    return M, J

HOME = {"left": [0.8333, 0, 0, -1.78, 0, -0.641, 0], "right": [-0.8333, 0, 0, -1.78, 0, -0.641, 0]}
ISAAC = {"left": [0.9984, -0.0523, 0.0235], "right": [0.1049, 0.0523, 0.9931]}
R = {}
for side in ("left", "right"):
    M, J = fk(side, HOME[side])
    approach = M[:3, 2]                              # gripper_base +z -> toward the fingertips
    R[side] = {"xacro_fk_approach": approach.round(4).tolist(),
               "isaac_measured_approach": ISAAC[side],
               "angle_between_deg": round(math.degrees(math.acos(np.clip(np.dot(approach, ISAAC[side]) /
                                                                            np.linalg.norm(ISAAC[side]), -1, 1))), 2),
               "gripper_base_pos_m": M[:3, 3].round(4).tolist(),
               "parsed_axes": {f"joint{k}": J[f"${{prefix}}joint{k}"]["axis"] for k in range(1, 8)}}
R["mirror_check_xacro"] = "left/right approach y-components should be opposite for a mirrored pose"
print(json.dumps(R, indent=1))
