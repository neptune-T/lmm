"""Third-person renders of the AlphaBot 2 model as Habitat simulates it (DIAGNOSTICS ONLY).

Adds a free camera (not one of the policy's cameras) and renders the robot from 4 sides in a real OVMM scene:
row 1 the reset / navigation tuck pose; rows 2-3 both arms set to the recorded joint configuration at the
grasp moment of an accepted left-arm and an accepted right-arm episode (kinematic, from proprio16). The last
column of each row is the policy's own [head | left wrist | right wrist] views at that pose.

    UNIT=16496#0 LEFT_DIR=... RIGHT_DIR=... OUT=png GPU=0 run_habitat.sh diag_robot_view.py
"""
import math, os, sys

import numpy as np

sys.path.insert(0, os.environ.get("LMM_REPO", "/workspace/lmm"))
import habitat_sim  # noqa: E402
import magnum as mn  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from lmm.embodiments.alphabot2.habitat import env as HE  # noqa: E402
from lmm.embodiments.alphabot2.habitat import nav as NAV  # noqa: E402

env = HE.make_env(gpu=int(os.environ.get("GPU", "0")), split="train")
e, r = HE.parse_unit(os.environ.get("UNIT", "16496#0"))
HE.reset_to(env, e, r)
HE.fix_start(env)
sim, rob = env.sim, HE.robot(env)
spec = habitat_sim.CameraSensorSpec()
spec.uuid, spec.sensor_type, spec.resolution = "tp_rgb", habitat_sim.SensorType.COLOR, [360, 480]
spec.hfov = 75
sim.add_sensor(spec)
node = sim._sensors["tp_rgb"]._sensor_object.node


def render_tp(eye, target):
    T = mn.Matrix4.look_at(mn.Vector3(*eye), mn.Vector3(*target), mn.Vector3(0, 1, 0))
    node.transformation = node.parent.absolute_transformation().inverted() @ T
    rob.update()
    o = sim.get_sensor_observations()
    return np.asarray(o["tp_rgb"])[..., :3].astype(np.uint8), o


def policy_views(o):
    return np.concatenate([np.asarray(o[k])[..., :3][::2, ::2] for k in ("head_rgb", "left_wrist_rgb", "right_wrist_rgb")], 1)


xz, yaw, p = NAV.pose2d(env)
c = np.array([p[0], p[1] + 0.8, p[2]])
fwd = np.array([math.cos(yaw), 0, -math.sin(yaw)])
left = np.array([-fwd[2], 0, fwd[0]]) * -1


def row(label):
    ims = []
    for name, d in (("front", fwd), ("back", -fwd), ("left", left), ("right", -left)):
        eye = c + 1.9 * d + np.array([0, 0.9, 0])
        im, o = render_tp(eye, c)
        im = Image.fromarray(im).resize((240, 180))
        ImageDraw.Draw(im).text((3, 3), f"{label} / {name}", fill=(255, 255, 0))
        ims.append(im)
    pv = Image.fromarray(policy_views(o)).resize((480, 120))
    return ims, pv


rows = [row("nav tuck (reset)")]
for side, dkey in (("left", "LEFT_DIR"), ("right", "RIGHT_DIR")):
    d = os.environ.get(dkey)
    if not d:
        continue
    prop = np.load(os.path.join(d, "lowdim.npz"))["proprio"]
    stg = [str(s) for s in np.load(os.path.join(d, "metadata.npz"))["stage"]]
    t = next((i for i, s in enumerate(stg) if s == "grasp"), len(stg) // 2)
    q = prop[t]
    rob.set_arm_q("left", q[0:7])
    rob.set_arm_q("right", q[8:15])
    rows.append(row(f"{side}-arm episode grasp pose (t={t})"))
W = 240 * 4
sheet = Image.new("RGB", (W, len(rows) * (180 + 120)), (30, 30, 30))
for i, (ims, pv) in enumerate(rows):
    for j, im in enumerate(ims):
        sheet.paste(im, (j * 240, i * 300))
    sheet.paste(pv, (0, i * 300 + 180))
    ImageDraw.Draw(sheet).text((490, i * 300 + 185), "policy views: head | left wrist | right wrist", fill=(255, 255, 0))
sheet.save(os.environ.get("OUT", "/dev/shm/robot_view.png"))
print("ROBOT_VIEW_OK", os.environ.get("OUT", "/dev/shm/robot_view.png"), "links", len(rob.sim_obj.get_link_ids()),
      "dofs", len(rob.sim_obj.joint_positions), flush=True)
env.close()
