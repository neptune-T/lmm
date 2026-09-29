"""Bounded canonical camera-posture sweep (torso fixed; neck only).

For each candidate neck pose (head_joint1 yaw, head_joint2 pitch) the head
camera (policy key camera_top <- cam_front) is scored on what it must show:

* navigation   -- floor corridor 0.6-3.0 m ahead (+-0.8 m) and the target
                  table top, robot 1.25 m from the table edge;
* object workspace -- table-top grid over the arms' reach
                  (chassis x 0.30-0.65, y +-0.45) at the grasp standoff;
* receptacle   -- box rim + floor points;
* object       -- top-face points of the object;
* both arms    -- pad midpoints and gripper-base points of BOTH arms at a
                  pre-grasp and at grasp height.

A 3-D point counts as visible if it projects inside the image and the rendered
depth there is not in front of it by more than 3 cm (so self-occlusion by the
arms/head shell counts). The chest camera (torso fixed, so constant) is scored
for reference. Montages of every candidate are saved.
"""
import json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import simworld as SW  # noqa: E402

app, ZD = SW.launch()
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from lmm.embodiments.alphabot2 import cartesian as C  # noqa: E402
from lmm.embodiments.alphabot2 import kinematics as K  # noqa: E402

OUT = os.environ.get("POSTURE_OUT", "/nix/yanjun/zts/alphabot2/stage1/posture")
os.makedirs(OUT, exist_ok=True)
W = SW.SimWorld(app, ZD, extra_camera_annotators=("depth",))
IA, B = W.IA, W.B
lim = B.dof_limits()
R = {"head_dof_limits_rad": {d: [round(float(v), 4) for v in lim[B.index[d]]] for d in IA.HEAD_DOFS}}
PITCHES = [float(v) for v in os.environ.get("PITCHES", "0,0.1,0.2,0.25,0.3,0.33,0.36,0.38").split(",")]
YAWS = [float(v) for v in os.environ.get("YAWS", "0").split(",")]
TABLE = (1.50, 0.0)
OBJ = np.array([1.40, 0.13])
BOX = np.array([1.40, -0.20])
TOP = SW.TABLE_TOP


def scene():
    W.place_table("table_a", TABLE[0], TABLE[1], 0.0)
    W.place_table("table_b", 0.0, 2.2, math.pi / 2)
    W.place_box(BOX[0], BOX[1], 0.0)
    W.place_object("red cube", OBJ[0], OBJ[1], 0.0)


def set_head(yaw, pitch):
    B.set_joint_targets({IA.HEAD_DOFS[0]: yaw, IA.HEAD_DOFS[1]: pitch})
    q = B._np(W.art.get_joint_positions()).astype(float)
    q[B.index[IA.HEAD_DOFS[0]]], q[B.index[IA.HEAD_DOFS[1]]] = yaw, pitch
    W.art.set_joint_positions(q)


def settle(n=6):
    for _ in range(n):
        W.world.step(render=True)


def visible(cam, pts_w):
    """Fraction of world points visible (in frame, unoccluded) in ``cam``."""
    p, r = W.camera_world_pose(cam)
    # read the annotator directly (synchronous, like get_rgba); get_current_frame() is a
    # callback-updated cache that can hold a frame from an earlier state
    depth = np.asarray(B.cameras[cam]._custom_annotators["distance_to_image_plane"].get_data())
    f = (SW.IMG_W / 2) / math.tan(math.radians(69.0 if cam in ("cam_front", "cam_chest") else 87.0) / 2)
    ok, inf = [], []
    for x in pts_w:
        c = r.T @ (np.asarray(x) - p)          # camera frame: +x right, +y up, -z forward
        z = -c[2]
        if z <= 0.05:
            ok.append(False); inf.append(False)
            continue
        u = SW.IMG_W / 2 + f * c[0] / z
        v = SW.IMG_H / 2 - f * c[1] / z
        if not (0 <= u < SW.IMG_W and 0 <= v < SW.IMG_H):
            ok.append(False); inf.append(False)
            continue
        inf.append(True)
        d = float(depth[int(v), int(u)]) if depth.ndim >= 2 else np.inf
        ok.append(bool(not np.isfinite(d) or d >= z - 0.03))
    VIS_LOG.append({"cam": cam, "in_frustum": float(np.mean(inf)), "unoccluded": float(np.mean(ok))})
    return float(np.mean(ok)), ok


def world_from_chassis(pc):
    cp, yaw = W.chassis_pose()
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([cp[0] + c * pc[0] - s * pc[1], cp[1] + s * pc[0] + c * pc[1], pc[2]])


def optical_axis(cam):
    _, r = W.camera_world_pose(cam)
    ax = -r[:, 2]
    return {"pitch_down_deg": round(float(math.degrees(math.asin(-ax[2]))), 1),
            "yaw_deg": round(float(math.degrees(math.atan2(ax[1], ax[0]))), 1)}


VIS_LOG = []
grid = lambda xs, ys, z: [np.array([x, y, z]) for x in xs for y in ys]
TD_R = {s: np.array([[0.0, 0, 1], [0, 1, 0], [-1, 0, 0]]) @ K.fk(s, np.array(IA.HOME_LEFT_ARM if s == "left" else IA.HOME_RIGHT_ARM))[:3, :3]
        for s in ("left", "right")}


def arm_q(side, pc_pad_mid):
    t = np.eye(4); t[:3, :3] = TD_R[side]; t[:3, 3] = np.asarray(pc_pad_mid) + np.array([0, 0, 0.1196])
    r = K.ik(side, t, np.array(IA.HOME_LEFT_ARM if side == "left" else IA.HOME_RIGHT_ARM))
    return r.q, r.pos_err_m


rows, montage = [], []
for yaw in YAWS:
    for pitch in PITCHES:
        row = {"head_joint1": yaw, "head_joint2": pitch}
        # --- navigation state: 1.25 m from the table edge, facing it
        W.reset_robot((0.0, 0.0), 0.0, scene=scene)
        set_head(yaw, pitch); settle()
        row["optical_axis"] = optical_axis("cam_front")
        floor = grid(np.linspace(0.6, 3.0, 9), np.linspace(-0.8, 0.8, 7), 0.0)
        tab = grid(np.linspace(TABLE[0] - 0.22, TABLE[0] + 0.22, 5), np.linspace(-0.37, 0.37, 7), TOP)
        row["nav_floor"], _ = visible("cam_front", floor)
        row["nav_table"], _ = visible("cam_front", tab)
        im_nav = B.render("cam_front")
        # --- approach states: 0.6 m and 0.3 m before the grasp standoff, facing the table
        base = OBJ - np.array([0.45, 0.25])
        for back in (0.6, 0.3):
            W.reset_robot((base[0] - back, base[1]), 0.0, scene=scene)
            set_head(yaw, pitch); settle()
            tab_near = grid(np.linspace(TABLE[0] - 0.24, TABLE[0] - 0.02, 4), np.linspace(-0.37, 0.37, 7), TOP)
            obj_top = [np.array([OBJ[0] + dx, OBJ[1] + dy, TOP + 0.05]) for dx in (-0.02, 0, 0.02) for dy in (-0.02, 0, 0.02)]
            box = [np.array([BOX[0] + dx, BOX[1] + dy, TOP + z]) for dx in (-0.07, 0, 0.07) for dy in (-0.07, 0, 0.07) for z in (0.01, 0.07)]
            row[f"approach{back}_table_near_half"] = visible("cam_front", tab_near)[0]
            row[f"approach{back}_object"] = visible("cam_front", obj_top)[0]
            row[f"approach{back}_receptacle"] = visible("cam_front", box)[0]
            if back == 0.3:
                im_app = B.render("cam_front")
        # --- manipulation state: object at the left arm's nominal point, both arms at pre-grasp / grasp
        sc = {}
        for stage, dz in (("pregrasp", 0.10), ("grasp", 0.012)):
            zc = TOP - W.chassis_z + 0.025 + dz          # chassis-frame height of the pad midpoint
            ql, el = arm_q("left", (0.45, 0.25, zc))
            qr, er = arm_q("right", (0.45, -0.25, zc))
            base = OBJ - np.array([0.45, 0.25])
            W.reset_robot((base[0], base[1]), 0.0, arm_left=ql, arm_right=qr, scene=scene)
            set_head(yaw, pitch); settle()
            pts_arm = []
            for S in ("Left", "Right"):
                a_, b_ = B.link_pose(f"Gripper_{S}/left_pad")[0], B.link_pose(f"Gripper_{S}/right_pad")[0]
                pts_arm += [(a_ + b_) / 2, B.link_pose(f"Gripper_{S}/gripper_base")[0]]
            sc[stage] = visible("cam_front", pts_arm)[0]
            if stage == "pregrasp":
                ws = [world_from_chassis(p) for p in grid(np.linspace(0.30, 0.65, 5), np.linspace(-0.45, 0.45, 7), TOP)]
                row["manip_workspace"] = visible("cam_front", ws)[0]
                obj_top = [np.array([OBJ[0] + dx, OBJ[1] + dy, TOP + 0.05]) for dx in (-0.02, 0, 0.02) for dy in (-0.02, 0, 0.02)]
                row["object"] = visible("cam_front", obj_top)[0]
                box = [np.array([BOX[0] + dx, BOX[1] + dy, TOP + z]) for dx in (-0.07, 0, 0.07) for dy in (-0.07, 0, 0.07)
                       for z in (0.01, 0.07)]
                row["receptacle"] = visible("cam_front", box)[0]
                row["chest_ref"] = {"manip_workspace": visible("cam_chest", ws)[0], "object": visible("cam_chest", obj_top)[0],
                                    "receptacle": visible("cam_chest", box)[0]}
                im_man = B.render("cam_front")
                row["ik_err_mm"] = [round(el * 1000, 1), round(er * 1000, 1)]
        row["both_arms"] = float(np.mean([sc["pregrasp"], sc["grasp"]]))
        cats = ("nav_floor", "nav_table", "manip_workspace", "object", "receptacle", "both_arms")
        row["min_category"] = min(row[c] for c in cats)
        row["mean_category"] = float(np.mean([row[c] for c in cats]))
        row["in_frustum_vs_unoccluded"] = VIS_LOG[-12:]
        rows.append(row)
        montage.append(np.concatenate([im_nav, im_man], axis=1))
        print(json.dumps(row), flush=True)
Image.fromarray(np.concatenate(montage, axis=0)).save(f"{OUT}/head_posture_montage.png")
# chest camera montage once (torso fixed -> identical for every neck pose)
W.reset_robot((0.0, 0.0), 0.0, scene=scene); settle()
c_nav = B.render("cam_chest")
W.reset_robot(tuple(OBJ - np.array([0.45, 0.25])), 0.0, scene=scene); settle()
Image.fromarray(np.concatenate([c_nav, B.render("cam_chest")], axis=1)).save(f"{OUT}/chest_reference.png")
CATS = ("nav_floor", "nav_table", "approach0.6_table_near_half", "approach0.6_object", "approach0.6_receptacle",
        "approach0.3_table_near_half", "approach0.3_object", "approach0.3_receptacle",
        "manip_workspace", "object", "receptacle", "both_arms")
# a category no admissible pose can see (max < 0.1) cannot discriminate: reported, not scored
scored = [c for c in CATS if max(r[c] for r in rows) >= 0.1]
structural = [c for c in CATS if c not in scored]
for r in rows:
    r["min_scored"] = min(r[c] for c in scored)
    r["mean_scored"] = float(np.mean([r[c] for c in scored]))
MARGIN = 0.02                      # rad inside the head_joint2 limit
lim_hi = float(lim[B.index[IA.HEAD_DOFS[1]]][1])
admissible = [r for r in rows if r["head_joint2"] <= lim_hi - MARGIN]
best = max(admissible, key=lambda r: (round(r["min_scored"], 2), round(r["mean_scored"], 3), -r["head_joint2"]))
R.update({"candidates": rows, "scored_categories": scored, "structurally_invisible": structural,
          "chosen": {"head_joint1": best["head_joint1"], "head_joint2": best["head_joint2"]},
          "rule": f"among poses >= {MARGIN} rad inside the joint limit: maximize the worst scored category, "
                  "ties by the mean, then the smaller pitch"})
print("###POSTURE###" + json.dumps(R, default=str), flush=True)
with open(f"{OUT}/camera_posture.json", "w") as fh:
    json.dump(R, fh, indent=1, default=str)
app.close()
