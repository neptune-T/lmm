"""Zero-shot smoke of the public AlphaBot pi0.5 checkpoint in Isaac (manipulation only, v = omega = 0).

Client of scripts/alphabot2_pi05/pi05_server.py (127.0.0.1:$PI05_PORT). The scene
is placed in the checkpoint's Cartesian frame B at its training workspace. Every
predicted chunk, executed Cartesian target, IK report, joint target, gripper
command, camera observation and task outcome is logged.
"""
import json, math, os, pickle, socket, struct, sys, time, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("LMM_REPO", os.path.dirname(os.path.dirname(HERE)))
USDS = os.environ.get("ROBOT_USDS", "/nix/yanjun/zts/alphabot2/deps/robot_usds")
EPS = os.environ.get("REPLAY_NPZ", "/nix/yanjun/zts/alphabot2/hf/replay_episodes.npz")
OUT = os.environ.get("SMOKE_OUT", "/nix/yanjun/zts/alphabot2/smoke/zeroshot")
PORT = int(os.environ.get("PI05_PORT", "5577"))
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, REPO)
from lmm.embodiments.alphabot2 import isaac_adapter as IA  # noqa: E402
from lmm.embodiments.alphabot2 import isaac_backend as IB  # noqa: E402
sys.argv.append(IB.RTX_DRIVER_CHECK_BYPASS)

from isaacsim import SimulationApp  # noqa: E402
app = SimulationApp({"headless": True})
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.api.materials import PhysicsMaterial  # noqa: E402
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid  # noqa: E402
from isaacsim.core.prims import SingleArticulation  # noqa: E402
from pxr import Usd, UsdGeom, UsdLux  # noqa: E402
from PIL import Image  # noqa: E402
from lmm.embodiments.alphabot2 import cartesian as C  # noqa: E402
from lmm.embodiments.alphabot2 import pi05_contract as P  # noqa: E402
from lmm.embodiments.alphabot2.contract import AlphaBot2Contract  # noqa: E402

O = {"episodes": []}


def r4(x):
    return [round(float(v), 4) for v in np.asarray(x).reshape(-1)]


# ---------------- policy client ----------------

def _enc(o):
    """numpy-version-neutral encoding (numpy 2 pickles reference numpy._core)."""
    if isinstance(o, np.ndarray):
        return {"__nd__": True, "dtype": o.dtype.str, "shape": o.shape, "data": o.tobytes()}
    if isinstance(o, dict):
        return {k: _enc(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return type(o)(_enc(v) for v in o)
    if isinstance(o, np.generic):
        return o.item()
    return o


def _dec(o):
    if isinstance(o, dict):
        if o.get("__nd__"):
            return np.frombuffer(o["data"], dtype=np.dtype(o["dtype"])).reshape(o["shape"]).copy()
        return {k: _dec(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return type(o)(_dec(v) for v in o)
    return o

def rpc(sock, obj):
    b = pickle.dumps(_enc(obj), protocol=4)
    sock.sendall(struct.pack("!Q", len(b)) + b)
    n = struct.unpack("!Q", sock.recv(8, socket.MSG_WAITALL))[0]
    buf = bytearray()
    while len(buf) < n:
        buf.extend(sock.recv(min(1 << 20, n - len(buf))))
    out = _dec(pickle.loads(bytes(buf)))
    if "error" in out:
        raise RuntimeError(out["error"] + "\n" + out.get("tb", ""))
    return out


sock = socket.create_connection(("127.0.0.1", PORT), timeout=600)
O["server_ping"] = rpc(sock, {"cmd": "ping"})["ok"]

# ---------------- world ----------------
world = World(stage_units_in_meters=1.0, physics_dt=1 / IA.PHYSICS_HZ, rendering_dt=1 / IA.PHYSICS_HZ)
stage = omni.usd.get_context().get_stage()
O["build"] = IB.build_stage(app, stage, USDS)
world.scene.add_default_ground_plane()
sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
sun.CreateIntensityAttr(3000.0)
sun.CreateAngleAttr(1.0)
UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set((-35.0, 20.0, 0.0))


def b_to_world(p_b):
    """Frame B -> chassis (robot spawns with chassis at the world origin, +x forward)."""
    return P.R_B_CHASSIS.T @ (np.asarray(p_b, float) - P.T_B_CHASSIS)


# table under the right arm's training workspace; top at z_B = 0.63
TOP_B = 0.63
tc = b_to_world((0.78, -0.25, TOP_B))
table_h = tc[2]
FixedCuboid(prim_path="/World/table", name="table", position=np.array([tc[0] + 0.05, tc[1], table_h / 2]),
            scale=np.array([0.55, 0.90, table_h]), color=np.array([0.62, 0.52, 0.40]))
mat = PhysicsMaterial(prim_path="/World/mat_obj", static_friction=1.0, dynamic_friction=0.9, restitution=0.0)
cube = DynamicCuboid(prim_path="/World/cube", name="cube", position=b_to_world((0.74, -0.32, TOP_B + 0.026)),
                     scale=np.array([0.05, 0.05, 0.05]), mass=0.05, color=np.array([0.9, 0.2, 0.15]), physics_material=mat)
bag = DynamicCuboid(prim_path="/World/bag", name="bag", position=b_to_world((0.80, -0.12, TOP_B + 0.021)),
                    scale=np.array([0.16, 0.11, 0.04]), mass=0.08, color=np.array([0.85, 0.95, 0.80]), physics_material=mat)
world.scene.add(cube)
world.scene.add(bag)
# open box on the table (5 static walls), interior 0.14 x 0.14 x 0.07
BOX_B = np.array([0.62, -0.42])
bw, bh, th = 0.14, 0.07, 0.01
for name, off, sc in (("floor", (0, 0, th / 2), (bw + 2 * th, bw + 2 * th, th)),
                      ("wx0", (-(bw + th) / 2, 0, bh / 2), (th, bw + 2 * th, bh)), ("wx1", ((bw + th) / 2, 0, bh / 2), (th, bw + 2 * th, bh)),
                      ("wy0", (0, -(bw + th) / 2, bh / 2), (bw, th, bh)), ("wy1", (0, (bw + th) / 2, bh / 2), (bw, th, bh))):
    c = b_to_world((BOX_B[0] + off[0], BOX_B[1] + off[1], TOP_B + off[2]))
    FixedCuboid(prim_path=f"/World/box_{name}", name=f"box_{name}", position=c, scale=np.array(sc), color=np.array([0.25, 0.35, 0.75]))
IB.enable_contact_reports(stage)
art = SingleArticulation(prim_path=IB.ROBOT_PRIM, name="bot2")
world.scene.add(art)
world.reset()
B = IB.IsaacArticulationBackend(world, art)
A = IA.AlphaBot2IsaacAdapter(AlphaBot2Contract(), backend=B)
B.register_object("cube", cube)
B.register_object("bag", bag)
EP = dict(np.load(EPS))
OBJ0 = {"cube": np.array(cube.get_world_pose()[0]), "bag": np.array(bag.get_world_pose()[0])}

# ---------------- cameras: chest, head, both wrists ----------------
H, W = 240, 320
xc = UsdGeom.XformCache()


def frame_axes(path, parent_link):
    rel = np.array(xc.GetLocalToWorldTransform(stage.GetPrimAtPath(path)) * xc.GetLocalToWorldTransform(
        stage.GetPrimAtPath(f"{IB.ROBOT_PRIM}/{parent_link}")).GetInverse()).T
    p, r = B.link_pose(parent_link)
    _, rc, _ = B.chassis_pose()
    return rc.T @ r @ rel[:3, :3]


cam_meta = {}
for name, path, link in (("cam_front", f"{IB.ROBOT_PRIM}/head_link2/head_camera", "head_link2"),
                         ("cam_chest", f"{IB.ROBOT_PRIM}/body_link4/front_camera", "body_link4")):
    ax = frame_axes(path, link)
    # optical-frame convention (+z optical axis, -y image up), as verified for head_camera
    B.add_camera(name, path, (0, 0, 0), IB.camera_rows((0, 0, 1), (0, -1, 0)), (W, H), hfov_deg=69.0, near_m=0.10)
    cam_meta[name] = {"optical_axis_chassis": r4(ax[:, 2]), "image_right_chassis": r4(ax[:, 0])}
for side in ("left", "right"):
    S = {"left": "Left", "right": "Right"}[side]
    d405 = [p for p in Usd.PrimRange(stage.GetPrimAtPath(f"{IB.ROBOT_PRIM}/Arm_{S}"), Usd.TraverseInstanceProxies()) if p.GetName() == "d405"][0]
    link7 = f"Arm_{S}/link7"
    p7, r7 = B.link_pose(link7)
    rel = np.array(xc.GetLocalToWorldTransform(d405) * xc.GetLocalToWorldTransform(stage.GetPrimAtPath(f"{IB.ROBOT_PRIM}/{link7}")).GetInverse()).T
    cam_w = p7 + r7 @ rel[:3, 3]
    pg, rg = B.link_pose(f"Gripper_{S}/gripper_base")
    a_, b_ = B.link_pose(f"Gripper_{S}/left_pad")[0], B.link_pose(f"Gripper_{S}/right_pad")[0]
    target = (a_ + b_) / 2 + 0.10 * rg[:, 2]
    up = cam_w - (pg + rg[:, 2] * ((cam_w - pg) @ rg[:, 2]))
    B.add_camera(f"cam_{side}_wrist", f"{IB.ROBOT_PRIM}/{link7}", r7.T @ (cam_w - p7),
                 IB.camera_rows(r7.T @ (target - cam_w), r7.T @ up), (W, H), hfov_deg=87.0, near_m=0.02)
O["cameras"] = cam_meta
O["checkpoint_camera_map"] = P.SIM_CAMERA_FOR
for _ in range(30):
    world.step(render=True)
B.render_on_step = True


def vendor_from_dataset(s):
    s = np.asarray(s, float)
    return P.DATASET_JOINT_SIGN["left"] * s[0:7], P.DATASET_JOINT_SIGN["right"] * s[7:14]


def reset_to(start_state):
    world.reset()
    B.refresh()
    ql, qr = vendor_from_dataset(start_state)
    m = {}
    for d, v in zip(IA.LEFT_ARM_DOFS, A.arm_vendor_to_sim("left", ql)):
        m[d] = float(v)
    for d, v in zip(IA.RIGHT_ARM_DOFS, A.arm_vendor_to_sim("right", qr)):
        m[d] = float(v)
    m[IA.LEFT_GRIPPER_DOF] = m[IA.RIGHT_GRIPPER_DOF] = A.gripper_fraction_to_joint(0.0)
    for d in (*IA.BODY_DOFS, *IA.HEAD_DOFS):
        m[d] = 0.0
    B.teleport(m)
    hold = np.zeros(18)
    hold[0:7], hold[8:15] = ql, qr
    for _ in range(8):
        A.apply_action(hold)
    return ql, qr


def tcp(side):
    p, r = B.link_pose(f"Gripper_{ {'left': 'Left', 'right': 'Right'}[side] }/gripper_base")
    return p + r @ P.T_GB_E[side]


def episode(tag, task, obj, start_state, seed, replan_every, n_actions):
    ql, qr = reset_to(start_state)
    ex = C.CartesianExecutor(q={"left": ql.copy(), "right": qr.copy()})
    o0 = np.array(B.objects[obj].get_world_pose()[0])
    log = {"chunks": [], "exec": []}
    min_d, lifted, grasp_contact = 9.0, 0.0, False
    n_done, ci = 0, 0
    epdir = f"{OUT}/{tag}"
    os.makedirs(epdir, exist_ok=True)
    while n_done < n_actions:
        imgs = {k: B.render(P.SIM_CAMERA_FOR[k]) for k in P.SIM_CAMERA_FOR}
        state = P.dataset_state_from_proprio(A.observe_proprio())
        resp = rpc(sock, {"cmd": "infer", "images": imgs, "state": state, "task": task, "seed": seed * 1000 + ci})
        chunk = resp["actions"]
        if ci < 6:
            Image.fromarray(np.concatenate([imgs[k] for k in ("camera_front", "camera_top", "camera_left", "camera_right")], axis=1)).save(f"{epdir}/chunk{ci:02d}.png")
        log["chunks"].append({"i": ci, "state_sent": r4(state), "latency_s": round(resp["latency_s"], 3),
                              "pred_first": r4(chunk[0]), "pred_last": r4(chunk[-1]),
                              "pred_right_xyz_range": [r4(chunk[:, 7:10].min(0)), r4(chunk[:, 7:10].max(0))],
                              "pred_right_grip": r4(chunk[:, 15][::5])})
        np.save(f"{epdir}/chunk{ci:02d}.npy", chunk)
        for k in range(min(replan_every, n_actions - n_done)):
            cart = np.r_[chunk[k], 0.0, 0.0]
            j, rep = ex.lower(cart)
            A.apply_action(j)
            op = np.array(B.objects[obj].get_world_pose()[0])
            d = float(np.linalg.norm(tcp("right") - op))
            min_d = min(min_d, d)
            lifted = max(lifted, float(op[2] - o0[2]))
            pairs = IB.contact_pairs()
            if any(f"/World/{obj}" in kk and "Gripper_Right" in kk and "pad" in kk for kk in pairs):
                grasp_contact = True
            log["exec"].append({"t": n_done, "cart": r4(cart[:16]), "ik_valid_R": rep["right"]["ik_valid"],
                                "ik_err_R_mm": round(rep["right"]["pos_err_m"] * 1000, 1), "ik_valid_L": rep["left"]["ik_valid"],
                                "joint_R": r4(j[8:15]), "grip_R": round(float(j[15]), 3), "tcp_obj_dist_m": round(d, 3)})
            n_done += 1
        ci += 1
    of = np.array(B.objects[obj].get_world_pose()[0])
    of_b = P.R_B_CHASSIS @ (of - B.chassis_pose()[0]) + P.T_B_CHASSIS
    in_box = bool(abs(of_b[0] - BOX_B[0]) < bw / 2 and abs(of_b[1] - BOX_B[1]) < bw / 2 and of_b[2] < TOP_B + bh)
    ex_log = log["exec"]
    res = {"tag": tag, "task": task, "object": obj, "seed": seed, "replan_every": replan_every, "actions": n_done,
           "chunks": ci, "ik_valid_frac_right": round(float(np.mean([e["ik_valid_R"] for e in ex_log])), 3),
           "ik_valid_frac_left": round(float(np.mean([e["ik_valid_L"] for e in ex_log])), 3),
           "min_tcp_object_dist_m": round(min_d, 3), "object_max_lift_m": round(lifted, 3),
           "grasp_contact_pads": grasp_contact, "object_in_box_end": in_box,
           "object_disp_m": round(float(np.linalg.norm(of - o0)), 3),
           "gripper_close_cmds": int(sum(e["grip_R"] >= 0.5 for e in ex_log)),
           "right_tcp_path_len_m": None, "median_latency_s": round(float(np.median([c["latency_s"] for c in log["chunks"]])), 3)}
    with open(f"{epdir}/log.json", "w") as f:
        json.dump({"result": res, **log}, f)
    O["episodes"].append(res)
    print(json.dumps(res), flush=True)


T_PICK = P.TASKS[3]
T_BAG_TOP = P.TASKS[0]
try:
    t0 = time.time()
    episode("pick_s0_c50", T_PICK, "cube", EP["start_median_task3"], 0, 50, 150)
    episode("pick_s1_c50", T_PICK, "cube", EP["start_median_task3"], 1, 50, 150)
    episode("pick_s0_r10", T_PICK, "cube", EP["start_median_task3"], 0, 10, 150)
    episode("bagtop_s0_c50", T_BAG_TOP, "bag", EP["start_median_task0"], 0, 50, 150)
    episode("bagtop_s1_r10", T_BAG_TOP, "bag", EP["start_median_task0"], 1, 10, 150)
    O["wall_s"] = round(time.time() - t0, 1)
except Exception as e:
    O["error"] = repr(e)
    O["tb"] = traceback.format_exc()[-2000:]
print("###JSON###" + json.dumps(O, default=str) + "\n", flush=True)
with open(f"{OUT}/zero_shot.json", "w") as f:
    json.dump(O, f, indent=1, default=str)
app.close()
