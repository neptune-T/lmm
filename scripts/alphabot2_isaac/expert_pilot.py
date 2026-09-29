"""Privileged mobile-manipulation expert and pilot demonstration recorder (OFFLINE ONLY).

The expert uses privileged simulator state (object/box/robot poses) to
navigate -> reach -> grasp -> (navigate -> place). It is a data generator, not
part of any policy. Every action it takes is a CartesianMobileAction18 lowered by
CartesianExecutor -> AlphaBot2IsaacAdapter; the executed action is the label.

Recorded at 10 Hz per step t (proprio read BEFORE action t; images read right
after action t, which -- because the annotators deliver the previous render --
show that same state; see sync_check.py):
  images: cam_front (canonical head), cam_left_wrist, cam_right_wrist (+ cam_chest extra)
  proprio16 (canonical), action CartesianMobileAction18, language instruction
Metadata only (never a policy input): stage labels, event times, object pose,
grasp state, chassis pose, success, initial configuration (for replay).

Then every saved episode is replayed from its initial configuration and checked.
"""
import json, math, os, sys, time, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("LMM_REPO", os.path.dirname(os.path.dirname(HERE)))
USDS = os.environ.get("ROBOT_USDS", "/nix/yanjun/zts/alphabot2/deps/robot_usds")
OUT = os.environ.get("PILOT_OUT", "/nix/yanjun/zts/alphabot2/pilot")
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
from lmm.embodiments.alphabot2 import cartesian as C  # noqa: E402
from lmm.embodiments.alphabot2 import kinematics as K  # noqa: E402
from lmm.embodiments.alphabot2 import pi05_contract as P  # noqa: E402
from lmm.embodiments.alphabot2.contract import AlphaBot2Contract  # noqa: E402

O = {"episodes": [], "replays": []}
DEBUG = bool(int(os.environ.get("PILOT_DEBUG", "0")))
HOME = {"left": np.array(IA.HOME_LEFT_ARM), "right": np.array(IA.HOME_RIGHT_ARM)}
SIDEP = {"left": "Left", "right": "Right"}
TABLE_TOP = 0.67
CUBE = 0.05
REACH = 0.1196          # gripper_base -> pad midpoint along the approach (measured live)


def r4(x):
    return [round(float(v), 4) for v in np.asarray(x).reshape(-1)]


# ---------------- scene (fixed world) ----------------
world = World(stage_units_in_meters=1.0, physics_dt=1 / IA.PHYSICS_HZ, rendering_dt=1 / IA.PHYSICS_HZ)
stage = omni.usd.get_context().get_stage()
O["build"] = IB.build_stage(app, stage, USDS)
world.scene.add_default_ground_plane()
sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
sun.CreateIntensityAttr(3000.0)
sun.CreateAngleAttr(1.0)
UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set((-35.0, 20.0, 0.0))
PICK_TABLE = {"center": np.array([1.60, 0.0]), "size": np.array([0.50, 0.80])}     # x-extent, y-extent
PLACE_TABLE = {"center": np.array([0.0, 1.75]), "size": np.array([0.80, 0.50])}
BOX_XY = np.array([0.0, 1.60])
for name, T in (("pick_table", PICK_TABLE), ("place_table", PLACE_TABLE)):
    FixedCuboid(prim_path=f"/World/{name}", name=name, position=np.array([*T["center"], TABLE_TOP / 2]),
                scale=np.array([*T["size"], TABLE_TOP]), color=np.array([0.62, 0.52, 0.40]))
bw, bh, th = 0.14, 0.07, 0.01
for name, off, sc in (("floor", (0, 0, th / 2), (bw + 2 * th, bw + 2 * th, th)),
                      ("wx0", (-(bw + th) / 2, 0, bh / 2), (th, bw + 2 * th, bh)), ("wx1", ((bw + th) / 2, 0, bh / 2), (th, bw + 2 * th, bh)),
                      ("wy0", (0, -(bw + th) / 2, bh / 2), (bw, th, bh)), ("wy1", (0, (bw + th) / 2, bh / 2), (bw, th, bh))):
    FixedCuboid(prim_path=f"/World/box_{name}", name=f"box_{name}",
                position=np.array([BOX_XY[0] + off[0], BOX_XY[1] + off[1], TABLE_TOP + off[2]]),
                scale=np.array(sc), color=np.array([0.25, 0.35, 0.75]))
mat = PhysicsMaterial(prim_path="/World/mat_obj", static_friction=1.2, dynamic_friction=1.0, restitution=0.0)
cube = DynamicCuboid(prim_path="/World/cube", name="cube", position=np.array([1.42, 0.0, TABLE_TOP + CUBE / 2]),
                     scale=np.array([CUBE] * 3), mass=0.05, color=np.array([0.9, 0.15, 0.1]), physics_material=mat)
world.scene.add(cube)
IB.enable_contact_reports(stage)
art = SingleArticulation(prim_path=IB.ROBOT_PRIM, name="bot2")
world.scene.add(art)
world.reset()
B = IB.IsaacArticulationBackend(world, art)
A = IA.AlphaBot2IsaacAdapter(AlphaBot2Contract(), backend=B)
B.register_object("cube", cube)

# cameras (identical construction to the zero-shot smoke)
H, W = 240, 320
xc = UsdGeom.XformCache()
for name, path in (("cam_front", f"{IB.ROBOT_PRIM}/head_link2/head_camera"), ("cam_chest", f"{IB.ROBOT_PRIM}/body_link4/front_camera")):
    B.add_camera(name, path, (0, 0, 0), IB.camera_rows((0, 0, 1), (0, -1, 0)), (W, H), hfov_deg=69.0, near_m=0.10)
for side in ("left", "right"):
    S = SIDEP[side]
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
for _ in range(30):
    world.step(render=True)
CAMS = ("cam_front", "cam_left_wrist", "cam_right_wrist", "cam_chest")

# root pose offset relative to the chassis (constant at zero torso)
world.reset()
B.refresh()
root_p, root_q = art.get_world_pose()
cp0, cr0, _ = B.chassis_pose()


def quat_wxyz_to_R(q):
    w, x, y, z = q
    return P.quat_xyzw_to_matrix((x, y, z, w))


T_CH_ROOT = np.eye(4)
_Tc = np.eye(4); _Tc[:3, :3], _Tc[:3, 3] = cr0, cp0
_Tr = np.eye(4); _Tr[:3, :3], _Tr[:3, 3] = quat_wxyz_to_R(np.asarray(root_q)), np.asarray(root_p)
T_CH_ROOT = np.linalg.inv(_Tc) @ _Tr


def yaw_R(y):
    return np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])


# ---------------- expert primitives ----------------
TD_R = {s: np.array([[0.0, 0, 1], [0, 1, 0], [-1, 0, 0]]) @ K.fk(s, HOME[s])[:3, :3] for s in ("left", "right")}


class Episode:
    def __init__(self, meta, record=True, recorded_actions=None):
        self.meta, self.record = meta, record
        self.recorded_actions = recorded_actions
        self.buf = {k: [] for k in ("proprio", "action", "obj_pose", "chassis", "grasp", "stage", "sim_time")}
        self.imgs = {k: [] for k in CAMS}
        self.events = []
        self.stage = "init"
        self.t = 0
        self.ik_invalid = 0
        self.ik_log = []
        self.arrival = {}

    # ---- scene reset ----
    def reset(self):
        cfg = self.meta["init"]
        world.reset()
        B.refresh()
        yaw = cfg["base_yaw"]
        Tc = np.eye(4)
        Tc[:3, :3] = yaw_R(yaw)
        Tc[:3, 3] = [cfg["base_xy"][0], cfg["base_xy"][1], cp0[2]]
        Tr = Tc @ T_CH_ROOT
        rq = P.matrix_to_quat_xyzw(Tr[:3, :3])
        art.set_world_pose(position=Tr[:3, 3], orientation=np.array([rq[3], rq[0], rq[1], rq[2]]))
        cube.set_world_pose(position=np.array([*cfg["object_xy"], TABLE_TOP + CUBE / 2]), orientation=np.array([1.0, 0, 0, 0]))
        cube.set_linear_velocity(np.zeros(3))
        cube.set_angular_velocity(np.zeros(3))
        m = {}
        for side, dofs in (("left", IA.LEFT_ARM_DOFS), ("right", IA.RIGHT_ARM_DOFS)):
            for d, v in zip(dofs, A.arm_vendor_to_sim(side, HOME[side])):
                m[d] = float(v)
        m[IA.LEFT_GRIPPER_DOF] = m[IA.RIGHT_GRIPPER_DOF] = A.gripper_fraction_to_joint(0.0)
        for d in (*IA.BODY_DOFS, *IA.HEAD_DOFS):
            m[d] = 0.0
        B.teleport(m)
        self.ex = C.CartesianExecutor(q={"left": HOME["left"].copy(), "right": HOME["right"].copy()})
        self.cart = C.cartesian_from_joint_action(np.r_[HOME["left"], 0.0, HOME["right"], 0.0, 0.0, 0.0])
        B.render_on_step = True
        for _ in range(8):          # settle on the wheels; not recorded
            j, _ = self.ex.lower(self.cart)
            A.apply_action(j)
        self.q_plan = {"left": self.ex.q["left"].copy(), "right": self.ex.q["right"].copy()}

    # ---- one recorded step ----
    def step(self, cart):
        if self.recorded_actions is not None:          # replay: the recorded label is executed verbatim
            cart = self.recorded_actions[self.t]
        if self.record:
            prop = A.observe_proprio()
            self.buf["proprio"].append(prop)
            self.buf["action"].append(np.asarray(cart, float).copy())
            op, oq = cube.get_world_pose()
            self.buf["obj_pose"].append(np.r_[op, oq])
            cp, _, cy = B.chassis_pose()
            self.buf["chassis"].append(np.r_[cp, cy])
            self.buf["grasp"].append(self.grasped())
            self.buf["stage"].append(self.stage)
            self.buf["sim_time"].append(world.current_time)
        j, rep = self.ex.lower(cart)
        self.ik_invalid += int(not (rep["left"]["ik_valid"] and rep["right"]["ik_valid"]))
        A.apply_action(j)
        if self.record:
            # The camera annotators return the PREVIOUS render (measured, sync_check.py):
            # images read now show the state the proprio above was read from.
            for k in CAMS:
                self.imgs[k].append(B.render(k))
        if DEBUG and self.stage in ("reach", "descend", "grasp") and self.record:
            side = self.meta["arm"]
            pads = [B.link_pose(f"Gripper_{SIDEP[side]}/{n}")[0] for n in ("left_pad", "right_pad")]
            hits = {k: n for k, n in IB.contact_pairs().items() if "pick_table" in k or "/World/cube" in k}
            O.setdefault("debug", []).append({"t": self.t, "stage": self.stage, "pad_mid_world": r4((pads[0] + pads[1]) / 2),
                                              "cube": r4(cube.get_world_pose()[0]), "contacts": hits})
        self.t += 1

    def grasped(self):
        return any("/World/cube" in k and "Gripper_" in k and "pad" in k and n > 0 for k, n in IB.contact_pairs().items())

    def set_stage(self, s):
        self.stage = s
        self.events.append((self.t, s))

    # ---- arm helpers (vendor joints; Cartesian labels = FK of the plan) ----
    def arm_joint_move(self, side, q_goal, n, grip=None):
        q0 = self.q_plan[side].copy()
        for i in range(n):
            q = q0 + (q_goal - q0) * (i + 1) / n
            self._set_arm(side, q, grip)
            self.step(self.cart)
        self.q_plan[side] = np.asarray(q_goal, float).copy()

    def arm_line(self, side, dp_chassis, n, grip=None):
        m0 = K.fk(side, self.q_plan[side])
        q = self.q_plan[side].copy()
        worst = 0.0
        for i in range(n):
            t = m0.copy()
            t[:3, 3] += np.asarray(dp_chassis) * (i + 1) / n
            r = K.ik(side, t, q)
            q, worst = r.q, max(worst, r.pos_err_m)
        self.ik_log.append({"what": f"line {r4(dp_chassis)}", "side": side, "worst_pos_err_mm": round(worst * 1000, 2)})
        q = self.q_plan[side].copy()
        for i in range(n):
            t = m0.copy()
            t[:3, 3] += np.asarray(dp_chassis) * (i + 1) / n
            q = K.ik(side, t, q).q
            self._set_arm(side, q, grip)
            self.step(self.cart)
        self.q_plan[side] = q

    def finger_contact(self, side, what):
        tag = f"Gripper_{SIDEP[side]}/"
        hits = [k for k, n in IB.contact_pairs().items() if tag in k and what in k and n > 0]
        if hits:
            self.ik_log.append({"what": "contact", "side": side, "pairs": hits[:4]})
        return bool(hits)

    def arm_line_guarded(self, side, dp_chassis, n, stop_on):
        """Straight line, stopping at the first finger contact with ``stop_on``; then back up 5 mm."""
        m0 = K.fk(side, self.q_plan[side])
        q = self.q_plan[side].copy()
        for i in range(n):
            t = m0.copy()
            t[:3, 3] += np.asarray(dp_chassis) * (i + 1) / n
            q = K.ik(side, t, q).q
            self._set_arm(side, q, None)
            self.step(self.cart)
            if self.finger_contact(side, stop_on):
                self.ik_log.append({"what": f"guarded stop on {stop_on}", "side": side, "at_step": i + 1, "of": n})
                t = K.fk(side, q)
                t[2, 3] += 0.005
                q = K.ik(side, t, q).q
                self._set_arm(side, q, None)
                self.step(self.cart)
                break
        self.q_plan[side] = q

    def close_until_held(self, side, min_steps=6, max_steps=20):
        gi = 7 if side == "left" else 15
        last = None
        for i in range(max_steps):
            self.hold(1, grip=1.0, side=side)
            f = float(A.observe_proprio()[gi])
            if i + 1 >= min_steps and last is not None and abs(f - last) < 0.005:
                break
            last = f
        self.ik_log.append({"what": "close", "side": side, "steps": i + 1, "closed_fraction": round(f, 3)})
        if not (0.25 < f < 0.95):
            raise RuntimeError(f"gripper did not close on the object ({side}): fraction {f:.3f}")

    def hold(self, n, v=0.0, w=0.0, grip=None, side=None):
        for _ in range(n):
            if grip is not None:
                self.cart[C.CART_LEFT_GRIPPER if side == "left" else C.CART_RIGHT_GRIPPER] = grip
            self.cart[16], self.cart[17] = v, w
            self.step(self.cart)
        self.cart[16] = self.cart[17] = 0.0

    def _set_arm(self, side, q, grip):
        xyz, quat = P.action_from_vendor_joints(side, q, self.ex.prev_quat[side])
        sx, sq, sg = ((C.CART_LEFT_XYZ, C.CART_LEFT_QUAT, C.CART_LEFT_GRIPPER) if side == "left"
                      else (C.CART_RIGHT_XYZ, C.CART_RIGHT_QUAT, C.CART_RIGHT_GRIPPER))
        self.cart[sx], self.cart[sq] = xyz, quat
        if grip is not None:
            self.cart[sg] = grip
        self.cart[16] = self.cart[17] = 0.0

    # ---- navigation (differential drive, privileged pose) ----
    def base_pose(self):
        p, _, yaw = B.chassis_pose()
        return p[:2], yaw

    def drive_to(self, xy, yaw_goal, pre=0.35, vmax=0.35, wmax=0.6):
        xy = np.asarray(xy, float)
        head = np.array([math.cos(yaw_goal), math.sin(yaw_goal)])
        for goal, final in ((xy - pre * head, False), (xy, True)):
            # rotate toward the goal point, drive, (final leg: keep heading)
            for _ in range(300):
                p, yaw = self.base_pose()
                d = goal - p
                dist = float(np.linalg.norm(d))
                along = float(d @ np.array([math.cos(yaw), math.sin(yaw)]))
                # a differential base cannot remove lateral error on a straight leg:
                # the final leg stops on longitudinal error; the arm absorbs the rest
                if (final and abs(along) < 0.01) or (not final and dist < 0.03):
                    break
                bearing = math.atan2(d[1], d[0]) if not final else yaw_goal
                err = (bearing - yaw + math.pi) % (2 * math.pi) - math.pi
                if not final and abs(err) > 0.25:
                    self.hold(1, 0.0, float(np.clip(1.5 * err, -wmax, wmax)))
                    continue
                v = float(np.clip(1.2 * along, -vmax, vmax))
                if abs(v) < 0.04:
                    v = 0.04 * np.sign(along) if along != 0 else 0.0
                self.hold(1, v, float(np.clip(1.5 * err, -wmax, wmax)))
            if not final:
                for _ in range(200):
                    _, yaw = self.base_pose()
                    err = (yaw_goal - yaw + math.pi) % (2 * math.pi) - math.pi
                    if abs(err) < 0.02:
                        break
                    self.hold(1, 0.0, float(np.clip(1.5 * err, -wmax, wmax)))
        self.hold(4)

    def back_off(self, dist=0.30):
        self.hold(int(dist / 0.2 / 0.1), v=-0.2)
        self.hold(3)

    # ---- manipulation ----
    def obj_in_chassis(self):
        op = np.asarray(cube.get_world_pose()[0])
        return B.in_chassis(op)

    def box_in_chassis(self):
        return B.in_chassis(np.array([BOX_XY[0], BOX_XY[1], TABLE_TOP + th]))

    def solve(self, side, target, what, pos_tol=0.004, rot_tol_deg=2.0):
        """IK from several seeds; the best solution within tolerance, else fail loudly."""
        best = None
        for seed in (self.q_plan[side], SEED_PRE[side], HOME[side]):
            r = K.ik(side, target, seed)
            if best is None or r.pos_err_m + 0.1 * r.rot_err_rad < best.pos_err_m + 0.1 * best.rot_err_rad:
                best = r
        self.ik_log.append({"what": what, "side": side, "pos_err_mm": round(best.pos_err_m * 1000, 2),
                            "rot_err_deg": round(float(np.degrees(best.rot_err_rad)), 2), "at_limit": best.at_limit})
        if best.pos_err_m > pos_tol or np.degrees(best.rot_err_rad) > rot_tol_deg:
            raise RuntimeError(f"{what} IK failed ({side}): {best.pos_err_m * 1000:.1f} mm, "
                               f"{np.degrees(best.rot_err_rad):.1f} deg, at_limit {best.at_limit}")
        return best.q

    def pick(self, side):
        o = self.obj_in_chassis()
        self.arrival["object_in_chassis"] = r4(o)
        pre = np.eye(4); pre[:3, :3] = TD_R[side]; pre[:3, 3] = o + np.array([0, 0, REACH + 0.08])
        q_pre = self.solve(side, pre, "pre_grasp")
        self.set_stage("reach")
        self.arm_joint_move(side, q_pre, 25, grip=0.0)
        self.set_stage("descend")
        self.arm_line_guarded(side, (0, 0, -0.07), 10, stop_on="pick_table")   # pad mid 1 cm above the cube centre
        self.set_stage("grasp")
        self.close_until_held(side)
        self.hold(2)
        self.set_stage("lift")
        self.arm_line(side, (0, 0, 0.10), 10)
        self.hold(3)

    def place(self, side):
        b = self.box_in_chassis()
        self.arrival["box_in_chassis"] = r4(b)
        above = np.eye(4); above[:3, :3] = TD_R[side]
        above[:3, 3] = b + np.array([0, 0, CUBE / 2 + 0.02 + REACH + 0.06])   # cube bottom 2 cm over the rim
        q_above = self.solve(side, above, "place_above", pos_tol=0.015, rot_tol_deg=3.0)
        self.set_stage("reach_box")
        self.arm_joint_move(side, q_above, 20)
        self.set_stage("lower")
        self.arm_line_guarded(side, (0, 0, -0.06), 8, stop_on="box_")
        self.set_stage("release")
        self.hold(10, grip=0.0, side=side)
        self.set_stage("retract")
        self.arm_line(side, (0, 0, 0.12), 10)
        self.arm_joint_move(side, HOME[side], 25)

    def run(self):
        self.reset()
        cfg = self.meta["init"]
        side = self.meta["arm"]
        o = np.array([*cfg["object_xy"]])
        ofs = np.array([0.45, 0.25 if side == "left" else -0.25])
        self.set_stage("navigate_to_object")
        self.drive_to(o - yaw_R(0.0)[:2, :2] @ ofs, 0.0)
        self.pick(side)
        self.set_stage("back_off")
        self.back_off()
        if self.meta["kind"] == "full":
            self.set_stage("navigate_to_box")
            goal_yaw = math.pi / 2
            self.drive_to(BOX_XY - yaw_R(goal_yaw)[:2, :2] @ ofs, goal_yaw)
            self.place(side)
            self.set_stage("back_off")
            self.back_off(0.2)
        self.set_stage("done")

    def outcome(self):
        op = np.asarray(cube.get_world_pose()[0])
        in_box = bool(abs(op[0] - BOX_XY[0]) < bw / 2 and abs(op[1] - BOX_XY[1]) < bw / 2 and op[2] < TABLE_TOP + bh)
        lifted_held = bool(op[2] > TABLE_TOP + CUBE / 2 + 0.05 and self.grasped())
        ok = in_box if self.meta["kind"] == "full" else lifted_held
        return {"success": ok, "object_final": r4(op), "object_in_box": in_box, "object_lifted_and_held": lifted_held}


def plan_seeds():
    seeds = {}
    for side in ("left", "right"):
        y = 0.25 if side == "left" else -0.25
        pre = np.eye(4); pre[:3, :3] = TD_R[side]; pre[:3, 3] = (0.45, y, TABLE_TOP + CUBE / 2 + REACH + 0.08)
        r = K.ik(side, pre, HOME[side], margin=0.05)
        seeds[side] = r.q
        O[f"pregrasp_seed_{side}"] = {"q": r4(r.q), "converged": r.converged, "pos_err_mm": round(r.pos_err_m * 1000, 2)}
    return seeds


SEED_PRE = plan_seeds()
INSTR = {("full", "left"): "Use the left arm to pick up the red cube and put it in the blue box.",
         ("full", "right"): "Use the right arm to pick up the red cube and put it in the blue box.",
         ("pick", "left"): "Use the left arm to pick up the red cube.",
         ("pick", "right"): "Use the right arm to pick up the red cube."}
PLAN = [("full", "left"), ("full", "right"), ("pick", "left"), ("pick", "right"),
        ("full", "left"), ("full", "right"), ("pick", "left"), ("pick", "right"), ("full", "right"), ("pick", "left")]


def save(ep, eid, out):
    d = f"{OUT}/ep{eid:03d}"
    os.makedirs(d, exist_ok=True)
    arrs = {k: np.asarray(v) for k, v in ep.buf.items() if k != "stage"}
    arrs["stage_id"] = np.array([STAGES.index(s) for s in ep.buf["stage"]], np.int16)
    np.savez_compressed(f"{d}/lowdim.npz", **arrs)
    np.savez_compressed(f"{d}/images.npz", **{k: np.asarray(v, np.uint8) for k, v in ep.imgs.items()})
    meta = {**ep.meta, "episode_id": eid, "fps": 10, "steps": ep.t, "events": ep.events, "stages": STAGES,
            "action_space": "CartesianMobileAction18", "action_labels": list(C.CARTESIAN_MOBILE_ACTION18_LABELS),
            "proprio": "canonical 16-D (vendor joints, closed fractions)", "cameras": list(CAMS),
            "policy_inputs": ["cam_front", "cam_left_wrist", "cam_right_wrist", "proprio", "instruction"],
            "metadata_only": ["stage_id", "events", "obj_pose", "chassis", "grasp", "sim_time", "success"],
            "ik_invalid_steps": ep.ik_invalid, "expert_ik": ep.ik_log, "arrival": ep.arrival, **out}
    with open(f"{d}/meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    return d


STAGES = ["init", "navigate_to_object", "reach", "descend", "grasp", "lift", "back_off", "navigate_to_box",
          "reach_box", "lower", "release", "retract", "done"]
rng = np.random.default_rng(2026)
t_all = time.time()
saved = []
MAX_EP = int(os.environ.get("PILOT_MAX_EPISODES", len(PLAN)))
for eid, (kind, side) in enumerate(PLAN[:MAX_EP]):
    init = {"base_xy": [float(rng.uniform(-0.25, 0.25)), float(rng.uniform(-0.25, 0.25))],
            "base_yaw": float(rng.uniform(-0.45, 0.45)),
            "object_xy": [float(1.42 + rng.uniform(-0.02, 0.04)), float(rng.uniform(-0.12, 0.12))]}
    meta = {"kind": kind, "arm": side, "instruction": INSTR[(kind, side)], "init": init, "task_id": f"{kind}_{side}"}
    t0 = time.time()
    ep = Episode(meta)
    try:
        ep.run()
        out = ep.outcome()
    except Exception as e:
        out = {"success": False, "error": repr(e), "tb": traceback.format_exc()[-1200:]}
    out["wall_s"] = round(time.time() - t0, 1)
    out["steps"] = ep.t
    d = save(ep, eid, out) if ep.t > 0 else None
    res = {"episode": eid, "kind": kind, "arm": side, "dir": d, "arrival": ep.arrival, "expert_ik": ep.ik_log,
           **{k: v for k, v in out.items() if k != "tb"}}
    O["episodes"].append(res)
    print(json.dumps(res), flush=True)
    if out.get("success"):
        saved.append((eid, meta, np.asarray(ep.buf["action"]), np.asarray(ep.buf["proprio"]), np.asarray(ep.buf["obj_pose"])))
O["record_wall_s"] = round(time.time() - t_all, 1)

# ---------------- replay validation ----------------
for eid, meta, acts, props, objs in saved:
    t0 = time.time()
    ep = Episode(meta, record=True, recorded_actions=acts)
    ep.reset()
    for _ in range(len(acts)):
        ep.step(None)
    out = ep.outcome()
    rp = np.asarray(ep.buf["proprio"])
    ro = np.asarray(ep.buf["obj_pose"])
    res = {"episode": eid, "replay_success": out["success"], "steps": len(acts),
           "proprio_max_abs_diff": float(np.max(np.abs(rp - props))),
           "object_pos_max_diff_m": float(np.max(np.linalg.norm(ro[:, :3] - objs[:, :3], axis=1))),
           "object_final_diff_m": float(np.linalg.norm(ro[-1, :3] - objs[-1, :3])),
           "bitwise_identical_proprio": bool(np.array_equal(rp, props)),
           "wall_s": round(time.time() - t0, 1)}
    O["replays"].append(res)
    print("REPLAY", json.dumps(res), flush=True)

print("###JSON###" + json.dumps(O, default=str) + "\n", flush=True)
with open(f"{OUT}/pilot_summary.json", "w") as f:
    json.dump(O, f, indent=1, default=str)
app.close()
