"""Stage-1 privileged expert (OFFLINE data generator only) + task predicates.

Generalizes the pilot expert (scripts/alphabot2_isaac/expert_pilot.py) to the
Stage-1 distribution (lmm/embodiments/alphabot2/stage1_tasks.py): arbitrary
table approach yaw, in-place reorientation, grasp yaw aligned with the
object's geometry, placement on the same or another table. It uses privileged
poses; it is never part of a policy.

Every action is a CartesianMobileAction18 executed through ``SimWorld.act``
(CartesianExecutor -> adapter); the executed action is the label. Observations
come only from ``SimWorld.observe()`` (the timing contract).

``task_state`` / ``Outcome`` are shared with policy evaluation, so expert
success and policy success are measured by the same code.
"""
from __future__ import annotations

import math

import numpy as np

import simworld as SW
from lmm.embodiments.alphabot2 import cartesian as C
from lmm.embodiments.alphabot2 import kinematics as K
from lmm.embodiments.alphabot2 import pi05_contract as P

HOME = None                     # filled by Expert.setup (needs the adapter module)
REACH = 0.1196                  # gripper_base -> pad midpoint along the approach (measured live)
SIDEP = {"left": "Left", "right": "Right"}
ROBOT = "/World/Bot2"
ENV_PREFIXES = ("/World/table_", "/World/box_", "/World/obj_")


# ---------------------------------------------------------------- predicates
def robot_env_collisions(pairs: dict) -> list[str]:
    """Contacts between NON-gripper robot links and tables/box/objects.

    Gripper fingers/pads touching the table, the box or the object are part of
    grasping/placing (guarded moves stop on them); anything else touching the
    environment (arm links, torso, chassis, wheels) is a collision."""
    bad = []
    for k in pairs:
        a, b = k.split(" <-> ")
        for r, e in ((a, b), (b, a)):
            if r.startswith(ROBOT) and e.startswith(ENV_PREFIXES) and "/Gripper_" not in r:
                bad.append(k)
    return bad


def grasped_by(pairs: dict, obj_path: str) -> dict:
    """Which grippers have pad/finger contact with the object."""
    out = {"left": False, "right": False}
    for k in pairs:
        if obj_path in k:
            for side in ("left", "right"):
                if f"Gripper_{SIDEP[side]}/" in k:
                    out[side] = True
    return out


def task_state(W, cfg: dict, obj_rest_z: float) -> dict:
    """Privileged task predicates (used by the expert, the gates and evaluation)."""
    op, _ = W.object_pose()
    pairs = W.contacts()
    g = grasped_by(pairs, W.obj_paths[cfg["object"]])
    in_box = False
    if W.box_pose is not None:
        bx, by, byaw, bz = W.box_pose
        d = op[:2] - np.array([bx, by])
        c, s = math.cos(-byaw), math.sin(-byaw)
        lx, ly = c * d[0] - s * d[1], s * d[0] + c * d[1]
        half = SW.BOX_INNER / 2
        in_box = bool(abs(lx) < half and abs(ly) < half and op[2] < bz + SW.BOX_WALL_H)
    held = g[cfg["arm"]]
    lifted = bool(op[2] > obj_rest_z + 0.05)
    return {"object_pos": op, "grasped": held, "grasped_any": bool(g["left"] or g["right"]),
            "lifted": lifted, "lifted_held": bool(lifted and (g["left"] or g["right"])),
            "in_box": in_box, "in_box_released": bool(in_box and not (g["left"] or g["right"])),
            "on_floor": bool(op[2] < 0.3), "collisions": robot_env_collisions(pairs)}


def success_flag(cfg: dict, st: dict) -> bool:
    if cfg["family"] in ("nav_pick_nav_place", "short_pick_place"):
        return st["in_box_released"]
    return st["lifted_held"]


def yaw_R(y):
    return np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class ExpertFailure(RuntimeError):
    pass


class Expert:
    """Runs one Stage-1 task config. ``on_step(obs, cart, info)`` is called for
    every recorded step BEFORE the action is applied (the recorder hook)."""

    MAX_STEPS = 900

    def __init__(self, W, cfg: dict, on_step=None, on_after=None):
        global HOME
        self.W, self.cfg = W, cfg
        IA = W.IA
        HOME = {"left": np.array(IA.HOME_LEFT_ARM), "right": np.array(IA.HOME_RIGHT_ARM)}
        self.home = HOME
        self.on_step, self.on_after = on_step, on_after
        self.td_r = {s: np.array([[0.0, 0, 1], [0, 1, 0], [-1, 0, 0]]) @ K.fk(s, HOME[s])[:3, :3] for s in ("left", "right")}
        self.stage = "init"
        self.events = []
        self.ik_log = []
        self.t = 0
        self.ik_invalid = 0
        self.max_joint_step = 0.0

    # ---------------------------------------------------------- scene / reset
    def scene(self):
        W, cfg = self.W, self.cfg
        W.place_table("table_a", *cfg["table_a"])
        W.place_table("table_b", *cfg["table_b"])
        bx, by, byaw, on = cfg["box"]
        W.place_box(bx, by, byaw)
        W.place_object(cfg["object"], cfg["object_xy"][0], cfg["object_xy"][1], cfg["object_yaw"])

    def reset(self):
        self.cart = self.W.reset_robot(self.cfg["base_xy"], self.cfg["base_yaw"], scene=self.scene)
        self.q_plan = {"left": self.W.executor.q["left"].copy(), "right": self.W.executor.q["right"].copy()}
        self.obj_rest_z = float(self.W.object_pose()[0][2])
        self.closing_axis_gb = {}
        for s in ("left", "right"):
            pg, rg = self.W.B.link_pose(f"Gripper_{SIDEP[s]}/gripper_base")
            a_, b_ = (self.W.B.link_pose(f"Gripper_{SIDEP[s]}/{n}")[0] for n in ("left_pad", "right_pad"))
            ax = rg.T @ (a_ - b_)
            self.closing_axis_gb[s] = ax / np.linalg.norm(ax)
        self.seed_pre = {}
        for s in ("left", "right"):
            pre = np.eye(4); pre[:3, :3] = self.td_r[s]
            pre[:3, 3] = (*[0.45, 0.25 if s == "left" else -0.25], SW.TABLE_TOP - self.W.chassis_z + 0.03 + REACH + 0.08)
            self.seed_pre[s] = K.ik(s, pre, HOME[s], margin=0.05).q
        return self.cart

    # ---------------------------------------------------------- stepping
    def step(self, cart):
        if self.t >= self.MAX_STEPS:
            raise ExpertFailure(f"step budget {self.MAX_STEPS} exhausted in stage {self.stage}")
        W = self.W
        obs = W.observe(need_images=self.on_step is not None)
        info = {"stage": self.stage}
        if self.on_step is not None:
            self.on_step(obs, np.asarray(cart, float).copy(), info)
        _, rep = W.act(cart)
        bad = not (rep["left"]["ik_valid"] and rep["right"]["ik_valid"])
        self.ik_invalid += int(bad)
        self.max_joint_step = max(self.max_joint_step, rep["left"]["max_joint_step_rad"], rep["right"]["max_joint_step_rad"])
        if self.on_after is not None:
            self.on_after(rep)
        self.t += 1

    def set_stage(self, s):
        self.stage = s
        self.events.append((self.t, s))

    def hold(self, n, v=0.0, w=0.0, grip=None, side=None):
        for _ in range(n):
            if grip is not None:
                self.cart[C.CART_LEFT_GRIPPER if side == "left" else C.CART_RIGHT_GRIPPER] = grip
            self.cart[16], self.cart[17] = v, w
            self.step(self.cart)
        self.cart[16] = self.cart[17] = 0.0

    def _set_arm(self, side, q, grip):
        xyz, quat = P.action_from_vendor_joints(side, q, self.W.executor.prev_quat[side])
        sx, sq, sg = ((C.CART_LEFT_XYZ, C.CART_LEFT_QUAT, C.CART_LEFT_GRIPPER) if side == "left"
                      else (C.CART_RIGHT_XYZ, C.CART_RIGHT_QUAT, C.CART_RIGHT_GRIPPER))
        self.cart[sx], self.cart[sq] = xyz, quat
        if grip is not None:
            self.cart[sg] = grip
        self.cart[16] = self.cart[17] = 0.0

    def arm_joint_move(self, side, q_goal, n, grip=None):
        q0 = self.q_plan[side].copy()
        for i in range(n):
            # smoothstep profile: gentle start/stop
            u = (i + 1) / n
            u = u * u * (3 - 2 * u)
            self._set_arm(side, q0 + (q_goal - q0) * u, grip)
            self.step(self.cart)
        self.q_plan[side] = np.asarray(q_goal, float).copy()

    def arm_line(self, side, dp, n, grip=None):
        m0 = K.fk(side, self.q_plan[side])
        q = self.q_plan[side].copy()
        for i in range(n):
            t = m0.copy()
            t[:3, 3] += np.asarray(dp) * (i + 1) / n
            q = K.ik(side, t, q).q
            self._set_arm(side, q, grip)
            self.step(self.cart)
        self.q_plan[side] = q

    def finger_contact(self, side, what):
        tag = f"Gripper_{SIDEP[side]}/"
        return any(tag in k and what in k and n > 0 for k, n in self.W.contacts().items())

    def arm_line_guarded(self, side, dp, n, stop_on):
        m0 = K.fk(side, self.q_plan[side])
        q = self.q_plan[side].copy()
        for i in range(n):
            t = m0.copy()
            t[:3, 3] += np.asarray(dp) * (i + 1) / n
            q = K.ik(side, t, q).q
            self._set_arm(side, q, None)
            self.step(self.cart)
            if any(self.finger_contact(side, s) for s in stop_on):
                self.ik_log.append({"what": f"guarded stop {stop_on}", "side": side, "at": i + 1, "of": n})
                t = K.fk(side, q)
                t[2, 3] += 0.005
                q = K.ik(side, t, q).q
                self._set_arm(side, q, None)
                self.step(self.cart)
                break
        self.q_plan[side] = q

    def close_until_held(self, side, min_steps=6, max_steps=20):
        gi = 7 if side == "left" else 15
        last, f = None, 0.0
        for i in range(max_steps):
            self.hold(1, grip=1.0, side=side)
            f = float(self.W.A.observe_proprio()[gi])
            if i + 1 >= min_steps and last is not None and abs(f - last) < 0.005:
                break
            last = f
        self.ik_log.append({"what": "close", "side": side, "steps": i + 1, "closed_fraction": round(f, 3)})
        if not (0.25 < f < 0.95):
            raise ExpertFailure(f"gripper did not close on the object ({side}): fraction {f:.3f}")

    def solve(self, side, target, what, pos_tol=0.004, rot_tol_deg=2.0):
        best = None
        for seed in (self.q_plan[side], self.seed_pre[side], HOME[side]):
            r = K.ik(side, target, seed)
            if best is None or r.pos_err_m + 0.1 * r.rot_err_rad < best.pos_err_m + 0.1 * best.rot_err_rad:
                best = r
        self.ik_log.append({"what": what, "side": side, "pos_err_mm": round(best.pos_err_m * 1000, 2),
                            "rot_err_deg": round(float(np.degrees(best.rot_err_rad)), 2)})
        if best.pos_err_m > pos_tol or np.degrees(best.rot_err_rad) > rot_tol_deg:
            raise ExpertFailure(f"{what} IK failed ({side}): {best.pos_err_m * 1000:.1f} mm, "
                                f"{np.degrees(best.rot_err_rad):.1f} deg")
        return best.q

    # ---------------------------------------------------------- navigation
    def base_pose(self):
        p, yaw = self.W.chassis_pose()
        return p[:2], yaw

    def turn_to(self, yaw_goal, wmax=0.6, tol=0.02):
        for _ in range(250):
            _, yaw = self.base_pose()
            err = wrap(yaw_goal - yaw)
            if abs(err) < tol:
                return
            self.hold(1, 0.0, float(np.clip(1.5 * err, -wmax, wmax)))

    def drive_to(self, xy, yaw_goal, pre=0.35, vmax=0.35, wmax=0.6):
        xy = np.asarray(xy, float)
        head = np.array([math.cos(yaw_goal), math.sin(yaw_goal)])
        for goal, final in ((xy - pre * head, False), (xy, True)):
            for _ in range(400):
                p, yaw = self.base_pose()
                d = goal - p
                dist = float(np.linalg.norm(d))
                along = float(d @ np.array([math.cos(yaw), math.sin(yaw)]))
                if (final and abs(along) < 0.01) or (not final and dist < 0.03):
                    break
                bearing = math.atan2(d[1], d[0]) if not final else yaw_goal
                err = wrap(bearing - yaw)
                if not final and abs(err) > 0.25:
                    self.hold(1, 0.0, float(np.clip(1.5 * err, -wmax, wmax)))
                    continue
                v = float(np.clip(1.2 * along, -vmax, vmax))
                if abs(v) < 0.04:
                    v = 0.04 * np.sign(along) if along != 0 else 0.0
                self.hold(1, v, float(np.clip(1.5 * err, -wmax, wmax)))
            else:
                raise ExpertFailure(f"navigation leg did not converge (final={final})")
            if not final:
                self.turn_to(yaw_goal, wmax)
        self.hold(4)

    def back_off(self, dist):
        self.hold(int(round(dist / 0.2 / 0.1)), v=-0.2)
        self.hold(3)

    # ---------------------------------------------------------- manipulation
    def in_chassis(self, p_world):
        return self.W.B.in_chassis(np.asarray(p_world, float))

    def grasp_yaw(self, side):
        """Wrist rotation about the chassis z that aligns the closing axis with the object."""
        W, name = self.W, self.cfg["object"]
        kind, size, _, _ = SW.OBJECTS[name]
        _, q = W.object_pose()
        obj_yaw = 2 * math.atan2(q[3], q[0])                    # wxyz, upright object
        _, cyaw = W.chassis_pose()
        th = wrap(obj_yaw - cyaw)
        c = self.td_r[side] @ self.closing_axis_gb[side]
        a0 = math.atan2(c[1], c[0])
        if kind == "cylinder":
            return [0.0]
        if size[0] > size[1] + 0.01:                             # bar: close across its short (local y) side
            base, period = th + math.pi / 2 - a0, math.pi
        else:                                                    # square cross-section
            base, period = th - a0, math.pi / 2
        cands = sorted({round(wrap(base + k * period), 6) for k in range(-4, 5)}, key=abs)
        return [c_ for c_ in cands if abs(c_) <= math.pi / 2 + 1e-6][:3]

    def top_down(self, side, psi, pad_mid_chassis):
        t = np.eye(4)
        t[:3, :3] = yaw_R(psi) @ self.td_r[side]
        t[:3, 3] = np.asarray(pad_mid_chassis) + np.array([0, 0, REACH])
        return t

    def pick(self, side):
        W, name = self.W, self.cfg["object"]
        o = self.in_chassis(W.object_pose()[0])
        pad_mid = o + np.array([0.0, 0.0, 0.01])        # pad midpoint 1 cm above the object centre
        err = None
        for psi in self.grasp_yaw(side):
            try:
                q_pre = self.solve(side, self.top_down(side, psi, pad_mid + np.array([0, 0, 0.07])), f"pre_grasp psi={psi:.2f}")
                self.solve(side, self.top_down(side, psi, pad_mid), f"grasp psi={psi:.2f}")
                self.psi = psi
                break
            except ExpertFailure as e:
                err = e
        else:
            raise err
        self.set_stage("reach")
        self.arm_joint_move(side, q_pre, 25, grip=0.0)
        self.set_stage("descend")
        self.arm_line_guarded(side, (0, 0, -0.07), 10, stop_on=("table_",))
        self.set_stage("grasp")
        self.close_until_held(side)
        self.hold(2)
        self.set_stage("lift")
        self.arm_line(side, (0, 0, 0.10), 10)
        self.hold(3)

    def place(self, side):
        W = self.W
        bx, by, byaw, bz = W.box_pose
        b = self.in_chassis(np.array([bx, by, bz + SW.BOX_T]))
        h = SW.object_height(self.cfg["object"])
        pad_mid = b + np.array([0, 0, h / 2 + 0.01 + SW.BOX_WALL_H + 0.01])     # object bottom 1 cm over the rim
        q_above = self.solve(side, self.top_down(side, self.psi, pad_mid), "place_above", pos_tol=0.015, rot_tol_deg=3.0)
        self.set_stage("reach_box")
        self.arm_joint_move(side, q_above, 20)
        self.set_stage("lower")
        self.arm_line_guarded(side, (0, 0, -0.06), 8, stop_on=("box_",))
        self.set_stage("release")
        self.hold(10, grip=0.0, side=side)
        self.set_stage("retract")
        # straight up and stop there: a joint-space return to HOME passes near its aligned-axis
        # posture, where the Cartesian executor cannot track the plan (smoke: up to 95 mm IK error)
        self.arm_line(side, (0, 0, 0.12), 10)

    # ---------------------------------------------------------- episode
    def approach_table_a(self, side, target_xy):
        s = np.asarray(target_xy) - yaw_R(0.0)[:2, :2] @ np.array([0.45, 0.25 if side == "left" else -0.25])
        self.drive_to(s, 0.0)

    def run(self):
        cfg = self.cfg
        side, fam = cfg["arm"], cfg["family"]
        obj0 = np.array(cfg["object_xy"])
        if fam == "short_pick_place":
            self.set_stage("adjust_base")
            self.approach_table_a(side, (obj0 + np.array(cfg["box"][:2])) / 2)
        else:
            self.set_stage("navigate_to_object")
            self.approach_table_a(side, obj0)
        self.pick(side)
        if fam == "nav_pick_nav_place":
            self.set_stage("back_off")
            self.back_off(float(np.random.default_rng(cfg["seed"]).uniform(0.25, 0.40)))
            self.set_stage("navigate_to_box")
            bx, by, _, _ = cfg["box"]
            tyaw = cfg["table_b"][2]
            s = np.array([bx, by]) - yaw_R(tyaw)[:2, :2] @ np.array([0.45, 0.25 if side == "left" else -0.25])
            self.drive_to(s, tyaw)
            self.place(side)
            self.set_stage("back_off")
            self.back_off(0.2)
        elif fam == "short_pick_place":
            self.place(side)
            self.hold(3)
        else:
            self.set_stage("back_off")
            self.back_off(float(np.random.default_rng(cfg["seed"]).uniform(0.20, 0.35)))
        self.set_stage("done")
        self.hold(3)
