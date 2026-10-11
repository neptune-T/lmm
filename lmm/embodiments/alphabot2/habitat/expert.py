"""Privileged dual-arm OVMM expert for AlphaBot 2 in Habitat (DATA GENERATOR ONLY, never a policy).

Stages (metadata only, never a policy input)::

    tuck -> navigate_to_object -> reach -> descend -> grasp -> lift -> carry
         -> navigate_to_receptacle -> reach_place -> lower -> release -> retract -> done

Privileged inputs: OVMM semantic state (target object, goal receptacles and their
surfaces), exact poses, the navmesh, IK. Every command is a
CartesianMobileAction18 in the ARM FRAME A (kinematics.py) -- left/right EE pose
targets, grippers, (v, omega) -- lowered by the Isaac-qualified CartesianExecutor
to the atomic Habitat action; the executed action is the label. Joint-space arm
moves are labelled with the FK of the commanded joints, so every label is
reachable and the executor reproduces the plan.

Arm selection (privileged, geometric): for each arm the expert searches navigable
standoff poses from which that arm reaches a grasp of the object (top-down or
tilted up to 60 deg, IK inside the vendor limits, aperture fits the 0.102 m gripper)
and the geodesic cost; the cheaper feasible arm is chosen and recorded as metadata.
"""
from __future__ import annotations

import math

import magnum as mn
import numpy as np

from lmm.embodiments.alphabot2 import cartesian as C
from lmm.embodiments.alphabot2 import kinematics as K
from lmm.embodiments.alphabot2 import pi05_contract as P

from . import env as HE
from . import nav as NAV
from .alphabot2_robot import HOME, NAV_TUCK, REACH_M, SIDES, _np4

MAX_APERTURE_M = 0.0804            # fully open pad-SURFACE gap (xacro collision meshes, gripper_gap.py)
SETTLE_REST_STEPS = 55             # > OVMM stability_steps (50) consecutive at-rest steps after release
SETTLE_MAX_STEPS = int(__import__("os").environ.get("ALPHABOT2_SETTLE_MAX_STEPS", "300"))   # 30 s cap
RETRACT_LIFT_M = 0.12             # vertical lift of the releasing hand before anything else moves
RETRACT_BACK_V, RETRACT_BACK_STEPS = 0.2, 10    # then back the base off 0.2 m before tucking
PLACE_ROBUST_M = 0.03              # place IK must survive +-3 cm (base arrival error), as for picks
MAX_GRASP_WIDTH_M = 0.072          # widest object planned across the fingers (0.8 cm margin)
MIN_GRASP_WIDTH_M = 0.016
PAD_MID_GB = np.array([0.0, -0.006, 0.1196])     # pad midpoint in gripper_base (URDF, open)
#: Rim pinch of flat objects: DISABLED. For an object lying on a surface the lower finger must pass
#: through the support (measured: 22-33 gripper-support contact steps per pinch), so it is not a
#: physically valid grasp; flat objects wider than the aperture are an EG2-4C2 limitation.
ENABLE_RIM_PINCH = False
#: standoff candidates (ranked by geodesic distance) that get an IK check
IK_CANDIDATES = 12
#: base-to-object distance window: arm frame A is 0.104 m behind the chassis, the crouched arm reaches
#: grasp points ~0.35-0.75 m ahead in A -> 0.25-0.64 m from the base centre
R_MIN, R_MAX = 0.25, 0.64
#: fingertips reach 2.85 cm past the pad midpoint along the approach (xacro meshes, measured);
#: the pad midpoint is kept this far (x cos tilt) + 6 mm above the supporting surface
FINGERTIP_BEYOND_PAD_MID_M = 0.0285
TD_R = {s: np.array([[0.0, 0, 1], [0, 1, 0], [-1, 0, 0]]) @ K.fk(s, np.array(HOME[s]))[:3, :3] for s in SIDES}
TILTS = (0.0, math.radians(30), math.radians(45), math.radians(60))


class ExpertFailure(RuntimeError):
    pass


def Ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def hab_to_ros(p):
    """Habitat world (y up) point -> ROS-style world (z up), same origin: (x, -z, y)."""
    return np.array([p[0], -p[2], p[1]])


class OVMMExpert:
    def __init__(self, env, on_step=None, force_arm: str | None = None, rng_seed: int = 0):
        self.env, self.on_step, self.force_arm = env, on_step, force_arm
        self.rob = HE.robot(env)
        self.stage = "init"
        self.events: list = []
        self.log: dict = {"ik": [], "arm_choice": None}
        self.ex = None
        self.rng = np.random.default_rng(rng_seed)
        self.t = 0
        # COMMANDED gripper state (the label semantics: 1 = close/hold, 0 = open). Never the measured closed
        # fraction: holding an object the fingers stop at its width (fraction < close threshold), and
        # re-commanding that measurement released the object on the first navigation step.
        p = self.rob.proprio16()
        self.grip_cmd = {"left": float(p[7] >= 0.5), "right": float(p[15] >= 0.5)}

    # ------------------------------------------------------------------ geometry
    def arm_T_world(self):
        """World (Habitat) pose of arm frame A as 4x4, ROS convention inside A (x fwd, y left, z up)."""
        return self.rob.arm_frame()

    def to_arm(self, p_hab):
        A = self.arm_T_world()
        return (np.linalg.inv(A) @ np.r_[np.asarray(p_hab, float), 1.0])[:3]

    @staticmethod
    def frame(approach, closing):
        """gripper_base rotation (arm frame) with +z along ``approach`` and the finger-closing axis
        (gripper_base -x, measured from the URDF pads) along ``closing`` (orthogonalized)."""
        a = np.asarray(approach, float)
        a /= np.linalg.norm(a)
        c = np.asarray(closing, float)
        c = c - a * (c @ a)
        n = np.linalg.norm(c)
        if n < 1e-6:
            return None
        x = -c / n                                 # gripper_base +x = -closing
        y = np.cross(a, x)
        return np.stack([x, y, a], axis=1)

    @staticmethod
    def pose(R, pad_target, backoff=0.0):
        """gripper_base pose whose pad midpoint is at ``pad_target`` (arm frame), backed off along -approach."""
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = np.asarray(pad_target) - R @ PAD_MID_GB - R[:, 2] * backoff
        return T

    def ik_ok(self, side, T, seeds=None):
        best = None
        for seed in seeds or (np.array(HOME[side]), np.array(NAV_TUCK[side])):
            r = K.ik(side, T, seed)
            if best is None or r.pos_err_m + 0.1 * r.rot_err_rad < best.pos_err_m + 0.1 * best.rot_err_rad:
                best = r
        return best, bool(best.pos_err_m < 0.004 and best.rot_err_rad < math.radians(2.0))

    # ------------------------------------------------------------------ stepping
    def step(self, cart18):
        self.grip_cmd = {"left": float(cart18[C.CART_LEFT_GRIPPER]), "right": float(cart18[C.CART_RIGHT_GRIPPER])}
        joint, rep = self.ex.lower(cart18)
        valid = bool(rep["left"]["ik_valid"] and rep["right"]["ik_valid"])
        if self.on_step is not None:
            self.on_step(np.asarray(cart18, float).copy(), self.stage, rep)
        obs = HE.step18(self.env, joint)
        self.t += 1
        if not valid:
            self.log.setdefault("ik_invalid_steps", 0)
            self.log["ik_invalid_steps"] += 1
        return obs

    def set_stage(self, s):
        self.stage = s
        self.events.append((self.t, s))

    def cart_now(self):
        """Hold the current arm pose; grippers keep their COMMANDED state."""
        c = C.cartesian_from_joint_action(np.r_[self.rob.proprio16(), 0.0, 0.0], self.ex.prev_quat)
        c[C.CART_LEFT_GRIPPER], c[C.CART_RIGHT_GRIPPER] = self.grip_cmd["left"], self.grip_cmd["right"]
        return c

    def joint_move(self, targets: dict, n: int, grips=None, v=0.0, w=0.0):
        """Interpolate arm joints (per side) to targets; label = FK of the commanded joints."""
        p0 = self.rob.proprio16()
        q0 = {"left": p0[0:7].copy(), "right": p0[8:15].copy()}
        g = dict(self.grip_cmd) if grips is None else {**self.grip_cmd, **grips}
        for i in range(n):
            u = (i + 1) / n
            u = u * u * (3 - 2 * u)
            j = np.zeros(18)
            for side, sl, gi in (("left", slice(0, 7), 7), ("right", slice(8, 15), 15)):
                qt = targets.get(side, q0[side])
                j[sl] = q0[side] + (np.asarray(qt) - q0[side]) * u
                j[gi] = g[side]
            j[16], j[17] = v, w
            self.step(C.cartesian_from_joint_action(j, self.ex.prev_quat))

    def line(self, side, dp_arm, n, grip=None):
        """Straight Cartesian line of one gripper_base (arm frame), the other arm held."""
        cart0 = self.cart_now()
        sx = C.CART_LEFT_XYZ if side == "left" else C.CART_RIGHT_XYZ
        sq = C.CART_LEFT_QUAT if side == "left" else C.CART_RIGHT_QUAT
        T0 = P.gripper_base_from_action(side, cart0[sx], cart0[sq])
        for i in range(n):
            T = T0.copy()
            T[:3, 3] += np.asarray(dp_arm) * (i + 1) / n
            xyz, quat = P.action_from_gripper_base(side, T, self.ex.prev_quat[side])
            c = self.cart_now()
            c[sx], c[sq] = xyz, quat
            if grip is not None:
                c[C.CART_LEFT_GRIPPER if side == "left" else C.CART_RIGHT_GRIPPER] = grip
            c[16] = c[17] = 0.0
            self.step(c)

    def hold(self, n, grips=None):
        for _ in range(n):
            c = self.cart_now()
            if grips:
                for side, g in grips.items():
                    c[C.CART_LEFT_GRIPPER if side == "left" else C.CART_RIGHT_GRIPPER] = g
            c[16] = c[17] = 0.0
            self.step(c)

    # ------------------------------------------------------------------ navigation (through the same step)
    def nav_to(self, goal_xyz, face_xz):
        rob_tuck = {"left": np.array(NAV_TUCK["left"]), "right": np.array(NAV_TUCK["right"])}
        expert = self

        class _Shim:                       # route NAV's steps through the expert (labels + recording)
            pass

        def step_vw(env_, v, w):
            c = expert.cart_now()
            c[16], c[17] = v, w
            return expert.step(c)
        st = NAV.follow_with(self.env, goal_xyz, step_vw, face_xz=face_xz, arrive_m=0.02)
        return st

    # ------------------------------------------------------------------ planning
    def target_object(self):
        sim = self.env.sim
        idx, goals = sim.get_targets()
        obj = sim.get_rigid_object_manager().get_object_by_id(sim.scene_obj_ids[int(idx[0])])
        return obj, np.asarray(goals[0], float)

    def object_geometry(self, obj):
        bb = obj.root_scene_node.cumulative_bb
        T = obj.transformation
        c_local = (np.array(bb.min) + np.array(bb.max)) / 2
        centre = np.array(T.transform_point(mn.Vector3(*c_local)))
        R = np.array(T.rotation())
        half = (np.array(bb.max) - np.array(bb.min)) / 2
        # horizontal extents of the rotated box (world x/z) and vertical half-height (world y)
        ext_world = np.abs(R) @ half
        return centre, ext_world, R, half

    def standoff_candidates(self, target_hab, side, dists=None, n=160, laterals=(0.14, 0.22, 0.30)):
        """Navigable base poses near ``target_hab`` (same navmesh island as the robot), each with the
        heading that puts the target at the arm's side (lateral offset in the arm frame)."""
        pf = self.env.sim.pathfinder
        start = self.rob.sim_obj.translation
        island = pf.get_island(start)
        floor_y = float(start[1])
        tgt = mn.Vector3(float(target_hab[0]), floor_y, float(target_hab[2]))
        sgn = 1.0 if side == "left" else -1.0
        out, seen, misses = [], set(), 0
        # existence is decided exactly, not by sampling luck: the nearest navigable point on the robot's
        # island must lie within the arm's planar reach of the target
        snap = np.array(pf.snap_point(tgt, island))
        self.last_nearest_navigable_m = (float(np.hypot(snap[0] - target_hab[0], snap[2] - target_hab[2]))
                                         if np.isfinite(snap).all() else None)
        if self.last_nearest_navigable_m is None or self.last_nearest_navigable_m > R_MAX:
            return out
        pf.seed(int(self.rng.integers(1 << 30)))
        pts = [snap]
        for _ in range(n):
            p = np.array(pf.get_random_navigable_point_near(tgt, R_MAX, 100, island))
            if not np.isfinite(p).all():
                misses += 1
                continue
            pts.append(p)
        for p in pts:
            d = np.array([target_hab[0] - p[0], target_hab[2] - p[2]])
            r = float(np.linalg.norm(d))
            if r < R_MIN or r > R_MAX:                    # object within the crouched arm's forward reach
                continue
            key = (round(p[0], 2), round(p[2], 2))
            if key in seen:
                continue
            seen.add(key)
            bearing = math.atan2(-d[1], d[0])                     # yaw convention of nav.pose2d
            for lat in laterals:
                if r <= lat + 0.05:
                    continue
                yaw = bearing - sgn * math.asin(lat / r)          # object ends up `lat` to the arm's side
                out.append({"xyz": p, "yaw": yaw, "d": r, "lat": lat})
        return out

    @staticmethod
    def width_along(R, R_obj, half):
        """Object box extent along the hand's closing axis (gripper_base -x): the width the fingers
        actually close across, computed exactly as the grasp action does at attach time."""
        c = -np.asarray(R)[:, 0]
        return float(2 * np.sum(np.abs(np.asarray(R_obj).T @ c) * np.asarray(half)))

    def grasp_candidates(self, obj_centre_arm, R_obj_arm, half, robot_xy_arm=(0.0, 0.0)):
        """Grasp options for a box object (arm frame): top/tilted grasps closing across a narrow
        horizontal axis, and a rim pinch closing across the thickness of a flat object."""
        out = []
        vert = int(np.argmax(np.abs(R_obj_arm[2, :])))             # object axis closest to vertical
        hz = [i for i in range(3) if i != vert]
        to_obj = np.asarray(obj_centre_arm[:2]) - np.asarray(robot_xy_arm)
        to_obj = to_obj / max(1e-6, np.linalg.norm(to_obj))
        u = np.array([to_obj[0], to_obj[1], 0.0])                   # horizontal direction robot -> object
        for i in hz:
            width = 2 * half[i]
            if not MIN_GRASP_WIDTH_M <= width <= MAX_GRASP_WIDTH_M:
                continue
            axis = R_obj_arm[:, i].copy()
            axis[2] = 0.0
            if np.linalg.norm(axis) < 1e-6:
                continue
            vz = abs(float(R_obj_arm[2, vert])) * half[vert] + sum(abs(float(R_obj_arm[2, j])) * half[j] for j in hz)
            bottom, top = obj_centre_arm[2] - vz, obj_centre_arm[2] + vz
            for tilt in TILTS:                                     # 0 = straight down, leaning forward toward the object
                appr = np.array([u[0] * math.sin(tilt), u[1] * math.sin(tilt), -math.cos(tilt)])
                pad = np.asarray(obj_centre_arm, float).copy()
                pad[2] = max(pad[2], bottom + FINGERTIP_BEYOND_PAD_MID_M * math.cos(tilt) + 0.006)
                if pad[2] > top + 0.003:                           # too flat: the pads would miss it
                    continue
                for sgn in (1.0, -1.0):
                    R = self.frame(appr, sgn * axis)
                    if R is None:
                        continue
                    # frame() makes the closing axis perpendicular to the (tilted) approach, which tilts it
                    # out of the horizontal when the object axis is not perpendicular to robot->object; the
                    # hand then closes across the box along THAT axis (what the grasp action measures)
                    w_eff = self.width_along(R, R_obj_arm, half)
                    if MIN_GRASP_WIDTH_M <= w_eff <= MAX_GRASP_WIDTH_M:
                        out.append({"mode": "top", "tilt": tilt, "R": R, "pad": pad, "width": w_eff})
        thick = 2 * half[vert]
        if ENABLE_RIM_PINCH and thick <= MAX_GRASP_WIDTH_M and min(2 * half[i] for i in hz) > MAX_GRASP_WIDTH_M:
            # flat object: pinch the rim nearest to the robot across its thickness
            # distance from the object centre to its rim along the robot->object direction
            r_edge = float(sum(abs(float(R_obj_arm[:2, i] @ u[:2])) * half[i] for i in hz))
            pad = np.asarray(obj_centre_arm, float) - u * max(0.0, r_edge - 0.025)
            for down in (math.radians(15), math.radians(35), math.radians(55)):
                appr = np.array([u[0] * math.cos(down), u[1] * math.cos(down), -math.sin(down)])
                for sgn in (1.0, -1.0):
                    R = self.frame(appr, sgn * np.array([0.0, 0.0, 1.0]))
                    if R is None:
                        continue
                    w_eff = self.width_along(R, R_obj_arm, half)
                    if MIN_GRASP_WIDTH_M <= w_eff <= MAX_GRASP_WIDTH_M:
                        out.append({"mode": "rim", "tilt": down, "R": R, "pad": pad, "width": w_eff})
        return out

    def feasible_grasp(self, side, cands, robust=0.0):
        """First grasp option whose grasp and pre-grasp are IK-feasible; with ``robust`` > 0 the grasp
        must also stay feasible with the object shifted by +-robust in x and y (base arrival error)."""
        for g in cands:
            rg, okg = self.ik_ok(side, self.pose(g["R"], g["pad"]))
            if not okg:
                continue
            rp, okp = self.ik_ok(side, self.pose(g["R"], g["pad"], backoff=0.10), seeds=(rg.q,))
            if not okp:
                continue
            if robust > 0 and not all(self.ik_ok(side, self.pose(g["R"], g["pad"] + np.array(d)), seeds=(rg.q,))[1]
                                      for d in ((robust, 0, 0), (-robust, 0, 0), (0, robust, 0), (0, -robust, 0))):
                continue
            return {**g, "q_grasp": rg.q, "q_pre": rp.q}
        return None

    def plan_pick(self, side, obj):
        centre, ext, R_obj, half = self.object_geometry(obj)
        cands = self.standoff_candidates(centre, side)
        floor_y = float(self.rob.sim_obj.translation[1])
        diag = {"nearest_navigable_m": self.last_nearest_navigable_m,
                "object_height_above_floor_m": round(float(centre[1] - floor_y), 3),
                "object_extent_local_m": (2 * half).round(3).tolist(), "candidates_navigable": len(cands),
                "reachable": 0, "grasp_options": 0, "ik_ok": 0, "sample_obj_arm": None}
        self.log.setdefault("pick_diag", {})[side] = diag
        best = None
        ranked = []
        for c in cands:                                   # geodesic first (cheap), IK only for the nearest
            ok, _, geo = NAV.plan(self.env, c["xyz"])
            if ok and np.isfinite(geo):
                ranked.append((geo, c))
        diag["reachable"] = len(ranked)
        ranked.sort(key=lambda gc: gc[0])
        for geo, c in ranked[:IK_CANDIDATES]:
            A = self._arm_frame_at(c["xyz"], c["yaw"])
            Ai = np.linalg.inv(A)
            o_arm = (Ai @ np.r_[centre, 1.0])[:3]
            R_arm = Ai[:3, :3] @ R_obj
            if diag["sample_obj_arm"] is None:
                diag["sample_obj_arm"] = o_arm.round(3).tolist()
            opts = self.grasp_candidates(o_arm, R_arm, half)
            diag["grasp_options"] = max(diag["grasp_options"], len(opts))
            g = self.feasible_grasp(side, opts, robust=0.03)
            if g is None:
                continue
            diag["ik_ok"] += 1
            best = {"standoff": c, "grasp": g, "cost": geo + 0.3 * g["tilt"], "geo": geo}
            break                                         # nearest feasible standoff
        return best

    def _arm_frame_at(self, xyz_hab, yaw):
        """Arm frame A world pose if the robot root stood at xyz_hab with heading yaw.

        Pure algebra (A is rigid w.r.t. the root): never writes the root transform. A transient
        teleport-and-restore perturbed Bullet's contact state and left the root ~1e-4 m off, so the
        expert run stopped being a pure replay of its recorded actions (collection replay mismatches)."""
        T_cur = _np4(self.rob.sim_obj.transformation)
        T_at = _np4(mn.Matrix4.from_(
            mn.Quaternion.rotation(mn.Rad(yaw), mn.Vector3(0, 1, 0)).to_matrix(), mn.Vector3(*xyz_hab)))
        return T_at @ np.linalg.inv(T_cur) @ np.asarray(self.rob.arm_frame(), float)

    @staticmethod
    def _shifted(T, d):
        Ts = np.array(T, float).copy()
        Ts[0, 3] += d[0]
        Ts[1, 3] += d[1]
        return Ts

    def place_pose(self, side, point_arm, half_h, yaw):
        """gripper_base target (arm frame) that holds the object (as grasped) with its centre
        ``half_h + 0.03`` above ``point_arm``, the grasp rotated by ``yaw`` about the vertical."""
        R0 = self.grasp_used["R"]
        R = Rz(yaw) @ R0
        c_gb = self.obj_in_gb[:3, 3]                                # object centre in gripper_base
        obj_c = np.asarray(point_arm) + np.array([0.0, 0.0, half_h + 0.03])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = obj_c - R @ c_gb
        return T

    def plan_place(self, side, half_h):
        """Standoff + hand pose over a sampled point of a goal-receptacle surface."""
        ep = self.env.current_episode
        receps = self.env.sim.receptacles
        best, tried = None, 0
        for g in ep.candidate_goal_receps:
            if g.object_name not in receps:
                continue
            rec = receps[g.object_name]
            for _ in range(6):
                p = np.array(rec.sample_uniform_global(self.env.sim, 0.6))
                ranked = []
                for c in self.standoff_candidates(p, side, n=40):
                    ok, _, geo = NAV.plan(self.env, c["xyz"])
                    if ok and np.isfinite(geo):
                        ranked.append((geo, c))
                ranked.sort(key=lambda gc: gc[0])
                for geo, c in ranked[:6]:
                    tried += 1
                    A = self._arm_frame_at(c["xyz"], c["yaw"])
                    p_arm = (np.linalg.inv(A) @ np.r_[p, 1.0])[:3]
                    for yaw in (0.0, 0.5, -0.5, 1.0, -1.0):
                        T = self.place_pose(side, p_arm, half_h, yaw)
                        rg, okg = self.ik_ok(side, T)
                        if not okg:
                            continue
                        Ta = T.copy()
                        Ta[2, 3] += 0.10
                        rp, okp = self.ik_ok(side, Ta, seeds=(rg.q,))
                        if okp and not all(self.ik_ok(side, self._shifted(T, d), seeds=(rg.q,))[1] for d in
                                           ((PLACE_ROBUST_M, 0), (-PLACE_ROBUST_M, 0), (0, PLACE_ROBUST_M), (0, -PLACE_ROBUST_M))):
                            okp = False
                        if okp:
                            if best is None or geo < best["cost"]:
                                best = {"standoff": c, "yaw": yaw, "cost": geo, "point": p, "recep": g.object_name}
                            break
                if best is not None:
                    break
            if best is not None:
                break
        self.log["place_candidates_tried"] = tried
        return best

    # ------------------------------------------------------------------ episode
    def run(self):
        env, rob = self.env, self.rob
        self.ex = C.CartesianExecutor.from_proprio(rob.proprio16())
        obj, _ = self.target_object()
        self.set_stage("tuck")
        self.joint_move({s: np.array(NAV_TUCK[s]) for s in SIDES}, 15)
        # arm choice by geometry
        plans = {}
        for side in ([self.force_arm] if self.force_arm else SIDES):
            plans[side] = self.plan_pick(side, obj)
        feas = {s: p for s, p in plans.items() if p is not None}
        self.log["arm_choice"] = {s: (None if p is None else round(p["cost"], 3)) for s, p in plans.items()}
        if not feas:
            raise ExpertFailure("no feasible pick standoff for any arm (reach/aperture/navmesh)")
        side = min(feas, key=lambda s: feas[s]["cost"])
        pk = feas[side]
        self.side = side
        self.log["arm"] = side
        self.log["pick_plan"] = {"mode": pk["grasp"]["mode"], "tilt_deg": round(math.degrees(pk["grasp"]["tilt"]), 1),
                                 "geo_m": pk["geo"], "width_m": pk["grasp"]["width"]}
        # navigate to the standoff (arms tucked), face the planned heading
        self.set_stage("navigate_to_object")
        st = pk["standoff"]
        c = np.array([obj.translation[0], obj.translation[2]])
        nav1 = self.nav_to(st["xyz"], None)
        self._turn_to(st["yaw"])
        self.log["nav_to_object"] = {k: nav1.get(k) for k in ("arrived", "arrival_dist_m", "steps", "collision_steps")}
        # re-plan the grasp from the actual pose (arrival is within a few cm)
        centre, ext, R_obj, half = self.object_geometry(obj)
        Ai = np.linalg.inv(self.arm_T_world())
        g = self.feasible_grasp(side, self.grasp_candidates((Ai @ np.r_[centre, 1.0])[:3], Ai[:3, :3] @ R_obj, half))
        if g is None:
            raise ExpertFailure(f"grasp IK infeasible after arrival ({side})")
        rg, rp = type("Q", (), {"q": g["q_grasp"]}), type("Q", (), {"q": g["q_pre"]})
        self.grasp_used = g
        self.set_stage("reach")
        self.joint_move({side: rp.q}, 25, grips={"left": 0.0, "right": 0.0})
        self.set_stage("descend")
        self.joint_move({side: rg.q}, 12, grips={"left": 0.0, "right": 0.0})
        self.set_stage("grasp")
        self.hold(4, grips={side: 1.0})
        mgr = self.env.sim.get_agent_data(0).grasp_mgrs[SIDES.index(side)]
        if not mgr.is_grasped:
            raise ExpertFailure(f"grasp did not attach ({side})")
        self.events.append((self.t, f"grasp_{side}"))
        self.set_stage("lift")
        self.joint_move({side: rp.q}, 12, grips={side: 1.0, ("right" if side == "left" else "left"): 0.0})
        self.set_stage("carry")
        self.joint_move({s: np.array(NAV_TUCK[s]) for s in SIDES}, 20, grips={side: 1.0, ("right" if side == "left" else "left"): 0.0})
        # place: keep the object's pose in the hand; object bottom 3 cm over the surface
        rel = np.linalg.inv(self.rob.link_frame(f"{side}_gripper_base")) @ np.r_[
            np.c_[np.array(obj.transformation.rotation()), np.array(obj.translation)], [[0, 0, 0, 1]]]
        self.obj_in_gb = rel
        _, ext, _, _ = self.object_geometry(obj)
        pl = self.plan_place(side, ext[1])
        if pl is None:
            raise ExpertFailure("no feasible place standoff on any goal receptacle")
        self.log["place_plan"] = {"yaw": pl["yaw"], "recep": pl["recep"]}
        self.set_stage("navigate_to_receptacle")
        nav2 = self.nav_to(pl["standoff"]["xyz"], None)
        self._turn_to(pl["standoff"]["yaw"])
        self.log["nav_to_receptacle"] = {k: nav2.get(k) for k in ("arrived", "arrival_dist_m", "steps", "collision_steps")}
        # re-plan the place yaw from the actual pose (arrival is within a few cm / degrees)
        okg = okp = False
        for yaw in [pl["yaw"]] + [y for y in (0.0, 0.5, -0.5, 1.0, -1.0) if y != pl["yaw"]]:
            T = self.place_pose(side, self.to_arm(pl["point"]), ext[1], yaw)
            rg, okg = self.ik_ok(side, T)
            if not okg:
                continue
            Ta = T.copy()
            Ta[2, 3] += 0.10
            rp, okp = self.ik_ok(side, Ta, seeds=(rg.q,))
            if okp:
                self.log["place_plan"]["yaw_after_arrival"] = yaw
                break
        if not (okg and okp):
            raise ExpertFailure(f"place IK infeasible after arrival ({side})")
        g = {side: 1.0, ("right" if side == "left" else "left"): 0.0}
        self.set_stage("reach_place")
        self.joint_move({side: rp.q}, 25, grips=g)
        self.set_stage("lower")
        self.joint_move({side: rg.q}, 10, grips=g)
        self.set_stage("release")
        t_release = self.t
        self.hold(6, grips={"left": 0.0, "right": 0.0})
        self.events.append((self.t, f"release_{side}"))
        self.set_stage("retract")
        # Measured: tucking straight from the place pose swept the arm back through the just-placed object
        # (object at rest on the goal, then knocked at up to 4 rad/s during the tuck). Lift vertically,
        # back the base away with the arms still, and only then tuck.
        self.line(side, [0.0, 0.0, RETRACT_LIFT_M], 12, grip=0.0)
        for _ in range(RETRACT_BACK_STEPS):
            c = self.cart_now()
            c[16], c[17] = -RETRACT_BACK_V, 0.0
            self.step(c)
        self.joint_move({s: np.array(NAV_TUCK[s]) for s in SIDES}, 20, grips={"left": 0.0, "right": 0.0})
        self.set_stage("settle")   # post-release settle: simulator/OVMM success only, not SFT behaviour
        # OVMM place success needs the object on the goal, at rest and released for stability_steps (50)
        # consecutive steps. Released objects keep micro-jittering on some receptacles for seconds
        # (measured: stable after 72 / 240 steps). Stay still until the object has been at rest (OVMM's
        # 1e-3 linear / angular velocity thresholds, physical state only -- never the success metric) for
        # SETTLE_REST_STEPS consecutive steps, at most SETTLE_MAX_STEPS after release.
        self.hold(10)
        rest = 0
        while rest < SETTLE_REST_STEPS and self.t - t_release < SETTLE_MAX_STEPS:
            lv = float(np.linalg.norm(np.array(obj.linear_velocity)))
            av = float(np.linalg.norm(np.array(obj.angular_velocity)))
            rest = rest + 1 if (lv < 1e-3 and av < 1e-3) else 0
            self.hold(1)
        self.log["settle"] = {"steps_after_release": self.t - t_release, "rest_run": rest}
        self.set_stage("done")

    def _turn_to(self, yaw_goal):
        for _ in range(150):
            _, yaw, _ = NAV.pose2d(self.env)
            err = NAV.wrap(yaw_goal - yaw)
            if abs(err) < 0.02:
                return
            c = self.cart_now()
            c[16], c[17] = 0.0, float(np.clip(1.5 * err, -0.8, 0.8))
            self.step(c)
