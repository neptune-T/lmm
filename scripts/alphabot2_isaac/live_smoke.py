"""AlphaBot 2 embodiment smoke on the public Ai2 Bot2 asset, live in Isaac Sim.

Run under Isaac Sim 4.5's ``python.sh`` (headless). The robot is driven ONLY
through the repository adapter and backend (``lmm.embodiments.alphabot2``),
so what passes here is the committed code.

Environment:
  LMM_REPO     repository root (default: two levels above this file)
  ROBOT_USDS   fiveages-sim/robot_usds checkout
  BOT2_XACRO   .../ai2_bot2_description/xacro/components
  SMOKE_OUT    output directory (json + png)

Phase 1 (empty scene): census, home, 16-D state, per-joint calibration of all
14 arm joints against xacro FK, gripper calibration, atomic 18-D action, base
calibration, reset reproducibility, timing.
Phase 2 (objects added, world reset): cameras, grasp + snapshot/restore under
contact, bounded navigation past off-path obstacles.
"""
import json
import math
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("LMM_REPO", os.path.dirname(os.path.dirname(HERE)))
USDS = os.environ.get("ROBOT_USDS", "/nix/yanjun/zts/alphabot2/deps/robot_usds")
XACRO = os.environ.get("BOT2_XACRO", "/nix/yanjun/zts/alphabot2/deps/robot_descriptions/humanoid/"
                       "Ai2Robotics/ai2_bot2_description/xacro/components")
OUT = os.environ.get("SMOKE_OUT", "/nix/yanjun/zts/alphabot2/smoke/live")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from lmm.embodiments.alphabot2 import isaac_adapter as IA  # noqa: E402  (numpy-only)
from lmm.embodiments.alphabot2 import isaac_backend as IB  # noqa: E402  (Isaac imported lazily)

sys.argv.append(IB.RTX_DRIVER_CHECK_BYPASS)
O = {"sections": {}}


def sec(name):
    def deco(fn):
        t = time.time()
        try:
            O["sections"][name] = {"ok": True, **(fn() or {})}
        except Exception as e:  # keep going: one failed section must not hide the rest
            O["sections"][name] = {"ok": False, "error": repr(e), "tb": traceback.format_exc()[-1500:]}
        O["sections"][name]["wall_s"] = round(time.time() - t, 2)
        print(f"[smoke] {name}: ok={O['sections'][name]['ok']} ({O['sections'][name]['wall_s']} s)", flush=True)
        return fn
    return deco


t_boot = time.time()
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})
O["boot_s"] = round(time.time() - t_boot, 1)

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.api.materials import PhysicsMaterial  # noqa: E402
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid  # noqa: E402
from isaacsim.core.prims import SingleArticulation  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

from lmm.embodiments.alphabot2.contract import AlphaBot2Contract  # noqa: E402
from xacro_fk import XacroArms, pose_error, rotvec  # noqa: E402

XF = XacroArms(XACRO)
DT = 1.0 / IA.PHYSICS_HZ
world = World(stage_units_in_meters=1.0, physics_dt=DT, rendering_dt=DT)
stage = omni.usd.get_context().get_stage()
O["build"] = IB.build_stage(app, stage, USDS)
world.scene.add_default_ground_plane()
O["contact_report_bodies_phase1"] = IB.enable_contact_reports(stage)
art = SingleArticulation(prim_path=IB.ROBOT_PRIM, name="bot2")
world.scene.add(art)
world.reset()

B = IB.IsaacArticulationBackend(world, art)
A = IA.AlphaBot2IsaacAdapter(AlphaBot2Contract(), backend=B)
SIDE_ARM = {"left": IA.LEFT_ARM_DOFS, "right": IA.RIGHT_ARM_DOFS}
SIDE_PREFIX = {"left": "Left", "right": "Right"}
HOME = {"left": np.array(IA.HOME_LEFT_ARM), "right": np.array(IA.HOME_RIGHT_ARM)}
ARM_SLICE = {"left": slice(0, 7), "right": slice(8, 15)}


def home_targets(grip_frac=0.0):
    m = {}
    for s in ("left", "right"):
        for d, v in zip(SIDE_ARM[s], A.arm_vendor_to_sim(s, HOME[s])):
            m[d] = float(v)
    m[IA.LEFT_GRIPPER_DOF] = m[IA.RIGHT_GRIPPER_DOF] = A.gripper_fraction_to_joint(grip_frac)
    for d in (*IA.BODY_DOFS, *IA.HEAD_DOFS):
        m[d] = 0.0
    return m


def act(q_left=None, q_right=None, g=(0.0, 0.0), v=0.0, w=0.0):
    a = A.home_action()
    if q_left is not None:
        a[0:7] = q_left
    if q_right is not None:
        a[8:15] = q_right
    a[7], a[15], a[16], a[17] = g[0], g[1], v, w
    return a


def run(action, n):
    for _ in range(n):
        A.apply_action(action)


def gb_pose(side):
    """gripper_base pose (4x4) in the chassis frame, from PhysX."""
    p, r = B.link_pose(f"Gripper_{SIDE_PREFIX[side]}/gripper_base")
    pc, rc = B.in_chassis(p, r)
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = rc, pc
    return m


def pads(side):
    S = SIDE_PREFIX[side]
    a, _ = B.link_pose(f"Gripper_{S}/left_pad")
    b, _ = B.link_pose(f"Gripper_{S}/right_pad")
    return a, b


def pad_sep(side):
    a, b = pads(side)
    return float(np.linalg.norm(a - b))


def robot_self_pairs(pairs):
    pre = IB.ROBOT_PRIM + "/"
    return sorted(k for k in pairs if all(x.startswith(pre) for x in k.split(" <-> ")))


def r4(x):
    return [round(float(v), 4) for v in np.asarray(x).reshape(-1)]


Q_HOME = home_targets()
full = np.array([Q_HOME.get(p, 0.0) for p in B.dof_paths])
art.set_joints_default_state(positions=full)
B.teleport(Q_HOME)
B.step(200)


# ============================== PHASE 1 ==============================

@sec("census")
def _():
    pv = B._pv()
    lim = B.dof_limits()
    masses = B._np(pv.get_masses()).reshape(-1)
    idx = world.current_time_step_index
    world.step(render=False)
    per_step_nr = world.current_time_step_index - idx
    idx = world.current_time_step_index
    world.step(render=True)
    per_step_r = world.current_time_step_index - idx
    return {
        "num_dof": len(B.dof_paths), "expected_dof": IA.EXPECTED_DOF_COUNT,
        "num_links": len(B.link_paths), "root_link": B.link_paths[0],
        "chassis_link": B.link_paths[B.chassis],
        "fixed_base": bool(pv.shared_metatype.fixed_base),
        "total_mass_kg": round(float(masses.sum()), 3),
        "physics_dt_s": world.get_physics_dt(), "rendering_dt_s": world.get_rendering_dt(),
        "physics_steps_per_world_step": {"render_false": int(per_step_nr), "render_true": int(per_step_r)},
        "gripper_dofs": [p for p in B.dof_paths if p.startswith("Gripper_")],
        "resolved_contract_dofs": {k: v for k, v in B.resolved.items() if k != v},
        "dof_paths": B.dof_paths,
        "right_arm_limits_live": {d: r4(lim[B.index[d]]) for d in IA.RIGHT_ARM_DOFS},
        "left_arm_limits_live": {d: r4(lim[B.index[d]]) for d in IA.LEFT_ARM_DOFS},
        "limits_match_mapped_vendor": bool(all(
            np.allclose(lim[B.index[d]], want, atol=2e-3)
            for s in ("left", "right") for d, want in zip(SIDE_ARM[s], IA.sim_arm_limits(s)))),
        "colliders_per_body": IB.collider_owners(stage),
    }


@sec("home_and_state16")
def _():
    run(A.home_action(), 10)
    s = A.observe_proprio()
    gl, gr = gb_pose("left"), gb_pose("right")
    fk_err = {side: pose_error(gb_pose(side), XF.fk(side, s[ARM_SLICE[side]])) for side in ("left", "right")}
    world.step(render=False)
    pairs = IB.contact_pairs()
    return {
        "state16": r4(s), "shape": list(s.shape),
        "left_arm_err_vs_vendor_home": round(float(np.max(np.abs(s[0:7] - HOME["left"]))), 5),
        "right_arm_err_vs_vendor_home": round(float(np.max(np.abs(s[8:15] - HOME["right"]))), 5),
        "grippers_fraction": [round(float(s[7]), 4), round(float(s[15]), 4)],
        "gb_left_chassis": r4(gl[:3, 3]), "gb_right_chassis": r4(gr[:3, 3]),
        "approach_left": r4(gl[:3, 2]), "approach_right": r4(gr[:3, 2]),
        "mirror_residual_m": round(float(np.linalg.norm(gl[:3, 3] * [1, -1, 1] - gr[:3, 3])), 4),
        "fk_vs_isaac_home": {k: [round(v[0], 4), round(v[1], 3)] for k, v in fk_err.items()},
        "self_contact_pairs": robot_self_pairs(pairs),
        "ground_contact_bodies": sorted({x for k in pairs for x in k.split(" <-> ")
                                         if x.startswith(IB.ROBOT_PRIM) and "GroundPlane" in k}),
    }


CAL = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "convention": "vendor (xacro) joint convention",
       "delta_rad": 0.15, "rows": []}


@sec("arm_joint_calibration")
def _():
    delta = CAL["delta_rad"]
    for side in ("left", "right"):
        other = "right" if side == "left" else "left"
        amap = IA.LEFT_ARM_MAP if side == "left" else IA.RIGHT_ARM_MAP
        for k in range(7):
            for sgn in (+1, -1):
                run(A.home_action(), 6)
                s0 = A.observe_proprio()
                m0, mo0 = gb_pose(side), gb_pose(other)
                q = HOME[side].copy()
                q[k] += sgn * delta
                run(act(**{f"q_{side}": q}), 10)
                s1 = A.observe_proprio()
                m1, mo1 = gb_pose(side), gb_pose(other)
                ds = s1 - s0
                sl = ARM_SLICE[side]
                achieved = float(ds[sl][k])
                others = np.abs(np.delete(ds[0:15], [sl.start + k, 7]))
                rv_isaac = rotvec(m1[:3, :3] @ m0[:3, :3].T)
                f0, f1 = XF.fk(side, s0[sl]), XF.fk(side, s1[sl])
                rv_fk = rotvec(f1[:3, :3] @ f0[:3, :3].T)
                dp_isaac, dp_fk = m1[:3, 3] - m0[:3, 3], f1[:3, 3] - f0[:3, 3]
                cos = float(rv_isaac @ rv_fk / (np.linalg.norm(rv_isaac) * np.linalg.norm(rv_fk) + 1e-12))
                row = {
                    "side": side, "joint": k + 1, "policy_index": sl.start + k,
                    "sim_dof": SIDE_ARM[side][k], "sim_sign": amap.sign[k], "sim_offset_rad": round(amap.offset[k], 4),
                    "cmd_delta": sgn * delta, "achieved_delta": round(achieved, 5),
                    "track_err": round(abs(achieved - sgn * delta), 5),
                    "crosstalk_max_rad": round(float(others.max()), 5),
                    "gb_rotation_isaac_rad": round(float(np.linalg.norm(rv_isaac)), 5),
                    "gb_rotation_fk_rad": round(float(np.linalg.norm(rv_fk)), 5),
                    "rotation_axis_cos_isaac_vs_fk": round(cos, 5),
                    "gb_disp_err_vs_fk_m": round(float(np.linalg.norm(dp_isaac - dp_fk)), 5),
                    "other_arm_gb_disp_m": round(float(np.linalg.norm(mo1[:3, 3] - mo0[:3, 3])), 6),
                }
                row["pass"] = bool(row["track_err"] < 0.01 and row["crosstalk_max_rad"] < 0.01
                                   and cos > 0.999 and row["gb_disp_err_vs_fk_m"] < 0.003
                                   and row["other_arm_gb_disp_m"] < 0.002)
                CAL["rows"].append(row)
    run(A.home_action(), 10)
    rows = CAL["rows"]
    CAL["summary"] = {
        "n": len(rows), "n_pass": sum(r["pass"] for r in rows),
        "max_track_err": max(r["track_err"] for r in rows),
        "max_crosstalk": max(r["crosstalk_max_rad"] for r in rows),
        "min_axis_cos": min(r["rotation_axis_cos_isaac_vs_fk"] for r in rows),
        "max_disp_err_vs_fk_m": max(r["gb_disp_err_vs_fk_m"] for r in rows),
        "max_other_arm_disp_m": max(r["other_arm_gb_disp_m"] for r in rows),
    }
    with open(f"{OUT}/joint_calibration.json", "w") as f:
        json.dump(CAL, f, indent=1)
    return CAL["summary"]


@sec("gripper_calibration")
def _():
    rows = []
    for g in (0.0, 0.25, 0.5, 0.75, 1.0):
        run(act(g=(g, g)), 10)
        s = A.observe_proprio()
        qj = B.get_joint_positions()
        rows.append({"cmd": g, "readback_left": round(float(s[7]), 4), "readback_right": round(float(s[15]), 4),
                     "joint_left": round(qj[IA.LEFT_GRIPPER_DOF], 4), "joint_right": round(qj[IA.RIGHT_GRIPPER_DOF], 4),
                     "pad_sep_left_m": round(pad_sep("left"), 5), "pad_sep_right_m": round(pad_sep("right"), 5),
                     "mimic_left": r4([qj[p] for p in B.dof_paths if p.startswith("Gripper_Left/") and p != IA.LEFT_GRIPPER_DOF]),
                     "arm_err_left": round(float(np.max(np.abs(s[0:7] - HOME["left"]))), 5)})
    run(A.home_action(), 10)
    sl = [r["pad_sep_left_m"] for r in rows]
    sr = [r["pad_sep_right_m"] for r in rows]
    return {"rows": rows,
            "monotonic_closing": bool(all(a > b for a, b in zip(sl, sl[1:])) and all(a > b for a, b in zip(sr, sr[1:]))),
            "policy_0_open_1_closed": bool(sl[0] > sl[-1] and sr[0] > sr[-1]),
            "left_right_sep_max_diff_m": round(max(abs(a - b) for a, b in zip(sl, sr)), 5),
            "max_readback_err": round(max(max(abs(r["readback_left"] - r["cmd"]), abs(r["readback_right"] - r["cmd"])) for r in rows), 4)}


@sec("atomic_18d_action")
def _():
    run(A.home_action(), 6)
    s0, (p0, _, y0) = A.observe_proprio(), B.chassis_pose()
    ql, qr = HOME["left"].copy(), HOME["right"].copy()
    ql[0] += 0.10
    qr[0] -= 0.10
    a = act(ql, qr, g=(0.5, 0.25), v=0.15, w=0.3)
    i0, t0, b0 = world.current_time_step_index, world.current_time, B.steps_taken
    info = A.apply_action(a)
    i1, t1, b1 = world.current_time_step_index, world.current_time, B.steps_taken
    s1, (p1, _, y1) = A.observe_proprio(), B.chassis_pose()
    moved = {"left_joint1": float(s1[0] - s0[0]), "right_joint1": float(s1[8] - s0[8]),
             "left_gripper": float(s1[7] - s0[7]), "right_gripper": float(s1[15] - s0[15]),
             "base_disp_m": float(np.linalg.norm(p1 - p0)), "base_dyaw_rad": float(y1 - y0)}
    run(A.home_action(), 20)
    return {"world_steps_consumed": int(i1 - i0), "backend_steps_consumed": int(b1 - b0),
            "adapter_reported": [info["sim_steps_before"], info["sim_steps_after"], info["physics_steps_consumed"]],
            "sim_time_advanced_s": round(float(t1 - t0), 6), "policy_hz": IA.POLICY_HZ,
            "channels": {k: round(v, 6) for k, v in moved.items()},
            "every_channel_moved": bool(all(abs(v) > 1e-5 for v in moved.values())),
            "joint_target_keys": sorted(info["joint_targets"])}


@sec("base_calibration")
def _():
    res = {}
    for name, v, w in (("forward", 0.2, 0.0), ("reverse", -0.2, 0.0), ("yaw_ccw", 0.0, 0.5), ("yaw_cw", 0.0, -0.5)):
        run(A.home_action(), 5)
        p0, r0, y0 = B.chassis_pose()
        s_before = A.observe_proprio()
        n = 20                                  # 2.0 s at 10 Hz
        run(act(v=v, w=w), n)
        run(A.home_action(), 5)                 # stop, settle 0.5 s
        p1, r1, y1 = B.chassis_pose()
        dp = p1 - p0
        dyaw = (y1 - y0 + math.pi) % (2 * math.pi) - math.pi
        T = n / IA.POLICY_HZ
        res[name] = {"cmd_v": v, "cmd_w": w, "duration_s": T,
                     "wheel_radps": r4(IA.twist_to_wheel_radps(v, w)),
                     "along_chassis_x_m": round(float(dp @ r0[:, 0]), 4),
                     "along_chassis_y_m": round(float(dp @ r0[:, 1]), 4),
                     "dz_m": round(float(dp[2]), 4), "dyaw_rad": round(float(dyaw), 4),
                     "expected_x_m": round(v * T, 4), "expected_dyaw_rad": round(w * T, 4),
                     "arm_drift_rad": round(float(np.max(np.abs(A.observe_proprio()[0:15] - s_before[0:15]))), 5)}
    f, r, c, cw = res["forward"], res["reverse"], res["yaw_ccw"], res["yaw_cw"]
    res["forward_ratio"] = round(f["along_chassis_x_m"] / f["expected_x_m"], 3)
    res["reverse_ratio"] = round(r["along_chassis_x_m"] / r["expected_x_m"], 3)
    res["yaw_ccw_ratio"] = round(c["dyaw_rad"] / c["expected_dyaw_rad"], 3)
    res["yaw_cw_ratio"] = round(cw["dyaw_rad"] / cw["expected_dyaw_rad"], 3)
    res["signs_ok"] = bool(f["along_chassis_x_m"] > 0 and r["along_chassis_x_m"] < 0
                           and c["dyaw_rad"] > 0 and cw["dyaw_rad"] < 0)
    res["convention"] = "+v along chassis +x; +omega counter-clockwise about world +z"
    return res


@sec("timing")
def _():
    run(A.home_action(), 2)
    t = time.perf_counter()
    run(A.home_action(), 20)
    wall = time.perf_counter() - t
    return {"physics_hz": IA.PHYSICS_HZ, "physics_steps_per_action": IA.PHYSICS_STEPS_PER_ACTION,
            "policy_hz": IA.POLICY_HZ, "wall_s_per_action_no_render": round(wall / 20, 4),
            "realtime_factor_no_render": round((20 / IA.POLICY_HZ) / wall, 3),
            "asset_declared_physics_hz": 200, "asset_declared_source": "humanoid/Ai2/Bot2/env/empty.usda",
            "declared_ros2_control_hz": IA.DECLARED_CONTROLLER_HZ}


@sec("reset_reproducibility")
def _():
    out = []
    for _ in range(2):
        world.reset()
        B.refresh()
        run(A.home_action(), 10)
        out.append((A.observe_proprio(), B.chassis_pose()[0]))
    return {"state_diff_max": float(np.max(np.abs(out[0][0] - out[1][0]))),
            "chassis_diff_m": float(np.linalg.norm(out[0][1] - out[1][1])),
            "home_err_max_rad": float(np.max(np.abs(out[0][0][0:15] - A.home_action()[0:15])))}


# ---------------- grasp planning (xacro IK, rehearsed live before any object exists) -------------
# A top-down pick with the left hand. Orientation: the home orientation pitched
# 90 deg down about the chassis y axis, so the approach points at the floor and
# the fingers (which separate vertically at home) separate horizontally.

def plan_topdown():
    run(A.home_action(), 10)
    gb = gb_pose("left")
    mid = B.in_chassis(sum(pads("left")) / 2)
    reach = float((mid - gb[:3, 3]) @ gb[:3, 2])            # gripper_base -> pad midpoint
    m0 = XF.fk("left", HOME["left"])
    rg = np.array([[0.0, 0, 1], [0, 1, 0], [-1, 0, 0]]) @ m0[:3, :3]
    for x, z in ((0.45, 0.70), (0.40, 0.70), (0.45, 0.75), (0.50, 0.75), (0.40, 0.75)):
        goal = np.eye(4)
        goal[:3, :3], goal[:3, 3] = rg, (x, 0.25, z + reach)
        pre = goal.copy()
        pre[2, 3] += 0.08
        q_pre = XF.ik_pose("left", HOME["left"], pre, margin=0.05)
        if max(pose_error(XF.fk("left", q_pre), pre)[0] * 1000, pose_error(XF.fk("left", q_pre), pre)[1]) > 0.1:
            continue
        path = [q_pre]
        for _ in range(8):                                  # straight 8 cm descent, 1 cm per waypoint
            path.append(XF.ik_translate("left", path[-1], (0.0, 0.0, -0.01), margin=0.02))
        e = pose_error(XF.fk("left", path[-1]), goal)
        if e[0] < 1e-3 and e[1] < 0.1:
            return {"cube_center_chassis": (x, 0.25, z), "reach_m": reach, "path": path,
                    "goal_err": e, "pre_err": pose_error(XF.fk("left", q_pre), pre)}
    return None


PLAN = plan_topdown()
O["grasp_plan"] = None if PLAN is None else {
    "cube_center_chassis": PLAN["cube_center_chassis"], "gripper_base_to_pad_mid_m": round(PLAN["reach_m"], 4),
    "pre_grasp_vendor_q": r4(PLAN["path"][0]), "grasp_vendor_q": r4(PLAN["path"][-1]),
    "goal_err_m_deg": r4(PLAN["goal_err"])}


@sec("grasp_rehearsal_no_object")
def _():
    if PLAN is None:
        raise RuntimeError("no feasible top-down grasp inside the vendor limits")
    path = PLAN["path"]
    q0 = HOME["left"]
    contacts = set()
    for i in range(30):                                     # home -> pre-grasp, joint space
        A.apply_action(act(q_left=q0 + (path[0] - q0) * (i + 1) / 30))
        contacts.update(robot_self_pairs(IB.contact_pairs()))
    for q in path[1:]:
        run(act(q_left=q), 2)
        contacts.update(robot_self_pairs(IB.contact_pairs()))
    run(act(q_left=path[-1]), 5)
    s = A.observe_proprio()
    a_, b_ = pads("left")
    mid_w = (a_ + b_) / 2
    sep = (a_ - b_) / np.linalg.norm(a_ - b_)
    gb = gb_pose("left")
    out = {"self_contacts_during_motion": sorted(contacts),
           "track_err_rad": round(float(np.max(np.abs(s[0:7] - path[-1]))), 5),
           "pad_mid_world": r4(mid_w), "pad_mid_chassis": r4(B.in_chassis(mid_w)),
           "pad_separation_axis_world": r4(sep), "approach_chassis": r4(gb[:3, 2]),
           "fk_vs_isaac_m_deg": r4(pose_error(gb, XF.fk("left", s[0:7])))}
    for q in path[::-1]:
        run(act(q_left=q), 2)
    for i in range(30):
        A.apply_action(act(q_left=path[0] + (q0 - path[0]) * (i + 1) / 30))
    run(A.home_action(), 10)
    return out


REH = O["sections"]["grasp_rehearsal_no_object"]
GRASP_OK = bool(REH.get("ok") and not REH["self_contacts_during_motion"]
                and REH["pad_mid_chassis"][0] > 0.35 and abs(REH["pad_separation_axis_world"][2]) < 0.2)
O["grasp_objects_placed"] = GRASP_OK
run(A.home_action(), 10)
PAD_R = sum(pads("right")) / 2
PC0, RC0, _ = B.chassis_pose()

# ============================== PHASE 2 ==============================
from pxr import UsdLux  # noqa: E402

sun = UsdLux.DistantLight.Define(stage, "/World/Sun")     # analytic light: no sampling noise
sun.CreateIntensityAttr(3000.0)
sun.CreateAngleAttr(1.0)
UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set((-35.0, 20.0, 0.0))
CUBE = 0.05
fwd, left = RC0[:, 0] * np.array([1, 1, 0]), RC0[:, 1] * np.array([1, 1, 0])


def at(d_fwd, d_left, z):
    return np.array([PC0[0], PC0[1], 0.0]) + d_fwd * fwd + d_left * left + np.array([0, 0, z])


cube = None
if GRASP_OK:
    mat = PhysicsMaterial(prim_path="/World/mat_grip", static_friction=1.2, dynamic_friction=1.0, restitution=0.0)
    c = np.array(REH["pad_mid_world"])
    top = c[2] - CUBE / 2 - 0.002
    FixedCuboid(prim_path="/World/pedestal_l", name="pedestal_l", position=np.array([c[0], c[1], top / 2]),
                scale=np.array([0.03, 0.03, top]), color=np.array([0.5, 0.5, 0.5]))
    cube = DynamicCuboid(prim_path="/World/cube", name="cube", position=c, scale=np.array([CUBE] * 3), mass=0.05,
                         color=np.array([1.0, 0.55, 0.0]), physics_material=mat)
    world.scene.add(cube)
ahead_r = PAD_R + 0.22 * fwd
top_r = PAD_R[2] - 0.03
FixedCuboid(prim_path="/World/pedestal_r", name="pedestal_r", position=np.array([ahead_r[0], ahead_r[1], top_r / 2]),
            scale=np.array([0.05, 0.05, top_r]), color=np.array([0.5, 0.5, 0.5]))
FixedCuboid(prim_path="/World/target_r", name="target_r", position=np.array([ahead_r[0], ahead_r[1], top_r + 0.03]),
            scale=np.array([0.06] * 3), color=np.array([1.0, 1.0, 0.0]))
FixedCuboid(prim_path="/World/mk_red", name="mk_red", position=at(1.6, 0.0, 0.1), scale=np.array([0.2] * 3),
            color=np.array([1.0, 0, 0]))
FixedCuboid(prim_path="/World/mk_green", name="mk_green", position=at(1.2, 0.5, 0.1), scale=np.array([0.16] * 3),
            color=np.array([0, 1.0, 0]))
FixedCuboid(prim_path="/World/mk_blue", name="mk_blue", position=at(2.2, -0.5, 0.1), scale=np.array([0.2] * 3),
            color=np.array([0, 0, 1.0]))
# route: reverse 0.6 m, turn +90 deg, forward 1.0 m, turn -90 deg, forward 0.5 m. The swept
# region (arms tucked, ~0.3 m half-width, ~0.45 m reach) stays >= 0.3 m from every obstacle.
OBST = {"obst_a": at(-1.5, 0.5, 0.25), "obst_b": at(0.9, 1.0, 0.25), "obst_c": at(0.1, 1.8, 0.25),
        "obst_d": at(-0.3, -0.75, 0.25)}
for k, p in OBST.items():
    FixedCuboid(prim_path=f"/World/{k}", name=k, position=p, scale=np.array([0.3, 0.3, 0.5]),
                color=np.array([0.3, 0.3, 0.8]))
O["contact_report_bodies_phase2"] = IB.enable_contact_reports(stage)


def default_state(q_left):
    full = np.array([Q_HOME.get(p, 0.0) for p in B.dof_paths])
    for d, v in zip(IA.LEFT_ARM_DOFS, A.arm_vendor_to_sim("left", q_left)):
        full[B.index[d]] = v
    art.set_joints_default_state(positions=full)


Q_PRE = PLAN["path"][0] if GRASP_OK else HOME["left"]
default_state(Q_PRE)                  # phase 2 resets straight into the pre-grasp pose
world.reset()
B.refresh()
if cube is not None:
    B.register_object("cube", cube)
run(act(q_left=Q_PRE), 10)
O["phase2_start"] = {"chassis_tilt_deg": round(math.degrees(math.acos(min(1.0, B.chassis_pose()[1][2, 2]))), 3),
                     "self_contacts": robot_self_pairs(IB.contact_pairs()),
                     "cube_world": r4(cube.get_world_pose()[0]) if cube is not None else None}


@sec("cameras")
def _():
    H, W = 240, 320
    xc = UsdGeom.XformCache()
    head_frame = stage.GetPrimAtPath(f"{IB.ROBOT_PRIM}/head_link2/head_camera")
    hl = np.array(xc.GetLocalToWorldTransform(head_frame) * xc.GetLocalToWorldTransform(
        stage.GetPrimAtPath(f"{IB.ROBOT_PRIM}/head_link2")).GetInverse()).T
    ph, rh = B.link_pose("head_link2")
    _, rc, _ = B.chassis_pose()
    head_axes_chassis = rc.T @ rh @ hl[:3, :3]            # columns: head_camera x, y, z in the chassis frame
    # head_camera is an optical frame (+z optical axis, +x image right, +y image down): its +x
    # points to the robot's right. Checked against the xacro cam_high pitch below.
    B.add_camera("cam_front", f"{IB.ROBOT_PRIM}/head_link2/head_camera", (0, 0, 0),
                 IB.camera_rows((0, 0, 1), (0, -1, 0)), (W, H), hfov_deg=69.0, near_m=0.10)
    head_fwd_chassis = head_axes_chassis[:, 2]
    wrist = {}
    for side in ("left", "right"):
        S = SIDE_PREFIX[side]
        d405 = [p for p in Usd.PrimRange(stage.GetPrimAtPath(f"{IB.ROBOT_PRIM}/Arm_{S}"), Usd.TraverseInstanceProxies())
                if p.GetName() == "d405"]
        link7 = f"Arm_{S}/link7"
        p7, r7 = B.link_pose(link7)
        rel = np.array(xc.GetLocalToWorldTransform(d405[0]) * xc.GetLocalToWorldTransform(
            stage.GetPrimAtPath(f"{IB.ROBOT_PRIM}/{link7}")).GetInverse()).T
        cam_w = p7 + r7 @ rel[:3, 3]
        pg, rg = B.link_pose(f"Gripper_{S}/gripper_base")
        mid = sum(pads(side)) / 2
        target = mid + 0.10 * rg[:, 2]
        fwd_w = target - cam_w
        up_w = cam_w - (pg + rg[:, 2] * ((cam_w - pg) @ rg[:, 2]))    # away from the gripper axis
        B.add_camera(f"cam_{side}_wrist", f"{IB.ROBOT_PRIM}/{link7}", r7.T @ (cam_w - p7),
                     IB.camera_rows(r7.T @ fwd_w, r7.T @ up_w), (W, H), hfov_deg=87.0, near_m=0.02)
        wrist[side] = {"d405_prim": str(d405[0].GetPath()),
                       "cam_offset_from_gripper_axis_m": round(float(np.linalg.norm(up_w)), 4),
                       "cam_to_pad_mid_m": round(float(np.linalg.norm(mid - cam_w)), 4)}
    for _ in range(40):                           # RTX warm-up while holding the pose
        world.step(render=True)
    B.render_on_step = True
    t = time.perf_counter()
    run(act(q_left=Q_PRE), 4)
    wall = (time.perf_counter() - t) / 4
    from PIL import Image
    res = {"head_camera_frame_axes_chassis": {"x": r4(head_axes_chassis[:, 0]), "y": r4(head_axes_chassis[:, 1]),
                                              "z": r4(head_axes_chassis[:, 2])},
           "head_camera_optical_axis_chassis": r4(head_fwd_chassis),
           "head_camera_pitch_down_deg": round(math.degrees(math.asin(-head_fwd_chassis[2])), 2),
           "xacro_cam_high_pitch_down_deg": round(math.degrees(0.6458), 2),
           "wrist_geometry": wrist, "wall_s_per_action_with_3_cameras": round(wall, 3)}
    frames = {}
    for k, cam in B.cameras.items():
        rgb = B.render(k)
        Image.fromarray(rgb).save(f"{OUT}/{k}.png")
        fr = cam.get_current_frame()
        r_, g_, b_ = [rgb[..., i].astype(int) for i in range(3)]
        masks = {"red": (r_ > 150) & (g_ < 80) & (b_ < 80), "green": (g_ > 150) & (r_ < 80) & (b_ < 80),
                 "blue": (b_ > 150) & (r_ < 80) & (g_ < 80),
                 "orange": (r_ > 170) & (g_ > 70) & (g_ < 170) & (b_ < 60),
                 "yellow": (r_ > 170) & (g_ > 170) & (b_ < 80)}
        cen = {}
        for col, m in masks.items():
            ys, xs = np.nonzero(m)
            cen[col] = None if len(xs) < 15 else {"x": round(float(xs.mean() / W), 3),
                                                  "y": round(float(ys.mean() / H), 3), "px": int(len(xs))}
        frames[k] = {"shape": list(rgb.shape), "dtype": str(rgb.dtype), "mean": round(float(rgb.mean()), 1),
                     "black_frac": round(float((rgb.max(axis=2) < 8).mean()), 3), "centroids": cen,
                     "rendering_frame": fr.get("rendering_frame"), "rendering_time": fr.get("rendering_time")}
    res["frames"] = frames
    f = frames["cam_front"]["centroids"]
    if f["green"] and f["blue"] and f["red"]:
        res["front_left_right_ok"] = bool(f["green"]["x"] < f["red"]["x"] < f["blue"]["x"])
        res["front_upright_ok"] = bool(f["green"]["y"] > f["blue"]["y"])   # nearer marker lower in the image
    res["front_red_centered"] = bool(f["red"] and abs(f["red"]["x"] - 0.5) < 0.1)
    res["left_wrist_sees_cube"] = frames["cam_left_wrist"]["centroids"]["orange"] is not None
    res["right_wrist_sees_target"] = frames["cam_right_wrist"]["centroids"]["yellow"] is not None
    res["synchronized_rendering_frame"] = len({v["rendering_frame"] for v in frames.values()}) == 1
    B.render_on_step = False
    return res


@sec("grasp_snapshot_restore")
def _():
    if not GRASP_OK:
        raise RuntimeError("grasp objects were not placed (rehearsal failed)")
    path = PLAN["path"]
    run(act(q_left=path[0]), 5)
    c0 = B._np(cube.get_world_pose()[0]).copy()
    for q in path[1:]:
        run(act(q_left=q), 2)                    # descend around the cube
    run(act(q_left=path[-1]), 3)
    run(act(q_left=path[-1], g=(1.0, 0.0)), 15)  # close
    sep_closed = pad_sep("left")
    pad_cube = sorted(k for k in IB.contact_pairs() if "/World/cube" in k and "Gripper_Left" in k)
    lift = [act(q_left=q, g=(1.0, 0.0)) for q in path[::-1] for _ in range(2)]
    for a in lift[:8]:                           # half the lift, then snapshot while holding the cube
        A.apply_action(a)
    snap = B.snapshot()
    rest = lift[8:] + [act(q_left=path[0], g=(1.0, 0.0), v=-0.15, w=0.2)] * 10 \
        + [act(q_left=path[0], g=(1.0, 0.0))] * 5

    def roll():
        traj = []
        for a in rest:
            A.apply_action(a)
            traj.append((B._np(art.get_joint_positions()).copy(), B.chassis_pose()[0].copy(),
                         B._np(cube.get_world_pose()[0]).copy()))
        return traj

    ctrl = [B.index[d] for d in IB.CONTROLLED_DOFS[:16]]
    wheels = [B.index[d] for d in IA.WHEEL_DOFS]
    mimic = [i for i, p in enumerate(B.dof_paths) if p.startswith("Gripper_") and i not in ctrl]
    caster = [i for i, p in enumerate(B.dof_paths) if p.startswith("Ai2_Bot2_Chassis/") and i not in wheels]
    groups = {"arm_and_gripper_actuated": ctrl, "gripper_mimic": mimic, "caster_passive": caster, "wheels": wheels}

    def diff(ta, tb):
        out = {f"{g}_max_rad": float(max(np.max(np.abs(a[0][ix] - b[0][ix])) for a, b in zip(ta, tb)))
               for g, ix in groups.items()}
        out["chassis_max_m"] = float(max(np.linalg.norm(a[1] - b[1]) for a, b in zip(ta, tb)))
        out["cube_max_m"] = float(max(np.linalg.norm(a[2] - b[2]) for a, b in zip(ta, tb)))
        return out

    orig = roll()
    held = bool(any("/World/cube" in k and "Gripper_Left" in k for k in IB.contact_pairs()))
    B.restore(snap)
    chk = B.snapshot()
    r1 = roll()
    B.restore(snap)
    r2 = roll()

    def canonical_restore():
        # world.reset() rebuilds the PhysX scene, discarding solver warm-start and contact
        # caches, so every branch -- including the first -- continues from the same solver state
        world.reset()
        B.refresh()
        B.restore(snap)

    t = time.perf_counter()
    canonical_restore()
    canon_s = time.perf_counter() - t
    k0 = roll()
    canonical_restore()
    k1 = roll()

    return {
        "cube_start_world": r4(c0), "pad_sep_closed_m": round(sep_closed, 4), "cube_size_m": CUBE,
        "pad_cube_contact_pairs_after_close": pad_cube,
        "cube_rise_at_snapshot_m": round(float(snap["objects"]["cube"]["pose"][0][2] - c0[2]), 4),
        "cube_rise_end_m": round(float(orig[-1][2][2] - c0[2]), 4),
        "cube_still_in_contact_at_end": held,
        "chassis_travel_after_snapshot_m": round(float(np.linalg.norm(orig[-1][1] - orig[0][1])), 4),
        "restore_readback": {"q_max_err": float(np.max(np.abs(chk["q"] - snap["q"]))),
                             "cube_pos_err_m": float(np.linalg.norm(chk["objects"]["cube"]["pose"][0]
                                                                     - snap["objects"]["cube"]["pose"][0]))},
        "n_actions_after_snapshot": len(rest),
        "replay_vs_replay": diff(r1, r2), "replay_vs_original": diff(r1, orig),
        "canonical_branch_vs_branch": diff(k0, k1), "canonical_vs_original": diff(k0, orig),
        "canonical_restore_wall_s": round(canon_s, 3), "cube_rise_end_canonical_m": round(float(k0[-1][2][2] - c0[2]), 4),
        "cube_rise_end_replay_m": round(float(r1[-1][2][2] - c0[2]), 4),
        "captured": ["articulation root pose + lin/ang velocity", "all 42 DOF positions + velocities",
                     "position targets of the 22 controlled DOFs", "wheel velocity targets",
                     "object poses + lin/ang velocities", "backend step counter", "sim time (recorded only)"],
        "not_captured": ["PhysX solver warm-start and contact caches", "sim clock (cannot be rewound)"],
    }


@sec("navigation_bounded")
def _():
    default_state(HOME["left"])               # navigate tucked
    world.reset()
    B.refresh()
    if cube is not None:
        cube.set_world_pose(position=at(-3.0, -2.5, 0.05))
    run(A.home_action(), 10)
    p0, _, y0 = B.chassis_pose()
    s0 = A.observe_proprio()
    legs = [("reverse", -0.2, 0.0, 30), ("turn_left", 0.0, 0.5, 33), ("forward", 0.25, 0.0, 40),
            ("turn_right", 0.0, -0.5, 33), ("forward", 0.25, 0.0, 20)]
    hits, trace, arm_dev, tilt = set(), [], 0.0, 0.0
    obst = tuple(f"/World/{k}" for k in (*OBST, "pedestal_l", "pedestal_r", "target_r", "mk_red", "mk_green", "mk_blue"))
    for name, v, w, n in legs:
        for _ in range(n):
            A.apply_action(act(v=v, w=w))
            for k in IB.contact_pairs():
                if any(o in k for o in obst) and IB.ROBOT_PRIM in k:
                    hits.add(k)
            s = A.observe_proprio()
            arm_dev = max(arm_dev, float(np.max(np.abs(s[0:15] - s0[0:15]))))
            _, r, _ = B.chassis_pose()
            tilt = max(tilt, math.degrees(math.acos(min(1.0, r[2, 2]))))
        p, _, y = B.chassis_pose()
        trace.append({"leg": name, "x": round(float(p[0] - p0[0]), 3), "y": round(float(p[1] - p0[1]), 3),
                      "yaw": round(float((y - y0 + math.pi) % (2 * math.pi) - math.pi), 3)})
    run(A.home_action(), 5)
    p1, _, y1 = B.chassis_pose()
    return {"legs": [{"leg": l, "v": v, "w": w, "actions": n} for l, v, w, n in legs], "trace": trace,
            "commanded_path_length_m": round(0.2 * 3 + 0.25 * 4 + 0.25 * 2, 3),
            "robot_obstacle_contacts": sorted(hits), "collision_free": not hits,
            "max_arm_deviation_rad": round(arm_dev, 5), "max_chassis_tilt_deg": round(tilt, 3),
            "obstacles_world": {k: r4(v) for k, v in OBST.items()},
            "final_disp_m": r4(p1 - p0),
            "final_dyaw_rad": round(float((y1 - y0 + math.pi) % (2 * math.pi) - math.pi), 3)}


print("###JSON###" + json.dumps(O, default=str) + "\n", flush=True)
with open(f"{OUT}/live_smoke.json", "w") as f:
    json.dump(O, f, indent=1, default=str)
app.close()
