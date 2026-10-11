"""Diagnostics for a blocked manipulation qualification (no data written besides the JSON report).

  reset   : reset each RESET_EPISODES id REPEAT times in a fresh env, then once after a short expert run
            on the same episode -> is OVMM's reset reachability assertion deterministic?
  grasp   : run the expert on GRASP_EPISODE with ARM forced; at the first 'grasp' step report the pad
            midpoint in the target object's box frame, the margin, object displacement since reset,
            and the hand's error to the planned grasp pose.

    RESET_EPISODES=28255,26702 GRASP_EPISODE=5707 ARM=left run_habitat.sh diag_episode.py
"""
import json, os, sys, traceback

import numpy as np

REPO = os.environ.get("LMM_REPO", "/workspace/lmm")
sys.path.insert(0, REPO)
import magnum as mn  # noqa: E402
from lmm.embodiments.alphabot2.habitat import env as HE  # noqa: E402
from lmm.embodiments.alphabot2.habitat import nav as NAV  # noqa: E402
from lmm.embodiments.alphabot2.habitat.expert import OVMMExpert, SIDES  # noqa: E402

env = HE.make_env(gpu=int(os.environ.get("GPU", "0")), split=os.environ.get("SPLIT", "train"))
rep = {"reset": {}, "grasp": None}


def try_reset(eid):
    try:
        HE.reset_to(env, eid)
        rob = HE.robot(env)
        return {"ok": True, "base": [round(float(x), 4) for x in rob.sim_obj.translation],
                "start_navigable": bool(env.sim.pathfinder.is_navigable(rob.sim_obj.translation))}
    except AssertionError as e:
        tb = traceback.extract_tb(e.__traceback__)
        return {"ok": False, "where": [f"{f.name}:{f.lineno}" for f in tb[-4:]]}


for eid in [e for e in os.environ.get("RESET_EPISODES", "").split(",") if e]:
    runs = [try_reset(eid) for _ in range(int(os.environ.get("REPEAT", "3")))]
    after = None
    if runs[-1]["ok"]:
        HE.fix_start(env)
        NAV.move_arms_to_tuck(env)
        for _ in range(20):
            HE.step18(env, NAV.tuck_action(env, 0.2, 0.3))
        after = try_reset(eid)
    rep["reset"][eid] = {"fresh": runs, "after_motion": after}
    print("RESET", eid, json.dumps(rep["reset"][eid]), flush=True)

def head_cam_vs_link():
    """Head RGB sensor node vs the robot's head_camera link (world): far apart = the camera is stale."""
    rob = HE.robot(env)
    sens = np.array(env.sim._sensors["head_rgb"]._sensor_object.node.absolute_translation)
    link = np.array(rob.sim_obj.get_link_scene_node(rob.link_id["head_camera"]).absolute_translation)
    return {"sensor_minus_head_link_m": round(float(np.linalg.norm(sens - link)), 3),
            "base_xz": [round(float(rob.sim_obj.translation[0]), 3), round(float(rob.sim_obj.translation[2]), 3)]}


if os.environ.get("ORDER_TEST"):                  # "A,B": reset A; run the expert on B; reset A twice more
    a_, b_ = os.environ["ORDER_TEST"].split(",")
    seq = []
    for step_ in ("fresh", "expert_on_b", "after", "again"):
        if step_ == "expert_on_b":
            HE.reset_to(env, b_)
            HE.fix_start(env)
            try:
                OVMMExpert(env).run()
                seq.append({"step": f"expert {b_}", "result": "completed"})
            except Exception as e:
                seq.append({"step": f"expert {b_}", "result": repr(e)[:120]})
            seq.append({"step": f"after expert {b_}", "cam": head_cam_vs_link()})
            continue
        r = try_reset(a_)
        seq.append({"step": f"reset {a_} ({step_})", **r, "cam": head_cam_vs_link()})
    rep["order_test"] = seq
    for x in seq:
        print("ORDER", json.dumps(x), flush=True)

eid = os.environ.get("GRASP_EPISODE")
if eid:
    side = os.environ.get("ARM", "left")
    HE.reset_to(env, eid)
    HE.fix_start(env)
    act = HE.action_obj(env)
    box = {}

    trace, ptrace = [], []

    def on_step(cart, stage, r):
        if os.environ.get("TRACE_HOLD") and stage in ("grasp", "lift", "carry", "navigate_to_receptacle"):
            ad = env.sim.get_agent_data(0)
            rob_ = HE.robot(env)
            obj_, _ = ex.target_object()
            row = {"t": ex.t, "stage": stage, "active_idx": getattr(ad, "active_grasp_manager_idx", None)}
            for i, m in enumerate(ad.grasp_mgrs):
                row[f"m{i}"] = {"grasped": bool(m.is_grasped), "ee_index": getattr(m, "ee_index", None),
                                "dist_own_ee": round(float(np.linalg.norm(np.array(rob_.ee_transform(m.ee_index).translation) - np.array(obj_.translation))), 4)}
            row["dist_left_ee"] = round(float(np.linalg.norm(np.array(rob_.ee_transform(0).translation) - np.array(obj_.translation))), 4)
            row["dist_right_ee"] = round(float(np.linalg.norm(np.array(rob_.ee_transform(1).translation) - np.array(obj_.translation))), 4)
            if not trace or trace[-1]["stage"] != stage or any(trace[-1][f"m{i}"]["grasped"] != row[f"m{i}"]["grasped"] for i in (0, 1)) \
                    or len(trace) < 3 or ex.t % 25 == 0:
                trace.append(row)
        if os.environ.get("TRACE_PLACE") and stage in ("release", "retract", "done"):
            m = env.get_metrics()
            obj_, _ = ex.target_object()
            pick = lambda k: (m.get(k) if not isinstance(m.get(k), dict) else {kk: float(vv) for kk, vv in m.get(k).items()})
            ptrace.append({"t": ex.t, "stage": stage, "picked_idx": getattr(env.task, "_picked_object_idx", None),
                           "active_mgr": getattr(env.sim.get_agent_data(0), "active_grasp_manager_idx", None),
                           "sim_grasp_mgr_grasped": bool(env.sim.grasp_mgr.is_grasped),
                           "lin_vel": round(float(np.linalg.norm(np.array(obj_.linear_velocity))), 5),
                           "ang_vel": round(float(np.linalg.norm(np.array(obj_.angular_velocity))), 5),
                           "motion": str(obj_.motion_type),
                           **{k: pick(k) for k in ("ovmm_placement_stability", "object_at_rest", "obj_anywhere_on_goal",
                                                    "ovmm_place_object_phase_success", "ovmm_place_success")}})
        if stage != "grasp" or "diag" in box:
            return
        rob = HE.robot(env)
        si = SIDES.index(side)
        ee = rob.ee_transform(si)
        p = np.array(ee.translation)
        obj, _ = ex.target_object()
        bb = obj.root_scene_node.cumulative_bb
        lo, hi = np.array(bb.min), np.array(bb.max)
        local = np.array(obj.transformation.inverted().transform_point(mn.Vector3(*p)))
        outside = np.maximum(lo - local, 0) + np.maximum(local - hi, 0)
        cand = act._grasp_candidate(si)
        from lmm.embodiments.alphabot2.habitat.alphabot2_robot import EE_LINK
        Tgb = rob.link_frame(EE_LINK[side])
        pads = [rob.link_frame(f"{side}_{n}")[:3, 3] for n in ("left_pad", "right_pad")]
        c_exec = (pads[0] - pads[1]) / np.linalg.norm(pads[0] - pads[1])
        c_plan = -Tgb[:3, 0]                                   # planner: closing = gripper_base -x
        Ro = np.array(obj.transformation.rotation())
        axes = {"closing_exec_pads": c_exec.round(4).tolist(), "closing_gb_minus_x": c_plan.round(4).tolist(),
                "angle_exec_vs_gb_deg": round(float(np.degrees(np.arccos(min(1.0, abs(c_exec @ c_plan))))), 2),
                "pads_in_gb_frame": [np.round((np.linalg.inv(Tgb) @ np.r_[q, 1.0])[:3], 4).tolist() for q in pads],
                "exec_axis_in_obj_frame": np.round(Ro.T @ c_exec, 4).tolist(),
                "gb_axis_in_obj_frame": np.round(Ro.T @ c_plan, 4).tolist()}
        g = getattr(ex, "grasp_used", None) or {}
        box["diag"] = {"pad_mid_world": p.round(4).tolist(), "pad_mid_in_obj_box": local.round(4).tolist(),
                       "box_lo": lo.round(4).tolist(), "box_hi": hi.round(4).tolist(),
                       "outside_box_by_m": outside.round(4).tolist(), "margin_m": act.grasp_margin,
                       "axes": axes,
                       "candidate": None if cand is None else {"idx": cand[0], "dist_m": round(cand[2], 4), "width_m": round(cand[3], 4)},
                       "object_moved_since_reset_m": round(float(np.linalg.norm(np.array(obj.translation) - box["obj0"])), 4),
                       "arm_q_now": np.round(rob.arm_q(side), 4).tolist() if hasattr(rob, "arm_q") else None,
                       "arm_q_planned": np.round(np.asarray(g.get("q_grasp")), 4).tolist() if g.get("q_grasp") is not None else None,
                       "grasp_plan": {k: (np.round(np.asarray(v), 4).tolist() if hasattr(v, "__len__") and not isinstance(v, str) else v)
                                      for k, v in g.items() if k in ("mode", "tilt", "width", "pad_target_arm", "approach")}}

    ex = OVMMExpert(env, on_step=on_step, force_arm=side)
    obj0, _ = ex.target_object()
    box["obj0"] = np.array(obj0.translation)
    try:
        ex.run()
        res = "completed"
    except Exception as e:
        res = repr(e)
    rep["grasp"] = {"episode": eid, "arm": side, "result": res, "events": ex.events, "hold_trace": trace[:80], "place_trace": ptrace[:10] + ptrace[-6:],
                    "place_summary": {"steps_after_release": len(ptrace),
                                      "first_at_rest": next((p_["t"] for p_ in ptrace if p_["object_at_rest"]), None),
                                      "at_rest_fraction": round(sum(bool(p_["object_at_rest"]) for p_ in ptrace) / max(1, len(ptrace)), 3),
                                      "first_stable": next((p_["t"] for p_ in ptrace if p_["ovmm_placement_stability"]), None),
                                      "first_place_success": next((p_["t"] for p_ in ptrace if p_["ovmm_place_success"]), None),
                                      "max_ang_vel_last50": max((p_["ang_vel"] for p_ in ptrace[-50:]), default=None)},
                    "hold_thresh": float(env.sim.get_agent_data(0).grasp_mgrs[0]._config.hold_thresh),
                    "nav_to_object": ex.log.get("nav_to_object"), "diag": box.get("diag")}
    print("GRASP", json.dumps(rep["grasp"], default=str), flush=True)
out = os.environ.get("OUT", "/workspace/runtime/runs/diag")
os.makedirs(out, exist_ok=True)
json.dump(rep, open(f"{out}/diag_{os.getpid()}.json", "w"), indent=1, default=str)
env.close()
