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

eid = os.environ.get("GRASP_EPISODE")
if eid:
    side = os.environ.get("ARM", "left")
    HE.reset_to(env, eid)
    HE.fix_start(env)
    act = HE.action_obj(env)
    box = {}

    trace = []

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
    rep["grasp"] = {"episode": eid, "arm": side, "result": res, "events": ex.events, "hold_trace": trace[:80],
                    "hold_thresh": float(env.sim.get_agent_data(0).grasp_mgrs[0]._config.hold_thresh),
                    "nav_to_object": ex.log.get("nav_to_object"), "diag": box.get("diag")}
    print("GRASP", json.dumps(rep["grasp"], default=str), flush=True)
out = os.environ.get("OUT", "/workspace/runtime/runs/diag")
os.makedirs(out, exist_ok=True)
json.dump(rep, open(f"{out}/diag_{os.getpid()}.json", "w"), indent=1, default=str)
env.close()
