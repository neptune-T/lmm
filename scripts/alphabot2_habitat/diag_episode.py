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

    def on_step(cart, stage, r):
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
        g = getattr(ex, "grasp_used", None) or {}
        box["diag"] = {"pad_mid_world": p.round(4).tolist(), "pad_mid_in_obj_box": local.round(4).tolist(),
                       "box_lo": lo.round(4).tolist(), "box_hi": hi.round(4).tolist(),
                       "outside_box_by_m": outside.round(4).tolist(), "margin_m": act.grasp_margin,
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
    rep["grasp"] = {"episode": eid, "arm": side, "result": res, "events": ex.events,
                    "nav_to_object": ex.log.get("nav_to_object"), "diag": box.get("diag")}
    print("GRASP", json.dumps(rep["grasp"], default=str), flush=True)
out = os.environ.get("OUT", "/workspace/runtime/runs/diag")
os.makedirs(out, exist_ok=True)
json.dump(rep, open(f"{out}/diag_{os.getpid()}.json", "w"), indent=1, default=str)
env.close()
