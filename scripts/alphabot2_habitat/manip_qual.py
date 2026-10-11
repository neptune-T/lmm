"""OVMM manipulation qualification (+ full privileged expert) for AlphaBot 2.

For each episode (normal OVMM initialization, start snapped to the AlphaBot
navmesh when needed) the privileged expert runs in MODES (fresh reset each):
  left   -- left arm forced;   right -- right arm forced;   auto -- arm chosen by geometry.
Per run: arm feasibility (reach/aperture/navmesh), stage reached, grasp event and
owning hand, stable transport (still held on arrival at the receptacle), release,
OVMM measures (pick/place phase successes, ovmm_place_success), robot-scene
collisions during manipulation, IK validity, steps.

    MODES=left,right,auto EPISODES=... OUT=... [RESULTS_TAG=left] run_habitat.sh manip_qual.py
"""
import json, os, sys, time, traceback
from collections import defaultdict

import numpy as np

REPO = os.environ.get("LMM_REPO", "/workspace/lmm")
sys.path.insert(0, REPO)
from lmm.embodiments.alphabot2.habitat import env as HE  # noqa: E402
from lmm.embodiments.alphabot2.habitat import fsutil as FS  # noqa: E402
FS.protect_stdio()
from lmm.embodiments.alphabot2.habitat import nav as NAV  # noqa: E402
from lmm.embodiments.alphabot2.habitat.expert import ExpertFailure, OVMMExpert  # noqa: E402

OUT = os.environ.get("OUT", "/workspace/runtime/runs/manip_qual")
os.makedirs(OUT, exist_ok=True)
MODES = os.environ.get("MODES", "left,right,auto").split(",")
WK, NW = int(os.environ.get("WORKER", 0)), int(os.environ.get("NWORKERS", 1))
env = HE.make_env(gpu=int(os.environ.get("GPU", "0")), split=os.environ.get("SPLIT", "val"))
if os.environ.get("EPISODES"):
    eps = os.environ["EPISODES"].split(",")
else:
    by_scene = defaultdict(list)
    for e in env.episodes:
        by_scene[e.scene_id].append(e.episode_id)
    per = int(os.environ.get("EPISODES_PER_SCENE", "3"))
    eps = [eid for s in sorted(by_scene)[:int(os.environ.get("MAX_SCENES", "12"))] for eid in by_scene[s][:per]]
eps = eps[WK::NW]
KEYS = ("ovmm_find_object_phase_success", "ovmm_pick_object_phase_success", "ovmm_find_recep_phase_success",
        "ovmm_place_object_phase_success", "ovmm_place_success", "obj_anywhere_on_goal", "did_pick_object",
        "pick_success", "ovmm_nav_to_pick_succ", "ovmm_nav_to_place_succ")


def metrics():
    m = env.get_metrics()
    out = {}
    for k in KEYS:
        v = m.get(k)
        if isinstance(v, dict):
            v = any(bool(x) for x in v.values())
        out[k] = float(v) if isinstance(v, (bool, int, float, np.floating)) else v
    return out


with open(f"{OUT}/results_{os.environ.get('RESULTS_TAG', f'w{WK:02d}')}.jsonl", "a") as fout:
    for eid in eps:
        for mode in MODES:
            t0 = time.time()
            rec = {"episode_id": eid, "mode": mode}
            coll = {"manip_steps": 0, "collision_steps": 0, "links": {}}
            held_during_carry = []
            try:
                HE.reset_to(env, eid)
                ep = env.current_episode
                rec.update({"scene": ep.scene_id.split("/")[-1].split(".")[0], "object": ep.object_category,
                            "start_recep": ep.start_recep_category, "goal_recep": ep.goal_recep_category})
                rec["start_fix"] = HE.fix_start(env)
                ex = None

                def on_step(cart, stage, rep):
                    if stage in ("reach", "descend", "grasp", "lift", "reach_place", "lower", "release", "retract"):
                        c, _ = NAV.scene_contacts(env)
                        coll["manip_steps"] += 1
                        if c:
                            coll["collision_steps"] += 1
                            for x in c:
                                coll["links"][x["link"]] = coll["links"].get(x["link"], 0) + 1
                    if stage == "navigate_to_receptacle":
                        mg = env.sim.get_agent_data(0).grasp_mgrs[0 if ex.side == "left" else 1]
                        held_during_carry.append(bool(mg.is_grasped))
                ex = OVMMExpert(env, on_step=on_step, force_arm=None if mode == "auto" else mode)
                try:
                    ex.run()
                    rec["expert_completed"] = True
                except ExpertFailure as e:
                    rec["expert_completed"] = False
                    rec["failure"] = str(e)
                rec["pick_diag"] = ex.log.get("pick_diag")
                rec["arm"] = ex.log.get("arm")
                rec["arm_choice_costs"] = ex.log.get("arm_choice")
                rec["stage_reached"] = ex.stage
                rec["events"] = ex.events
                rec["steps"] = ex.t
                rec["ik_invalid_steps"] = ex.log.get("ik_invalid_steps", 0)
                rec["plans"] = {k: ex.log.get(k) for k in ("pick_plan", "place_plan", "nav_to_object", "nav_to_receptacle")}
                rec["grasp_events"] = HE.action_obj(env).events
                rec["transport_held_fraction"] = float(np.mean(held_during_carry)) if held_during_carry else None
                rec["manip_collisions"] = coll
                rec["ovmm"] = metrics()
            except Exception as e:
                rec["engineering_error"] = repr(e)
                rec["tb"] = traceback.format_exc()[-1500:]
            rec["wall_s"] = round(time.time() - t0, 1)
            FS.append_line(fout.name, json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
            print("MANIP", json.dumps({k: rec.get(k) for k in ("episode_id", "mode", "arm", "expert_completed", "failure",
                                                               "stage_reached", "engineering_error")}),
                  (rec.get("ovmm") or {}).get("ovmm_place_success"), flush=True)
env.close()
