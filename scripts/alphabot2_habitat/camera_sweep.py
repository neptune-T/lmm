"""Head-camera (neck pitch) sweep in real OVMM scenes for AlphaBot 2 (crouched canonical torso).

Per candidate neck pitch (a URDF variant with that pitch baked in, HEAD_URDFS),
per episode: the expert plans the pick standoff (geometric arm choice) and the
robot drives there with (v, omega); then to a place standoff. OVMM's own
occlusion-aware masks give visibility each step:
  object_segmentation      (target object)      -> visible if >= MIN_PIX pixels
  goal_recep_segmentation  (goal receptacles)   -> visible if >= MIN_PIX pixels
Reported per pitch: fraction of steps with the target visible over the last
1.5 m of the pick approach, at the pick standoff, and the same for the goal
receptacle on the place approach / standoff.

    HEAD_URDFS=p0.0:/path/a.urdf,p0.2:/path/b.urdf EPISODES=... run_habitat.sh camera_sweep.py
"""
import json, os, sys

import numpy as np

REPO = os.environ.get("LMM_REPO", "/nix/yanjun/zts/alphabot2/lmm_repo")
sys.path.insert(0, REPO)
from lmm.embodiments.alphabot2.habitat import env as HE  # noqa: E402
from lmm.embodiments.alphabot2.habitat import nav as NAV  # noqa: E402
from lmm.embodiments.alphabot2.habitat.expert import OVMMExpert  # noqa: E402

MIN_PIX = 20
OUT = os.environ.get("OUT", "/nix/yanjun/zts/alphabot2/habitat_runs/camera_sweep")
os.makedirs(OUT, exist_ok=True)
variants = [v.split(":", 1) for v in os.environ["HEAD_URDFS"].split(",")]
EPS = os.environ["EPISODES"].split(",")
res = {}
for tag, urdf in variants:
    env = HE.make_env(gpu=int(os.environ.get("GPU", "0")), split=os.environ.get("SPLIT", "train"),
                      overrides=[f"habitat.simulator.agents.main_agent.articulated_agent_urdf={urdf}"])
    stats = {"approach_obj": [], "standoff_obj": [], "approach_rec": [], "standoff_rec": [], "episodes": 0}
    for eid in EPS:
        try:
            HE.reset_to(env, eid)
        except AssertionError:
            continue
        HE.fix_start(env)
        NAV.move_arms_to_tuck(env)
        ex = OVMMExpert(env)
        obj, _ = ex.target_object()
        pk = ex.plan_pick("left", obj) or ex.plan_pick("right", obj)
        if pk is None:
            continue
        stats["episodes"] += 1
        seen = []

        def on_obs(obs, key, goal):
            m = np.asarray(obs.get(key))
            xz = NAV.pose2d(env)[0]
            seen.append((float(np.linalg.norm(xz - np.array([goal[0], goal[2]]))), bool(m.sum() >= MIN_PIX)))
        goal = pk["standoff"]["xyz"]
        NAV.follow(env, goal, on_step=lambda o: on_obs(o, "object_segmentation", goal), arrive_m=0.02)
        # turn to the planned heading and look
        for _ in range(80):
            _, yaw, _ = NAV.pose2d(env)
            e = NAV.wrap(pk["standoff"]["yaw"] - yaw)
            if abs(e) < 0.03:
                break
            o = HE.step18(env, NAV.tuck_action(env, 0.0, float(np.clip(1.5 * e, -0.8, 0.8))))
        o = HE.step18(env, NAV.tuck_action(env))
        stats["approach_obj"] += [v for d, v in seen if d <= 1.5]
        stats["standoff_obj"].append(bool(np.asarray(o["object_segmentation"]).sum() >= MIN_PIX))
        # place approach: nearest goal-receptacle view point
        pf = env.sim.pathfinder
        vps = [np.array(pf.snap_point(np.array(v.agent_state.position))) for g in env.current_episode.candidate_goal_receps
               for v in g.view_points[:20]]
        vps = [v for v in vps if np.isfinite(v).all() and pf.is_navigable(v)]
        if vps:
            tgt = min(vps, key=lambda v: NAV.plan(env, v)[2])
            seen = []
            gp = env.current_episode.candidate_goal_receps[0].position
            NAV.follow(env, tgt, face_xz=np.array([gp[0], gp[2]]),
                       on_step=lambda o: on_obs(o, "goal_recep_segmentation", tgt))
            o = HE.step18(env, NAV.tuck_action(env))
            stats["approach_rec"] += [v for d, v in seen if d <= 1.5]
            stats["standoff_rec"].append(bool(np.asarray(o["goal_recep_segmentation"]).sum() >= MIN_PIX))
    res[tag] = {k: (round(float(np.mean(v)), 3) if isinstance(v, list) and v else v) for k, v in stats.items()}
    res[tag]["n"] = {k: len(v) for k, v in stats.items() if isinstance(v, list)}
    print("SWEEP", tag, json.dumps(res[tag]), flush=True)
    env.close()
json.dump(res, open(f"{OUT}/camera_sweep.json", "w"), indent=1)
print("###SWEEP###" + json.dumps(res))
