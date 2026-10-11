"""Per-object fresh-reset replay divergence diagnosis (pilot replay rejections).

For each UNIT: reset, run the expert recording every step's executed JointMobileAction18 and the pose of
every OVMM scene object; reset again and replay the joint actions recording the same. Reports per object:
max translation / rotation divergence, first step with > 1 cm translation, whether it is the target,
whether it already moved before the robot did anything (first STILL_STEPS steps of the original), and its
initial velocity right after reset (an unsettled object at reset makes physics history-dependent).

    UNITS=13271#0,24636#0 GPU=1 OUT=... run_habitat.sh diag_replay.py
"""
import json, os, sys

import numpy as np

REPO = os.environ.get("LMM_REPO", "/workspace/lmm")
sys.path.insert(0, REPO)
from lmm.embodiments.alphabot2.habitat import env as HE  # noqa: E402
from lmm.embodiments.alphabot2.habitat import fsutil as FS  # noqa: E402
from lmm.embodiments.alphabot2.habitat.expert import OVMMExpert  # noqa: E402

FS.protect_stdio()
STILL = 15
env = HE.make_env(gpu=int(os.environ.get("GPU", "0")), split=os.environ.get("SPLIT", "train"))
out = os.environ.get("OUT", "/workspace/runtime/runs/diag")
os.makedirs(out, exist_ok=True)


def objs():
    rom = env.sim.get_rigid_object_manager()
    res = []
    for o in env.sim.scene_obj_ids:
        ob = rom.get_object_by_id(o)
        T = np.array(ob.transformation)
        res.append((T[:3, 3].copy(), T[:3, :3].copy(), float(np.linalg.norm(np.array(ob.linear_velocity))),
                    float(np.linalg.norm(np.array(ob.angular_velocity)))))
    return res


def handles():
    rom = env.sim.get_rigid_object_manager()
    return [rom.get_object_by_id(o).handle for o in env.sim.scene_obj_ids]


rep = {}
for unit in os.environ["UNITS"].split(","):
    e, r = HE.parse_unit(unit)
    HE.reset_to(env, e, r)
    rob_ = HE.robot(env)
    start_pose = [round(float(x), 5) for x in rob_.sim_obj.translation] + [round(float(rob_.base_rot), 5)]
    HE.fix_start(env)
    o0 = objs()
    hs = handles()
    tgt = int(env.sim.get_targets()[0][0])
    traj, acts = [o0], []
    orig = HE.step18

    def rec(env_, a):
        acts.append(np.asarray(a, float).copy())
        ob = orig(env_, a)
        traj.append(objs())
        return ob
    HE.step18 = rec
    try:
        OVMMExpert(env).run()
    except Exception:
        pass
    HE.step18 = orig
    HE.reset_to(env, e, r)
    HE.fix_start(env)
    rtraj = [objs()]
    for a in acts:
        HE.step18(env, a)
        rtraj.append(objs())
    per = []
    for i, h in enumerate(hs):
        dt = np.array([np.linalg.norm(a_[i][0] - b_[i][0]) for a_, b_ in zip(traj, rtraj)])
        dr = np.array([np.degrees(np.arccos(np.clip((np.trace(a_[i][1].T @ b_[i][1]) - 1) / 2, -1, 1))) for a_, b_ in zip(traj, rtraj)])
        moved_early = float(np.linalg.norm(traj[min(STILL, len(traj) - 1)][i][0] - traj[0][i][0]))
        per.append({"idx": i, "handle": h.split("_:")[0][-40:], "target": i == tgt,
                    "max_trans_div_m": round(float(dt.max()), 4), "max_rot_div_deg": round(float(dr.max()), 2),
                    "first_step_trans_div_gt_1cm": int(np.argmax(dt > 0.01)) if (dt > 0.01).any() else None,
                    "moved_in_first_steps_m": round(moved_early, 4), "lin_vel_at_reset": round(o0[i][2], 4),
                    "ang_vel_at_reset": round(o0[i][3], 4),
                    "total_motion_in_original_m": round(float(np.linalg.norm(traj[-1][i][0] - traj[0][i][0])), 4)})
    bad = [p for p in per if p["max_trans_div_m"] > 0.025 or p["max_rot_div_deg"] > 5]
    rep[unit] = {"steps": len(acts), "objects": len(per), "diverging": bad, "start_pose": start_pose,
                 "clean_reset": os.environ.get("ALPHABOT2_CLEAN_RESET", "1"),
                 "unsettled_at_reset": [p["idx"] for p in per if p["lin_vel_at_reset"] > 1e-3 or p["ang_vel_at_reset"] > 1e-3]}
    print("REPLAYDIAG", unit, json.dumps(rep[unit]), flush=True)
json.dump(rep, open(f"{out}/diag_replay_{os.getpid()}.json", "w"), indent=1)
env.close()
