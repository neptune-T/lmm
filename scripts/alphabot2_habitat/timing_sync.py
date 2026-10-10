"""Dynamic RGB / proprio / action synchronization test for AlphaBot 2 in Habitat.

One policy action = one env.step (20 x 0.005 s). The observation returned by
env.step(a_t) must show the state the robot is in when a_{t+1} is decided.

Test 1 (robot motion): base turning at 0.8 rad/s + both arms swinging
+-0.35 rad on joint 1 and 4, 30 steps. For each observation we store the RGB of
the three canonical cameras plus a full snapshot of the state; afterwards every
recorded state (and its -1/+1 neighbours) is restored and re-rendered, and the
recorded image must match the claimed state (offset 0) best.

Test 2 (physics motion): the episode's target object is lifted 0.5 m into the
air in front of the head camera and released as a DYNAMIC body; while it
falls, the recorded head image is compared with re-renders of the recorded
object poses at offsets -1/0/+1 (the object's pose recorded with the proprio).

Run with CONCUR_RENDER=0 and =1 (OVMM default renders concurrently with the
physics substeps, i.e. BEFORE them).
"""
import json, math, os, sys

import magnum as mn
import numpy as np

REPO = os.environ.get("LMM_REPO", "/workspace/lmm")
sys.path.insert(0, REPO)
from lmm.embodiments.alphabot2.habitat import env as HE  # noqa: E402
from lmm.embodiments.alphabot2.habitat import snapshot as SN  # noqa: E402

CONCUR = bool(int(os.environ.get("CONCUR_RENDER", "0")))
env = HE.make_env(gpu=int(os.environ.get("GPU", "0")), split=os.environ.get("SPLIT", "train"),
                  overrides=[f"habitat.simulator.concur_render={CONCUR}"])
EP = os.environ.get("EPISODE", env.episodes[0].episode_id)
CAMS = ("head_rgb", "left_wrist_rgb", "right_wrist_rgb")
OBSERVABLE = 4.0


def small(im):
    im = np.asarray(im)[..., :3].astype(float)
    return im.reshape(15, 16, 20, 16, 3).mean(axis=(1, 3))


def render_now():
    HE.robot(env).update()
    obs = env.sim.get_sensor_observations()
    return {k: small(obs[k]) for k in CAMS}


R = {"concur_render": CONCUR, "episode": EP}
HE.reset_to(env, EP)
HE.fix_start(env)
rob = HE.robot(env)
# ---------------- test 1: robot motion
recs = []
base = HE.hold_action(env)
for t in range(30):
    a = base.copy()
    a[0] += 0.35 * math.sin(2 * math.pi * t / 7)
    a[3] += 0.35 * math.sin(2 * math.pi * t / 5)
    a[8] -= 0.35 * math.sin(2 * math.pi * t / 6)
    a[11] += 0.35 * math.cos(2 * math.pi * t / 5)
    a[17] = 0.8
    obs = HE.step18(env, a)
    recs.append({"img": {k: small(obs[k]) for k in CAMS}, "snap": SN.capture(env),
                 "proprio_obs": np.asarray(obs["alphabot2_proprio"]), "proprio_state": rob.proprio16()})
rows, ok = [], {k: [] for k in CAMS}
refs = [None] * len(recs)
for i in range(len(recs)):
    SN.restore(env, recs[i]["snap"])
    refs[i] = render_now()
for i in range(1, len(recs) - 1):
    row = {"t": i}
    for k in CAMS:
        d = {o: float(np.abs(recs[i]["img"][k] - refs[i + o][k]).mean()) for o in (-1, 0, 1)}
        sep = min(float(np.abs(refs[i + o][k] - refs[i][k]).mean()) for o in (-1, 1))
        best = min(d, key=d.get)
        row[k] = {"best": best, "err0": round(d[0], 3), "sep": round(sep, 2)}
        if sep >= OBSERVABLE:
            ok[k].append(best == 0)
    rows.append(row)
R["robot_motion"] = {"per_camera_best_at_0": {k: (sum(v), len(v)) for k, v in ok.items()},
                     "proprio_obs_equals_state": bool(all(np.allclose(r["proprio_obs"], r["proprio_state"], atol=1e-5) for r in recs)),
                     "median_err0": {k: float(np.median([r[k]["err0"] for r in rows])) for k in CAMS},
                     "rows_sample": rows[:4]}
# ---------------- test 2: falling object (physics)
HE.reset_to(env, EP)
HE.fix_start(env)
sim = env.sim
rom = sim.get_rigid_object_manager()
obj = rom.get_object_by_id(sim.scene_obj_ids[int(sim.get_targets()[0][0])])
cam = rob.sim_obj.get_link_scene_node(rob.link_id["head_camera"]).absolute_transformation()
fwd = np.array(cam.transform_vector(mn.Vector3(0, 0, 1)))           # head optical axis (+z optical frame)
p0 = np.array(cam.translation) + 0.9 * fwd / np.linalg.norm(fwd)
p0[1] = float(rob.sim_obj.translation[1]) + 1.3
obj.motion_type = __import__("habitat_sim").physics.MotionType.DYNAMIC
obj.collidable = True
obj.translation = mn.Vector3(*p0)
obj.linear_velocity = mn.Vector3(0, 0, 0)
fall = []
hold = HE.hold_action(env)
for t in range(8):
    obs = HE.step18(env, hold)
    fall.append({"img": small(obs["head_rgb"]), "pose": np.array(obj.translation), "rot": obj.rotation,
                 "y": float(obj.translation[1])})
f_ok, f_rows = [], []
for i in range(1, len(fall) - 1):
    refs_f = {}
    for o in (-1, 0, 1):
        obj.translation = mn.Vector3(*fall[i + o]["pose"])
        obj.rotation = fall[i + o]["rot"]
        refs_f[o] = render_now()["head_rgb"]
    d = {o: float(np.abs(fall[i]["img"] - refs_f[o]).mean()) for o in refs_f}
    best = min(d, key=d.get)
    f_rows.append({"t": i, "best": best, "err": {o: round(v, 3) for o, v in d.items()}, "drop_m": round(fall[i - 1]["y"] - fall[i]["y"], 3)})
    f_ok.append(best == 0)
R["falling_object"] = {"best_at_0": (sum(f_ok), len(f_ok)), "rows": f_rows}
R["synchronized"] = bool(all(len(v) >= 5 and all(v) for v in ok.values()) and all(f_ok)
                         and R["robot_motion"]["proprio_obs_equals_state"])
print("###SYNC###" + json.dumps(R, default=str), flush=True)
env.close()
