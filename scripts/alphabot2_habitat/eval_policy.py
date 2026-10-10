"""End-to-end evaluation of an AlphaBot 2 OVMM policy from normal OVMM initialization (Habitat client).

The policy (pi0.5 policy_server.py, local socket) sees exactly what it was trained on: camera_top <-
head_rgb, camera_left <- right_wrist_rgb, camera_right <- left_wrist_rgb (240x320 RGB), the 16-D state
(checkpoint convention) and the instruction. Its CartesianMobileAction18 chunk is lowered by the same
CartesianExecutor the expert used and executed as one atomic 18-D action per 10 Hz step; the chunk is
re-queried every EXEC_STEPS steps. No privileged information reaches the policy.

Funnel per episode, from OVMM's own measures at the end of the episode:
  find/approach  ovmm_find_object_phase_success
  grasp          ovmm_pick_object_phase_success
  carry          ovmm_find_recep_phase_success (holding the object at the goal receptacle)
  place          ovmm_place_object_phase_success
  full success   ovmm_place_success
plus the hand that grasped (from the grasp action's events), steps, IK-invalid steps, and per-arm /
per-scene breakdowns from eval_summary.py.

    EPISODES=... SPLIT=val PORT=5600 OUT=... run_habitat.sh eval_policy.py
"""
import json, os, socket, sys, time, traceback

import numpy as np

REPO = os.environ.get("LMM_REPO", "/workspace/lmm")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "alphabot2_pi05"))
from lmm.embodiments.alphabot2 import cartesian as C  # noqa: E402
from lmm.embodiments.alphabot2 import pi05_contract as P  # noqa: E402
from lmm.embodiments.alphabot2.habitat import env as HE  # noqa: E402
from wire import recv_msg, send_msg  # noqa: E402

OUT = os.environ["OUT"]
os.makedirs(OUT, exist_ok=True)
SPLIT = os.environ.get("SPLIT", "val")
EXEC = int(os.environ.get("EXEC_STEPS", "10"))
SEED_OFFSET = int(os.environ.get("SEED_OFFSET", "0"))       # policy noise seed pass (stability across passes)
MAX_STEPS = int(os.environ.get("MAX_STEPS", "3000"))
WK, NW = int(os.environ.get("WORKER", 0)), int(os.environ.get("NWORKERS", 1))
CAM_MAP = {"camera_top": "head_rgb", "camera_left": "right_wrist_rgb", "camera_right": "left_wrist_rgb"}
FUNNEL = (("find", "ovmm_find_object_phase_success"), ("grasp", "ovmm_pick_object_phase_success"),
          ("carry", "ovmm_find_recep_phase_success"), ("place", "ovmm_place_object_phase_success"),
          ("full", "ovmm_place_success"))
units = [u for u in os.environ["EPISODES"].replace("\n", ",").split(",") if u][WK::NW]
env = HE.make_env(gpu=int(os.environ.get("GPU", "0")), split=SPLIT)
sock = socket.create_connection(("127.0.0.1", int(os.environ.get("PORT", "5600"))))
send_msg(sock, {"cmd": "ping"})
srv = recv_msg(sock)
assert set(srv.get("cams", [])) == set(CAM_MAP), f"server cameras {srv.get('cams')} != {sorted(CAM_MAP)}"


def observe():
    HE.robot(env).update()
    o = env.sim.get_sensor_observations()
    return {k: np.asarray(o[c])[..., :3].astype(np.uint8) for k, c in CAM_MAP.items()}


def metrics():
    m = env.get_metrics()
    out = {}
    for _, k in FUNNEL:
        v = m.get(k)
        if isinstance(v, dict):
            v = any(bool(x) for x in v.values())
        out[k] = float(bool(v)) if v is not None else 0.0
    return out


def instruction(ep):
    nice = lambda s: str(s).replace("_", " ")
    return f"Move the {nice(ep.object_category)} from the {nice(ep.start_recep_category)} to the {nice(ep.goal_recep_category)}."


fn = f"{OUT}/eval_w{WK:02d}.jsonl"
done = {json.loads(l)["unit"] for l in open(fn)} if os.path.exists(fn) else set()
for unit in units:
    if unit in done:
        continue
    ep_id, rep = HE.parse_unit(unit)
    rec = {"unit": unit, "episode": ep_id, "replica": rep, "split": SPLIT}
    t0 = time.time()
    try:
        HE.reset_to(env, ep_id, rep)
        ep = env.current_episode
        rec.update({"scene": ep.scene_id.split("/")[-1].split(".")[0], "object": ep.object_category,
                    "instruction": instruction(ep), "start_fix": HE.fix_start(env)})
        rob = HE.robot(env)
        ex = C.CartesianExecutor.from_proprio(rob.proprio16())
        steps = ik_invalid = 0
        chunk, k = None, EXEC
        while steps < MAX_STEPS and not env.episode_over:
            if k >= EXEC:
                send_msg(sock, {"cmd": "infer", "images": observe(), "task": rec["instruction"],
                                "state": P.dataset_state_from_proprio(rob.proprio16()).astype(np.float32),
                                "seed": int(HE.episode_seed(ep_id, rep)) + 1_000_003 * SEED_OFFSET + steps})
                r = recv_msg(sock)
                if "error" in r:
                    raise RuntimeError("policy server: " + r["error"])
                chunk, k = r["actions"], 0
            joint, rep_ik = ex.lower(chunk[k])
            ik_invalid += int(not (rep_ik["left"]["ik_valid"] and rep_ik["right"]["ik_valid"]))
            HE.step18(env, joint)
            steps, k = steps + 1, k + 1
        grasps = [e for e in HE.action_obj(env).events if e.get("type") == "grasp"]
        rec.update({"steps": steps, "ik_invalid_steps": ik_invalid, "episode_over": bool(env.episode_over),
                    "hand": grasps[0]["hand"] if grasps else None, "n_grasps": len(grasps), "ovmm": metrics()})
        rec["funnel"] = {name: rec["ovmm"][k_] for name, k_ in FUNNEL}
    except Exception as e:
        rec["engineering_error"] = repr(e)[:300]
        rec["tb"] = traceback.format_exc()[-1200:]
    rec["wall_s"] = round(time.time() - t0, 1)
    with open(fn, "a") as f:
        f.write(json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
    print("EVAL", json.dumps({k_: rec.get(k_) for k_ in ("unit", "scene", "hand", "funnel", "engineering_error")}), flush=True)
env.close()

