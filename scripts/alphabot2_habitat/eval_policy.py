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

Diagnostics (TRACE_DIR set): per unit TRACE_DIR/<unit>.npz with the per-step base pose (x, z, yaw), the
executed CartesianMobileAction18 (incl. v, omega), IK validity, the base-to-target-object distance
(privileged; logged only, never sent to the policy) and, every VIDEO_EVERY steps, a downscaled
[camera_top | camera_left | camera_right] frame strip. The policy's inputs are unchanged.
"""
import json, os, socket, sys, time, traceback

import numpy as np

REPO = os.environ.get("LMM_REPO", "/workspace/lmm")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "alphabot2_pi05"))
from lmm.embodiments.alphabot2 import cartesian as C  # noqa: E402
from lmm.embodiments.alphabot2 import ovmm_policy_io as IO  # noqa: E402
from lmm.embodiments.alphabot2.habitat import env as HE  # noqa: E402
from lmm.embodiments.alphabot2.habitat import nav as NAV  # noqa: E402
from wire import recv_msg, send_msg  # noqa: E402

OUT = os.environ["OUT"]
os.makedirs(OUT, exist_ok=True)
SPLIT = os.environ.get("SPLIT", "val")
EXEC = int(os.environ.get("EXEC_STEPS", "10"))
SEED_OFFSET = int(os.environ.get("SEED_OFFSET", "0"))       # policy noise seed pass (stability across passes)
MAX_STEPS = int(os.environ.get("MAX_STEPS", "3000"))
WK, NW = int(os.environ.get("WORKER", 0)), int(os.environ.get("NWORKERS", 1))
CAM_MAP = IO.CAMERA_MAP                                       # the one interface (crossed wrists)
FUNNEL = (("find", "ovmm_find_object_phase_success"), ("grasp", "ovmm_pick_object_phase_success"),
          ("carry", "ovmm_find_recep_phase_success"), ("place", "ovmm_place_object_phase_success"),
          ("full", "ovmm_place_success"))
# v2 B short history (ovmm_model.history): HISTORY='{"camera": "camera_top", "offsets": [5, 10, 15], "state": false} (v2_run.sh passes sft_B.yaml history verbatim)'.
# Past observations are those of earlier queries (queries every EXEC steps), so offsets must be multiples of
# EXEC; before the episode start the first observation repeats (lerobot clamps the same way in training).
HISTORY = json.loads(os.environ["HISTORY"]) if os.environ.get("HISTORY") else None
if HISTORY:
    assert all(k % EXEC == 0 for k in HISTORY["offsets"]), f"history offsets {HISTORY['offsets']} not multiples of H_exec {EXEC}"
TRACE_DIR = os.environ.get("TRACE_DIR")
VIDEO_EVERY = int(os.environ.get("VIDEO_EVERY", "0"))
if TRACE_DIR:
    os.makedirs(TRACE_DIR, exist_ok=True)
units = [u for u in os.environ["EPISODES"].replace("\n", ",").split(",") if u][WK::NW]
env = HE.make_env(gpu=int(os.environ.get("GPU", "0")), split=SPLIT)
sock = socket.create_connection(("127.0.0.1", int(os.environ.get("PORT", "5600"))))
send_msg(sock, {"cmd": "ping"})
srv = recv_msg(sock)
_hk = {f"{HISTORY['camera']}_h{k}" for k in HISTORY["offsets"]} if HISTORY else set()
assert set(srv.get("cams", [])) == set(CAM_MAP) | _hk, f"server cameras {srv.get('cams')} != {sorted(set(CAM_MAP) | _hk)}"
assert 1 <= EXEC <= IO.H_PRED, f"H_exec {EXEC} outside 1..{IO.H_PRED}"


def observe():
    HE.robot(env).update()
    o = env.sim.get_sensor_observations()
    return IO.policy_images(o)


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
    return IO.instruction(ep.object_category, ep.start_recep_category, ep.goal_recep_category)


fn = f"{OUT}/eval_w{WK:02d}.jsonl"
done = {json.loads(l)["unit"] for l in open(fn)} if os.path.exists(fn) else set()
for unit in units:
    if unit in done:
        continue
    ep_id, rep = HE.parse_unit(unit)
    rec = {"unit": unit, "episode": ep_id, "replica": rep, "split": SPLIT, "h_exec": EXEC, "h_pred": IO.H_PRED,
           "seed_offset": SEED_OFFSET}
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
        tr = {"pose": [], "act": [], "ik": [], "dist": [], "frames": [], "frame_steps": []}
        past = []                                               # (image of HISTORY camera, state) per earlier query
        if TRACE_DIR:
            sim = env.sim
            tgt = sim.get_rigid_object_manager().get_object_by_id(sim.scene_obj_ids[int(sim.get_targets()[0][0])])
        while steps < MAX_STEPS and not env.episode_over:
            if k >= EXEC:
                imgs = observe()
                if TRACE_DIR and VIDEO_EVERY and steps % VIDEO_EVERY == 0:
                    tr["frames"].append(np.concatenate([imgs[c][::2, ::2] for c in ("camera_top", "camera_left", "camera_right")], 1))
                    tr["frame_steps"].append(steps)
                st16 = IO.state_from_proprio(rob.proprio16()).astype(np.float32)
                state_in = st16
                if HISTORY:
                    cam = HISTORY["camera"]
                    hist_obs = []
                    for k in HISTORY["offsets"]:
                        j = k // EXEC
                        hist_obs.append(past[-j] if len(past) >= j else (past[0] if past else (imgs[cam], st16)))
                    for k, (im_k, _) in zip(HISTORY["offsets"], hist_obs):
                        imgs[f"{cam}_h{k}"] = im_k
                    if HISTORY.get("state"):
                        state_in = np.concatenate([st16] + [s_k for _, s_k in hist_obs]).astype(np.float32)
                    past.append((imgs[cam], st16))
                send_msg(sock, {"cmd": "infer", "images": imgs, "task": rec["instruction"],
                                "state": state_in,
                                "seed": int(HE.episode_seed(ep_id, rep)) + 1_000_003 * SEED_OFFSET + steps})
                r = recv_msg(sock)
                if "error" in r:
                    raise RuntimeError("policy server: " + r["error"])
                chunk, k = r["actions"], 0
            joint, rep_ik = ex.lower(chunk[k])
            ik_ok = bool(rep_ik["left"]["ik_valid"] and rep_ik["right"]["ik_valid"])
            ik_invalid += int(not ik_ok)
            if TRACE_DIR:
                xz, yaw, _ = NAV.pose2d(env)
                tp = np.array(tgt.translation)
                tr["pose"].append([float(xz[0]), float(xz[1]), float(yaw)])
                tr["act"].append(np.asarray(chunk[k], np.float32))
                tr["ik"].append(ik_ok)
                tr["dist"].append(float(np.hypot(tp[0] - xz[0], tp[2] - xz[1])))
            HE.step18(env, joint)
            steps, k = steps + 1, k + 1
        grasps = [e for e in HE.action_obj(env).events if e.get("type") == "grasp"]
        rec.update({"steps": steps, "ik_invalid_steps": ik_invalid, "episode_over": bool(env.episode_over),
                    "hand": grasps[0]["hand"] if grasps else None, "n_grasps": len(grasps), "ovmm": metrics()})
        rec["funnel"] = {name: rec["ovmm"][k_] for name, k_ in FUNNEL}
        if TRACE_DIR:
            np.savez_compressed(os.path.join("/dev/shm", f"trace_{unit.replace('#', '_r')}.npz"),
                                pose=np.asarray(tr["pose"], np.float32), act=np.asarray(tr["act"], np.float32),
                                ik=np.asarray(tr["ik"]), dist=np.asarray(tr["dist"], np.float32),
                                frames=np.asarray(tr["frames"], np.uint8).reshape(-1, 120, 480, 3),
                                frame_steps=np.asarray(tr["frame_steps"]), funnel=json.dumps(rec["funnel"]))
            import shutil                                     # bosfs rejects seeks while writing: build in /dev/shm
            shutil.copy(os.path.join("/dev/shm", f"trace_{unit.replace('#', '_r')}.npz"), TRACE_DIR)
    except Exception as e:
        rec["engineering_error"] = repr(e)[:300]
        rec["tb"] = traceback.format_exc()[-1200:]
    rec["wall_s"] = round(time.time() - t0, 1)
    with open(fn, "a") as f:
        f.write(json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
    print("EVAL", json.dumps({k_: rec.get(k_) for k_ in ("unit", "scene", "hand", "funnel", "engineering_error")}), flush=True)
env.close()

