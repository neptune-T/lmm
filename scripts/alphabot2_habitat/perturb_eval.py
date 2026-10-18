"""Matched-state perturbation rollouts (Habitat client; causal NAV diagnostic, no training).

Each job = (unit, t, level, seed) from JOBS (perturb_plan.py):
  1. reproducible expert NAV state: reset_to(unit) + fix_start + the unit's recorded joint18[0:t] (the
     collector's validated fresh-reset replay); the reached base pose is checked against the recorded
     base_xz_yaw[t] (job skipped as "replay_mismatch" beyond REPLAY_TOL_M / REPLAY_TOL_DEG);
  2. safe perturbation of that state: "none" (same code path, zero offset), "pos" (lateral offset d metres
     perpendicular to the heading; + side first, - side if unsafe) or "yaw" (rotation by deg about +y; + first).
     Safe = navigable on the AlphaBot navmesh, same navmesh island as the unperturbed pose, and no robot-scene
     contact after the teleport (nav.scene_contacts); a level unsafe on both sides is recorded and skipped;
  3. ROLLOUT_STEPS policy steps exactly as eval_policy.py (same server, H_exec, executor, history convention:
     history before the rollout start repeats the first observation, as at an episode start), with policy noise
     seeds that depend only on (unit, t, seed, step) -- identical across perturbation levels (matched).
Trace per job: per-step base pose, executed CartesianMobileAction18, IK flag, distance to the target object;
plus the replay check, the applied perturbation (measured) and safety facts in the jsonl row.
    JOBS=jobs.json OUT=... TRACE_DIR=... PORT=... GPU=... WORKER=.. NWORKERS=.. EXEC_STEPS=5 [HISTORY=json] \
        run_habitat.sh perturb_eval.py
"""
import json, math, os, socket, sys, time, traceback

import numpy as np

REPO = os.environ.get("LMM_REPO", "/workspace/lmm")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "alphabot2_pi05"))
import magnum as mn  # noqa: E402
from lmm.embodiments.alphabot2 import cartesian as C  # noqa: E402
from lmm.embodiments.alphabot2 import ovmm_policy_io as IO  # noqa: E402
from lmm.embodiments.alphabot2.habitat import env as HE  # noqa: E402
from lmm.embodiments.alphabot2.habitat import nav as NAV  # noqa: E402
from wire import recv_msg, send_msg  # noqa: E402

OUT, TRACE_DIR = os.environ["OUT"], os.environ["TRACE_DIR"]
os.makedirs(OUT, exist_ok=True); os.makedirs(TRACE_DIR, exist_ok=True)
EXEC = int(os.environ.get("EXEC_STEPS", "5"))
ROLLOUT = int(os.environ.get("ROLLOUT_STEPS", "400"))
WK, NW = int(os.environ.get("WORKER", 0)), int(os.environ.get("NWORKERS", 1))
REPLAY_TOL_M, REPLAY_TOL_DEG = 0.02, 1.0
HISTORY = json.loads(os.environ["HISTORY"]) if os.environ.get("HISTORY") else None
if HISTORY:
    assert all(k % EXEC == 0 for k in HISTORY["offsets"])
plan = json.load(open(os.environ["JOBS"]))
jobs = plan["jobs"][WK::NW]
env = HE.make_env(gpu=int(os.environ.get("GPU", "0")), split=plan.get("split", "train"))
sock = socket.create_connection(("127.0.0.1", int(os.environ.get("PORT", "5600"))))
send_msg(sock, {"cmd": "ping"})
srv = recv_msg(sock)
_hk = {f"{HISTORY['camera']}_h{k}" for k in HISTORY["offsets"]} if HISTORY else set()
assert set(srv.get("cams", [])) == set(IO.CAMERA_MAP) | _hk, srv.get("cams")


def observe():
    HE.robot(env).update()
    return IO.policy_images(env.sim.get_sensor_observations())


def job_id(j):
    return f"{j['unit'].replace('#', '_r')}_t{j['t']}_{j['level']}_s{j['seed']}"


def set_pose(rob, R, p):
    rob.sim_obj.transformation = mn.Matrix4.from_(R, p)
    rob.update()


def perturb(level):
    """apply a safe perturbation; returns the record (or raises ValueError('unsafe'))"""
    rob = HE.robot(env); pf = env.sim.pathfinder
    T0 = rob.sim_obj.transformation
    R0, p0 = T0.rotation(), T0.translation
    xz0, yaw0, _ = NAV.pose2d(env)
    island0 = pf.get_island(p0)
    kind, mag = level["kind"], float(level.get("mag", 0.0))
    tried = []
    for sign in ((1.0,) if kind == "none" else (1.0, -1.0)):
        if kind == "pos":
            left = np.array([-math.sin(yaw0), -math.cos(yaw0)])          # +90 deg from the heading in (x, z)
            off = sign * mag * left
            p = mn.Vector3(p0[0] + off[0], p0[1], p0[2] + off[1]); R = R0
        elif kind == "yaw":
            p = p0; R = mn.Matrix4.rotation_y(mn.Rad(sign * math.radians(mag))).rotation() @ R0
        else:
            p, R = p0, R0
        nav_ok = bool(pf.is_navigable(p)) and pf.get_island(p) == island0
        coll = []
        if nav_ok:
            set_pose(rob, R, p)
            coll, _ = NAV.scene_contacts(env)
        tried.append({"sign": sign, "navigable_same_island": nav_ok, "contacts": len(coll)})
        if nav_ok and not coll:
            xz1, yaw1, _ = NAV.pose2d(env)
            return {"sign": sign, "tried": tried, "moved_m": float(np.linalg.norm(xz1 - xz0)),
                    "dyaw_deg": float(math.degrees(NAV.wrap(yaw1 - yaw0)))}
        set_pose(rob, R0, p0)                                             # undo before the other side
    raise ValueError(json.dumps({"unsafe": tried}))


fn = f"{OUT}/perturb_w{WK:02d}.jsonl"
done = {json.loads(l)["job"] for l in open(fn)} if os.path.exists(fn) else set()
for j in jobs:
    jid = job_id(j)
    if jid in done:
        continue
    rec = {"job": jid, **j, "h_exec": EXEC, "rollout_steps": ROLLOUT}
    t0 = time.time()
    try:
        ep_id, rep = HE.parse_unit(j["unit"])
        src = j["source"]
        low, md = np.load(f"{src}/lowdim.npz"), np.load(f"{src}/metadata.npz")
        HE.reset_to(env, ep_id, rep)
        rec["start_fix"] = HE.fix_start(env)
        for a in low["joint18"][: j["t"]]:                                # the expert's own executed actions
            HE.step18(env, a)
        xz, yaw, _ = NAV.pose2d(env)
        exz, eyaw = md["base_xz_yaw"][j["t"], :2], md["base_xz_yaw"][j["t"], 2]
        rec["replay"] = {"pos_err_m": float(np.linalg.norm(xz - exz)), "yaw_err_deg": float(abs(math.degrees(NAV.wrap(yaw - eyaw))))}
        if rec["replay"]["pos_err_m"] > REPLAY_TOL_M or rec["replay"]["yaw_err_deg"] > REPLAY_TOL_DEG:
            raise ValueError("replay_mismatch")
        rec["perturbation"] = perturb(j["level_spec"])
        ep = env.current_episode
        task = IO.instruction(ep.object_category, ep.start_recep_category, ep.goal_recep_category)
        rob = HE.robot(env)
        ex = C.CartesianExecutor.from_proprio(rob.proprio16())
        sim = env.sim
        tgt = sim.get_rigid_object_manager().get_object_by_id(sim.scene_obj_ids[int(sim.get_targets()[0][0])])
        tr = {"pose": [], "act": [], "ik": [], "dist": []}
        past, chunk, k, steps = [], None, EXEC, 0
        seed0 = int(HE.episode_seed(ep_id, rep)) + 1_000_003 * int(j["seed"]) + 7919 * int(j["t"])
        while steps < ROLLOUT and not env.episode_over:
            if k >= EXEC:
                imgs = observe()
                st16 = IO.state_from_proprio(rob.proprio16()).astype(np.float32)
                state_in = st16
                if HISTORY:
                    cam = HISTORY["camera"]
                    hist_obs = [past[-(kk // EXEC)] if len(past) >= kk // EXEC else (past[0] if past else (imgs[cam], st16))
                                for kk in HISTORY["offsets"]]
                    for kk, (im_k, _) in zip(HISTORY["offsets"], hist_obs):
                        imgs[f"{cam}_h{kk}"] = im_k
                    if HISTORY.get("state"):
                        state_in = np.concatenate([st16] + [s_k for _, s_k in hist_obs]).astype(np.float32)
                    past.append((imgs[cam], st16))
                send_msg(sock, {"cmd": "infer", "images": imgs, "task": task, "state": state_in, "seed": seed0 + steps})
                r = recv_msg(sock)
                if "error" in r:
                    raise RuntimeError("policy server: " + r["error"])
                chunk, k = r["actions"], 0
            joint, rep_ik = ex.lower(chunk[k])
            xz, yaw, _ = NAV.pose2d(env)
            tp = np.array(tgt.translation)
            tr["pose"].append([float(xz[0]), float(xz[1]), float(yaw)])
            tr["act"].append(np.asarray(chunk[k], np.float32))
            tr["ik"].append(bool(rep_ik["left"]["ik_valid"] and rep_ik["right"]["ik_valid"]))
            tr["dist"].append(float(np.hypot(tp[0] - xz[0], tp[2] - xz[1])))
            HE.step18(env, joint)
            steps, k = steps + 1, k + 1
        rec["steps"], rec["episode_over"] = steps, bool(env.episode_over)
        tmp = os.path.join("/dev/shm", f"ptrace_{WK}_{jid}.npz")
        np.savez_compressed(tmp, pose=np.asarray(tr["pose"], np.float32), act=np.asarray(tr["act"], np.float32),
                            ik=np.asarray(tr["ik"]), dist=np.asarray(tr["dist"], np.float32))
        import shutil                                                    # bosfs: build in /dev/shm, then copy
        shutil.copy(tmp, os.path.join(TRACE_DIR, f"{jid}.npz")); os.remove(tmp)
    except ValueError as e:
        rec["skipped"] = str(e)[:600]
    except Exception as e:
        rec["engineering_error"] = repr(e)[:300]
        rec["tb"] = traceback.format_exc()[-1200:]
    rec["wall_s"] = round(time.time() - t0, 1)
    with open(fn, "a") as f:
        f.write(json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
    print("PERTURB", json.dumps({k_: rec.get(k_) for k_ in ("job", "replay", "perturbation", "skipped", "engineering_error", "wall_s")}), flush=True)
env.close()
