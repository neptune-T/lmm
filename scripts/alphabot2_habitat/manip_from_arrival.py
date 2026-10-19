"""Manipulation from expert arrival (Habitat client; diagnostic, no training): does the policy grasp / carry / place
when navigation is already solved?

Job = (unit, entry, mode, seed) from JOBS (manip_plan.py). State restore = the collector's validated fresh-reset
replay: reset_to(unit) + fix_start + the expert's recorded joint18[0:t]; the base pose is checked against the
recorded base_xz_yaw[t] (skip "replay_mismatch" beyond 2 cm / 1 deg); for entry "recep" the target object must be
held after the replay (skip "replay_not_holding"). The head-camera frames at t-15 / t-10 / t-5 of the replay are the
policy's history at its first query (the frames the expert episode really had there; training convention).
  expert_rest  continue replaying the expert's own recorded joint18[t:] (restored state is completable)
  policy       ROLLOUT_STEPS policy steps exactly as eval_policy.py (same server, H_exec, executor, history,
               instruction); noise seeds depend only on (unit, entry, t, seed, step)
Per step: base pose, executed CartesianMobileAction18, IK flag, target held (grasp manager snapped to the target),
target height, max |arm q - NAV_TUCK| per arm, OVMM phase flags; every query a [top|left|right] half-res frame
strip. Row: new grasp / release events during the rollout (expert replay events excluded), outcome flags (see
OUTCOMES), OVMM funnel at the end and "ever" over the rollout. Privileged state is logged only, never sent.
    JOBS=jobs.json OUT=... PORT=... GPU=.. WORKER=.. NWORKERS=.. EXEC_STEPS=5 [HISTORY=json] run_habitat.sh manip_from_arrival.py
"""
import json, math, os, shutil, socket, sys, time, traceback

import numpy as np

REPO = os.environ.get("LMM_REPO", "/workspace/lmm")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "alphabot2_pi05"))
from lmm.embodiments.alphabot2 import cartesian as C  # noqa: E402
from lmm.embodiments.alphabot2 import ovmm_policy_io as IO  # noqa: E402
from lmm.embodiments.alphabot2.habitat import env as HE  # noqa: E402
from lmm.embodiments.alphabot2.habitat import nav as NAV  # noqa: E402
from lmm.embodiments.alphabot2.habitat.constants import NAV_TUCK  # noqa: E402
from wire import recv_msg, send_msg  # noqa: E402

OUT = os.environ["OUT"]; os.makedirs(f"{OUT}/traces", exist_ok=True)
EXEC = int(os.environ.get("EXEC_STEPS", "5"))
STEPS = {"obj": int(os.environ.get("OBJ_STEPS", "600")), "recep": int(os.environ.get("RECEP_STEPS", "500"))}
WK, NW = int(os.environ.get("WORKER", 0)), int(os.environ.get("NWORKERS", 1))
REPLAY_TOL_M, REPLAY_TOL_DEG = 0.02, 1.0
LIFT_M, TUCK_RAD = 0.05, 0.30
HISTORY = json.loads(os.environ["HISTORY"]) if os.environ.get("HISTORY") else None
if HISTORY:
    assert all(k % EXEC == 0 for k in HISTORY["offsets"])
FUNNEL = (("find", "ovmm_find_object_phase_success"), ("grasp", "ovmm_pick_object_phase_success"),
          ("carry", "ovmm_find_recep_phase_success"), ("place", "ovmm_place_object_phase_success"), ("full", "ovmm_place_success"))
plan = json.load(open(os.environ["JOBS"]))
jobs = plan["jobs"][WK::NW]
env = HE.make_env(gpu=int(os.environ.get("GPU", "0")), split=plan.get("split", "train"))
sock = socket.create_connection(("127.0.0.1", int(os.environ.get("PORT", "5600"))))
send_msg(sock, {"cmd": "ping"}); srv = recv_msg(sock)
_hk = {f"{HISTORY['camera']}_h{k}" for k in HISTORY["offsets"]} if HISTORY else set()
assert set(srv.get("cams", [])) == set(IO.CAMERA_MAP) | _hk, srv.get("cams")
TUCK = {s: np.array(v) for s, v in NAV_TUCK.items()}


def observe():
    HE.robot(env).update()
    return IO.policy_images(env.sim.get_sensor_observations())


def target():
    sim = env.sim
    return sim.get_rigid_object_manager().get_object_by_id(sim.scene_obj_ids[int(sim.get_targets()[0][0])])


def held():
    oid = target().object_id
    return any(m.is_grasped and m.snap_idx == oid for m in env.sim.get_agent_data(0).grasp_mgrs)


def phases():
    m = env.get_metrics(); out = {}
    for name, k in FUNNEL:
        v = m.get(k)
        if isinstance(v, dict):
            v = any(bool(x) for x in v.values())
        out[name] = bool(v) if v is not None else False
    return out


def job_id(j):
    return f"{j['unit'].replace('#', '_r')}_{j['entry']}_{j['mode']}_s{j['seed']}"


def restore(j):
    ep_id, rep = HE.parse_unit(j["unit"])
    low, md = np.load(f"{j['source']}/lowdim.npz"), np.load(f"{j['source']}/metadata.npz")
    HE.reset_to(env, ep_id, rep); HE.fix_start(env)
    t = j["t"]
    hist_steps = sorted({t - k for k in (HISTORY["offsets"] if HISTORY else []) if t - k >= 0})
    past = []
    for i, a in enumerate(low["joint18"][:t]):
        if i in hist_steps:
            past.append((observe()[HISTORY["camera"]], None))
        HE.step18(env, a)
    xz, yaw, _ = NAV.pose2d(env)
    exz, eyaw = md["base_xz_yaw"][t, :2], md["base_xz_yaw"][t, 2]
    err = {"pos_m": float(np.linalg.norm(xz - exz)), "yaw_deg": float(abs(math.degrees(NAV.wrap(yaw - eyaw))))}
    return low, past, err


def step_record(tr, cart18, ik_ok, h0):
    rob = HE.robot(env)
    xz, yaw, _ = NAV.pose2d(env)
    tp = np.array(target().translation)
    tr["pose"].append([float(xz[0]), float(xz[1]), float(yaw)])
    tr["act"].append(np.asarray(cart18, np.float32) if cart18 is not None else np.full(18, np.nan, np.float32))
    tr["ik"].append(ik_ok)
    tr["held"].append(held())
    tr["obj_dh"].append(float(tp[1] - h0))
    tr["tuck_err"].append([float(np.max(np.abs(rob.arm_q(s) - TUCK[s]))) for s in ("left", "right")])
    tr["dist"].append(float(np.hypot(tp[0] - xz[0], tp[2] - xz[1])))
    ph = phases()
    tr["phase"].append([ph[n] for n, _ in FUNNEL])


def run(j, low, past):
    rob = HE.robot(env); ep = env.current_episode
    task = IO.instruction(ep.object_category, ep.start_recep_category, ep.goal_recep_category)
    tr = {k: [] for k in ("pose", "act", "ik", "held", "obj_dh", "tuck_err", "dist", "phase", "frames", "frame_steps")}
    n_ev0 = len(HE.action_obj(env).events)
    h0 = float(np.array(target().translation)[1])
    held0 = held()
    steps, limit = 0, STEPS[j["entry"]]
    if j["mode"] == "expert_rest":
        for a in low["joint18"][j["t"]:]:
            if env.episode_over:
                break
            HE.step18(env, a); step_record(tr, None, True, h0); steps += 1
    else:
        ep_id, rep = HE.parse_unit(j["unit"])
        seed0 = int(HE.episode_seed(ep_id, rep)) + 1_000_003 * int(j["seed"]) + 7919 * int(j["t"]) + (0 if j["entry"] == "obj" else 104_729)
        ex = C.CartesianExecutor.from_proprio(rob.proprio16())
        chunk, k = None, EXEC
        while steps < limit and not env.episode_over:
            if k >= EXEC:
                imgs = observe()
                tr["frames"].append(np.concatenate([imgs[c][::2, ::2] for c in ("camera_top", "camera_left", "camera_right")], 1))
                tr["frame_steps"].append(steps)
                st16 = IO.state_from_proprio(rob.proprio16()).astype(np.float32)
                if HISTORY:
                    cam = HISTORY["camera"]
                    hist = []
                    for kk in HISTORY["offsets"]:
                        jj = kk // EXEC
                        hist.append(past[-jj] if len(past) >= jj else (past[0] if past else (imgs[cam], st16)))
                    for kk, (im, _) in zip(HISTORY["offsets"], hist):
                        imgs[f"{cam}_h{kk}"] = im
                    past.append((imgs[cam], st16))
                send_msg(sock, {"cmd": "infer", "images": imgs, "task": task, "state": st16, "seed": seed0 + steps})
                r = recv_msg(sock)
                if "error" in r:
                    raise RuntimeError("policy server: " + r["error"])
                chunk, k = r["actions"], 0
            joint, rep_ik = ex.lower(chunk[k])
            HE.step18(env, joint)
            step_record(tr, chunk[k], bool(rep_ik["left"]["ik_valid"] and rep_ik["right"]["ik_valid"]), h0)
            steps, k = steps + 1, k + 1
    ev = HE.action_obj(env).events[n_ev0:]
    return tr, ev, task, held0, steps


def outcomes(j, tr, ev, held0):
    H = np.asarray(tr["held"], bool); dh = np.asarray(tr["obj_dh"]); te = np.asarray(tr["tuck_err"]).reshape(-1, 2)
    P = np.asarray(tr["phase"], bool).reshape(-1, len(FUNNEL)); pose = np.asarray(tr["pose"]).reshape(-1, 3)
    o = {"n_grasp_events": sum(e.get("type") == "grasp" for e in ev), "n_release_events": sum(e.get("type") == "release" for e in ev),
         "held_end": bool(H[-1]) if len(H) else False}
    if j["entry"] == "obj":
        g = np.flatnonzero(H)
        o["grasp"] = bool(len(g))
        o["grasp_step"] = int(g[0]) if len(g) else None
        lift = H & (dh >= LIFT_M)
        o["lift"] = bool(lift.any())
        carry = lift & (te.max(1) <= TUCK_RAD)
        o["carry_ready"] = bool(carry.any())                         # held, lifted, both arms back near the NAV tuck
        o["carry_ready_step"] = int(np.flatnonzero(carry)[0]) if carry.any() else None
        if carry.any():                                              # base motion after carry-ready while still holding
            c0 = int(np.flatnonzero(carry)[0])
            o["held_travel_m_after_carry_ready"] = float(np.sum(np.linalg.norm(np.diff(pose[c0:, :2], axis=0), axis=1) * H[c0 + 1:]))
        o["dropped"] = bool(len(g) and not H[-1])
    else:
        o["held_at_start"] = bool(held0)
        o["released"] = o["n_release_events"] > 0
    o["ever"] = {n: bool(P[:, i].any()) for i, (n, _) in enumerate(FUNNEL)} if len(P) else {}
    o["end"] = {n: bool(P[-1, i]) for i, (n, _) in enumerate(FUNNEL)} if len(P) else {}
    o["min_tuck_err_rad"] = float(te.max(1).min()) if len(te) else None
    o["max_obj_dh_m"] = float(dh.max()) if len(dh) else None
    return o


fn = f"{OUT}/manip_w{WK:02d}.jsonl"
done = {json.loads(l)["job"] for l in open(fn)} if os.path.exists(fn) else set()
for j in jobs:
    jid = job_id(j)
    if jid in done:
        continue
    rec = {"job": jid, **j, "h_exec": EXEC, "policy_ckpt": os.environ.get("POLICY_CKPT")}
    t0 = time.time()
    try:
        low, past, err = restore(j)
        rec["replay"] = err
        if err["pos_m"] > REPLAY_TOL_M or err["yaw_deg"] > REPLAY_TOL_DEG:
            raise ValueError("replay_mismatch")
        if j["entry"] == "recep" and not held():
            raise ValueError("replay_not_holding")
        if j["entry"] == "obj" and held():
            raise ValueError("replay_already_holding")
        rec["dist_to_target_at_entry_m"] = float(np.hypot(*(np.array(target().translation)[[0, 2]] - NAV.pose2d(env)[0])))
        rec["phase_at_entry"] = phases()
        tr, ev, task, held0, steps = run(j, low, past)
        rec.update({"instruction": task, "steps": steps, "episode_over": bool(env.episode_over), "events": ev,
                    "outcome": outcomes(j, tr, ev, held0)})
        tmp = f"/dev/shm/mtrace_{WK}_{jid}.npz"
        fr = np.asarray(tr.pop("frames"), np.uint8).reshape(-1, 120, 480, 3)
        np.savez_compressed(tmp, frames=fr, **{k: np.asarray(v) for k, v in tr.items()})
        dst = f"{OUT}/traces/{jid}.npz"
        shutil.copy(tmp, dst); os.remove(tmp); rec["trace_out"] = dst
    except ValueError as e:
        rec["skipped"] = str(e)[:300]
    except Exception as e:
        rec["engineering_error"] = repr(e)[:300]; rec["tb"] = traceback.format_exc()[-1200:]
    rec["wall_s"] = round(time.time() - t0, 1)
    with open(fn, "a") as f:
        f.write(json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
    print("MANIP", json.dumps({k: rec.get(k) for k in ("job", "replay", "outcome", "skipped", "engineering_error", "wall_s")}), flush=True)
env.close()
