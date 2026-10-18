"""Expert-recovery feasibility pilot (Habitat client; no training). Job = (root, mode) from ROOTS (recovery_roots.py).

Root reconstruction (exactly the original v2-B rollout of perturb_eval.py, level "none"): reset_to(unit) + fix_start
+ the expert's recorded joint18[0:t] + the same zero-offset teleport and contact pass as perturb_eval.perturb() +
B's RECORDED executed CartesianMobileAction18 rows [0:r] lowered through a fresh CartesianExecutor (observing at
the original query steps, which also rebuilds B's real history frames). The reached pose is checked against the
trace pose at r (skip as "root_mismatch" beyond 2 cm / 1 deg). Then one of:
  expert       OVMMExpert(env).run() from the root: the privileged expert re-plans arm, standoff, path, grasp and
               place from the ACTUAL state; navigation with arms tucked (nav.follow: shortest path, back-off/replan).
               Every executed step is recorded with its stage and Cartesian label (the collector's label path).
  policy_orig  v2-B fp32 continuation for POLICY_STEPS with the ORIGINAL rollout noise seeds (reproduction check)
  policy_alt   the same with a fresh noise seed
Per step: base pose, executed / labelled CartesianMobileAction18, distance to the target object, robot-scene
contact count (nav.scene_contacts); end: OVMM funnel measures. Provenance in every row.
    ROOTS=roots.json OUT=... PORT=... GPU=.. WORKER=.. NWORKERS=.. [HISTORY=json] run_habitat.sh recovery_eval.py
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
from lmm.embodiments.alphabot2.habitat.expert import OVMMExpert, ExpertFailure  # noqa: E402
from wire import recv_msg, send_msg  # noqa: E402

OUT = os.environ["OUT"]; os.makedirs(f"{OUT}/traces", exist_ok=True)
EXEC, POLICY_STEPS = 5, int(os.environ.get("POLICY_STEPS", "600"))
WK, NW = int(os.environ.get("WORKER", 0)), int(os.environ.get("NWORKERS", 1))
HISTORY = json.loads(os.environ["HISTORY"]) if os.environ.get("HISTORY") else None
FUNNEL = (("find", "ovmm_find_object_phase_success"), ("grasp", "ovmm_pick_object_phase_success"),
          ("carry", "ovmm_find_recep_phase_success"), ("place", "ovmm_place_object_phase_success"), ("full", "ovmm_place_success"))
plan = json.load(open(os.environ["ROOTS"]))
MODES = tuple(os.environ.get("MODES", "expert,policy_orig,policy_alt").split(","))
POLICY_CKPT = os.environ.get("POLICY_CKPT")                             # recorded for provenance (the server loads it)
jobs = [(r, m) for r in plan["roots"] for m in MODES][WK::NW]
env = HE.make_env(gpu=int(os.environ.get("GPU", "0")), split="train")
sock = socket.create_connection(("127.0.0.1", int(os.environ.get("PORT", "5600"))))
send_msg(sock, {"cmd": "ping"}); srv = recv_msg(sock)


def observe():
    HE.robot(env).update()
    return IO.policy_images(env.sim.get_sensor_observations())


def target():
    sim = env.sim
    return sim.get_rigid_object_manager().get_object_by_id(sim.scene_obj_ids[int(sim.get_targets()[0][0])])


def tdist():
    xz, _, _ = NAV.pose2d(env); tp = np.array(target().translation)
    return float(np.hypot(tp[0] - xz[0], tp[2] - xz[1]))


def metrics():
    m = env.get_metrics(); out = {}
    for name, k in FUNNEL:
        v = m.get(k)
        if isinstance(v, dict):
            v = any(bool(x) for x in v.values())
        out[name] = float(bool(v)) if v is not None else 0.0
    return out


def reconstruct(root):
    ep_id, rep = HE.parse_unit(root["unit"])
    low = np.load(f"{root['source']}/lowdim.npz")
    tr = np.load(root["trace"]); act, pose = tr["act"], tr["pose"]
    HE.reset_to(env, ep_id, rep); HE.fix_start(env)
    for a in low["joint18"][: root["t"]]:
        HE.step18(env, a)
    rob = HE.robot(env)
    observe()                                                           # perturb_eval.perturb("none"): same calls
    T0 = rob.sim_obj.transformation
    rob.sim_obj.transformation = mn.Matrix4.from_(T0.rotation(), T0.translation); rob.update()
    NAV.scene_contacts(env)
    observe()
    ex = C.CartesianExecutor.from_proprio(rob.proprio16())
    past = []
    for k in range(root["r"]):
        if k % EXEC == 0:
            imgs = observe()
            past.append((imgs[HISTORY["camera"]] if HISTORY else None, IO.state_from_proprio(rob.proprio16()).astype(np.float32)))
        joint, _ = ex.lower(act[k])
        HE.step18(env, joint)
    xz, yaw, _ = NAV.pose2d(env)
    err = {"pos_m": float(np.linalg.norm(xz - pose[root["r"], :2])),
           "yaw_deg": float(abs(math.degrees(NAV.wrap(yaw - pose[root["r"], 2]))))}
    return ex, past, err, pose


def run_policy(root, ex, past, seed_extra):
    rob = HE.robot(env); ep = env.current_episode
    task = IO.instruction(ep.object_category, ep.start_recep_category, ep.goal_recep_category)
    ep_id, rep = HE.parse_unit(root["unit"])
    seed0 = int(HE.episode_seed(ep_id, rep)) + 1_000_003 * int(root["seed"]) + 7919 * int(root["t"]) + seed_extra
    tr = {"pose": [], "act": [], "dist": [], "contacts": []}
    chunk, k, steps = None, EXEC, root["r"]
    while steps < root["r"] + POLICY_STEPS and not env.episode_over:
        if k >= EXEC:
            imgs = observe(); st16 = IO.state_from_proprio(rob.proprio16()).astype(np.float32)
            if HISTORY:
                cam = HISTORY["camera"]
                hist = [past[-(kk // EXEC)] if len(past) >= kk // EXEC else past[0] for kk in HISTORY["offsets"]]
                for kk, (im, _) in zip(HISTORY["offsets"], hist):
                    imgs[f"{cam}_h{kk}"] = im
                past.append((imgs[cam], st16))
            send_msg(sock, {"cmd": "infer", "images": imgs, "task": task, "state": st16, "seed": seed0 + steps})
            r = recv_msg(sock)
            if "error" in r:
                raise RuntimeError("policy server: " + r["error"])
            chunk, k = r["actions"], 0
        joint, _ = ex.lower(chunk[k])
        xz, yaw, _ = NAV.pose2d(env)
        tr["pose"].append([float(xz[0]), float(xz[1]), float(yaw)]); tr["act"].append(np.asarray(chunk[k], np.float32))
        tr["dist"].append(tdist())
        HE.step18(env, joint)
        tr["contacts"].append(len(NAV.scene_contacts(env)[0]))
        steps, k = steps + 1, k + 1
    return tr


def run_expert():
    tr = {"pose": [], "act": [], "dist": [], "stage": [], "ik_ok": [], "contacts": []}

    def on_step(cart18, stage, rep):
        xz, yaw, _ = NAV.pose2d(env)
        tr["pose"].append([float(xz[0]), float(xz[1]), float(yaw)]); tr["act"].append(np.asarray(cart18, np.float32))
        tr["dist"].append(tdist()); tr["stage"].append(stage)
        tr["ik_ok"].append(bool(rep["left"]["ik_valid"] and rep["right"]["ik_valid"]))
    ex = OVMMExpert(env, on_step=on_step)
    status = "completed"
    try:
        ex.run()
    except ExpertFailure as e:
        status = f"expert_failure: {e}"[:300]
    return tr, status, {k: ex.log.get(k) for k in ("arm", "arm_choice", "nav_to_object", "ik_invalid_steps")}, \
        [(int(t), str(s)) for t, s in ex.events]


fn = f"{OUT}/recovery_w{WK:02d}.jsonl"
done = {json.loads(l)["job"] for l in open(fn)} if os.path.exists(fn) else set()
for root, mode in jobs:
    jid = f"{root['root_id']}:{mode}"
    if jid in done:
        continue
    rec = {"job": jid, "mode": mode, **{k: root[k] for k in ("root_id", "category", "unit", "base_episode", "group", "t", "seed", "r",
                                                            "job", "trace", "trace_sha256", "source", "trigger_step")}}
    rec["job"] = jid; rec["source_job"] = root["job"]; rec["policy_ckpt"] = POLICY_CKPT
    t0 = time.time()
    try:
        ex, past, err, orig_pose = reconstruct(root)
        rec["root_check"] = err
        if err["pos_m"] > 0.02 or err["yaw_deg"] > 1.0:
            raise ValueError("root_mismatch")
        rec["root_dist_to_target_m"] = tdist()
        if mode == "expert":
            tr, status, log, events = run_expert()
            rec.update({"expert_status": status, "expert_log": log, "expert_events": events})
        else:
            tr = run_policy(root, ex, past, 0 if mode == "policy_orig" else 1_000_003 * 97)
            if mode == "policy_orig":                                    # reproduction of the original continuation
                n = min(50, len(tr["pose"]), len(orig_pose) - root["r"])
                rec["reproduction_max_pos_err_m_first50"] = float(np.max(np.linalg.norm(
                    np.asarray(tr["pose"])[:n, :2] - orig_pose[root["r"]:root["r"] + n, :2], axis=1))) if n else None
        rec["steps"] = len(tr["pose"]); rec["funnel_end"] = metrics()
        tmp = f"/dev/shm/rtrace_{WK}_{abs(hash(jid))}.npz"
        np.savez_compressed(tmp, **{k: np.asarray(v) for k, v in tr.items()})
        import shutil
        dst = f"{OUT}/traces/{jid.replace(':', '__')}.npz"
        shutil.copy(tmp, dst); os.remove(tmp); rec["trace_out"] = dst
    except ValueError as e:
        rec["skipped"] = str(e)
    except Exception as e:
        rec["engineering_error"] = repr(e)[:300]; rec["tb"] = traceback.format_exc()[-1200:]
    rec["wall_s"] = round(time.time() - t0, 1)
    with open(fn, "a") as f:
        f.write(json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
    print("RECOVERY", json.dumps({k: rec.get(k) for k in ("job", "root_check", "expert_status", "steps", "funnel_end",
                                                          "reproduction_max_pos_err_m_first50", "skipped", "engineering_error")}), flush=True)
env.close()
