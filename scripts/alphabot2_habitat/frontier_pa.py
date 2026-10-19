"""Manipulation frontier feasibility pilot: exact root restore + matched noise rollouts under the FROZEN PA-OG + base-lock
contract with the FROZEN v2-B policy (Habitat client; no training, no RL). Controller = pa_eval.py's default contract
(MANIP1_MAX 600, MANIP2_MAX 500, SETTLE 120, HOLD_GAP 0.05, TUCK_RAD 0.30, HOLD_K 3, H_exec 5, FRAME_EVERY 2 render
schedule, base lock), ported into a resumable form; nothing about the policy, the switches or the planner changes.

Determinism requirements (enforced): every policy response must come back with batch_size == 1 (serve with
policy_server.py --max-batch 1); only v2 TRAIN-manifest units on the OVMM train split (held-out v2 SFT-val units and
OVMM-val are refused in code).

MODE=source   PA episodes from normal OVMM init. Besides the pa_eval trace, every side-effecting simulator call is
              logged in order (EV: 0 = env step with JointMobileAction18 #i, 1 = render (robot.update + sensors),
              2 = contact query (perform_discrete_collision_detection)), and at every policy query a controller
              snapshot is stored: event index, phase, phase step, consecutive-condition count, executor IK seed and
              quaternion memory, commanded grippers, grasp rotation, the event indices of the history frames, seed base.
              Per step (privileged, logged only, for root labelling): held, target height above its start, nearest
              gripper-pad-to-target distance, max |arm q - NAV tuck|.
MODE=rollout  JOBS (frontier_plan.py): (root, noise index j). Restore = fresh reset_to + fix_start + exact replay of the
              source event log up to the root's query (history frames re-rendered at their original events), controller
              state from the snapshot; then the controller continues from that query to the end of the episode. Noise:
              query seed = seed_base + t, seed_base shifted by NOISE_STRIDE * (j + 1) for j >= 0 (j = -1 reproduces the
              source exactly: the restore check). Row: OVMM funnel at the end, what was reached after the root,
              steps, wall time; trace: event log + joints of the whole episode (bitwise comparisons).
    MODE=source UNITS=u1,u2 SEED=100 OUT=... PORT=... GPU=.. WORKER=.. NWORKERS=.. HISTORY=json run_habitat.sh frontier_pa.py
    MODE=rollout JOBS=jobs.json OUT=... PORT=... GPU=.. WORKER=.. NWORKERS=.. HISTORY=json run_habitat.sh frontier_pa.py
"""
import glob, json, os, shutil, socket, sys, time, traceback

import numpy as np

REPO = os.environ.get("LMM_REPO", "/workspace/lmm")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "alphabot2_pi05"))
from lmm.embodiments.alphabot2 import cartesian as C  # noqa: E402
from lmm.embodiments.alphabot2 import ovmm_policy_io as IO  # noqa: E402
from lmm.embodiments.alphabot2.habitat import env as HE  # noqa: E402
from lmm.embodiments.alphabot2.habitat import nav as NAV  # noqa: E402
from lmm.embodiments.alphabot2.habitat.constants import NAV_TUCK  # noqa: E402
from lmm.embodiments.alphabot2.habitat.expert import OVMMExpert, ExpertFailure, SIDES, PAD_MID_GB  # noqa: E402
from wire import recv_msg, send_msg  # noqa: E402

MODE = os.environ["MODE"]
OUT = os.environ["OUT"]; os.makedirs(f"{OUT}/traces", exist_ok=True)
EXEC, MANIP1_MAX, MANIP2_MAX, SETTLE_STEPS = 5, 600, 500, 120          # frozen PA-OG default contract
HOLD_GAP, TUCK_RAD, HOLD_K, FRAME_EVERY = 0.05, 0.30, 3, 2
STORE_EVERY = int(os.environ.get("STORE_EVERY", "2" if MODE == "source" else "10"))   # stored frames only
NOISE_STRIDE = 1_000_003 * 7919
WK, NW = int(os.environ.get("WORKER", 0)), int(os.environ.get("NWORKERS", 1))
HISTORY = json.loads(os.environ["HISTORY"])
assert all(k % EXEC == 0 for k in HISTORY["offsets"])
FUNNEL = (("find", "ovmm_find_object_phase_success"), ("grasp", "ovmm_pick_object_phase_success"),
          ("carry", "ovmm_find_recep_phase_success"), ("place", "ovmm_place_object_phase_success"), ("full", "ovmm_place_success"))
TUCK = {s: np.array(v) for s, v in NAV_TUCK.items()}
TRAIN_UNITS = set(json.load(open(os.environ.get("MANIFEST", "/workspace/runtime/runs/manifests/v2.json")))["train"])


def allowed(unit):
    if unit not in TRAIN_UNITS:
        raise SystemExit(f"refused: {unit} is not a v2 TRAIN-manifest unit (held-out / val never enter the pilot)")


# ------------------------------------------------------------------ side-effect event log
EV, JOINTS = [], []
_step18, _contacts = HE.step18, NAV.scene_contacts


def _rec_step18(env_, joint18):
    EV.append(0); JOINTS.append(np.asarray(joint18, np.float32).reshape(18).copy())
    return _step18(env_, joint18)


def _rec_contacts(env_):
    EV.append(2)
    return _contacts(env_)


HE.step18, NAV.scene_contacts = _rec_step18, _rec_contacts
env = HE.make_env(gpu=int(os.environ.get("GPU", "0")), split="train")
sock = socket.create_connection(("127.0.0.1", int(os.environ.get("PORT", "5600"))))
send_msg(sock, {"cmd": "ping"}); srv = recv_msg(sock)
assert set(srv.get("cams", [])) == set(IO.CAMERA_MAP) | {f"{HISTORY['camera']}_h{k}" for k in HISTORY["offsets"]}, srv.get("cams")
SERVER_S = [0.0]


def observe():
    EV.append(1)
    HE.robot(env).update()
    return IO.policy_images(env.sim.get_sensor_observations())


def infer(imgs, task, st16, seed):
    send_msg(sock, {"cmd": "infer", "images": imgs, "task": task, "state": st16, "seed": int(seed)})
    r = recv_msg(sock)
    if "error" in r:
        raise RuntimeError("policy server: " + r["error"])
    if r.get("batch_size") != 1:
        raise RuntimeError(f"non-deterministic serving: batch_size {r.get('batch_size')} (need --max-batch 1)")
    SERVER_S[0] += float(r.get("latency_s", 0.0))
    return r["actions"]


def target():
    sim = env.sim
    return sim.get_rigid_object_manager().get_object_by_id(sim.scene_obj_ids[int(sim.get_targets()[0][0])])


def held_priv():
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


def pad_dist():
    """privileged, logged only: nearest gripper pad midpoint to the target centre (m)"""
    rob, tp = HE.robot(env), np.array(target().translation)
    d = []
    for s in SIDES:
        T = np.asarray(rob.link_frame(f"{s}_gripper_base"))
        d.append(float(np.linalg.norm((T @ np.r_[PAD_MID_GB, 1.0])[:3] - tp)))
    return min(d)


class Run:
    def __init__(self, unit):
        self.unit, self.phase = unit, "nav1"
        self.head = []                                     # (t, ev, img) render buffer (history at a phase start)
        self.tr = {k: [] for k in ("phase", "act", "held", "obj_dh", "pad_d", "tuck_err", "frames", "frame_steps")}
        self.grip_cmd = {"left": 0.0, "right": 0.0}
        self.ever = {n: False for n, _ in FUNNEL}
        self.grasp_R = None
        self.snaps = []
        self.h0 = None
        self.after = {"carry_ready": False, "manip2": False, "released_in_manip2": False}

    @property
    def t(self):
        return len(JOINTS)

    def record(self, cart18):
        """after every env step (pa_eval.Run.record: same render schedule)"""
        t = self.t - 1                                     # pa_eval's self.t before its increment
        rob = HE.robot(env)
        self.tr["phase"].append(self.phase); self.tr["act"].append(np.asarray(cart18, np.float32))
        self.tr["held"].append(held_priv()); self.tr["obj_dh"].append(float(np.array(target().translation)[1] - self.h0))
        self.tr["pad_d"].append(pad_dist())
        self.tr["tuck_err"].append([float(np.max(np.abs(rob.arm_q(s) - TUCK[s]))) for s in SIDES])
        if t % FRAME_EVERY == 0 or t % EXEC == 0 or len(self.head) < 1:
            ev = len(EV); imgs = observe()
            if t % STORE_EVERY == 0:
                self.tr["frames"].append(np.concatenate([imgs[c][::2, ::2] for c in ("camera_top", "camera_left", "camera_right")], 1))
                self.tr["frame_steps"].append(t)
            self.head.append((t, ev, imgs["camera_top"]))
            self.head = self.head[-24:]
        for n, v in phases().items():
            self.ever[n] = self.ever[n] or v

    def holding(self):
        p = HE.robot(env).proprio16()
        meas = {"left": p[7], "right": p[15]}
        return [s for s in SIDES if self.grip_cmd[s] >= 0.5 and self.grip_cmd[s] - meas[s] > HOLD_GAP]

    def tucked(self):
        rob = HE.robot(env)
        return all(float(np.max(np.abs(rob.arm_q(s) - TUCK[s]))) <= TUCK_RAD for s in SIDES)


def planner(run):
    xp = OVMMExpert(env, on_step=lambda c, st, rep: None)
    orig_step = xp.step

    def step(c):
        obs = orig_step(c)
        run.grip_cmd = dict(xp.grip_cmd)
        run.record(c)
        return obs
    xp.step = step
    return xp


def policy_phase(run, task, seed0, limit, done_fn, resume=None):
    """pa_eval.policy_phase; resume = (snapshot, {event index: image}) restores the state at a query boundary"""
    rob = HE.robot(env)
    if resume is None:
        ex = C.CartesianExecutor.from_proprio(rob.proprio16())
        past = []
        byt = {t: (ev, img) for t, ev, img in run.head}
        for kk in sorted(HISTORY["offsets"], reverse=True):
            tk = run.t - kk                                 # pa_eval: steps so far - kk
            cand = [t for t in byt if t <= tk]
            ev, img = byt[max(cand)] if cand else (run.head[0][1], run.head[0][2])
            past.append((img, ev))
        n, ok_run = 0, 0
    else:
        s, imgs_by_ev = resume
        ex = C.CartesianExecutor.from_proprio(rob.proprio16())
        ex.q = {k: np.asarray(v, np.float64) for k, v in s["ex_q"].items()}
        ex.prev_quat = {k: (None if v is None else np.asarray(v, np.float64)) for k, v in s["ex_prev_quat"].items()}
        past = [(imgs_by_ev[e], e) for e in s["past_ev"]]
        n, ok_run = s["phase_n"], s["ok_run"]
    chunk, k, reason = None, EXEC, "timeout"
    while n < limit and not env.episode_over:
        if k >= EXEC:
            run.snaps.append({"ev": len(EV), "t": run.t, "phase": run.phase, "phase_n": n, "ok_run": ok_run,
                              "ex_q": {kk: ex.q[kk].tolist() for kk in SIDES},
                              "ex_prev_quat": {kk: (None if ex.prev_quat[kk] is None else np.asarray(ex.prev_quat[kk]).tolist()) for kk in SIDES},
                              "grip_cmd": dict(run.grip_cmd), "grasp_R": None if run.grasp_R is None else np.asarray(run.grasp_R).tolist(),
                              "past_ev": [e for _, e in past], "seed_base": int(seed0), "ever": dict(run.ever)})
            ev = len(EV); imgs = observe()
            st16 = IO.state_from_proprio(rob.proprio16()).astype(np.float32)
            cam = HISTORY["camera"]
            for kk in HISTORY["offsets"]:
                jj = kk // EXEC
                imgs[f"{cam}_h{kk}"] = (past[-jj] if len(past) >= jj else past[0])[0]
            past.append((imgs[cam], ev))
            chunk, k = infer(imgs, task, st16, seed0 + run.t), 0
        a = np.asarray(chunk[k], float).copy()
        a[16] = a[17] = 0.0                                 # base lock (frozen contract)
        joint, _ = ex.lower(a)
        HE.step18(env, joint)
        run.grip_cmd = {"left": float(a[C.CART_LEFT_GRIPPER]), "right": float(a[C.CART_RIGHT_GRIPPER])}
        run.record(a)
        k, n = k + 1, n + 1
        if run.grasp_R is None and run.holding():
            hs = run.holding()[0]
            run.grasp_R = (np.linalg.inv(rob.arm_frame()) @ rob.link_frame(f"{hs}_gripper_base"))[:3, :3]
        ok_run = ok_run + 1 if done_fn() else 0
        if ok_run >= HOLD_K:
            reason = "condition"
            break
    if env.episode_over:
        reason = "episode_over"
    return {"steps": n, "end": reason}


def replay_prefix(src, upto, need):
    """exact replay of a source event log up to event index `upto`; returns the renders at the `need` events"""
    ev_t, J = src["ev"], src["joint18"]
    out, j = {}, 0
    for i in range(upto):
        e = int(ev_t[i])
        if e == 0:
            HE.step18(env, J[j]); j += 1
        elif e == 1:
            imgs = observe()
            if i in need:
                out[i] = imgs[HISTORY["camera"]]
        else:
            NAV.scene_contacts(env)
    return out


def episode(unit, rec, seed, root=None, shift=0):
    ep_id, rep = HE.parse_unit(unit)
    HE.reset_to(env, ep_id, rep)
    EV.clear(); JOINTS.clear()
    ep = env.current_episode
    task = IO.instruction(ep.object_category, ep.start_recep_category, ep.goal_recep_category)
    rec.update({"scene": ep.scene_id.split("/")[-1].split(".")[0], "object": ep.object_category, "instruction": task,
                "start_fix": HE.fix_start(env)})
    run = Run(unit)
    run.h0 = float(np.array(target().translation)[1])
    seed_m1 = int(HE.episode_seed(ep_id, rep)) + 1_000_003 * seed
    ph = rec["phases"] = {}
    rob = HE.robot(env); obj = target()
    resume, start = None, "nav1"
    if root is not None:
        src = np.load(root["source_trace"])
        snap = json.loads(str(src["snaps"]))[root["snap"]]
        imgs = replay_prefix(src, snap["ev"], set(snap["past_ev"]))
        rec["restore"] = {"events_replayed": snap["ev"], "steps_replayed": len(JOINTS), "t_snap": snap["t"]}
        assert len(JOINTS) == snap["t"], (len(JOINTS), snap["t"])
        run.phase, start = snap["phase"], snap["phase"]
        run.grip_cmd, run.ever = dict(snap["grip_cmd"]), dict(snap["ever"])
        run.grasp_R = None if snap["grasp_R"] is None else np.asarray(snap["grasp_R"], np.float64)
        resume = (snap, imgs)
    if start == "nav1":
        run.phase = "nav1"
        xp = planner(run)
        xp.ex = C.CartesianExecutor.from_proprio(rob.proprio16())
        xp.grip_cmd = {s: 0.0 for s in SIDES}
        xp.joint_move({s: TUCK[s] for s in SIDES}, 15, grips={s: 0.0 for s in SIDES})
        plans = {s: xp.plan_pick(s, obj) for s in SIDES}
        feas = {s: p for s, p in plans.items() if p is not None}
        if not feas:
            rec["end"] = "planner_no_pick_standoff"; return run
        side = min(feas, key=lambda s: feas[s]["cost"])
        st = feas[side]["standoff"]
        nav1 = xp.nav_to(st["xyz"], None); xp._turn_to(st["yaw"])
        ph["nav1"] = {"planned_arm": side, **{kk: nav1.get(kk) for kk in ("arrived", "arrival_dist_m", "steps", "collision_steps")}}
        if not nav1.get("arrived"):
            rec["end"] = "nav1_not_arrived"; return run
        start = "manip1"
    if start == "manip1":
        run.phase = "manip1"
        m1 = policy_phase(run, task, seed_m1 + shift, MANIP1_MAX, lambda: bool(run.holding()) and run.tucked(),
                          resume=resume if resume and resume[0]["phase"] == "manip1" else None)
        m1["holding_obs"] = run.holding(); m1["held_priv"] = held_priv()
        ph["manip1"] = m1
        if m1["end"] != "condition":
            rec["end"] = "manip1_" + m1["end"]; return run
        run.after["carry_ready"] = True
        run.phase = "nav2"
        hside = run.holding()[0]
        xp = planner(run)
        xp.ex = C.CartesianExecutor.from_proprio(rob.proprio16())
        xp.grip_cmd = {s: (1.0 if s == hside else 0.0) for s in SIDES}
        xp.side = hside
        xp.joint_move({s: TUCK[s] for s in SIDES}, 10)
        A = xp.arm_T_world(); gb = xp.rob.link_frame(f"{hside}_gripper_base")
        xp.obj_in_gb = np.linalg.inv(gb) @ np.r_[np.c_[np.array(obj.transformation.rotation()), np.array(obj.translation)], [[0, 0, 0, 1]]]
        _, ext, _, _ = xp.object_geometry(obj)
        pl = None
        for R0 in [r for r in (run.grasp_R, (np.linalg.inv(A) @ gb)[:3, :3]) if r is not None]:
            xp.grasp_used = {"R": R0}
            xp.rng = np.random.default_rng(0)
            pl = xp.plan_place(hside, ext[1])
            if pl is not None:
                break
        if pl is None:
            rec["end"] = "planner_no_place_standoff"; return run
        nav2 = xp.nav_to(pl["standoff"]["xyz"], None); xp._turn_to(pl["standoff"]["yaw"])
        ph["nav2"] = {"hand": hside, "recep": pl["recep"], **{kk: nav2.get(kk) for kk in ("arrived", "arrival_dist_m", "steps", "collision_steps")},
                      "held_priv_at_arrival": held_priv()}
        if not nav2.get("arrived"):
            rec["end"] = "nav2_not_arrived"; return run
        start, resume = "manip2", None
    if start == "manip2":
        run.phase = "manip2"; run.after["manip2"] = True
        m2 = policy_phase(run, task, seed_m1 + 7_777_777 + shift, MANIP2_MAX, lambda: (not run.holding()) and run.tucked(),
                          resume=resume)
        ph["manip2"] = m2
        run.after["released_in_manip2"] = not run.holding()
    run.phase = "settle"
    hold = np.r_[rob.proprio16(), 0.0, 0.0]
    for g, s in ((7, "left"), (15, "right")):
        hold[g] = run.grip_cmd[s]
    for _ in range(SETTLE_STEPS):
        if env.episode_over:
            break
        HE.step18(env, hold); run.record(np.full(18, np.nan))
    rec["end"] = "manip2_" + ph["manip2"]["end"]
    return run


def save(run, rec, jid):
    tmp = f"/dev/shm/ftrace_{WK}_{jid}.npz"
    fr = np.asarray(run.tr.pop("frames"), np.uint8).reshape(-1, 120, 480, 3)
    np.savez_compressed(tmp, frames=fr, ev=np.asarray(EV, np.uint8), joint18=np.asarray(JOINTS, np.float32).reshape(-1, 18),
                        snaps=json.dumps(run.snaps), **{k: np.asarray(v) for k, v in run.tr.items()})
    shutil.copy(tmp, f"{OUT}/traces/{jid}.npz"); os.remove(tmp)
    return f"{OUT}/traces/{jid}.npz"


if MODE == "source":
    SEED = int(os.environ["SEED"])
    work = [(u, f"{u.replace('#', '_r')}_s{SEED}", None) for u in os.environ["UNITS"].split(",") if u][WK::NW]
else:
    plan = json.load(open(os.environ["JOBS"]))
    work = [(j["unit"], j["job"], j) for j in plan["jobs"]][WK::NW]
fn = f"{OUT}/{MODE}_w{WK:02d}.jsonl"
done = {json.loads(l)["job"] for f in glob.glob(f"{OUT}/{MODE}_w*.jsonl") for l in open(f)}
for unit, jid, job in work:
    if jid in done:
        continue
    allowed(unit)
    rec = {"job": jid, "unit": unit, "mode": MODE, "contract": "PA-OG + base lock (frozen pa_eval default)",
           "policy_ckpt": os.environ.get("POLICY_CKPT"), "server_port": int(os.environ.get("PORT", "5600"))}
    if job:
        rec.update({k: job[k] for k in ("root_id", "noise", "subtask", "source_job", "snap", "t_root") if k in job})
    t0 = time.time(); SERVER_S[0] = 0.0
    run = None
    try:
        if MODE == "source":
            rec["seed"] = SEED
            run = episode(unit, rec, SEED)
        else:
            rec["seed"] = job["source_seed"]
            run = episode(unit, rec, job["source_seed"], root=job, shift=0 if job["noise"] < 0 else NOISE_STRIDE * (job["noise"] + 1))
        rec["funnel"] = phases(); rec["ever"] = run.ever; rec["steps"] = run.t; rec["after"] = run.after
        rec["events"] = HE.action_obj(env).events
    except ExpertFailure as e:
        rec["end"] = f"planner_failure: {e}"[:300]
    except Exception as e:
        rec["engineering_error"] = repr(e)[:300]; rec["tb"] = traceback.format_exc()[-1500:]
    if run is not None and JOINTS:
        rec["trace_out"] = save(run, rec, jid)
    rec["wall_s"] = round(time.time() - t0, 1); rec["server_gpu_s"] = round(SERVER_S[0], 1); rec["t_end"] = time.time()
    with open(fn, "a") as f:
        f.write(json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
    print("FRONTIER", json.dumps({k: rec.get(k) for k in ("job", "end", "funnel", "after", "engineering_error", "wall_s")}), flush=True)
env.close()
