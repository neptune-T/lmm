"""Planner-assisted OVMM evaluation (Habitat client): planner NAV + pi0.5 manipulation, from normal OVMM init.

LABEL: "PA-OG" = planner-assisted, ORACLE GOAL. NOT an end-to-end result. The planner's navigation goals come from
privileged simulator state (target object pose, goal receptacle surfaces, navmesh, IK feasibility -- the expert's
own standoff planner); in a deployment they would come from a detector / semantic map. Everything the POLICY sees
and every phase switch is observable on the robot:

  NAV1   planner: arms to the NAV tuck, shortest navmesh path to the pick standoff, turn to its heading (the
         expert's exact NAV code; the cheaper feasible arm's standoff, like the training data).
         Switch -> MANIP1 when the planner reports arrival (odometry vs its own waypoint).
  MANIP1 pi0.5 (v2-B, fp32, H_exec 5, same 18-D CartesianMobileAction18 / executor / cameras / state / history /
         instruction as eval_policy.py) runs unchanged. Switch -> NAV2 at CARRY_READY: holding (a gripper commanded
         closed whose measured closure stops > HOLD_GAP short of the command: fingers blocked by an object) AND both
         arms within TUCK_RAD of the NAV tuck, for HOLD_K consecutive steps. Timeout MANIP1_MAX -> episode ends.
  NAV2   planner: arms to the exact NAV tuck (grippers keep their commands), path to the place standoff (planned
         from the object as actually held), turn. Switch -> MANIP2 on arrival.
  MANIP2 pi0.5 again. Switch -> DONE at PLACE_DONE: the holding gripper opened (not holding any more) AND both arms
         within TUCK_RAD of the tuck (retracted), or MANIP2_MAX.
  DONE   hold still SETTLE_STEPS (fixed; no object-velocity check), then the episode ends.
Funnel from OVMM's own measures at the end (find, grasp, carry, place, full = ovmm_place_success), plus per-phase
step counts, switch reasons, and "ever" flags. Trace: per step phase, base pose, executed action, held (privileged,
logged only), half-res [top|left|right] frame strip every FRAME_EVERY steps.
    EPISODES=u1,u2 SPLIT=train SEED_OFFSET=0 PORT=... GPU=.. OUT=... [HISTORY=json] run_habitat.sh pa_eval.py
"""
import json, os, shutil, socket, sys, time, traceback

import numpy as np

REPO = os.environ.get("LMM_REPO", "/workspace/lmm")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "alphabot2_pi05"))
from lmm.embodiments.alphabot2 import cartesian as C  # noqa: E402
from lmm.embodiments.alphabot2 import ovmm_policy_io as IO  # noqa: E402
from lmm.embodiments.alphabot2.habitat import env as HE  # noqa: E402
from lmm.embodiments.alphabot2.habitat import nav as NAV  # noqa: E402
from lmm.embodiments.alphabot2.habitat.constants import NAV_TUCK  # noqa: E402
from lmm.embodiments.alphabot2.habitat.expert import OVMMExpert, ExpertFailure, SIDES  # noqa: E402
from wire import recv_msg, send_msg  # noqa: E402

OUT = os.environ["OUT"]; os.makedirs(f"{OUT}/traces", exist_ok=True)
SPLIT = os.environ.get("SPLIT", "train")
EXEC = int(os.environ.get("EXEC_STEPS", "5"))
SEED_OFFSET = int(os.environ.get("SEED_OFFSET", "0"))
MANIP1_MAX, MANIP2_MAX = int(os.environ.get("MANIP1_MAX", "600")), int(os.environ.get("MANIP2_MAX", "500"))
SETTLE_STEPS = int(os.environ.get("SETTLE_STEPS", "120"))
HOLD_GAP, TUCK_RAD, HOLD_K = 0.05, float(os.environ.get("TUCK_RAD", "0.30")), 3
FRAME_EVERY = int(os.environ.get("FRAME_EVERY", "2"))
# BASE_LOCK=1: during MANIP phases the planner owns the base -- the policy's base.v / base.omega (dims 16, 17) are
# replaced by 0 before execution (arms + grippers executed unchanged); the policy's raw v / omega are traced.
BASE_LOCK = os.environ.get("BASE_LOCK", "0") == "1"
WK, NW = int(os.environ.get("WORKER", 0)), int(os.environ.get("NWORKERS", 1))
HISTORY = json.loads(os.environ["HISTORY"]) if os.environ.get("HISTORY") else None
if HISTORY:
    assert all(k % EXEC == 0 for k in HISTORY["offsets"])
FUNNEL = (("find", "ovmm_find_object_phase_success"), ("grasp", "ovmm_pick_object_phase_success"),
          ("carry", "ovmm_find_recep_phase_success"), ("place", "ovmm_place_object_phase_success"), ("full", "ovmm_place_success"))
TUCK = {s: np.array(v) for s, v in NAV_TUCK.items()}
# every executed JointMobileAction18 (planner and policy alike) is recorded for open-loop replay (pa_replay.py)
JOINTS = []
_step18 = HE.step18


def _rec_step18(env_, joint18):
    JOINTS.append(np.asarray(joint18, np.float32).reshape(18).copy())
    return _step18(env_, joint18)


HE.step18 = _rec_step18
units = [u for u in os.environ["EPISODES"].replace("\n", ",").split(",") if u][WK::NW]
env = HE.make_env(gpu=int(os.environ.get("GPU", "0")), split=SPLIT)
sock = socket.create_connection(("127.0.0.1", int(os.environ.get("PORT", "5600"))))
send_msg(sock, {"cmd": "ping"}); srv = recv_msg(sock)
_hk = {f"{HISTORY['camera']}_h{k}" for k in HISTORY["offsets"]} if HISTORY else set()
assert set(srv.get("cams", [])) == set(IO.CAMERA_MAP) | _hk, srv.get("cams")


def observe():
    HE.robot(env).update()
    return IO.policy_images(env.sim.get_sensor_observations())


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


class Run:
    """Per-episode state: the policy's history buffer, the trace, the last commanded grippers (for the observable
    holding test)."""

    def __init__(self, unit):
        self.unit = unit
        self.phase = "nav1"
        self.t = 0
        self.head = []                                    # head frame per step (history for the first policy query)
        self.tr = {k: [] for k in ("phase", "pose", "act", "held", "frames", "frame_steps", "raw_vw")}
        self.grip_cmd = {"left": 0.0, "right": 0.0}
        self.ever = {n: False for n, _ in FUNNEL}
        self.grasp_R = None

    def record(self, cart18, imgs=None):
        """after every env step"""
        xz, yaw, _ = NAV.pose2d(env)
        self.tr["phase"].append(self.phase); self.tr["pose"].append([float(xz[0]), float(xz[1]), float(yaw)])
        self.tr["act"].append(np.asarray(cart18, np.float32)); self.tr["held"].append(held_priv())
        if self.t % FRAME_EVERY == 0 or self.t % EXEC == 0 or len(self.head) < 1:
            imgs = imgs or observe()
            if self.t % FRAME_EVERY == 0:
                self.tr["frames"].append(np.concatenate([imgs[c][::2, ::2] for c in ("camera_top", "camera_left", "camera_right")], 1))
                self.tr["frame_steps"].append(self.t)
            self.head.append((self.t, imgs["camera_top"]))
            self.head = self.head[-24:]
        for n, v in phases().items():
            self.ever[n] = self.ever[n] or v
        self.t += 1

    def holding(self):
        """observable: a gripper commanded closed (>= 0.5) whose measured closure stops > HOLD_GAP short of it"""
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


def policy_phase(run, task, seed0, limit, done_fn):
    rob = HE.robot(env)
    ex = C.CartesianExecutor.from_proprio(rob.proprio16())
    # history at the first query: the head frames 5/10/15 steps earlier (planner steps), as in an expert episode
    past = []
    if HISTORY:
        byt = dict(run.head)
        for kk in sorted(HISTORY["offsets"], reverse=True):
            tk = run.t - kk
            cand = [t for t in byt if t <= tk]
            past.append(((byt[max(cand)] if cand else run.head[0][1]), None))
    chunk, k, n, ok_run, reason = None, EXEC, 0, 0, "timeout"
    while n < limit and not env.episode_over:
        if k >= EXEC:
            imgs = observe()
            st16 = IO.state_from_proprio(rob.proprio16()).astype(np.float32)
            if HISTORY:
                cam = HISTORY["camera"]
                for kk in HISTORY["offsets"]:
                    jj = kk // EXEC
                    imgs[f"{cam}_h{kk}"] = (past[-jj] if len(past) >= jj else past[0])[0]
                past.append((imgs[cam], st16))
            send_msg(sock, {"cmd": "infer", "images": imgs, "task": task, "state": st16, "seed": seed0 + run.t})
            r = recv_msg(sock)
            if "error" in r:
                raise RuntimeError("policy server: " + r["error"])
            chunk, k = r["actions"], 0
        a = np.asarray(chunk[k], float).copy()
        run.tr["raw_vw"].append([float(a[16]), float(a[17])])
        if BASE_LOCK:
            a[16] = a[17] = 0.0
        joint, _ = ex.lower(a)
        HE.step18(env, joint)
        run.grip_cmd = {"left": float(a[C.CART_LEFT_GRIPPER]), "right": float(a[C.CART_RIGHT_GRIPPER])}
        run.record(a)
        k, n = k + 1, n + 1
        if run.grasp_R is None and run.holding():                 # hand orientation at the first observed hold
            hs = run.holding()[0]
            run.grasp_R = (np.linalg.inv(rob.arm_frame()) @ rob.link_frame(f"{hs}_gripper_base"))[:3, :3]
        ok_run = ok_run + 1 if done_fn() else 0
        if ok_run >= HOLD_K:
            reason = "condition"
            break
    if env.episode_over:
        reason = "episode_over"
    return {"steps": n, "end": reason}


def episode(unit, rec):
    ep_id, rep = HE.parse_unit(unit)
    HE.reset_to(env, ep_id, rep)
    JOINTS.clear()
    ep = env.current_episode
    task = IO.instruction(ep.object_category, ep.start_recep_category, ep.goal_recep_category)
    rec.update({"scene": ep.scene_id.split("/")[-1].split(".")[0], "object": ep.object_category, "instruction": task,
                "start_fix": HE.fix_start(env)})
    run = Run(unit)
    seed0 = int(HE.episode_seed(ep_id, rep)) + 1_000_003 * SEED_OFFSET
    ph = rec["phases"] = {}
    # ---- NAV1 (planner, oracle goal)
    xp = planner(run)
    rob = HE.robot(env)
    xp.ex = C.CartesianExecutor.from_proprio(rob.proprio16())
    obj, _ = xp.target_object()
    xp.joint_move({s: TUCK[s] for s in SIDES}, 15)
    plans = {s: xp.plan_pick(s, obj) for s in SIDES}
    feas = {s: p for s, p in plans.items() if p is not None}
    if not feas:
        rec["end"] = "planner_no_pick_standoff"; return run
    side = min(feas, key=lambda s: feas[s]["cost"])
    st = feas[side]["standoff"]
    nav1 = xp.nav_to(st["xyz"], None); xp._turn_to(st["yaw"])
    ph["nav1"] = {"planned_arm": side, **{k: nav1.get(k) for k in ("arrived", "arrival_dist_m", "steps", "collision_steps")}}
    if not nav1.get("arrived"):
        rec["end"] = "nav1_not_arrived"; return run
    # ---- MANIP1 (policy) until carry-ready
    run.phase = "manip1"
    ph["manip1"] = policy_phase(run, task, seed0, MANIP1_MAX, lambda: bool(run.holding()) and run.tucked())
    ph["manip1"]["holding_obs"] = run.holding(); ph["manip1"]["held_priv"] = held_priv()
    if ph["manip1"]["end"] != "condition":
        rec["end"] = "manip1_" + ph["manip1"]["end"]; return run
    # ---- NAV2 (planner, oracle goal): plan the place from the object as actually held
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
    for R0 in [r for r in (run.grasp_R, (np.linalg.inv(A) @ gb)[:3, :3]) if r is not None]:   # grasp-time, then current hand
        xp.grasp_used = {"R": R0}
        pl = xp.plan_place(hside, ext[1])
        if pl is not None:
            break
    if pl is None:
        rec["end"] = "planner_no_place_standoff"; return run
    nav2 = xp.nav_to(pl["standoff"]["xyz"], None); xp._turn_to(pl["standoff"]["yaw"])
    ph["nav2"] = {"hand": hside, "recep": pl["recep"], **{k: nav2.get(k) for k in ("arrived", "arrival_dist_m", "steps", "collision_steps")},
                  "held_priv_at_arrival": held_priv()}
    if not nav2.get("arrived"):
        rec["end"] = "nav2_not_arrived"; return run
    # ---- MANIP2 (policy) until released + retracted
    run.phase = "manip2"
    ph["manip2"] = policy_phase(run, task, seed0 + 7_777_777, MANIP2_MAX, lambda: (not run.holding()) and run.tucked())
    # ---- DONE: hold still
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


fn = f"{OUT}/pa_w{WK:02d}_s{SEED_OFFSET}{'_lock' if BASE_LOCK else ''}.jsonl"
done = {json.loads(l)["unit"] for l in open(fn)} if os.path.exists(fn) else set()
for unit in units:
    if unit in done:
        continue
    rec = {"unit": unit, "split": SPLIT, "seed_offset": SEED_OFFSET, "label": "PA-OG (planner-assisted, oracle goal)" + (" + base lock in manipulation" if BASE_LOCK else ""),
           "h_exec": EXEC, "policy_ckpt": os.environ.get("POLICY_CKPT"),
           "contract": {"BASE_LOCK": BASE_LOCK, "MANIP1_MAX": MANIP1_MAX, "MANIP2_MAX": MANIP2_MAX, "SETTLE_STEPS": SETTLE_STEPS, "HOLD_GAP": HOLD_GAP,
                        "TUCK_RAD": TUCK_RAD, "HOLD_K": HOLD_K}}
    t0 = time.time()
    run = None
    try:
        run = episode(unit, rec)
        rec["funnel"] = phases(); rec["ever"] = run.ever; rec["steps"] = run.t
        rec["events"] = HE.action_obj(env).events
    except ExpertFailure as e:
        rec["end"] = f"planner_failure: {e}"[:300]
    except Exception as e:
        rec["engineering_error"] = repr(e)[:300]; rec["tb"] = traceback.format_exc()[-1500:]
    if run is not None and run.tr["pose"]:
        jid = f"{unit.replace('#', '_r')}_s{SEED_OFFSET}{'_lock' if BASE_LOCK else ''}"
        tmp = f"/dev/shm/patrace_{WK}_{jid}.npz"
        fr = np.asarray(run.tr.pop("frames"), np.uint8).reshape(-1, 120, 480, 3)
        np.savez_compressed(tmp, frames=fr, joint18=np.asarray(JOINTS, np.float32).reshape(-1, 18),
                            **{k: np.asarray(v) for k, v in run.tr.items()})
        shutil.copy(tmp, f"{OUT}/traces/{jid}.npz"); os.remove(tmp); rec["trace_out"] = f"{OUT}/traces/{jid}.npz"
    rec["wall_s"] = round(time.time() - t0, 1)
    with open(fn, "a") as f:
        f.write(json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
    print("PA", json.dumps({k: rec.get(k) for k in ("unit", "end", "funnel", "phases", "engineering_error", "wall_s")}), flush=True)
env.close()
