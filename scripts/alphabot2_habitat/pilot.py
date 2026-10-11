"""AlphaBot 2 OVMM pilot recorder (10-20 episodes) with replay validation.

Normal OVMM initialization of geometrically feasible TRAIN episodes; the
privileged expert (metadata only) drives; every step records, BEFORE action t:
  head_rgb, left_wrist_rgb, right_wrist_rgb   (the synchronized render of the current state)
  proprio16                                   (canonical)
  action: CartesianMobileAction18 label       (what the policy will be trained to output)
  joint18: the lowered JointMobileAction18    (what was executed; used for replay)
  timing: policy_step, sim world time
Episode metadata: OVMM-style instruction (object / start receptacle / goal receptacle
categories -- no arm token), arm used, stages (metadata only), OVMM metrics, start fix.

Replay: reset_to(episode) + the same start fix, the recorded joint18 sequence
re-executed; per-step proprio / base / object / held-object comparison and OVMM success.

The same script is the continuous collector (one code path for pilot and scale):
  EPISODES=a,b,c | EPISODE_FILE=ids.txt   unit order: "EPISODE#REPLICA" (seed-bank replica; bare id = #0),
                                          one per line or comma separated
  WORKER/NWORKERS                         shard of the order (one Habitat process per shard)
  MAX_OK                                  stop after this many accepted episodes (default unlimited)
  ARM_MODE=auto|left|right                auto = arm chosen by geometry (default)
  CLAIM_DIR=/dev/shm/...                  dynamic sharding: each episode is claimed with an O_EXCL file
                                          (pid while running, "done" after), so workers can be added at any
                                          time; a claim whose pid is dead is taken over
Episodes with a results row from any worker are skipped (resume after a crash/restart).

    EPISODES=... OUT=... run_habitat.sh pilot.py
"""
import glob, json, os, subprocess, sys, time, traceback, zlib

import numpy as np

REPO = os.environ.get("LMM_REPO", "/workspace/lmm")
sys.path.insert(0, REPO)
from lmm.embodiments.alphabot2.habitat import env as HE  # noqa: E402
from lmm.embodiments.alphabot2.habitat import fsutil as FS  # noqa: E402
FS.protect_stdio()
from lmm.embodiments.alphabot2.habitat import nav as NAV  # noqa: E402
from lmm.embodiments.alphabot2.habitat import snapshot as SN  # noqa: E402
from lmm.embodiments.alphabot2.habitat.expert import OVMMExpert  # noqa: E402

OUT = os.environ.get("OUT", "/workspace/runtime/runs/pilot")
os.makedirs(f"{OUT}/episodes", exist_ok=True)
CAMS = ("head_rgb", "left_wrist_rgb", "right_wrist_rgb")
# fresh-reset replay tolerances; positions and rotations separately (no mixed max over matrix entries)
REPLAY_TOL = {"root_pos_m": 1e-5, "root_rot_deg": 1e-3, "proprio_max_abs": 1e-6,
              "objects_pos_max_m": 0.025, "objects_rot_max_deg": 5.0}


def save_npz(path, **arrays):
    """np.savez_compressed into memory, then ONE sequential write: bosfs rejects the zip writer's seeks
    (OSError 22) when writing in place."""
    import io
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    data = buf.getvalue()

    def w():
        with open(path, "wb") as f:
            f.write(data)
    FS._retry(w)                                              # transient bosfs FUSE ENOMEM
    return len(data)


MAX_OK = int(os.environ.get("MAX_OK", "1000000000"))
ARM_MODE = os.environ.get("ARM_MODE", "auto")
WK, NW = int(os.environ.get("WORKER", 0)), int(os.environ.get("NWORKERS", 1))
if os.environ.get("EPISODE_FILE"):
    EPS = open(os.environ["EPISODE_FILE"]).read().replace(",", "\n").split()
else:
    EPS = os.environ["EPISODES"].split(",")
# static sharding only without dynamic claims: with CLAIM_DIR, WORKER just names the results file (e.g. 90,
# or gpu*10+slot) and every worker walks the whole list, claiming units atomically
if not os.environ.get("CLAIM_DIR"):
    EPS = EPS[WK::NW]
DONE = set()
for f in sorted(n for n in set(glob.glob(f"{OUT}/results*.jsonl")) | {f"{OUT}/results_w{i:02d}.jsonl" for i in range(100)}
               if os.path.exists(n)):                     # bosfs readdir lags: probe worker files too
    for line in open(f):
        try:
            DONE.add(str(json.loads(line)["episode_id"]))
        except (ValueError, KeyError):
            pass                                             # a torn last line from a killed worker
EPS = [e for e in EPS if e not in DONE]
print("COLLECT", json.dumps({"worker": WK, "nworkers": NW, "todo": len(EPS), "skipped_done": len(DONE),
                             "arm_mode": ARM_MODE}), flush=True)
env = HE.make_env(gpu=int(os.environ.get("GPU", "0")), split="train")
KEYS = ("ovmm_find_object_phase_success", "ovmm_pick_object_phase_success", "ovmm_find_recep_phase_success",
        "ovmm_place_object_phase_success", "ovmm_place_success", "obj_anywhere_on_goal")


def metrics():
    m = env.get_metrics()
    out = {}
    for k in KEYS:
        v = m.get(k)
        if isinstance(v, dict):
            v = any(bool(x) for x in v.values())
        out[k] = float(v) if isinstance(v, (bool, int, float, np.floating)) else v
    return out


def instruction(ep):
    nice = lambda s: str(s).replace("_", " ")
    return f"Move the {nice(ep.object_category)} from the {nice(ep.start_recep_category)} to the {nice(ep.goal_recep_category)}."


def render_now():
    HE.robot(env).update()
    o = env.sim.get_sensor_observations()
    return {k: np.asarray(o[k])[..., :3].astype(np.uint8) for k in CAMS}


CLAIM_DIR = os.environ.get("CLAIM_DIR")
if CLAIM_DIR:
    os.makedirs(CLAIM_DIR, exist_ok=True)


def claim(eid):
    """True if this worker owns the episode now (always True without CLAIM_DIR)."""
    if not CLAIM_DIR:
        return True
    f = os.path.join(CLAIM_DIR, str(eid))
    for _ in range(2):
        try:
            fd = os.open(f, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                c = open(f).read().strip()
            except OSError:
                continue
            if c == "done" or not c.isdigit() or os.path.exists(f"/proc/{c}"):
                return False
            try:
                os.remove(f)                                      # owner died mid-episode: take over
            except OSError:
                pass
    return False


def release(eid):
    if CLAIM_DIR:
        with open(os.path.join(CLAIM_DIR, str(eid)), "w") as f:
            f.write("done")


try:                                                               # provenance: the code this worker runs
    CODE_COMMIT = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True,
                                 text=True).stdout.strip()
except OSError:
    CODE_COMMIT = ""
n_ok = 0
res_f = open(f"{OUT}/results_w{WK:02d}.jsonl", "a")
for eid in EPS:
    if n_ok >= MAX_OK:
        break
    if not claim(eid):
        continue
    t0 = time.time()
    ep_id, rep = HE.parse_unit(eid)
    rec = {"episode_id": eid, "episode": ep_id, "replica": rep, "seed": HE.episode_seed(ep_id, rep),
           "worker": WK, "arm_mode": ARM_MODE, "code_commit": CODE_COMMIT}
    buf = {"imgs": {k: [] for k in CAMS}, "proprio": [], "action": [], "joint18": [], "stage": [], "sim_time": [],
           "state": [], "base": []}
    try:
        HE.reset_to(env, ep_id, rep)
        ep = env.current_episode
        rec.update({"scene": ep.scene_id.split("/")[-1].split(".")[0], "object": ep.object_category,
                    "start_recep": ep.start_recep_category, "goal_recep": ep.goal_recep_category,
                    "instruction": instruction(ep)})
        rob0 = HE.robot(env).sim_obj.translation
        rec["start_pose"] = [float(rob0[0]), float(rob0[1]), float(rob0[2]), float(HE.robot(env).base_rot)]
        rec["start_fix"] = HE.fix_start(env)
        orig = HE.step18
        pending = {}

        def on_step(cart, stage, rep):
            im = render_now()
            for k in CAMS:
                buf["imgs"][k].append(im[k])
            buf["proprio"].append(HE.robot(env).proprio16())
            buf["action"].append(np.asarray(cart, np.float64))
            buf["stage"].append(stage)
            buf["sim_time"].append(float(env.sim.get_world_time()))
            xz, yaw, _ = NAV.pose2d(env)                         # metadata: base pose before action t
            buf["base"].append([float(xz[0]), float(xz[1]), float(yaw)])
            pending["armed"] = True

        def rec_step(env_, a):
            if pending.pop("armed", False):
                buf["joint18"].append(np.asarray(a, np.float64).copy())
            obs = orig(env_, a)
            if len(buf["joint18"]) == len(buf["state"]) + 1:
                buf["state"].append(SN.state_vector(env_))
            return obs
        HE.step18 = rec_step
        ex = OVMMExpert(env, on_step=on_step, force_arm=None if ARM_MODE == "auto" else ARM_MODE)
        try:
            ex.run()
            rec["expert_completed"] = True
        except Exception as e:
            rec["expert_completed"] = False
            rec["failure"] = repr(e)
        finally:
            HE.step18 = orig
        rec.update({"arm": ex.log.get("arm"), "steps": len(buf["action"]), "events": ex.events,
                    "grasp_events": HE.action_obj(env).events, "ovmm": metrics(),
                    "plans": {k: ex.log.get(k) for k in ("pick_plan", "place_plan", "nav_to_object", "nav_to_receptacle")}})
        # success label is written only after the simulator has settled (end of the settle interval)
        rec["success"] = bool(rec["expert_completed"] and rec["ovmm"].get("ovmm_place_success"))
        st_ = [i for i, x in enumerate(buf["stage"]) if x == "settle"]
        rec["settle"] = {"start_t": st_[0] if st_ else None, "len": len(st_), **(ex.log.get("settle") or {}),
                         "note": "post-release settle interval: OVMM stability window only; truncated/near-zero weight in SFT"}
        if rec["success"]:
            # base travel before and after the grasp (real navigation on both sides)
            t_grasp = next(t for t, s in ex.events if str(s).startswith("grasp_"))
            base = np.array([s["root"][:3, 3] for s in buf["state"]])
            rec["base_travel_before_grasp_m"] = float(np.sum(np.linalg.norm(np.diff(base[:t_grasp], axis=0), axis=1)))
            rec["base_travel_after_grasp_m"] = float(np.sum(np.linalg.norm(np.diff(base[t_grasp:], axis=0), axis=1)))
            # ---- replay validation
            t1 = time.time()
            HE.reset_to(env, ep_id, rep)
            fix2 = HE.fix_start(env)
            rep_states = []
            for a in buf["joint18"]:
                HE.step18(env, a)
                rep_states.append(SN.state_vector(env))
            d = [SN.divergence(a, b) for a, b in zip(rep_states, buf["state"])]
            md = SN.max_divergence(d)
            rec["replay"] = {"start_fix_equal": fix2 == rec["start_fix"], "steps": len(d), **md,
                             "held_equal_all": md["held_equal"],
                             "place_success": metrics().get("ovmm_place_success"), "wall_s": round(time.time() - t1, 1)}
            rec["replay"]["tolerances"] = REPLAY_TOL
            rec["accepted"] = bool(rec["replay"]["place_success"] and rec["replay"]["held_equal_all"]
                                   and all(rec["replay"][k] <= v for k, v in REPLAY_TOL.items()))
            d_ep = f"{OUT}/episodes/{str(eid).replace('#', '_r')}"
            rec["dir"] = d_ep
            os.makedirs(d_ep, exist_ok=True)
            nbytes = save_npz(f"{d_ep}/images.npz", **{k: np.asarray(v, np.uint8) for k, v in buf["imgs"].items()})
            nbytes += save_npz(f"{d_ep}/lowdim.npz", proprio=np.asarray(buf["proprio"]), action=np.asarray(buf["action"]),
                                joint18=np.asarray(buf["joint18"]), sim_time=np.asarray(buf["sim_time"]))
            # metadata only, kept out of the policy-visible arrays (images.npz, lowdim.npz)
            nbytes += save_npz(f"{d_ep}/metadata.npz", stage=np.asarray(buf["stage"]),
                                base_xz_yaw=np.asarray(buf["base"]),      # pose2d convention (yaw about +y, 0 = +x)
                                stage_id=np.asarray([zlib.crc32(str(s).encode()) for s in buf["stage"]], np.uint32))
            with open(f"{d_ep}/meta.json", "w") as f:
                json.dump({**rec, "stages": buf["stage"], "policy_inputs": list(CAMS) + ["proprio16", "instruction"],
                           "metadata_only": ["stage", "events", "arm", "plans", "ovmm", "grasp_events"]},
                          f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
            rec["bytes"] = nbytes                                  # in-memory sizes (bosfs reports stale sizes)
            n_ok += int(rec["accepted"])
    except Exception as e:
        rec["engineering_error"] = repr(e)
        rec["tb"] = traceback.format_exc()[-1500:]
    rec["wall_s"] = round(time.time() - t0, 1)
    rec["finished_at"] = time.strftime("%F %T")
    FS.append_line(res_f.name, json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    release(eid)
    print("PILOT", json.dumps({k: rec.get(k) for k in ("episode_id", "arm", "success", "accepted", "failure",
                                                        "engineering_error")}), flush=True)
env.close()
