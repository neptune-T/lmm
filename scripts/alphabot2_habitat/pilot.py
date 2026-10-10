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
  EPISODES=a,b,c | EPISODE_FILE=ids.txt   episode order (one id per line or comma separated)
  WORKER/NWORKERS                         shard of the order (one Habitat process per shard)
  MAX_OK                                  stop after this many accepted episodes (default unlimited)
  ARM_MODE=auto|left|right                auto = arm chosen by geometry (default)
  CLAIM_DIR=/dev/shm/...                  dynamic sharding: each episode is claimed with an O_EXCL file
                                          (pid while running, "done" after), so workers can be added at any
                                          time; a claim whose pid is dead is taken over
Episodes with a results row from any worker are skipped (resume after a crash/restart).

    EPISODES=... OUT=... run_habitat.sh pilot.py
"""
import glob, json, os, sys, time, traceback, zlib

import numpy as np

REPO = os.environ.get("LMM_REPO", "/workspace/lmm")
sys.path.insert(0, REPO)
from lmm.embodiments.alphabot2.habitat import env as HE  # noqa: E402
from lmm.embodiments.alphabot2.habitat import snapshot as SN  # noqa: E402
from lmm.embodiments.alphabot2.habitat.expert import OVMMExpert  # noqa: E402

OUT = os.environ.get("OUT", "/workspace/runtime/runs/pilot")
os.makedirs(f"{OUT}/episodes", exist_ok=True)
CAMS = ("head_rgb", "left_wrist_rgb", "right_wrist_rgb")
MAX_OK = int(os.environ.get("MAX_OK", "1000000000"))
ARM_MODE = os.environ.get("ARM_MODE", "auto")
WK, NW = int(os.environ.get("WORKER", 0)), int(os.environ.get("NWORKERS", 1))
if os.environ.get("EPISODE_FILE"):
    EPS = open(os.environ["EPISODE_FILE"]).read().replace(",", "\n").split()
else:
    EPS = os.environ["EPISODES"].split(",")
EPS = EPS[WK::NW]
DONE = set()
for f in glob.glob(f"{OUT}/results*.jsonl"):
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


n_ok = 0
res_f = open(f"{OUT}/results_w{WK:02d}.jsonl", "a")
for eid in EPS:
    if n_ok >= MAX_OK:
        break
    if not claim(eid):
        continue
    t0 = time.time()
    rec = {"episode_id": eid, "worker": WK, "arm_mode": ARM_MODE}
    buf = {"imgs": {k: [] for k in CAMS}, "proprio": [], "action": [], "joint18": [], "stage": [], "sim_time": [],
           "state": []}
    try:
        HE.reset_to(env, eid)
        ep = env.current_episode
        rec.update({"scene": ep.scene_id.split("/")[-1].split(".")[0], "object": ep.object_category,
                    "start_recep": ep.start_recep_category, "goal_recep": ep.goal_recep_category,
                    "instruction": instruction(ep)})
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
        rec["success"] = bool(rec["expert_completed"] and rec["ovmm"].get("ovmm_place_success"))
        if rec["success"]:
            # base travel before and after the grasp (real navigation on both sides)
            t_grasp = next(t for t, s in ex.events if str(s).startswith("grasp_"))
            base = np.array([s["root"][:3, 3] for s in buf["state"]])
            rec["base_travel_before_grasp_m"] = float(np.sum(np.linalg.norm(np.diff(base[:t_grasp], axis=0), axis=1)))
            rec["base_travel_after_grasp_m"] = float(np.sum(np.linalg.norm(np.diff(base[t_grasp:], axis=0), axis=1)))
            # ---- replay validation
            t1 = time.time()
            HE.reset_to(env, eid)
            fix2 = HE.fix_start(env)
            rep_states = []
            for a in buf["joint18"]:
                HE.step18(env, a)
                rep_states.append(SN.state_vector(env))
            d = [SN.divergence(a, b) for a, b in zip(rep_states, buf["state"])]
            rec["replay"] = {"start_fix_equal": fix2 == rec["start_fix"], "steps": len(d),
                             "root_max_abs": max(x["root_max_abs"] for x in d),
                             "proprio_max_abs": max(x["proprio_max_abs"] for x in d),
                             "objects_max_abs": max(x["objects_max_abs"] for x in d),
                             "held_equal_all": all(x["held_equal"] for x in d),
                             "place_success": metrics().get("ovmm_place_success"), "wall_s": round(time.time() - t1, 1)}
            rec["accepted"] = bool(rec["replay"]["place_success"] and rec["replay"]["held_equal_all"]
                                   and rec["replay"]["proprio_max_abs"] < 1e-6 and rec["replay"]["root_max_abs"] < 1e-6)
            d_ep = f"{OUT}/episodes/{eid}"
            os.makedirs(d_ep, exist_ok=True)
            np.savez_compressed(f"{d_ep}/images.npz", **{k: np.asarray(v, np.uint8) for k, v in buf["imgs"].items()})
            np.savez_compressed(f"{d_ep}/lowdim.npz", proprio=np.asarray(buf["proprio"]), action=np.asarray(buf["action"]),
                                joint18=np.asarray(buf["joint18"]), sim_time=np.asarray(buf["sim_time"]))
            # metadata only, kept out of the policy-visible arrays (images.npz, lowdim.npz)
            np.savez_compressed(f"{d_ep}/metadata.npz", stage=np.asarray(buf["stage"]),
                                stage_id=np.asarray([zlib.crc32(str(s).encode()) for s in buf["stage"]], np.uint32))
            with open(f"{d_ep}/meta.json", "w") as f:
                json.dump({**rec, "stages": buf["stage"], "policy_inputs": list(CAMS) + ["proprio16", "instruction"],
                           "metadata_only": ["stage", "events", "arm", "plans", "ovmm", "grasp_events"]},
                          f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
            n_ok += int(rec["accepted"])
    except Exception as e:
        rec["engineering_error"] = repr(e)
        rec["tb"] = traceback.format_exc()[-1500:]
    rec["wall_s"] = round(time.time() - t0, 1)
    res_f.write(json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
    res_f.flush()
    release(eid)
    print("PILOT", json.dumps({k: rec.get(k) for k in ("episode_id", "arm", "success", "accepted", "failure",
                                                        "engineering_error")}), flush=True)
env.close()
