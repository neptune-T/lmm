"""Record expert-recovery demonstrations with full observations (Habitat client; no policy, no training).

For each root of ROOTS restricted to GROUP (default train_unit -- held-out roots are refused): reconstruct the root
exactly as recovery_eval.py (expert replay to t, zero-offset teleport, B's recorded executed actions to r; pose checked
against the trace), recording the LAST PREFIX steps of B's own approach (raw cameras + proprio before each action,
B's executed CartesianMobileAction18 as the label), then run OVMMExpert(env).run() from the actual state, recording
every step exactly like the collector (pilot.py): raw head / left wrist / right wrist RGB rendered before the action,
proprio16, the expert's Cartesian label, stage, base pose, the executed joint18, sim time.
Stage names (metadata only, never a policy input) mark the recovery structure for the sampler:
  recovery_prefix       B's own frames before the root (history context only; sampling weight 0)
  recovery_tuck         the expert's start-of-run tuck (base still; weight 0)
  recovery_navigate     the expert's navigate_to_object from the off-path root (the corrective demonstration)
  recovery_reach_onset  the first REACH_ONSET frames after arrival (the correct navigation -> reach switch)
  recovery_post         the rest of the expert task (chunk continuation only; weight 0), "settle" kept as is
Output: the collector's episode format (images.npz / lowdim.npz / metadata.npz / meta.json + results_wXX.jsonl rows with
accepted=True and full provenance), so ovmm_to_lerobot.py converts it with the v2 conventions.
    ROOTS=roots.json OUT=collect_dir GPU=.. WORKER=.. NWORKERS=.. [GROUP=train_unit] run_habitat.sh recovery_collect.py
"""
import io, json, math, os, sys, time, traceback, zlib

import numpy as np

REPO = os.environ.get("LMM_REPO", "/workspace/lmm")
sys.path.insert(0, REPO)
import magnum as mn  # noqa: E402
from lmm.embodiments.alphabot2 import cartesian as C  # noqa: E402
from lmm.embodiments.alphabot2 import ovmm_policy_io as IO  # noqa: E402
from lmm.embodiments.alphabot2.habitat import env as HE  # noqa: E402
from lmm.embodiments.alphabot2.habitat import nav as NAV  # noqa: E402
from lmm.embodiments.alphabot2.habitat.expert import OVMMExpert  # noqa: E402

OUT = os.environ["OUT"]; os.makedirs(f"{OUT}/episodes", exist_ok=True)
GROUP = os.environ.get("GROUP", "train_unit")
assert GROUP == "train_unit", "recovery demonstrations are recorded for training-unit roots only (held-out isolation)"
WK, NW = int(os.environ.get("WORKER", 0)), int(os.environ.get("NWORKERS", 1))
PREFIX, REACH_ONSET, EXEC = 15, 15, 5
CAMS = ("head_rgb", "left_wrist_rgb", "right_wrist_rgb")
CODE_COMMIT = os.popen(f"git -C {REPO} rev-parse --short HEAD").read().strip()
plan = json.load(open(os.environ["ROOTS"]))
roots = [r for r in plan["roots"] if r["group"] == GROUP][WK::NW]
env = HE.make_env(gpu=int(os.environ.get("GPU", "0")), split="train")


def render_now():
    HE.robot(env).update()
    o = env.sim.get_sensor_observations()
    return {k: np.asarray(o[k])[..., :3].astype(np.uint8) for k in CAMS}


def save_npz(path, **arrays):
    buf = io.BytesIO(); np.savez_compressed(buf, **arrays); data = buf.getvalue()
    with open(path, "wb") as f:
        f.write(data)
    return len(data)


def recovery_stage(stage, after_nav):
    if stage == "tuck":
        return "recovery_tuck"
    if stage == "navigate_to_object":
        return "recovery_navigate"
    if stage == "settle":
        return "settle"
    return "recovery_post"


res_f = f"{OUT}/results_w{WK:02d}.jsonl"
done = {json.loads(l)["episode_id"] for l in open(res_f)} if os.path.exists(res_f) else set()
for root in roots:
    eid = f"rec:{root['root_id']}"
    if eid in done:
        continue
    t0 = time.time()
    ep_id, rep = HE.parse_unit(root["unit"])
    rec = {"episode_id": eid, "episode": f"rec_{ep_id}", "replica": rep, "source_unit": root["unit"], "root": root,
           "group": root["group"], "code_commit": CODE_COMMIT, "kind": "expert_recovery"}
    buf = {"imgs": {k: [] for k in CAMS}, "proprio": [], "action": [], "joint18": [], "stage": [], "sim_time": [], "base": []}
    try:
        low = np.load(f"{root['source']}/lowdim.npz")
        tr = np.load(root["trace"]); act, pose = tr["act"], tr["pose"]
        HE.reset_to(env, ep_id, rep)
        ep = env.current_episode
        rec.update({"scene": ep.scene_id.split("/")[-1].split(".")[0], "object": ep.object_category,
                    "start_recep": ep.start_recep_category, "goal_recep": ep.goal_recep_category,
                    "instruction": IO.instruction(ep.object_category, ep.start_recep_category, ep.goal_recep_category)})
        HE.fix_start(env)
        for a in low["joint18"][: root["t"]]:
            HE.step18(env, a)
        rob = HE.robot(env)
        render_now()                                                     # perturb_eval.perturb("none"): same calls
        T0 = rob.sim_obj.transformation
        rob.sim_obj.transformation = mn.Matrix4.from_(T0.rotation(), T0.translation); rob.update()
        NAV.scene_contacts(env)
        render_now()
        ex = C.CartesianExecutor.from_proprio(rob.proprio16())
        r = root["r"]
        for k in range(r):
            if k % EXEC == 0:
                render_now()                                             # the original rollout observed here
            joint, _ = ex.lower(act[k])
            if k >= r - PREFIX:                                          # B's own approach: history context
                im = render_now()
                for c in CAMS:
                    buf["imgs"][c].append(im[c])
                buf["proprio"].append(rob.proprio16()); buf["action"].append(np.asarray(act[k], np.float64))
                buf["joint18"].append(np.asarray(joint, np.float64)); buf["stage"].append("recovery_prefix")
                buf["sim_time"].append(float(env.sim.get_world_time()))
                xz, yaw, _ = NAV.pose2d(env); buf["base"].append([float(xz[0]), float(xz[1]), float(yaw)])
            HE.step18(env, joint)
        xz, yaw, _ = NAV.pose2d(env)
        rec["root_check"] = {"pos_m": float(np.linalg.norm(xz - pose[r, :2])), "yaw_deg": float(abs(math.degrees(NAV.wrap(yaw - pose[r, 2]))))}
        if rec["root_check"]["pos_m"] > 0.02 or rec["root_check"]["yaw_deg"] > 1.0:
            raise ValueError("root_mismatch")
        orig, pending = HE.step18, {}

        def on_step(cart, stage, rep_):
            im = render_now()
            for c in CAMS:
                buf["imgs"][c].append(im[c])
            buf["proprio"].append(HE.robot(env).proprio16()); buf["action"].append(np.asarray(cart, np.float64))
            buf["stage"].append(stage); buf["sim_time"].append(float(env.sim.get_world_time()))
            xz_, yaw_, _ = NAV.pose2d(env); buf["base"].append([float(xz_[0]), float(xz_[1]), float(yaw_)])
            pending["armed"] = True

        def rec_step(env_, a):
            if pending.pop("armed", False):
                buf["joint18"].append(np.asarray(a, np.float64).copy())
            return orig(env_, a)
        HE.step18 = rec_step
        expert = OVMMExpert(env, on_step=on_step)
        try:
            expert.run(); rec["expert_completed"] = True
        except Exception as e:
            rec["expert_completed"] = False; rec["failure"] = repr(e)[:300]
        finally:
            HE.step18 = orig
        m = env.get_metrics()
        rec["ovmm"] = {k: float(bool(any(v.values()) if isinstance(v, dict) else v)) for k, v in m.items()
                       if k.startswith("ovmm_") and (isinstance(v, (int, float, bool)) or isinstance(v, dict))}
        # recovery stage names: prefix | tuck | navigate | reach onset (first REACH_ONSET after navigation) | post | settle
        raw = buf["stage"]; out_st, nav_end = [], None
        for i, s in enumerate(raw):
            if s == "navigate_to_object":
                nav_end = i
        for i, s in enumerate(raw):
            if s == "recovery_prefix":
                out_st.append(s)
            elif nav_end is not None and nav_end < i <= nav_end + REACH_ONSET and s not in ("navigate_to_object", "settle"):
                out_st.append("recovery_reach_onset")
            else:
                out_st.append(recovery_stage(s, nav_end is not None and i > nav_end))
        rec.update({"arm": expert.log.get("arm"), "steps": len(buf["action"]), "prefix_steps": PREFIX, "events": expert.events,
                    "nav_to_object": expert.log.get("nav_to_object"),
                    "stage_counts": {s: out_st.count(s) for s in sorted(set(out_st))}})
        lens = {len(buf["proprio"]), len(buf["action"]), len(buf["joint18"]), len(buf["stage"]), len(buf["base"])} | {len(v) for v in buf["imgs"].values()}
        if len(lens) != 1:
            raise RuntimeError(f"misaligned buffers {lens}")
        rec["success"] = bool(rec["expert_completed"] and (rec["nav_to_object"] or {}).get("arrived"))
        rec["accepted"] = rec["success"]
        d_ep = f"{OUT}/episodes/{root['unit'].replace('#', '_r')}_t{root['t']}_s{root['seed']}_r{r}"
        os.makedirs(d_ep, exist_ok=True); rec["dir"] = d_ep
        nb = save_npz(f"{d_ep}/images.npz", **{c: np.asarray(v, np.uint8) for c, v in buf["imgs"].items()})
        nb += save_npz(f"{d_ep}/lowdim.npz", proprio=np.asarray(buf["proprio"]), action=np.asarray(buf["action"]),
                       joint18=np.asarray(buf["joint18"]), sim_time=np.asarray(buf["sim_time"]))
        nb += save_npz(f"{d_ep}/metadata.npz", stage=np.asarray(out_st), stage_raw=np.asarray(raw), base_xz_yaw=np.asarray(buf["base"]),
                       stage_id=np.asarray([zlib.crc32(str(s).encode()) for s in out_st], np.uint32))
        with open(f"{d_ep}/meta.json", "w") as f:
            json.dump({**rec, "stages": out_st, "policy_inputs": list(CAMS) + ["proprio16", "instruction"]}, f, indent=1,
                      default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        rec["bytes"] = nb
    except Exception as e:
        rec["engineering_error"] = repr(e)[:300]; rec["tb"] = traceback.format_exc()[-1200:]; rec["accepted"] = False
    rec["wall_s"] = round(time.time() - t0, 1)
    with open(res_f, "a") as f:
        f.write(json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
    print("RECCOLLECT", json.dumps({k: rec.get(k) for k in ("episode_id", "root_check", "accepted", "steps", "stage_counts",
                                                            "failure", "engineering_error")}), flush=True)
env.close()
