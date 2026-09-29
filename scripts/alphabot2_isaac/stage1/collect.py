"""Stage-1 demonstration collection worker: record -> save -> replay -> gate.

One Isaac process per worker (CPU PhysX does not scale inside one scene).
Configs come from a JSON list (stage1_tasks.generate); worker k of n takes
every n-th config. Resumable: configs with a result line are skipped.

Per step t the recorder stores, from ``SimWorld.observe()`` only:
images (4 cameras), canonical 16-D proprio, and the timing record
(policy_step, image/proprio/obs frame + time), then the CartesianMobileAction18
label with its action frame/time. Privileged metadata (object pose, chassis
pose, grasp, stage, contacts) is stored separately and never a policy input.

Every saved episode is replayed from its initial configuration with the
recorded actions (rendering off; physics is unaffected) and gated
(lmm/embodiments/alphabot2/stage1_gates.py). Failures go to quarantine/.

    CONFIGS=... OUT=... WORKER=k NWORKERS=n  python.sh collect.py
"""
import json, os, shutil, sys, time, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import simworld as SW  # noqa: E402

app, ZD = SW.launch()
import numpy as np  # noqa: E402
import expert as E  # noqa: E402
from lmm.embodiments.alphabot2 import cartesian as C  # noqa: E402
from lmm.embodiments.alphabot2 import stage1_gates as G  # noqa: E402

CONFIGS = os.environ["CONFIGS"]
OUT = os.environ["OUT"]
WK, NW = int(os.environ.get("WORKER", 0)), int(os.environ.get("NWORKERS", 1))
LIMIT = int(os.environ.get("LIMIT", 0))
SAVE_FAILED_IMAGES = bool(int(os.environ.get("SAVE_FAILED_IMAGES", "0")))
STAGES = ["init", "adjust_base", "navigate_to_object", "reach", "descend", "grasp", "lift", "back_off",
          "navigate_to_box", "reach_box", "lower", "release", "retract", "done"]
os.makedirs(f"{OUT}/results", exist_ok=True)
RES = f"{OUT}/results/worker{WK:02d}.jsonl"
done = set()
if os.path.exists(RES):
    for line in open(RES):
        try:
            done.add(json.loads(line)["config_id"])
        except Exception:
            pass
cfgs = json.load(open(CONFIGS))[WK::NW]
if LIMIT:
    cfgs = cfgs[:LIMIT]
W = SW.SimWorld(app, ZD)


class Recorder:
    def __init__(self):
        self.low = {k: [] for k in ("proprio", "action", "policy_step", "image_frame", "proprio_frame", "obs_frame",
                                    "image_time", "proprio_time", "action_frame", "action_time", "obj_pose", "chassis",
                                    "grasp", "stage_id", "ik_valid", "ik_pos_err_m", "collision")}
        self.imgs = {k: [] for k in SW.CAMS}
        self.collision_pairs = {}

    def on_step(self, ob, cart, info):
        L = self.low
        for k in SW.CAMS:
            self.imgs[k].append(ob.images[k])
        L["proprio"].append(ob.proprio)
        L["action"].append(cart)
        for k in ("policy_step", "image_frame", "proprio_frame", "obs_frame", "image_time", "proprio_time"):
            L[k].append(getattr(ob, k))
        L["action_frame"].append(W.physics_frame)          # the action starts applying at this physics frame
        L["action_time"].append(W.sim_time())
        op, oq = W.object_pose()
        L["obj_pose"].append(np.r_[op, oq])
        cp, cy = W.chassis_pose()
        L["chassis"].append(np.r_[cp, cy])
        st = E.task_state(W, CFG, self.rest_z)
        L["grasp"].append(st["grasped"])
        L["stage_id"].append(STAGES.index(info["stage"]))
        L["collision"].append(bool(st["collisions"]))
        for p in st["collisions"]:
            self.collision_pairs[p] = self.collision_pairs.get(p, 0) + 1

    def on_after(self, rep):
        self.low["ik_valid"].append([rep["left"]["ik_valid"], rep["right"]["ik_valid"]])
        self.low["ik_pos_err_m"].append([rep["left"]["pos_err_m"], rep["right"]["pos_err_m"]])


def outcome(ex, cfg):
    st = E.task_state(W, cfg, ex.obj_rest_z)
    ok = E.success_flag(cfg, st)
    return {"success": bool(ok), "reason": None if ok else "final state fails the task predicate",
            "object_final": [round(float(v), 4) for v in st["object_pos"]], "lifted_held": st["lifted_held"],
            "in_box_released": st["in_box_released"], "on_floor": st["on_floor"]}


def replay(cfg, actions, proprio_rec):
    ex = E.Expert(W, cfg)
    ex.reset()
    props = []
    try:
        W.set_render(False)
        for a in actions:
            props.append(W.observe(need_images=False).proprio)
            W.act(a)
    finally:
        W.set_render(True)
    ok = E.success_flag(cfg, E.task_state(W, cfg, ex.obj_rest_z))
    props = np.asarray(props)
    return {"success": bool(ok), "proprio_max_abs_diff": float(np.max(np.abs(props - proprio_rec))),
            "bitwise_identical": bool(np.array_equal(props, proprio_rec)), "sync_check": W.last_sync_check["ok"]}


for cfg in cfgs:
    if cfg["config_id"] in done:
        continue
    CFG = cfg
    t0 = time.time()
    rec = Recorder()
    ex = E.Expert(W, cfg, on_step=rec.on_step, on_after=rec.on_after)
    err = None
    try:
        ex.reset()
        rec.rest_z = ex.obj_rest_z
        sync = W.last_sync_check
        ex.run()
        out = outcome(ex, cfg)
    except Exception as e:
        err = repr(e)
        out = {"success": False, "reason": err, "tb": traceback.format_exc()[-1500:]}
        sync = getattr(W, "last_sync_check", {})
    rec_s = time.time() - t0
    T = len(rec.low["action"])
    low = {k: np.asarray(v) for k, v in rec.low.items()}
    meta = {"config": cfg, "instruction": cfg["instruction"], "family": cfg["family"], "arm": cfg["arm"],
            "object": cfg["object"], "split": cfg["split"], "cells": cfg["cells"], "steps": T, "fps": 10,
            "zero_delay": ZD, "render_lag": W.render_lag, "sync_check": sync, "outcome": out,
            "events": ex.events, "expert_ik": ex.ik_log, "stages": STAGES, "collision_pairs": rec.collision_pairs,
            "action_space": "CartesianMobileAction18", "action_labels": list(C.CARTESIAN_MOBILE_ACTION18_LABELS),
            "cameras": list(SW.CAMS), "head_pose": list(W.head),
            "policy_inputs": ["cam_front", "cam_left_wrist", "cam_right_wrist", "cam_chest", "proprio", "instruction"],
            "metadata_only": ["stage_id", "events", "obj_pose", "chassis", "grasp", "collision", "config", "cells"],
            "record_wall_s": round(rec_s, 1)}
    rp = None
    if out["success"] and T > 0:
        t1 = time.time()
        try:
            rp = replay(cfg, low["action"], low["proprio"])
        except Exception as e:
            rp = {"success": False, "error": repr(e)}
        rp["wall_s"] = round(time.time() - t1, 1)
    imgs = {k: np.asarray(v, np.uint8) for k, v in rec.imgs.items()} if T > 0 else None
    gates = G.gate_episode(low, imgs, meta, rp) if T > 0 else {"task_success": (False, out.get("reason"))}
    ok = G.passed(gates)
    meta["replay"] = rp
    meta["gates"] = {k: {"pass": bool(v[0]), "detail": v[1]} for k, v in gates.items()}
    meta["accepted"] = ok
    d = f"{OUT}/{'episodes' if ok else 'quarantine'}/{cfg['config_id']}"
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)
    if T > 0:
        np.savez_compressed(f"{d}/lowdim.npz", **low)
        if ok or SAVE_FAILED_IMAGES:
            np.savez_compressed(f"{d}/images.npz", **imgs)
    with open(f"{d}/meta.json", "w") as f:
        json.dump(meta, f, indent=1, default=str)
    res = {"config_id": cfg["config_id"], "split": cfg["split"], "family": cfg["family"], "arm": cfg["arm"],
           "object": cfg["object"], "cells": cfg["cells"], "steps": T, "success": out["success"], "accepted": ok,
           "failing_gates": G.failing(gates), "error": err, "record_wall_s": round(rec_s, 1),
           "replay_wall_s": rp.get("wall_s") if rp else None, "wall_s": round(time.time() - t0, 1)}
    with open(RES, "a") as f:
        f.write(json.dumps(res) + "\n")
    print("EPISODE", json.dumps(res), flush=True)
app.close()
