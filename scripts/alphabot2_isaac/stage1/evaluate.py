"""Closed-loop evaluation of a pi0.5 policy from NORMAL task initialization.

Policy inputs are only what ``SimWorld.observe()`` returns -- the four camera
images (checkpoint keys, same mapping for every checkpoint), the 16-D state in
the checkpoint convention -- plus the language instruction. No expert stage,
privileged pose, scripted navigation, grasp oracle or waypoint reaches the
policy. The policy drives arms, grippers AND base through CartesianMobileAction18.

Receding horizon: the policy predicts a 50-step chunk; the first ``H_EXEC``
actions are executed, then a new observation is taken and the policy replans.

Metrics (privileged, evaluation only, shared predicates with the expert):
success (sustained 5 steps), reach / grasp / lift / place progress, reaching
the table-B area, collisions, base path length (-> path efficiency vs the
expert on the same config), action discontinuity at replanning boundaries,
inference latency and wall cost.

    CONFIGS=... OUT=... PORTS=5600,5601 H_EXEC=10 TAG=sft_h10 WORKER=k NWORKERS=n python.sh evaluate.py
"""
import json, math, os, sys, time, traceback, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import simworld as SW  # noqa: E402

app, ZD = SW.launch()
import numpy as np  # noqa: E402
import expert as E  # noqa: E402
from policy_client import PolicyClient  # noqa: E402
from lmm.embodiments.alphabot2 import pi05_contract as P  # noqa: E402

CONFIGS, OUT, TAG = os.environ["CONFIGS"], os.environ["OUT"], os.environ["TAG"]
WK, NW = int(os.environ.get("WORKER", 0)), int(os.environ.get("NWORKERS", 1))
PORTS = [int(p) for p in os.environ["PORTS"].split(",")]
H_EXEC = int(os.environ.get("H_EXEC", "10"))
SEED = int(os.environ.get("SEED", "0"))
SAVE_FRAMES = int(os.environ.get("SAVE_FRAMES", "0"))          # save head/wrist frames every N steps (0 = off)
MAX_STEPS = {"nav_pick_lift": 320, "nav_pick_nav_place": 700, "short_pick_place": 320, "reorient_pick_lift": 380}
SUSTAIN = 5
os.makedirs(f"{OUT}/{TAG}/episodes", exist_ok=True)
RES = f"{OUT}/{TAG}/results_w{WK:02d}.jsonl"
done = set()
if os.path.exists(RES):
    done = {json.loads(l)["config_id"] for l in open(RES) if l.strip()}
cfgs = [c for c in json.load(open(CONFIGS))[WK::NW] if c["config_id"] not in done]
W = SW.SimWorld(app, ZD)
client = PolicyClient(PORTS[WK % len(PORTS)])
GROUPS = {"left_xyz_m": slice(0, 3), "right_xyz_m": slice(7, 10), "grip": slice(14, 16), "v": slice(16, 17), "omega": slice(17, 18)}


def quat_deg(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    return float(np.degrees(2 * np.arccos(min(1.0, abs(float(a @ b))))))


def act_diff(a, b):
    d = {g: float(np.max(np.abs(a[s] - b[s]))) for g, s in GROUPS.items()}
    d["left_quat_deg"] = quat_deg(a[3:7], b[3:7])
    d["right_quat_deg"] = quat_deg(a[10:14], b[10:14])
    return d


def table_b_standoff(cfg):
    bx, by, _, _ = cfg["box"]
    tyaw = cfg["table_b"][2]
    c, s = math.cos(tyaw), math.sin(tyaw)
    nx, ny = 0.45, (0.25 if cfg["arm"] == "left" else -0.25)
    return np.array([bx - (c * nx - s * ny), by - (s * nx + c * ny)]), tyaw


def run(cfg):
    ex = E.Expert(W, cfg)
    ex.reset()
    side, fam = cfg["arm"], cfg["family"]
    rest_z = ex.obj_rest_z
    tb_xy, tb_yaw = table_b_standoff(cfg)
    T_max = MAX_STEPS[fam]
    chunk, k_in_chunk, prev_exec = None, 0, None
    m = {"reached": False, "grasped": False, "lifted": False, "placed": False, "at_table_b": False,
         "collision_steps": 0, "collision_pairs": {}, "ik_invalid_steps": 0, "path_m": 0.0, "infer_calls": 0,
         "infer_latency_s": [], "roundtrip_s": [], "batch_sizes": [], "disc_boundary": [], "disc_within": [],
         "quat_norm_dev_max": 0.0, "min_tcp_obj_m": 9.0}
    trace = {"t": [], "chassis": [], "obj": [], "action": [], "success_flag": []}
    streak, success, stop = 0, False, None
    p_prev = W.chassis_pose()[0][:2]
    t_wall = time.time()
    frames = []
    for t in range(T_max):
        ob = W.observe()
        if chunk is None or k_in_chunk >= H_EXEC:
            imgs = {k: ob.images[src] for k, src in P.SIM_CAMERA_FOR.items()}
            seed = zlib.crc32(f"{cfg['config_id']}:{t}:{SEED}".encode())
            r = client.infer(imgs, P.dataset_state_from_proprio(ob.proprio), cfg["instruction"], seed)
            new = np.asarray(r["actions"], float)
            if chunk is not None and prev_exec is not None:
                m["disc_boundary"].append(act_diff(new[0], prev_exec))
            chunk, k_in_chunk = new, 0
            m["infer_calls"] += 1
            m["infer_latency_s"].append(r["latency_s"])
            m["roundtrip_s"].append(r["roundtrip_s"])
            m["batch_sizes"].append(r.get("batch_size", 1))
            qn = np.r_[np.linalg.norm(chunk[:, 3:7], axis=1), np.linalg.norm(chunk[:, 10:14], axis=1)]
            m["quat_norm_dev_max"] = max(m["quat_norm_dev_max"], float(np.max(np.abs(qn - 1))))
        a = chunk[k_in_chunk]
        if prev_exec is not None and k_in_chunk > 0:
            m["disc_within"].append(act_diff(a, prev_exec))
        if not np.isfinite(a).all():
            stop = "non-finite action"
            break
        _, rep = W.act(a)
        prev_exec, k_in_chunk = a, k_in_chunk + 1
        m["ik_invalid_steps"] += int(not (rep["left"]["ik_valid"] and rep["right"]["ik_valid"]))
        st = E.task_state(W, cfg, rest_z)
        cp, cyaw = W.chassis_pose()
        m["path_m"] += float(np.linalg.norm(cp[:2] - p_prev))
        p_prev = cp[:2]
        pads = [W.B.link_pose(f"Gripper_{E.SIDEP[side]}/{n}")[0] for n in ("left_pad", "right_pad")]
        d = float(np.linalg.norm((pads[0] + pads[1]) / 2 - st["object_pos"]))
        m["min_tcp_obj_m"] = min(m["min_tcp_obj_m"], d)
        m["reached"] |= d < 0.04
        m["grasped"] |= bool(st["grasped"])
        m["lifted"] |= bool(st["lifted_held"])
        m["placed"] |= bool(st["in_box_released"])
        if np.linalg.norm(cp[:2] - tb_xy) < 0.35 and abs(E.wrap(cyaw - tb_yaw)) < 0.4:
            m["at_table_b"] = True
        if st["collisions"]:
            m["collision_steps"] += 1
            for p in st["collisions"]:
                m["collision_pairs"][p] = m["collision_pairs"].get(p, 0) + 1
        ok = E.success_flag(cfg, st)
        streak = streak + 1 if ok else 0
        for k_, v_ in (("t", t), ("chassis", [*cp[:2], cyaw]), ("obj", st["object_pos"]), ("action", a), ("success_flag", ok)):
            trace[k_].append(v_)
        if SAVE_FRAMES and t % SAVE_FRAMES == 0:
            frames.append(np.concatenate([ob.images["cam_front"], ob.images["cam_left_wrist"], ob.images["cam_right_wrist"]], axis=1))
        if streak >= SUSTAIN:
            success = True
            stop = "success"
            break
        if st["on_floor"]:
            stop = "object dropped to the floor"
            break
    steps = len(trace["t"])
    wall = time.time() - t_wall

    def agg(lst):
        if not lst:
            return None
        keys = lst[0].keys()
        return {k: float(np.mean([x[k] for x in lst])) for k in keys}

    res = {"config_id": cfg["config_id"], "tag": TAG, "h_exec": H_EXEC, "family": fam, "arm": side, "object": cfg["object"],
           "cells": cfg["cells"], "split": cfg["split"], "success": success, "stop": stop or "step budget",
           "steps": steps, "reached": m["reached"], "grasped": m["grasped"], "lifted": m["lifted"], "placed": m["placed"],
           "at_table_b": m["at_table_b"], "collision": m["collision_steps"] > 0, "collision_steps": m["collision_steps"],
           "collision_pairs": m["collision_pairs"], "ik_invalid_steps": m["ik_invalid_steps"], "path_m": m["path_m"],
           "min_tcp_obj_m": m["min_tcp_obj_m"], "infer_calls": m["infer_calls"],
           "infer_latency_s_mean": float(np.mean(m["infer_latency_s"])) if m["infer_latency_s"] else None,
           "roundtrip_s_mean": float(np.mean(m["roundtrip_s"])) if m["roundtrip_s"] else None,
           "batch_size_mean": float(np.mean(m["batch_sizes"])) if m["batch_sizes"] else None,
           "disc_boundary_mean": agg(m["disc_boundary"]), "disc_within_mean": agg(m["disc_within"]),
           "quat_norm_dev_max": m["quat_norm_dev_max"], "wall_s": round(wall, 1), "wall_s_per_step": wall / max(1, steps),
           "sync_check_ok": W.last_sync_check["ok"]}
    np.savez_compressed(f"{OUT}/{TAG}/episodes/{cfg['config_id']}.npz",
                        **{k: np.asarray(v) for k, v in trace.items()},
                        **({"frames": np.asarray(frames, np.uint8)} if frames else {}))
    return res


for cfg in cfgs:
    try:
        res = run(cfg)
    except Exception as e:
        res = {"config_id": cfg["config_id"], "tag": TAG, "family": cfg["family"], "success": False,
               "engineering_error": repr(e), "tb": traceback.format_exc()[-1500:]}
    with open(RES, "a") as f:
        f.write(json.dumps(res, default=str) + "\n")
    print("EVAL", json.dumps({k: res.get(k) for k in ("config_id", "success", "stop", "steps", "reached", "grasped",
                                                      "lifted", "placed", "collision", "wall_s", "engineering_error")}),
          flush=True)
app.close()
