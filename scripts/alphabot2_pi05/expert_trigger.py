#!/usr/bin/env python3
"""Teacher-forced NAV mode-confusion probe: the policy server is fed the EXPERT's own navigate_to_object
observations (dataset frames, built exactly as diag_audit / training: same cameras, state, history) and its
predicted chunk is checked for a false "base stop + arm move" (the expert's NAV -> REACH transition) while the
expert is still far from where it stopped to reach.

Per query (every --stride frames of each episode's navigate_to_object segment), first H_EXEC predicted steps:
  stop        mean |v| < 0.03 and mean |omega| < 0.05
  arm_move    either arm's commanded xyz > 5 cm from the expert NAV (tucked) pose at any of the H_EXEC steps
  trigger     stop and arm_move
Distance = |base xz - base xz at the last navigate_to_object frame| (the expert's stop point at the object).
The expert's own labels on the same frames are scored too (must be ~0: sanity of the definition).
    expert_trigger.py --dataset DS_ROOT --episodes 0-39 --port 5651 [--history JSON] --out OUT.json
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ovmm_model  # noqa: E402
from ovmm_model import IO  # noqa: E402
from wire import recv_msg, send_msg  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

H_EXEC, STOP_V, STOP_W, ARM_DEV_M = 5, 0.03, 0.05, 0.05
BINS = [(0, 1), (1, 2), (2, 4), (4, 8), (8, 99)]
LXYZ, RXYZ = slice(0, 3), slice(7, 10)


def classify(chunk, ref):
    c = np.asarray(chunk)[:H_EXEC]
    stop = bool(np.abs(c[:, 16]).mean() < STOP_V and np.abs(c[:, 17]).mean() < STOP_W)
    dev = max(float(np.linalg.norm(c[:, LXYZ] - ref["left"], axis=1).max()), float(np.linalg.norm(c[:, RXYZ] - ref["right"], axis=1).max()))
    return stop, dev > ARM_DEV_M, dev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--episodes", default=None, help="a-b (dataset episode indices) or comma list of units")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--history", default=None)
    ap.add_argument("--expert-pose", required=True, help="stall_diag output json (expert_nav_pose)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    hist = json.loads(a.history) if a.history else None
    ref = {k: np.asarray(v) for k, v in json.load(open(a.expert_pose))["expert_nav_pose"].items() if k in ("left", "right")}
    index = json.load(open(Path(a.dataset) / "ovmm_episode_index.json"))["episodes"]
    if a.episodes and "-" in a.episodes and "#" not in a.episodes:
        lo, hi = map(int, a.episodes.split("-")); eps = list(range(lo, min(hi, len(index) - 1) + 1))
    elif a.episodes:
        want = set(a.episodes.split(",")); eps = [e for e, u in enumerate(index) if str(u["unit"]) in want]
    else:
        eps = list(range(len(index)))
    ds = LeRobotDataset(repo_id="local/trigger", root=a.dataset,
                        delta_timestamps=ovmm_model.history_delta_timestamps({"action": [i / 10 for i in range(50)]}, hist))
    starts = np.asarray(ds.meta.episodes["dataset_from_index"])
    sock = __import__("socket").create_connection(("127.0.0.1", a.port))
    send_msg(sock, {"cmd": "ping"}); recv_msg(sock)
    u8 = lambda x: (x.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
    Q = []
    for e in eps:
        ent = index[e]; src = Path(ent["source"])
        md = np.load(src / "metadata.npz"); st = [str(x) for x in md["stage"]][: ent["frames"]]
        nav = [t for t, s in enumerate(st) if s == "navigate_to_object"]
        if len(nav) < 2:
            continue
        xz = md["base_xz_yaw"][:, :2]; stop_pt = xz[nav[-1]]
        for t in nav[:: a.stride]:
            it = ds[int(starts[e]) + t]
            if hist:
                hx = it[f"observation.images.{hist['camera']}"]
                it = {**it, f"observation.images.{hist['camera']}": hx[-1]}
            imgs = {c: u8(it[f"observation.images.{c}"]) for c in IO.CAMERA_MAP}
            if hist:
                for j, k in enumerate(sorted(hist["offsets"], reverse=True)):
                    imgs[f"{hist['camera']}_h{k}"] = u8(hx[j])
            send_msg(sock, {"cmd": "infer", "images": imgs, "state": it["observation.state"].numpy().astype(np.float32),
                            "task": it["task"], "seed": 1000 + int(starts[e]) + t})
            r = recv_msg(sock)
            ps, pa, pdev = classify(r["actions"], ref)
            es, ea, edev = classify(it["action"].numpy(), ref)
            Q.append({"ep": e, "unit": ent["unit"], "t": t, "dist": float(np.linalg.norm(xz[t] - stop_pt)),
                      "stop": ps, "arm": pa, "trig": ps and pa, "dev": round(pdev, 4),
                      "v": float(np.mean(np.asarray(r["actions"])[:H_EXEC, 16])), "expert_trig": es and ea, "expert_stop": es})
        print(f"ep {e} {ent['unit']}: {sum(1 for q in Q if q['ep'] == e)} queries", flush=True)
    rep = {"dataset": a.dataset, "episodes": len({q["ep"] for q in Q}), "queries": len(Q), "history": hist, "bins": {}}
    for lo, hi in BINS:
        s = [q for q in Q if lo <= q["dist"] < hi]
        if s:
            rep["bins"][f"{lo}-{hi}m"] = {"queries": len(s), "trigger": round(np.mean([q["trig"] for q in s]), 4),
                                         "stop": round(np.mean([q["stop"] for q in s]), 4), "arm_move": round(np.mean([q["arm"] for q in s]), 4),
                                         "v_mean": round(float(np.mean([q["v"] for q in s])), 4),
                                         "expert_trigger": round(np.mean([q["expert_trig"] for q in s]), 4),
                                         "expert_stop": round(np.mean([q["expert_stop"] for q in s]), 4)}
    far = [q for q in Q if q["dist"] > 2]
    rep["far_gt_2m"] = {"queries": len(far), "trigger": round(np.mean([q["trig"] for q in far]), 4) if far else None,
                        "episodes_with_any_trigger": len({q["ep"] for q in far if q["trig"]})}
    rep["queries_detail"] = Q
    Path(a.out).write_text(json.dumps(rep, indent=1, default=str))
    print("###TRIGGER###" + json.dumps({k: v for k, v in rep.items() if k != "queries_detail"}), flush=True)


if __name__ == "__main__":
    main()
