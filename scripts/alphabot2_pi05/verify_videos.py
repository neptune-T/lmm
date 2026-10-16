#!/usr/bin/env python3
"""Hard video-integrity gate for converted OVMM LeRobot datasets (run after conversion, before any SFT).

1. decoded frames: EVERY video file of every camera is fully decoded (PyAV); its frame count must equal the sum
   of its episodes' lengths, and per episode the decoded frames inside [from_timestamp, to_timestamp) must equal
   the episode length, at 1/fps spacing.
2. dropped-frame logs: no conversion log may contain lerobot's "dropped ... frame" warning, and every shard log
   must end with its ###CONVERT### record.
3. alignment: across shards (first / middle / last episode of every shard) and across video files (first / last
   episode of every camera_top file), sampled frames read through LeRobotDataset (the training loader) must match
   the source images.npz for all cameras (PSNR >= 30 dB, no offset within +-3 frames matching > 1 dB better) and the
   source proprio/action labels.
    verify_videos.py --dataset TRAIN_DIR:24 VAL_DIR:2 --logs 'LOGDIR/v2_convert_*.log' --out verify.json
"""
import argparse, glob, json, math, os, re, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = os.environ.get("LMM_REPO", str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, REPO)
from lmm.embodiments.alphabot2 import ovmm_policy_io as IO  # noqa: E402

PSNR_MIN, OFFSETS, SHIFT_DB = 30.0, range(-3, 4), 1.0     # a shift = another offset matching > 1 dB better


def decode_times(path):
    import av
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        return str(path), [float(f.time) for f in c.decode(s)]


def psnr(a, b):
    m = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if m == 0 else 10 * math.log10(255.0 ** 2 / m)


def align_episode(job):
    root, e, nfr = job
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset("local/verify", root=root)
    ent = json.load(open(Path(root) / "ovmm_episode_index.json"))["episodes"][e]
    src = Path(ent["source"]); ims = np.load(src / "images.npz"); low = np.load(src / "lowdim.npz")
    T, s0 = ent["frames"], int(ds.meta.episodes["dataset_from_index"][e])
    out = {"episode": e, "unit": ent["unit"], "frames": T, "checked": [], "fail": []}
    for t in sorted(set(np.linspace(0, T - 1, nfr).astype(int).tolist())):
        it = ds[s0 + t]
        if int(it["episode_index"]) != e or int(it["frame_index"]) != t:
            out["fail"].append({"t": t, "why": "index"}); continue
        st = float(np.abs(it["observation.state"].numpy() - IO.state_from_proprio(low["proprio"][t])).max())
        ac = float(np.abs(it["action"].numpy() - low["action"][t]).max())
        rec = {"t": t, "state_err": st, "action_err": ac}
        for k, cam in IO.CAMERA_MAP.items():
            img = it[f"observation.images.{k}"].permute(1, 2, 0).numpy() * 255.0
            sc = {o: psnr(img, ims[cam][t + o]) for o in OFFSETS if 0 <= t + o < T}
            best = max(sc, key=sc.get)
            gain = sc[best] - sc[0]                                       # static frames (settle) tie across offsets
            rec[k] = {"psnr": round(sc[0], 2), "best_offset": best, "best_gain_db": round(gain, 2)}
            if sc[0] < PSNR_MIN or gain > SHIFT_DB:
                out["fail"].append({"t": t, "camera": k, "psnr": round(sc[0], 2), "best_offset": best, "best_gain_db": round(gain, 2)})
        if st > 1e-5 or ac > 1e-5:
            out["fail"].append({"t": t, "why": "labels", "state_err": st, "action_err": ac})
        out["checked"].append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", required=True, help="DIR:NSHARDS")
    ap.add_argument("--logs", required=True, help="glob of the conversion shard logs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--align-workers", type=int, default=16)
    ap.add_argument("--frames", type=int, default=8, help="frames per alignment episode (incl. first and last)")
    a = ap.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    rep, ok = {"datasets": {}}, {}

    logs = sorted(glob.glob(a.logs))
    drops = {l: len(re.findall(r"dropped.*frame", open(l, errors="replace").read())) for l in logs}
    unfinished = [l for l in logs if "###CONVERT###" not in open(l, errors="replace").read()]
    rep["logs"] = {"n": len(logs), "with_drops": {l: n for l, n in drops.items() if n}, "without_convert_record": unfinished}
    ok["logs_no_dropped_frames"] = len(logs) > 0 and not any(drops.values())
    ok["logs_all_shards_finished"] = len(logs) > 0 and not unfinished

    for spec in a.dataset:
        root, nsh = spec.rsplit(":", 1); nsh = int(nsh); name = Path(root).name
        meta = LeRobotDatasetMetadata("local/verify", root=root)
        E, fps = meta.episodes, meta.fps
        n_ep = len(E["dataset_from_index"])
        index = json.load(open(Path(root) / "ovmm_episode_index.json"))
        files = {}                                                     # (key, path) -> [episode ...]
        for e in range(n_ep):
            for key in meta.video_keys:
                p = Path(root) / meta.get_video_file_path(e, key)
                files.setdefault((key, str(p)), []).append(e)
        with Pool(a.workers) as pool:
            times = dict(pool.map(decode_times, sorted({p for _, p in files})))
        bad_files, bad_eps, n_frames = [], [], 0
        for (key, p), eps in sorted(files.items()):
            t = np.asarray(times[p]); n_frames += len(t)
            want = sum(int(E["length"][e]) for e in eps)
            if len(t) != want:
                bad_files.append({"file": p, "decoded": len(t), "expected": want})
            for e in eps:
                lo, hi = float(E[f"videos/{key}/from_timestamp"][e]), float(E[f"videos/{key}/to_timestamp"][e])
                sel = t[(t >= lo - 0.5 / fps) & (t < hi - 0.5 / fps)]
                L = int(E["length"][e])
                spacing_ok = len(sel) < 2 or float(np.abs(np.diff(sel) - 1.0 / fps).max()) < 0.25 / fps
                if len(sel) != L or not spacing_ok:
                    bad_eps.append({"episode": e, "key": key, "decoded_in_window": len(sel), "length": L, "spacing_ok": spacing_ok})
        lengths_ok = [int(E["length"][e]) for e in range(n_ep)] == [int(u["frames"]) for u in index["episodes"]]
        # alignment sample: first / middle / last of every shard + first / last of every camera_top file
        per = -(-n_ep // nsh)
        pick = set()
        for k in range(nsh):
            lo, hi = k * per, min(n_ep, (k + 1) * per) - 1
            if lo <= hi:
                pick |= {lo, (lo + hi) // 2, hi}
        for (key, p), eps in files.items():
            if key == "observation.images.camera_top":
                pick |= {eps[0], eps[-1]}
        with Pool(a.align_workers) as pool:
            al = pool.map(align_episode, [(root, e, a.frames) for e in sorted(pick)])
        al_fail = [x for x in al if x["fail"]]
        psnrs = [r[k]["psnr"] for x in al for r in x["checked"] for k in IO.CAMERA_MAP]
        rep["datasets"][name] = {"episodes": n_ep, "video_files": len(files), "decoded_frames_all_cameras": n_frames,
                                 "expected_frames_all_cameras": len(meta.video_keys) * int(sum(int(x) for x in E["length"])),
                                 "bad_files": bad_files[:50], "n_bad_files": len(bad_files),
                                 "bad_episodes": bad_eps[:50], "n_bad_episode_videos": len(bad_eps),
                                 "lengths_match_index": lengths_ok, "shards": nsh,
                                 "alignment": {"episodes": len(al), "frames": sum(len(x["checked"]) for x in al),
                                               "psnr_min": min(psnrs) if psnrs else None,
                                               "psnr_median": float(np.median(psnrs)) if psnrs else None,
                                               "failures": al_fail[:20], "n_fail_episodes": len(al_fail)}}
        ok[f"{name}_decoded_frame_counts"] = not bad_files and not bad_eps and lengths_ok
        ok[f"{name}_alignment"] = bool(al) and not al_fail
        print(name, json.dumps({k: v for k, v in rep["datasets"][name].items() if k not in ("bad_files", "bad_episodes")}, default=str)[:900], flush=True)
    rep.update({"checks": ok, "pass": all(ok.values())})
    Path(a.out).write_text(json.dumps(rep, indent=1, default=str))
    print("###VIDEO_VERIFY###" + json.dumps({"pass": rep["pass"], "checks": ok}), flush=True)
    sys.exit(0 if rep["pass"] else 1)


if __name__ == "__main__":
    main()
