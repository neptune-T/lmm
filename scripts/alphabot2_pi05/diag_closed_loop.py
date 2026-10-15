"""Closed-loop drift analysis of traced eval rollouts (eval_policy.py TRACE_DIR) vs the expert (diagnostics).

Per traced unit: the policy's base trajectory and commanded v / omega against, for train-manifest units,
the expert's own recorded trajectory of the same unit (same seeded start; collection episode dir):
  min_dist_m / first_step_within_1m   distance from the base to the target object (start pose)
  path_m                              base path length
  stall_frac                          fraction of steps with |v| < 0.02 and |omega| < 0.05
  omega_flip_rate                     sign flips of omega per 100 steps where |omega| > 0.05 (oscillation)
  v_mean / abs_omega_mean             commanded base velocities
Also renders each trace's frame strip to an mp4 (PyAV) with the step, v, omega and distance burnt in.

    python diag_closed_loop.py --traces DIR --manifest M.json --out OUT.json [--videos VDIR]
"""
import argparse, glob, json, os

import numpy as np


def stats(pose, v, w, dist):
    d = np.diff(pose[:, :2], axis=0)
    big = np.abs(w) > 0.05
    sw = np.sign(w[big])
    within = np.flatnonzero(dist < 1.0)
    return {"steps": int(len(v)), "min_dist_m": round(float(dist.min()), 3), "start_dist_m": round(float(dist[0]), 3),
            "first_step_within_1m": int(within[0]) if len(within) else None,
            "path_m": round(float(np.linalg.norm(d, axis=1).sum()), 2),
            "stall_frac": round(float(((np.abs(v) < 0.02) & (np.abs(w) < 0.05)).mean()), 3),
            "omega_flip_rate_per100": round(float((sw[1:] != sw[:-1]).sum() * 100 / max(1, len(w))), 2),
            "v_mean": round(float(v.mean()), 3), "abs_omega_mean": round(float(np.abs(w).mean()), 3)}


def video(npz, path, fps=10):
    import av
    from PIL import Image, ImageDraw
    t = np.load(npz)
    fr, fs, act, dist = t["frames"], t["frame_steps"], t["act"], t["dist"]
    if not len(fr):
        return False
    c = av.open(path, "w")
    s = c.add_stream("libx264", rate=fps)
    s.width, s.height, s.pix_fmt = fr.shape[2], fr.shape[1] + 20, "yuv420p"
    for f, k in zip(fr, fs):
        im = Image.new("RGB", (fr.shape[2], fr.shape[1] + 20))
        im.paste(Image.fromarray(f), (0, 20))
        k = min(int(k), len(act) - 1)
        ImageDraw.Draw(im).text((4, 4), f"step {k}  v {act[k, 16]:+.2f}  w {act[k, 17]:+.2f}  dist {dist[k]:.2f} m  "
                                        f"grip L{act[k, 14]:.1f} R{act[k, 15]:.1f}", fill=(255, 255, 0))
        for p in s.encode(av.VideoFrame.from_image(im)):
            c.mux(p)
    for p in s.encode():
        c.mux(p)
    c.close()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--videos", default=None)
    a = ap.parse_args()
    m = json.load(open(a.manifest))
    dirs = {d["unit"]: d["dir"] for d in m["train_detail"] + m["val_detail"]}
    rep = {}
    for f in sorted(glob.glob(os.path.join(a.traces, "*.npz"))):
        unit = os.path.basename(f)[:-4].replace("_r", "#")
        t = np.load(f)
        pose, act, dist = t["pose"], t["act"], t["dist"]
        r = {"funnel": json.loads(str(t["funnel"])), "policy": stats(pose, act[:, 16], act[:, 17], dist)}
        if unit in dirs:                                          # trained-on unit: the expert's own run
            md = np.load(os.path.join(dirs[unit], "metadata.npz"))
            low = np.load(os.path.join(dirs[unit], "lowdim.npz"))
            ep, ea = np.asarray(md["base_xz_yaw"], float), np.asarray(low["action"], float)
            stg = [str(s) for s in md["stage"]]
            nav_end = next((i for i, s in enumerate(stg) if s not in ("tuck", "navigate_to_object", "init")), len(stg))
            r["expert"] = {"steps": len(ea), "steps_until_reach": nav_end, "path_m_until_reach": round(float(
                np.linalg.norm(np.diff(ep[:nav_end, :2], axis=0), axis=1).sum()), 2),
                "v_mean_until_reach": round(float(ea[:nav_end, 16].mean()), 3),
                "abs_omega_mean_until_reach": round(float(np.abs(ea[:nav_end, 17]).mean()), 3),
                "start_pose_equal": bool(np.allclose(ep[0], pose[0], atol=1e-3))}
        if a.videos:
            os.makedirs(a.videos, exist_ok=True)
            r["video"] = os.path.join(a.videos, os.path.basename(f)[:-4] + ".mp4") if video(f, os.path.join(
                "/dev/shm", os.path.basename(f)[:-4] + ".mp4")) else None
            if r["video"]:
                import shutil
                shutil.copy(os.path.join("/dev/shm", os.path.basename(f)[:-4] + ".mp4"), r["video"])
        rep[unit] = r
        print(unit, json.dumps(r["policy"]), json.dumps(r.get("expert")), flush=True)
    json.dump(rep, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
