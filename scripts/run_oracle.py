#!/usr/bin/env python3
"""Run the WXAI oracle over a few OVMM episodes and record a qualification set."""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, "/nix/yanjun/zts/lmm/canonical_lmm")
import habitat
from habitat_baselines.config.default import get_config
from lmm.oracle_wxai import WXAIOracle, CONTROL_HZ, DATASET_FPS, TICKS_PER_ACTION

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True, choices=["left", "right"])
ap.add_argument("--episodes", type=int, default=3)
ap.add_argument("--out", required=True)
ap.add_argument("--save-images", action="store_true")
a = ap.parse_args()

cfg = get_config(os.environ["OVMM_BIMANUAL_CONFIG"], overrides=[
    "habitat.simulator.habitat_sim_v0.gpu_device_id=0",
    "habitat.environment.max_episode_steps=200000"])
env = habitat.Env(config=cfg)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
reports = []
for i in range(a.episodes):
    obs = env.reset()
    t0 = time.time()
    orc = WXAIOracle(env, a.arm, np.random.default_rng(1000 + i))
    try:
        rep = orc.run()
    except Exception as e:
        import traceback
        rep = {"arm": a.arm, "error": repr(e)[:300],
               "traceback": traceback.format_exc()[-1200:],
               "n_dataset_steps": len(orc.frames), "success": False,
               "episode_id": str(env.current_episode.episode_id)}
    rep["wall_s"] = round(time.time() - t0, 1)
    rep["control_hz"] = CONTROL_HZ
    rep["dataset_fps"] = DATASET_FPS
    rep["ticks_per_action"] = TICKS_PER_ACTION
    ep_dir = out / f"{a.arm}_ep{i}_{rep.get('episode_id','na')}"
    ep_dir.mkdir(exist_ok=True)
    (ep_dir / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    with (ep_dir / "frames.jsonl").open("w") as fh:
        for f in orc.frames:
            fh.write(json.dumps(f, default=str) + "\n")
    if a.save_images and orc.frames:
        o = orc._last_obs
        np.savez_compressed(ep_dir / "last_obs.npz",
                            head_rgb=np.asarray(o["head_rgb"]),
                            left_wrist_rgb=np.asarray(o["left_wrist_rgb"]),
                            right_wrist_rgb=np.asarray(o["right_wrist_rgb"]))
    reports.append({k: rep.get(k) for k in
                    ("arm","episode_id","success","grasped","reach_dist_m",
                     "nav_pick_ok","nav_place_ok","n_dataset_steps","n_sim_ticks",
                     "wall_s","error")})
    print(f"[{a.arm} ep{i}] success={rep.get('success')} grasped={rep.get('grasped')} "
          f"steps={rep.get('n_dataset_steps')} err={rep.get('error')}", flush=True)
env.close()
(out / f"summary_{a.arm}.json").write_text(json.dumps(reports, indent=1, default=str))
print("###JSON###"); print(json.dumps(reports, indent=1, default=str))
