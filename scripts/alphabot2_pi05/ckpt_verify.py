"""Verify that a mirrored SFT run directory on /workspace is a complete, loadable copy of its /dev/shm source.

Per item (best/, ckpt_step*/):
  content    sha256 of every file, mirror vs source (not just sizes)
  load       the policy loaded from the MIRROR and from the SOURCE (float32, separate loads) predicts identical
             action chunks on K recorded frames with identical seeds (max abs diff must be 0)
  trainer    for checkpoints: trainer.pt and every optim_rank*.pt load with torch, with the expected keys
Writes a JSON verdict; exit 1 on any failure.

    python ckpt_verify.py --mirror /workspace/runtime/sft_mirror/RUN --source /dev/shm/lmm_runtime/sft/RUN \
        --items best ckpt_step0006000 --data DATASET_ROOT --out OUT.json
"""
import argparse, hashlib, json, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ovmm_model import IO  # noqa: E402


def sha(path, bs=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bs)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def frames(root, k=8):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(repo_id="local/verify", root=root)
    idx = np.linspace(0, len(ds) - 1, k).astype(int)
    out = []
    for i in idx:
        it = ds[int(i)]
        out.append({"cmd": "infer", "images": {c: (it[f"observation.images.{c}"].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                                               for c in IO.CAMERA_MAP},
                    "state": it["observation.state"].numpy().astype(np.float32), "task": it["task"], "seed": 4242 + int(i)})
    return out


def predict(policy_dir, reqs):
    from policy_server import Server
    s = Server(policy_dir, device="cuda", dtype="float32")
    a = np.stack([s.infer_batch([r])[0][0] for r in reqs])
    del s
    torch.cuda.empty_cache()
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mirror", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--items", nargs="+", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rep, ok_all = {}, True
    reqs = frames(a.data)
    for item in a.items:
        src, mir = os.path.join(a.source, item), os.path.join(a.mirror, item)
        r = {}
        files = sorted(os.path.relpath(os.path.join(dp, f), src) for dp, _, fs in os.walk(src) for f in fs)
        mism = [f for f in files if not os.path.exists(os.path.join(mir, f)) or sha(os.path.join(src, f)) != sha(os.path.join(mir, f))]
        r["files"], r["sha256_mismatch"] = len(files), mism
        pol = "policy" if os.path.isdir(os.path.join(src, "policy")) else ""
        A_m = predict(os.path.join(mir, pol), reqs)
        A_s = predict(os.path.join(src, pol), reqs)
        r["load_from_mirror_max_abs_diff_vs_source"] = float(np.abs(A_m - A_s).max())
        r["predictions_finite"] = bool(np.isfinite(A_m).all())
        if os.path.exists(os.path.join(mir, "trainer.pt")):
            t = torch.load(os.path.join(mir, "trainer.pt"), map_location="cpu", weights_only=False)
            r["trainer_keys"] = sorted(t)
            r["trainer_step"] = t.get("step")
            shards = sorted(f for f in os.listdir(mir) if f.startswith("optim_rank"))
            r["optimizer_shards"] = {}
            for f in shards:
                sh = torch.load(os.path.join(mir, f), map_location="cpu", weights_only=False)
                r["optimizer_shards"][f] = {"rank": sh.get("rank"), "world": sh.get("world"),
                                            "state_tensors": len(sh["optimizer_shard"]["state"])}
            r["trainer_ok"] = bool(r["trainer_step"]) and (t.get("sharded_world") in (None, len(shards)))
        r["pass"] = (not mism and r["load_from_mirror_max_abs_diff_vs_source"] == 0.0 and r["predictions_finite"]
                     and r.get("trainer_ok", True))
        ok_all &= r["pass"]
        rep[item] = r
        print(item, json.dumps(r), flush=True)
    rep["ALL_PASS"] = bool(ok_all)
    json.dump(rep, open(a.out, "w"), indent=1)
    print("###VERIFY###" + json.dumps({"ALL_PASS": rep["ALL_PASS"]}), flush=True)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
