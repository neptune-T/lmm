"""Where does a policy-server vs reference action deviation come from? (diagnostics; one checkpoint)

On K recorded frames with fixed seeds, action chunks from:
  srv_bf16 / srv_fp32   policy_server.py subprocesses (the eval path) over the socket
  in_bf16 / in_fp32     the same Server class in-process (x2 each: determinism)
  train_path_fp32       ovmm_model.build with the TRAINING dataset's exact stats (instead of the checkpoint's saved
                        processors) -- checks processors / normalization
plus the checkpoint's saved normalizer stats vs the dataset's exact stats and the input hash. Differences are
reported as max and mean |a - b| / (q99 - q01) of each action dim (first H_exec steps and the full chunk).

    python server_consistency.py --ckpt BEST --data DATASET_ROOT --out OUT.json [--k 16] [--port 5650]
"""
import argparse, hashlib, json, os, socket, subprocess, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ovmm_model import IO  # noqa: E402
from wire import recv_msg, send_msg  # noqa: E402


def frames(root, k):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(repo_id="local/consistency", root=root)
    idx = np.linspace(0, len(ds) - 1, k).astype(int)
    reqs = []
    for i in idx:
        it = ds[int(i)]
        reqs.append({"cmd": "infer", "images": {c: (it[f"observation.images.{c}"].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                                                for c in IO.CAMERA_MAP},
                     "state": it["observation.state"].numpy().astype(np.float32), "task": it["task"], "seed": 7000 + int(i)})
    return reqs


def in_process(ckpt, dtype, reqs):
    from policy_server import Server
    s = Server(ckpt, device="cuda", dtype=dtype)
    out = np.stack([s.infer_batch([r])[0][0] for r in reqs])
    del s
    torch.cuda.empty_cache()
    return out


def via_server(ckpt, dtype, reqs, port):
    log = open(f"/dev/shm/consistency_server_{port}.log", "w")
    p = subprocess.Popen([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "policy_server.py"),
                          "--ckpt", ckpt, "--port", str(port), "--dtype", dtype], stdout=log, stderr=subprocess.STDOUT)
    for _ in range(180):
        if "READY" in open(log.name).read():
            break
        time.sleep(5)
    sock = socket.create_connection(("127.0.0.1", port))
    out = []
    for r in reqs:
        send_msg(sock, r)
        out.append(np.asarray(recv_msg(sock)["actions"]))
    sock.close()
    p.terminate()
    p.wait(60)
    torch.cuda.empty_cache()
    return np.stack(out)


def train_path(ckpt, data, reqs):
    """The training-time processors (dataset exact stats) instead of the checkpoint's saved processors."""
    import ovmm_model
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    cfg = ovmm_model.build_config(ckpt, device="cuda", dtype="float32", gradient_checkpointing=False)
    meta = LeRobotDatasetMetadata("local/consistency", root=data)
    policy, pre, post = ovmm_model.build(cfg, ckpt, ckpt, meta.stats, device="cuda")
    policy.eval()
    out = []
    with torch.no_grad():
        for r in reqs:
            b = {"observation.state": torch.as_tensor(r["state"])[None], "task": [r["task"]]}
            for c in IO.CAMERA_MAP:
                b[f"observation.images.{c}"] = torch.as_tensor(r["images"][c])[None].permute(0, 3, 1, 2).float() / 255.0
            noise = torch.randn(cfg.chunk_size, cfg.max_action_dim, generator=torch.Generator().manual_seed(r["seed"]))[None].cuda()
            a = post(policy.predict_action_chunk(pre(b), noise=noise)).float().cpu().numpy()[0]
            out.append(a)
    del policy
    torch.cuda.empty_cache()
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--port", type=int, default=5650)
    ap.add_argument("--h-exec", type=int, default=5)
    a = ap.parse_args()
    stats = json.load(open(os.path.join(a.data, "meta", "stats.json")))
    rng = np.maximum(np.asarray(stats["action"]["q99"]) - np.asarray(stats["action"]["q01"]), 1e-6)
    reqs = frames(a.data, a.k)
    rep = {"ckpt": a.ckpt, "k": a.k,
           "input_sha256": hashlib.sha256(b"".join(r["state"].tobytes() + b"".join(r["images"][c].tobytes() for c in sorted(r["images"]))
                                                   for r in reqs)).hexdigest()[:16]}
    # saved processor stats vs dataset exact stats
    from safetensors.torch import load_file
    f = next(os.path.join(a.ckpt, x) for x in os.listdir(a.ckpt) if x.startswith("policy_preprocessor") and x.endswith(".safetensors"))
    ns = {k: v.numpy() for k, v in load_file(f).items()}
    rep["saved_normalizer_vs_dataset_max_abs"] = {k: float(np.abs(ns[k] - np.asarray(stats[k.rsplit(".", 1)[0]][k.rsplit(".", 1)[1]])).max())
                                                  for k in ns if k.rsplit(".", 1)[0] in ("action", "observation.state")
                                                  and k.rsplit(".", 1)[1] in ("q01", "q99")}
    A = {"in_fp32": in_process(a.ckpt, "float32", reqs), "in_fp32_again": in_process(a.ckpt, "float32", reqs),
         "in_bf16": in_process(a.ckpt, "bfloat16", reqs), "in_bf16_again": in_process(a.ckpt, "bfloat16", reqs),
         "srv_bf16": via_server(a.ckpt, "bfloat16", reqs, a.port), "srv_fp32": via_server(a.ckpt, "float32", reqs, a.port + 1),
         "train_path_fp32": train_path(a.ckpt, a.data, reqs)}

    def d(x, y):
        e = np.abs(A[x] - A[y]) / rng
        return {"max_first_hexec": round(float(e[:, :a.h_exec].max()), 6), "mean_first_hexec": round(float(e[:, :a.h_exec].mean()), 6),
                "max_chunk": round(float(e.max()), 6), "worst_dim": IO.ACTION_LABELS[int(e.max(axis=(0, 1)).argmax())]}
    pairs = [("in_fp32", "in_fp32_again"), ("in_bf16", "in_bf16_again"), ("srv_bf16", "in_bf16"), ("srv_fp32", "in_fp32"),
             ("in_bf16", "in_fp32"), ("srv_bf16", "srv_fp32"), ("train_path_fp32", "in_fp32")]
    rep["diffs"] = {f"{x} vs {y}": d(x, y) for x, y in pairs}
    rep["action_scale"] = {k: round(float(np.abs(v).max()), 3) for k, v in A.items()}
    rep["action_range_violation_fp32"] = round(float(((A["in_fp32"] - np.asarray(stats["action"]["q01"])) / rng).min()), 3), \
        round(float(((A["in_fp32"] - np.asarray(stats["action"]["q99"])) / rng).max()), 3)
    json.dump(rep, open(a.out, "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
