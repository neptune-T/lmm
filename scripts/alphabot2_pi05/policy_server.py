"""Batched pi0.5 policy server for the Stage-1 Isaac clients (127.0.0.1 only).

Serves either the public AlphaBot checkpoint (16-D output; the base dims are
returned as v = omega = 0) or a Stage-1 SFT checkpoint (18-D output). The
observation mapping is identical for both: the client sends the four
checkpoint camera keys, the 16-D state in the checkpoint convention and the
task string. Nothing else (no stages, no privileged state) is accepted.

Many Isaac workers connect at once; a single GPU thread batches whatever
requests are waiting (up to --max-batch, waiting at most --batch-wait-ms).
Each request carries its own seed; its flow-matching noise is drawn from that
seed alone, so a request's result does not depend on what it was batched with
(up to batched-matmul numerics).

Request:  {"cmd": "infer", "images": {key: HxWx3 uint8}, "state": (16,), "task": str, "seed": int}
Response: {"actions": (50, 18) float32, "latency_s", "batch_size", "queue_s"}
"""
import argparse, os, queue, socket, sys, threading, time, traceback

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pi05_runtime import SPM, SentencePieceTokenizer  # noqa: E402
from pi05_server import recv_msg, send_msg  # noqa: E402
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa: E402

CAMS = ("camera_front", "camera_left", "camera_right", "camera_top")   # public 4-camera checkpoint (Isaac stage)


class Server:
    def __init__(self, ckpt, device="cuda", dtype="bfloat16"):
        cfg = PreTrainedConfig.from_pretrained(ckpt)
        cfg.compile_model = False
        cfg.gradient_checkpointing = False
        cfg.device = device
        cfg.dtype = dtype
        self.cfg = cfg
        self.policy = PI05Policy.from_pretrained(ckpt, config=cfg).to(device).eval()
        self.pre, self.post = make_pre_post_processors(
            cfg, pretrained_path=ckpt,
            preprocessor_overrides={"tokenizer_processor": {"tokenizer": SentencePieceTokenizer(SPM)},
                                    "device_processor": {"device": device}})
        self.adim = int(cfg.output_features["action"].shape[0])
        # cameras from the checkpoint's own config (the OVMM model has 3: camera_top/left/right)
        self.cams = tuple(k.split(".")[-1] for k in cfg.image_features)
        self.device = device
        self.q: queue.Queue = queue.Queue()
        self.n_served = 0
        self.gpu_s = 0.0

    @torch.no_grad()
    def infer_batch(self, reqs):
        B = len(reqs)
        batch = {"observation.state": torch.as_tensor(np.stack([np.asarray(r["state"], np.float32) for r in reqs])),
                 "task": [r["task"] for r in reqs]}
        for c in self.cams:
            im = torch.as_tensor(np.stack([np.ascontiguousarray(r["images"][c]) for r in reqs]))
            batch[f"observation.images.{c}"] = im.permute(0, 3, 1, 2).float() / 255.0
        noise = torch.stack([torch.randn(self.cfg.chunk_size, self.cfg.max_action_dim,
                                         generator=torch.Generator().manual_seed(int(r.get("seed", 0))))
                             for r in reqs]).to(self.device)
        t = time.perf_counter()
        b = self.pre(batch)
        a = self.policy.predict_action_chunk(b, noise=noise)
        a = self.post(a)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        a = a.float().cpu().numpy()
        if self.adim == 16:                      # public checkpoint: no base output -> v = omega = 0
            a = np.concatenate([a, np.zeros((B, a.shape[1], 2), np.float32)], axis=2)
        return a, dt

    def gpu_loop(self, max_batch, wait_s):
        while True:
            first = self.q.get()
            reqs = [first]
            t_end = time.time() + wait_s
            while len(reqs) < max_batch:
                try:
                    reqs.append(self.q.get(timeout=max(0.0, t_end - time.time())))
                except queue.Empty:
                    break
            try:
                acts, dt = self.infer_batch([r[0] for r in reqs])
                self.n_served += len(reqs)
                self.gpu_s += dt
                for i, (msg, box) in enumerate(reqs):
                    box.put({"actions": acts[i].astype(np.float32), "latency_s": dt, "batch_size": len(reqs),
                             "queue_s": time.time() - msg["_t_in"] - dt})
            except Exception as e:
                for _, box in reqs:
                    box.put({"error": repr(e), "tb": traceback.format_exc()[-1500:]})

    def client(self, conn):
        with conn:
            while True:
                msg = recv_msg(conn)
                if msg is None:
                    return
                cmd = msg.get("cmd")
                if cmd == "ping":
                    send_msg(conn, {"ok": True, "action_dim": self.adim, "cams": list(self.cams), "n_served": self.n_served, "gpu_s": self.gpu_s})
                elif cmd == "infer":
                    msg["_t_in"] = time.time()
                    box: queue.Queue = queue.Queue(maxsize=1)
                    self.q.put((msg, box))
                    send_msg(conn, box.get())
                else:
                    send_msg(conn, {"error": f"unknown cmd {cmd!r}"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--port", type=int, default=5600)
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--batch-wait-ms", type=float, default=15.0)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    s = Server(args.ckpt, dtype=args.dtype)
    dummy = {"images": {k: np.zeros((240, 320, 3), np.uint8) for k in s.cams}, "state": np.zeros(16), "task": "warmup", "seed": 0}
    for n in (1, 2):
        s.infer_batch([dummy] * n)
    threading.Thread(target=s.gpu_loop, args=(args.max_batch, args.batch_wait_ms / 1000), daemon=True).start()
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", args.port))
    srv.listen(64)
    print(f"READY port={args.port} action_dim={s.adim} ckpt={args.ckpt} "
          f"gpu_mem_gb={torch.cuda.max_memory_allocated() / 1e9:.2f}", flush=True)
    while True:
        conn, _ = srv.accept()
        threading.Thread(target=s.client, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    main()
