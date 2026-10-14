"""End-to-end SFT of the public AlphaBot pi0.5 checkpoint on AlphaBot 2 + OVMM demonstrations.

Contract: inputs 3 RGB (camera_top / camera_left / camera_right) + 16-D proprio + language; output
CartesianMobileAction18 (dims 0-15 the pretrained ordering, 16 = v, 17 = omega). Full fine-tune.
Recipe from sft_stage1.py (qualified on Isaac): float32 parameters with bf16 autocast (bf16-only
updates are rounded away), gradient checkpointing, normalization stats from the TRAIN dataset only,
VAL flow-matching loss with fixed noise for checkpoint selection. Added here:
  * DDP over the visible GPUs; Adam state sharded with ZeroRedundancyOptimizer;
  * sampling from sampling.py (phase_balanced or uniform; frame_meta.npz is sampler-only);
  * atomic checkpoints (model + optimizer + step) every ckpt_every steps; on start the newest complete
    checkpoint is resumed; best-by-VAL and last are kept as plain policy dirs;
  * heartbeat.json (step, time) for the scheduler; train_log.jsonl;
  * loss masked on action_is_pad (chunk steps past the episode end are clamped copies of the last frame;
    PI05Policy.forward would train them as targets).
Exit codes: 0 done, 3 non-finite loss (not retried by the scheduler), other = crash (retried).

    torchrun --nproc_per_node N sft_ovmm.py --config configs/alphabot2_pi05/sft_ovmm_balanced.yaml
"""
import argparse, json, math, os, shutil, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ovmm_model  # noqa: E402
import sampling  # noqa: E402
from lerobot.datasets.factory import resolve_delta_timestamps  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata  # noqa: E402

GROUPS = {"left_xyz": slice(0, 3), "left_quat": slice(3, 7), "right_xyz": slice(7, 10), "right_quat": slice(10, 14),
          "grippers": slice(14, 16), "base_v": slice(16, 17), "base_omega": slice(17, 18)}


def lr_at(step, peak, warmup, decay_steps, decay_lr):
    if step < warmup:
        return peak * (step + 1) / warmup
    t = min(step - warmup, decay_steps) / max(1, decay_steps)
    return decay_lr + (peak - decay_lr) * 0.5 * (1 + math.cos(math.pi * t))


class WeightedBatches(torch.utils.data.Sampler):
    """Infinite stream of batches drawn with replacement by per-frame weight; per-rank, per-step seeded
    (resuming at step s continues the same stream)."""

    def __init__(self, w, batch, seed, rank, start_step):
        self.w = torch.as_tensor(w, dtype=torch.double)
        self.batch, self.seed, self.rank, self.step = batch, seed, rank, start_step

    def __iter__(self):
        while True:
            g = torch.Generator().manual_seed(self.seed * 1_000_003 + self.rank * 7919 + self.step)
            self.step += 1
            yield torch.multinomial(self.w, self.batch, replacement=True, generator=g).tolist()


def newest_ckpt(out):
    c = sorted(p for p in out.glob("ckpt_step*") if (p / "COMPLETE").exists())
    return c[-1] if c else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, default=0, help="override steps (smoke runs)")
    args = ap.parse_args()
    C = sampling.load_config(args.config)
    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        rank, world = 0, 1
    dev = "cuda"
    out = Path(C["out"])
    out.mkdir(parents=True, exist_ok=True)
    steps = args.max_steps or int(C["steps"])
    O = C["optim"]
    torch.manual_seed(C["seed"] + rank)
    np.random.seed(C["seed"] + rank)

    resume = newest_ckpt(out)
    src = str(resume / "policy") if resume else C["init_checkpoint"]
    cfg = ovmm_model.build_config(C["init_checkpoint"], device=dev, dtype=C["precision"]["params"],
                                  gradient_checkpointing=bool(C.get("gradient_checkpointing", True)))
    mtr = LeRobotDatasetMetadata("local/alphabot2_ovmm_train", root=C["train_data"])
    stats = mtr.stats                                                    # TRAIN ONLY
    dts = resolve_delta_timestamps(cfg, mtr)
    tr = LeRobotDataset("local/alphabot2_ovmm_train", root=C["train_data"], delta_timestamps=dts)
    va = LeRobotDataset("local/alphabot2_ovmm_val", root=C["val_data"], delta_timestamps=dts)
    fm = dict(np.load(Path(C["train_data"]) / "frame_meta.npz"))
    if len(fm["episode_index"]) != len(tr):
        raise RuntimeError(f"frame_meta has {len(fm['episode_index'])} frames, dataset {len(tr)}")
    w, winfo = sampling.weights(fm, C["sampling"])

    policy, pre, post = ovmm_model.build(cfg, src, C["init_checkpoint"], stats, device=dev)
    params = [p for p in policy.parameters() if p.requires_grad]
    # loss excludes chunk steps past the episode end (action_is_pad); see ovmm_model.masked_loss
    model = ovmm_model.MaskedLoss(policy)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()],
                                                          gradient_as_bucket_view=True,
                                                          find_unused_parameters=bool(C.get("ddp_find_unused", True)))
        from torch.distributed.optim import ZeroRedundancyOptimizer
        opt = ZeroRedundancyOptimizer(params, optimizer_class=torch.optim.AdamW, lr=O["lr"], betas=tuple(O["betas"]),
                                      eps=1e-8, weight_decay=O["weight_decay"])
    else:
        opt = torch.optim.AdamW(params, lr=O["lr"], betas=tuple(O["betas"]), eps=1e-8, weight_decay=O["weight_decay"])
    start = 0
    best = (float("inf"), -1)
    seen0 = 0
    if resume:
        st = torch.load(resume / "trainer.pt", map_location="cpu", weights_only=False)
        opt.load_state_dict(st["optimizer"])
        start, best = st["step"], tuple(st["best"])
        # samples actually seen before this attempt (the world size may differ between attempts);
        # checkpoints written before this field existed fall back to SAMPLES_SEEN_AT_RESUME, else step*batch*world
        seen0 = int(st.get("samples_seen") or os.environ.get("SAMPLES_SEEN_AT_RESUME") or start * int(C["batch_per_gpu"]) * world)
    if rank == 0:
        info0 = {"config": C, "policy_io": ovmm_model.IO.describe(), "image_features": list(cfg.image_features),
                 "resume_from": str(resume) if resume else None, "start_step": start, "world": world, "samples_seen_at_start": seen0,
                 "global_batch": int(C["batch_per_gpu"]) * world,
                 "planned_samples": int(C["batch_per_gpu"]) * world * steps,
                 "planned_effective_epochs": int(C["batch_per_gpu"]) * world * steps / max(1, len(tr)),
                 "train_episodes": mtr.total_episodes, "train_frames": mtr.total_frames, "val_frames": len(va),
                 "trainable_params": sum(p.numel() for p in params), "sampling": winfo}
        (out / f"run_info_step{start:06d}.json").write_text(json.dumps(info0, indent=1, default=str))
        print("INFO", json.dumps({k: v for k, v in info0.items() if k != "config"}, default=str), flush=True)

    loader = torch.utils.data.DataLoader(tr, batch_sampler=WeightedBatches(w, int(C["batch_per_gpu"]), int(C["seed"]), rank, start),
                                         num_workers=int(C["num_workers"]), pin_memory=True, persistent_workers=True)
    vidx = np.random.default_rng(0).choice(len(va), size=min(512, len(va)), replace=False)
    vloader = torch.utils.data.DataLoader(torch.utils.data.Subset(va, vidx.tolist()), batch_size=int(C["batch_per_gpu"]),
                                          shuffle=False, num_workers=4)
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=C["precision"].get("autocast") == "bfloat16")
    log = open(out / "train_log.jsonl", "a") if rank == 0 else None

    @torch.no_grad()
    def validate():
        policy.eval()
        cs, gs = torch.get_rng_state(), torch.cuda.get_rng_state()
        torch.manual_seed(1234)                                          # identical noise/time draws every time
        tot, n, pdim = 0.0, 0, np.zeros(18)
        for b in vloader:
            with amp:
                loss, inf = ovmm_model.masked_loss(policy, pre(b), b["action_is_pad"])
            k = len(b["action"])
            tot, n = tot + float(loss) * k, n + k
            pdim += np.asarray(inf["loss_per_dim"]) * k
        torch.set_rng_state(cs)
        torch.cuda.set_rng_state(gs)
        policy.train()
        return tot / n, {g: float((pdim / n)[s].mean()) for g, s in GROUPS.items()}

    def save_policy(d):
        tmp = d.parent / f".{d.name}.tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        policy.save_pretrained(tmp)
        pre.save_pretrained(tmp)
        post.save_pretrained(tmp)
        if d.exists():
            shutil.rmtree(d)
        tmp.rename(d)

    def checkpoint(step):
        if ddp:
            opt.consolidate_state_dict(to=0)
        if rank == 0:
            d = out / f"ckpt_step{step:07d}"
            save_policy(d / "policy")
            torch.save({"optimizer": opt.state_dict(), "step": step, "best": list(best),
                        "samples_seen": seen0 + (step - start) * int(C["batch_per_gpu"]) * world}, d / "trainer.pt")
            (d / "COMPLETE").write_text(time.strftime("%F %T"))
            for old in sorted(p for p in out.glob("ckpt_step*") if (p / "COMPLETE").exists())[:-int(C["keep_ckpts"])]:
                shutil.rmtree(old, ignore_errors=True)
        if ddp:
            dist.barrier()

    policy.train()
    it = iter(loader)
    t_log, n_log = time.time(), 0
    for step in range(start + 1, steps + 1):
        batch = next(it)
        lr = lr_at(step - 1, O["lr"], O["warmup"], steps, O["decay_lr"])
        for g in opt.param_groups:
            g["lr"] = lr
        with amp:
            loss, inf = model(pre(batch), batch["action_is_pad"])
        if not torch.isfinite(loss):
            print(f"NONFINITE loss at step {step}", flush=True)
            sys.exit(3)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(params, O["grad_clip"]))
        opt.step()
        opt.zero_grad(set_to_none=True)
        n_log += len(batch["action"]) * world
        if rank == 0 and (step % 25 == 0 or step == start + 1):
            dt = time.time() - t_log
            seen = seen0 + (step - start) * int(C["batch_per_gpu"]) * world
            rec = {"step": step, "samples_seen": seen, "effective_epochs": seen / len(tr), "loss": float(loss), "group_loss": {g: float(np.asarray(inf["loss_per_dim"])[s].mean())
                                                                    for g, s in GROUPS.items()},
                   "grad_norm": gn, "lr": lr, "samples_per_s": n_log / max(dt, 1e-9),
                   "gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
            log.write(json.dumps(rec) + "\n"); log.flush()
            (out / "heartbeat.json").write_text(json.dumps({"step": step, "time": time.time(), "loss": float(loss)}))
            t_log, n_log = time.time(), 0
        if step % int(C["val_every"]) == 0 or step == steps:
            if rank == 0:
                vl, vg = validate()
                rec = {"step": step, "val_loss": vl, "val_group": vg}
                if vl < best[0]:
                    best = (vl, step)
                    save_policy(out / "best")
                    rec["saved_best"] = True
                log.write(json.dumps(rec) + "\n"); log.flush()
                print("VAL", json.dumps(rec), flush=True)
            if ddp:
                b = torch.tensor([best[0], best[1]], dtype=torch.double, device=dev)
                dist.broadcast(b, 0)
                best = (float(b[0]), int(b[1]))
        if step % int(C["ckpt_every"]) == 0 or step == steps:
            checkpoint(step)
    if rank == 0:
        save_policy(out / "last")
        seen = seen0 + (steps - start) * int(C["batch_per_gpu"]) * world
        summary = {"best_val_loss": best[0], "best_step": best[1], "steps": steps, "world": world,
                   "global_batch": int(C["batch_per_gpu"]) * world, "samples_seen": seen,
                   "train_frames": len(tr), "train_episodes": mtr.total_episodes, "effective_epochs": seen / len(tr),
                   "unique_frame_coverage_note": "phase_balanced sampling draws with replacement by weight",
                   "selection": "lowest SFT-VAL flow-matching loss (fixed noise); evaluation episodes never used"}
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        (out / "DONE").write_text(time.strftime("%F %T"))
        print("###SFT###" + json.dumps(summary), flush=True)
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
