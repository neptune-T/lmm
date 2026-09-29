"""Stage-1 SFT of the public AlphaBot pi0.5 checkpoint on CartesianMobileAction18.

Same model/processor construction as lerobot-train (and sft_dryrun.py): the
public checkpoint ``alphabot2/aibot2_grab_bag_combined_over_under_pi05`` with an
18-D action output (dims 0-15 exactly the pretrained ordering, 16 = v,
17 = omega; the 32-D padded action space already has room), normalizer stats
from the TRAIN dataset only, the public PaliGemma tokenizer. Differences from
lerobot-train, each deliberate:

* parameters are held in float32 (the public checkpoint trained bf16
  parameters at lr 2.5e-5, below half a bf16 ulp for most weights -- updates
  there are rounded away); compute runs under bf16 autocast;
* validation loss on the held-out VAL dataset with fixed noise/time draws,
  predicted-action diagnostics per action group, and checkpoint selection on
  VAL loss only (no test data is ever touched);
* per-group training losses, gradient norm, LR and throughput logged as JSONL.
"""
import argparse, json, math, os, shutil, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pi05_runtime import CKPT, SPM, SentencePieceTokenizer  # noqa: E402
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.datasets.factory import resolve_delta_timestamps  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa: E402

GROUPS = {"left_xyz": slice(0, 3), "left_quat": slice(3, 7), "right_xyz": slice(7, 10), "right_quat": slice(10, 14),
          "grippers": slice(14, 16), "base_v": slice(16, 17), "base_omega": slice(17, 18)}


def quat_angle_deg(a, b):
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    d = np.clip(np.abs(np.sum(a * b, axis=-1)), 0, 1)
    return np.degrees(2 * np.arccos(d))


def lr_at(step, peak, warmup, decay_steps, decay_lr):
    if step < warmup:
        return peak * (step + 1) / warmup
    t = min(step - warmup, decay_steps) / max(1, decay_steps)
    return decay_lr + (peak - decay_lr) * 0.5 * (1 + math.cos(math.pi * t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2.5e-5)
    ap.add_argument("--decay-lr", type=float, default=2.5e-6)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--decay-steps", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--val-samples", type=int, default=512)
    ap.add_argument("--diag-samples", type=int, default=48)
    ap.add_argument("--save-every", type=int, default=0, help="also keep step checkpoints (0 = best + last only)")
    ap.add_argument("--num-workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--freeze-vision", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda"

    cfg = PreTrainedConfig.from_pretrained(CKPT)
    cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(18,))}
    cfg.compile_model = False
    cfg.gradient_checkpointing = True
    cfg.dtype = "float32"
    cfg.device = dev
    cfg.freeze_vision_encoder = bool(args.freeze_vision)
    mtr = LeRobotDatasetMetadata("local/alphabot2_stage1_train", root=args.train)
    mva = LeRobotDatasetMetadata("local/alphabot2_stage1_val", root=args.val)
    stats = mtr.stats                                     # TRAIN ONLY
    dts = resolve_delta_timestamps(cfg, mtr)
    tr = LeRobotDataset("local/alphabot2_stage1_train", root=args.train, delta_timestamps=dts)
    va = LeRobotDataset("local/alphabot2_stage1_val", root=args.val, delta_timestamps=dts)
    policy = PI05Policy.from_pretrained(CKPT, config=cfg).to(dev)
    feats = {**cfg.input_features, **cfg.output_features}
    pre, post = make_pre_post_processors(
        cfg, pretrained_path=CKPT, dataset_stats=stats,
        preprocessor_overrides={"device_processor": {"device": dev},
                                "normalizer_processor": {"stats": stats, "features": feats, "norm_map": cfg.normalization_mapping},
                                "tokenizer_processor": {"tokenizer": SentencePieceTokenizer(SPM)}},
        postprocessor_overrides={"unnormalizer_processor": {"stats": stats, "features": cfg.output_features,
                                                            "norm_map": cfg.normalization_mapping}})
    params = [p for p in policy.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    dtypes = sorted({str(p.dtype) for p in params})
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)
    decay_steps = args.decay_steps or args.steps
    info0 = {"train_root": args.train, "val_root": args.val, "train_episodes": mtr.total_episodes, "train_frames": mtr.total_frames,
             "val_episodes": mva.total_episodes, "val_frames": mva.total_frames, "trainable_params": n_train,
             "param_dtypes": dtypes, "amp_bf16": not args.no_amp, "args": vars(args),
             "action_stats_train": {k: np.asarray(v).tolist() for k, v in stats["action"].items() if k in ("mean", "std", "min", "max")},
             "state_stats_train": {k: np.asarray(v).tolist() for k, v in stats["observation.state"].items() if k in ("mean", "std")}}
    (out / "run_info.json").write_text(json.dumps(info0, indent=1, default=str))
    print("INFO", json.dumps({k: v for k, v in info0.items() if "stats" not in k}, default=str), flush=True)

    loader = torch.utils.data.DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=args.num_workers,
                                         drop_last=True, pin_memory=True, persistent_workers=True,
                                         generator=torch.Generator().manual_seed(args.seed))
    rng = np.random.default_rng(0)
    vidx = rng.choice(len(va), size=min(args.val_samples, len(va)), replace=False)
    didx = vidx[: args.diag_samples]
    vloader = torch.utils.data.DataLoader(torch.utils.data.Subset(va, vidx.tolist()), batch_size=args.batch,
                                          shuffle=False, num_workers=4)
    dloader = torch.utils.data.DataLoader(torch.utils.data.Subset(va, didx.tolist()), batch_size=min(16, args.batch),
                                          shuffle=False, num_workers=2)
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=not args.no_amp)
    log = open(out / "train_log.jsonl", "a")

    def group_losses(per_dim):
        pd = np.asarray(per_dim)
        return {g: float(pd[s].mean()) for g, s in GROUPS.items()}

    @torch.no_grad()
    def validate(step):
        policy.eval()
        cpu_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state()
        torch.manual_seed(1234)                           # identical noise/time draws at every validation
        tot, n, pdim = 0.0, 0, np.zeros(18)
        for b in vloader:
            with amp:
                loss, inf = policy.forward(pre(b))
            k = len(b["action"])
            tot += float(loss) * k
            pdim += np.asarray(inf["loss_per_dim"]) * k
            n += k
        # predicted-action diagnostics (10 flow steps, fixed noise)
        errs = {g: [] for g in GROUPS}
        qn, basep, baset = [], [], []
        for b in dloader:
            gt = b["action"].numpy()                      # (B, 50, 18) unnormalized
            with amp:
                a = policy.predict_action_chunk(pre(b))
            a = post(a).float().cpu().numpy()
            for g, s in GROUPS.items():
                if "quat" in g:
                    errs[g].append(quat_angle_deg(a[..., s], gt[..., s]).ravel())
                else:
                    errs[g].append(np.abs(a[..., s] - gt[..., s]).ravel())
            qn.append(np.linalg.norm(a[..., 3:7], axis=-1).ravel())
            qn.append(np.linalg.norm(a[..., 10:14], axis=-1).ravel())
            basep.append(a[:, 0, 16:18])
            baset.append(gt[:, 0, 16:18])
        torch.set_rng_state(cpu_state)
        torch.cuda.set_rng_state(cuda_state)
        qn, basep, baset = np.concatenate(qn), np.concatenate(basep), np.concatenate(baset)
        moving = np.abs(baset[:, 0]) > 0.02
        turning = np.abs(baset[:, 1]) > 0.05
        diag = {f"mae_{g}" + ("_deg" if "quat" in g else ""): float(np.mean(np.concatenate(v))) for g, v in errs.items()}
        diag.update({"pred_quat_norm_mean": float(qn.mean()), "pred_quat_norm_max_dev": float(np.max(np.abs(qn - 1))),
                     "pred_v_abs_mean": float(np.abs(basep[:, 0]).mean()), "gt_v_abs_mean": float(np.abs(baset[:, 0]).mean()),
                     "pred_omega_abs_mean": float(np.abs(basep[:, 1]).mean()), "gt_omega_abs_mean": float(np.abs(baset[:, 1]).mean()),
                     "v_sign_agreement_when_moving": float(np.mean(np.sign(basep[moving, 0]) == np.sign(baset[moving, 0]))) if moving.any() else None,
                     "omega_sign_agreement_when_turning": float(np.mean(np.sign(basep[turning, 1]) == np.sign(baset[turning, 1]))) if turning.any() else None,
                     "all_finite": bool(np.isfinite(qn).all() and np.isfinite(basep).all())})
        policy.train()
        return tot / n, group_losses(pdim / n), diag

    def save(tag):
        d = out / tag
        tmp = out / f".{tag}.tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        policy.save_pretrained(tmp)
        pre.save_pretrained(tmp)
        post.save_pretrained(tmp)
        if d.exists():
            shutil.rmtree(d)
        tmp.rename(d)

    best = (float("inf"), -1)
    it = iter(loader)
    policy.train()
    t_log, n_log = time.time(), 0
    v0 = validate(0)
    rec = {"step": 0, "val_loss": v0[0], "val_group": v0[1], "val_diag": v0[2]}
    log.write(json.dumps(rec) + "\n"); log.flush()
    print("VAL", json.dumps(rec), flush=True)
    w_probe = {n: p.detach().float().clone() for n, p in list(policy.named_parameters())[:: max(1, len(params) // 40)][:40]}
    for step in range(1, args.steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        lr = lr_at(step - 1, args.lr, args.warmup, decay_steps, args.decay_lr)
        for g in opt.param_groups:
            g["lr"] = lr
        with amp:
            loss, inf = policy.forward(pre(batch))
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {float(loss)}")
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        opt.step()
        opt.zero_grad(set_to_none=True)
        n_log += len(batch["action"])
        if step % args.log_every == 0 or step == 1:
            dt = time.time() - t_log
            rec = {"step": step, "loss": float(loss), "group_loss": group_losses(inf["loss_per_dim"]), "grad_norm": gn,
                   "lr": lr, "samples_per_s": n_log / max(dt, 1e-9), "gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
                   "batch_action_abs_mean": {g: float(batch["action"][..., s].abs().mean()) for g, s in GROUPS.items()}}
            log.write(json.dumps(rec) + "\n"); log.flush()
            print("TRAIN", json.dumps(rec), flush=True)
            t_log, n_log = time.time(), 0
        if step == 50:
            dp = max(float((p.detach().float() - w_probe[n]).abs().max()) for n, p in policy.named_parameters() if n in w_probe)
            print("PARAM_UPDATE_CHECK", json.dumps({"step": step, "max_abs_param_change_probe": dp}), flush=True)
            log.write(json.dumps({"step": step, "max_abs_param_change_probe": dp}) + "\n"); log.flush()
        if step % args.val_every == 0 or step == args.steps:
            vl, vg, vd = validate(step)
            rec = {"step": step, "val_loss": vl, "val_group": vg, "val_diag": vd}
            if vl < best[0]:
                best = (vl, step)
                save("best")
                rec["saved_best"] = True
            if args.save_every and step % args.save_every == 0:
                save(f"step{step:06d}")
            log.write(json.dumps(rec) + "\n"); log.flush()
            print("VAL", json.dumps(rec), flush=True)
    save("last")
    summary = {"best_val_loss": best[0], "best_step": best[1], "steps": args.steps,
               "selection": "lowest VAL flow-matching loss (fixed noise); test split never used"}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print("###SFT###" + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
