"""Policy-interface gate: must pass before any AlphaBot 2 OVMM pi0.5 SFT round starts.

Runs in the pi0.5 env on the round's converted TRAIN dataset and the real collected episodes.
  A  state     proprio16 <-> checkpoint state16: exact round trip (random + real frames), per-slot ordering
               (unit vectors incl. right-arm sign flips), dataset state == state_from_proprio(source proprio),
               and Habitat ordering consistency: proprio[t+1] arm joints == executed JointMobileAction18[t]
               arm joints (left and right slots; a left/right or slot permutation would fail this).
  B  images    dataset image keys == the fixed camera map; pixels equal their source camera (and are clearly
               farther from the crossed alternative); model image order == pretrained relative order
               (camera_left, camera_right, camera_top); camera_front masked == absent (identical fp32 loss
               with fixed noise/time); one-frame visualization PNG.
  C  action    dataset names == CartesianMobileAction18 labels; quantile stats (q01/q99) present, finite,
               non-degenerate for all 18 dims and equal to the exact data quantiles (exact_stats.py; incl.
               base v, omega, which have no checkpoint stats); normalize -> unnormalize round trip.
  D  horizon   chunk_size 50 (prediction) vs explicit 1 <= H_exec <= 50 (execution); recorded.
  E  smoke     one finite training loss; predict_action_chunk -> (1, 50, 18) finite.
  F  horizon   post-release settle truncated at conversion (<= keep_settle frames, at the tail) i.e. before
     boundary  any 50-step chunk exists; tail chunks: action_is_pad pattern exact, padded steps clamped to
               the last frame; PADDING CONTRACT of the SFT loss (ovmm_model): padded steps get no target
               supervision AND give no information to valid steps (padded action tokens masked out of
               attention: valid predictions invariant to padded inputs, while lerobot's unmasked forward
               leaks), masked forward == lerobot's forward when nothing is padded; no training window has
               more than keep_settle real settle targets.
Writes OUT/interface_gate.json and OUT/interface_frame.png; prints ###GATE###; exit 1 on any failure.

    python interface_gate.py --config ROUND_SFT_YAML --collect COLLECT_DIR --h-exec 5 --out DIR
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ovmm_model  # noqa: E402
import sampling  # noqa: E402
from ovmm_model import IO  # noqa: E402
from lmm.embodiments.alphabot2.ovmm_chain_check import lowering_chain  # noqa: E402
from lmm.embodiments.alphabot2.ovmm_sft_windows import expected_pad, window_exposure  # noqa: E402
from lerobot.datasets.factory import resolve_delta_timestamps  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata  # noqa: E402
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS  # noqa: E402




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--collect", required=True)
    ap.add_argument("--h-exec", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=64)
    args = ap.parse_args()
    C = sampling.load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rep, ok = {}, {}
    rng = np.random.default_rng(0)

    # ---------------- A: state
    p = np.c_[rng.uniform(-2, 2, (1000, 7)), rng.uniform(0, 1, (1000, 1)), rng.uniform(-2, 2, (1000, 7)), rng.uniform(0, 1, (1000, 1))]
    rt = max(float(np.abs(IO.proprio_from_state(IO.state_from_proprio(x)) - x).max()) for x in p)
    slots = {}
    for src in list(range(7)) + list(range(8, 15)):
        x = np.zeros(16)
        x[7] = x[15] = 1.0
        x[src] = 0.5
        s = IO.state_from_proprio(x)
        slots[src] = [int(i) for i in np.flatnonzero(np.abs(s) > 1e-12)]
    expect = {i: [i] for i in range(7)} | {8 + i: [7 + i] for i in range(7)}
    ok["A_state_round_trip"] = rt < 1e-12
    ok["A_state_slot_ordering"] = slots == expect
    meta = LeRobotDatasetMetadata("local/alphabot2_ovmm_train", root=C["train_data"])
    index = json.load(open(Path(C["train_data"]) / "ovmm_episode_index.json"))
    hist, loss_cfg = C.get("history"), C.get("loss")
    cfg = ovmm_model.build_config(C["init_checkpoint"], h_exec=args.h_exec, history=hist)
    ds = LeRobotDataset("local/alphabot2_ovmm_train", root=C["train_data"],
                        delta_timestamps=ovmm_model.history_delta_timestamps(resolve_delta_timestamps(cfg, meta), hist))

    def get(i):
        """dataset item as the SFT loop sees it (history stacked tensors -> per-key inputs), batch dim 1"""
        it = ds[i]
        b = {k: (v[None] if torch.is_tensor(v) else [v]) for k, v in it.items() if k.startswith(("observation", "action", "task"))}
        return ovmm_model.apply_history(b, hist)
    ep0 = index["episodes"][0]
    src = Path(ep0["source"])
    low = np.load(src / "lowdim.npz")
    ims = np.load(src / "images.npz")
    T = ep0["frames"]
    start = int(ds.meta.episodes["dataset_from_index"][0]) if "dataset_from_index" in ds.meta.episodes.column_names else 0
    ts = sorted(set(np.linspace(0, T - 1, min(args.frames, T)).astype(int).tolist()))
    st_err, act_err, psnr_ok, psnr_cross = 0.0, 0.0, [], []
    h_img_psnr, h_state_err = [], 0.0
    for t in ts:
        b = get(start + t)
        item = {k: v[0] for k, v in b.items()}
        st_err = max(st_err, float(np.abs(item["observation.state"].numpy()[:16] - IO.state_from_proprio(low["proprio"][t])).max()))
        act_err = max(act_err, float(np.abs(item["action"][0].numpy() - low["action"][t]).max()))     # chunk step 0 = label t
        if hist:                                                  # history inputs == the source frames at t - k
            for j, k in enumerate(hist["offsets"]):
                tk = max(0, t - k)
                hi = item[f"observation.images.{hist['camera']}_h{k}"].permute(1, 2, 0).numpy() * 255.0
                mse = float(np.mean((hi - ims[IO.CAMERA_MAP[hist['camera']]][tk].astype(np.float64)) ** 2))
                h_img_psnr.append(99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse))
                if hist.get("state"):
                    blk = item["observation.state"].numpy()[16 * (j + 1):16 * (j + 2)]
                    h_state_err = max(h_state_err, float(np.abs(blk - IO.state_from_proprio(low["proprio"][tk])).max()))
        for k, cam in IO.CAMERA_MAP.items():
            img = (item[f"observation.images.{k}"].permute(1, 2, 0).numpy() * 255.0)
            def psnr(ref):
                mse = float(np.mean((img - ref.astype(np.float64)) ** 2))
                return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
            psnr_ok.append(psnr(ims[cam][t]))
            if k in ("camera_left", "camera_right"):
                other = "left_wrist_rgb" if cam == "right_wrist_rgb" else "right_wrist_rgb"
                psnr_cross.append(psnr(ims[other][t]) - psnr(ims[cam][t]))
    ok["A_dataset_state_equals_mapped_proprio"] = st_err < 1e-5
    if hist:
        ok["H_history_alignment"] = bool(min(h_img_psnr) >= 30.0 and h_state_err < 1e-5)
        rep["H"] = {"history": hist, "image_psnr_min_db": float(min(h_img_psnr)), "state_max_abs_err": h_state_err,
                    "model_state_dim": ovmm_model.state_dim(hist)}
    ok["C_dataset_action_equals_source_label"] = act_err < 1e-5
    # Full lowering chain on the source episode (never state joint dims vs Cartesian action dims):
    # CartesianMobileAction18 label -> CartesianExecutor IK -> JointMobileAction18 -> Habitat next state
    chain = lowering_chain(low, np.load(src / "metadata.npz"))
    ok["A_chain_lowering_reproduces_executed"] = chain["lowering"]["pass"]
    ok["A_chain_joint_tracking"] = chain["joint_tracking"]["pass"]
    ok["A_chain_fk_next_state_matches_label"] = chain["fk_next_state_matches_label"]["pass"]
    ok["A_chain_gripper_round_trip"] = chain["gripper"]["pass"]
    ok["A_chain_base_v_omega"] = chain["base"]["pass"]
    rep["A"] = {"round_trip_max_err": rt, "slot_map": slots, "dataset_state_max_err": st_err, "dataset_action_max_err": act_err,
                "lowering_chain": chain, "state_labels": list(IO.STATE_LABELS), "chain_episode": str(src)}

    # ---------------- B: images
    keys = sorted(k for k in meta.features if k.startswith("observation.images."))
    ok["B_dataset_image_keys"] = keys == sorted(f"observation.images.{k}" for k in IO.CAMERA_MAP)
    ok["B_pixels_match_source_camera"] = float(np.min(psnr_ok)) >= 30.0
    ok["B_not_crossed"] = float(np.max(psnr_cross)) <= -3.0                 # the crossed source is clearly worse
    model_order = [k.split(".")[-1] for k in cfg.image_features]
    ok["B_model_image_order_pretrained"] = model_order == [c for c in IO.MODEL_CAMERA_ORDER if c in IO.CAMERA_MAP] + ovmm_model.history_keys(hist)
    rep["B"] = {"dataset_keys": keys, "psnr_min_db": float(np.min(psnr_ok)), "crossed_minus_true_db_max": float(np.max(psnr_cross)),
                "model_image_order": model_order, "camera_map": IO.CAMERA_MAP, "masked": list(IO.MASKED_CAMERAS),
                "video_storage": bool(index.get("video"))}

    # ---------------- C: action
    names = list(meta.features["action"]["names"])
    ok["C_action_labels"] = names == list(IO.ACTION_LABELS)
    ok["C_state_labels"] = list(meta.features["observation.state"]["names"]) == list(IO.STATE_LABELS)
    sa = {k: np.asarray(v, float) for k, v in meta.stats["action"].items()}
    acts = np.stack([np.asarray(x, float) for x in ds.hf_dataset["action"]])
    q01, q99 = np.quantile(acts, 0.01, axis=0), np.quantile(acts, 0.99, axis=0)
    rngs = np.maximum(acts.max(0) - acts.min(0), 1e-9)
    have = "q01" in sa and "q99" in sa
    ok["C_quantile_stats_present_finite"] = have and bool(np.isfinite(sa["q01"]).all() and np.isfinite(sa["q99"]).all())
    spread = (sa["q99"] - sa["q01"]) if have else np.zeros(18)
    ok["C_quantiles_non_degenerate"] = have and bool((spread > 1e-6).all())
    qerr = (np.maximum(np.abs(sa["q01"] - q01), np.abs(sa["q99"] - q99)) / rngs) if have else np.ones(18)
    ok["C_quantiles_match_data"] = bool((qerr <= 0.02).all())
    rep["C"] = {"labels": names, "q01": sa.get("q01", np.zeros(0)).tolist(), "q99": sa.get("q99", np.zeros(0)).tolist(),
                "base_v_q01_q99": [float(sa["q01"][16]), float(sa["q99"][16])] if have else None,
                "base_omega_q01_q99": [float(sa["q01"][17]), float(sa["q99"][17])] if have else None,
                "quantile_rel_err_max": float(qerr.max()), "degenerate_dims": [names[i] for i in np.flatnonzero(spread <= 1e-6)],
                "normalization_mapping": {k: str(v) for k, v in cfg.normalization_mapping.items()},
                "stats_source": "OVMM TRAIN dataset (all 18 dims)"}

    # ---------------- D: horizon
    ok["D_horizons"] = cfg.chunk_size == IO.H_PRED and 1 <= args.h_exec <= cfg.chunk_size
    ok["D_state_within_pi05_contract"] = ovmm_model.state_dim(hist) <= int(getattr(cfg, "max_state_dim", 32))
    rep["D"] = {"h_pred": cfg.chunk_size, "h_exec": args.h_exec, "n_action_steps_recorded": cfg.n_action_steps,
                "num_inference_steps": getattr(cfg, "num_inference_steps", None)}

    # ---------------- E: model (+ B masked == absent, C normalization round trip)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.device = dev
    policy, pre, post = ovmm_model.build(cfg, C["init_checkpoint"], C["init_checkpoint"], ovmm_model.history_stats(meta.stats, hist), device=dev)
    policy.eval()
    batch = get(start + ts[len(ts) // 2])
    pb = pre(dict(batch))
    a_rt = post(pb["action"].clone()).float().cpu()
    ok["C_normalize_round_trip"] = float((a_rt - batch["action"].float()).abs().max()) <= 1e-3 * float(rngs.max())
    # prompt budget: pi0.5 writes the discretized state into the prompt ("Task: ..., State: b1 b2 ...;\nAction: ")
    # and truncates at tokenizer_max_length -- worst case = the longest instruction with every state bin at 255
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK as _M
    tasks = list(meta.tasks.index) if hasattr(meta.tasks, "index") else list(meta.tasks)
    longest = max(tasks, key=len)
    wb = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    wb["observation.state"] = torch.full_like(batch["observation.state"], 1e6)     # normalizes to the top bin
    wb["task"] = [longest]
    ntok = int(pre(dict(wb))[_M].sum())
    lim = int(getattr(cfg, "tokenizer_max_length", 200))
    ok["D_prompt_within_tokenizer_limit"] = ntok < lim - 8
    rep["D_prompt"] = {"worst_case_tokens": ntok, "tokenizer_max_length": lim, "longest_task": longest,
                       "state_dim": ovmm_model.state_dim(hist), "max_state_dim": int(getattr(cfg, "max_state_dim", 32))}
    def seeded_loss():
        b = pre(dict(batch))
        torch.manual_seed(1234)                       # forward() draws its own noise/time from the global RNG
        torch.cuda.manual_seed_all(1234)
        return policy.forward(b)[0]

    torch.manual_seed(0)
    noise = torch.randn(1, cfg.chunk_size, cfg.max_action_dim, device=dev)
    # masked == absent is a statement about the math (pad-masked image tokens get zero attention weight
    # and do not shift positions), so it is checked in float32: under bf16 autocast the two sequence
    # lengths alone change reduction order and the loss by ~3e-4 relative.
    with torch.no_grad():
        loss3 = seeded_loss()
        imgs3, masks3 = policy._preprocess_images(pre(dict(batch)))
        from lerobot.configs.types import FeatureType, PolicyFeature
        policy.config.input_features["observation.images.camera_front"] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 240, 320))
        imgs4, masks4 = policy._preprocess_images(pre(dict(batch)))
        loss4 = seeded_loss()
        del policy.config.input_features["observation.images.camera_front"]
    with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
        loss_bf16 = seeded_loss()
        chunk = post(policy.predict_action_chunk(pre(dict(batch)), noise=noise)).float().cpu()
    ok["B_masked_equals_absent"] = (len(imgs4) == 4 and [bool(m[0]) for m in masks4] == [True, True, True, False]
                                    and abs(float(loss3) - float(loss4)) <= 1e-4 * max(1.0, abs(float(loss3))))
    ok["E_loss_finite"] = bool(torch.isfinite(loss3))
    ok["E_chunk_shape_finite"] = tuple(chunk.shape) == (1, cfg.chunk_size, 18) and bool(torch.isfinite(chunk).all())
    rep["E"] = {"loss_absent_fp32": float(loss3), "loss_masked_fp32": float(loss4), "loss_absent_bf16_autocast": float(loss_bf16),
                "masks_with_front": [bool(m[0]) for m in masks4],
                "n_images_absent": len(imgs3), "chunk_shape": list(chunk.shape),
                "chunk_first_step": chunk[0, 0].tolist(), "label_first_step": batch["action"][0, 0].tolist()}

    # ---------------- F: action-horizon boundary (settle truncation, tail padding, masked loss)
    keep = int(index.get("keep_settle", 5))
    fm = dict(np.load(Path(C["train_data"]) / "frame_meta.npz"))
    names = [str(x) for x in fm["stage_names"]]
    sid = names.index("settle") if "settle" in names else -1
    epi, stg = np.asarray(fm["episode_index"]), np.asarray(fm["stage"])
    trunc_ok, tail_ok = True, True
    for e in index["episodes"]:
        m = epi == e["episode_index"]
        s_ = stg[m] == sid
        if s_.sum() > keep or (s_.any() and not s_[np.argmax(s_):].all()):
            tail_ok = False
        if e.get("frames_raw") is not None and e["frames"] != e["frames_raw"] - e.get("settle_frames_dropped", 0):
            trunc_ok = False
    ok["F_settle_truncated_before_chunking"] = trunc_ok and tail_ok and len(fm["episode_index"]) == len(ds)
    # tail padding of the real episode used above: pattern and clamped values
    pad_ok, clamp_ok = True, True
    last_action = low["action"][T - 1]
    for t in (T - 1, T - 10, T - 30):
        if t < 0:
            continue
        it = ds[start + t]
        pad = it["action_is_pad"].numpy().astype(bool)
        pad_ok &= bool((pad == expected_pad(T, t, cfg.chunk_size)).all())
        if pad.any():
            clamp_ok &= bool(np.abs(it["action"].numpy()[pad] - last_action).max() < 1e-5)
    ok["F_tail_pad_pattern"] = pad_ok
    ok["F_tail_pad_clamped_to_last_frame"] = clamp_ok
    # Padding contract: chunk steps past the episode end give (1) no target supervision and (2) no
    # information to valid steps. (2) is attention: lerobot's PI05 action tokens attend to each other
    # bidirectionally, so padded noisy actions (built from the clamped tail action) would reach valid
    # steps; ovmm_model.flow_losses masks padded keys. Probes on a real tail frame, fp32, fixed noise/time:
    #   invariance  corrupt the padded actions -> valid-step predictions v_t unchanged (masked forward)
    #   sensitivity the same corruption DOES move valid v_t under lerobot's unmasked forward (the probe
    #               detects leakage; this is the leak the mask removes)
    #   reduction   corrupt the padded steps' losses -> masked reduction bit-identical, = valid-step mean
    #   equivalence nothing padded -> masked forward bit-identical to lerobot's PI05Pytorch.forward
    #   training    ovmm_model.masked_loss (the SFT loss) == reduce_masked(flow_losses(masked)) same seed
    raw = get(start + max(0, T - 10))
    pm = raw["action_is_pad"][0].bool()
    corrupt = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in raw.items()}
    corrupt["action"][0, pm] = 100.0
    xr, xc = pre(dict(raw)), pre(dict(corrupt))
    pad = raw["action_is_pad"].to(dev)
    vm = ~pm.to(dev)
    torch.manual_seed(77)
    nz = torch.randn(policy.prepare_action(xr).shape, device=dev)
    inv, sens = [], []
    with torch.no_grad():
        for tv in (0.1, 0.5, 0.9):
            tm = torch.full((1,), tv, device=dev)
            _, v_r = ovmm_model.flow_losses(policy, xr, pad, noise=nz, time=tm, return_v=True)
            _, v_c = ovmm_model.flow_losses(policy, xc, pad, noise=nz, time=tm, return_v=True)
            _, u_r = ovmm_model.flow_losses(policy, xr, None, noise=nz, time=tm, return_v=True)
            _, u_c = ovmm_model.flow_losses(policy, xc, None, noise=nz, time=tm, return_v=True)
            inv.append(float((v_r[0, vm] - v_c[0, vm]).abs().max()))
            sens.append(float((u_r[0, vm] - u_c[0, vm]).abs().max()))
        tm = torch.full((1,), 0.5, device=dev)
        losses = ovmm_model.flow_losses(policy, xr, pad, noise=nz, time=tm).float()
        torch.manual_seed(78)
        torch.cuda.manual_seed_all(78)
        lm_api = float(ovmm_model.masked_loss(policy, xr, pad)[0])
        torch.manual_seed(78)
        torch.cuda.manual_seed_all(78)
        lm_ref = float(ovmm_model.reduce_masked(ovmm_model.flow_losses(policy, xr, pad), pad)[0])
        # equivalence on an unpadded frame (mid-episode item from E)
        xb = pre(dict(batch))
        nopad = batch.get("action_is_pad", torch.zeros(1, cfg.chunk_size, dtype=torch.bool))
        images, img_masks = policy._preprocess_images(xb)
        acts = policy.prepare_action(xb)
        nz2 = torch.randn(acts.shape, device=dev)
        lerobot_l = policy.model.forward(images, img_masks, xb[OBS_LANGUAGE_TOKENS], xb[OBS_LANGUAGE_ATTENTION_MASK],
                                         acts, nz2, tm)[:, :, :18]
        ours_l = ovmm_model.flow_losses(policy, xb, nopad.to(dev), noise=nz2, time=tm)
    bad = losses.clone()
    bad[0, pm.to(bad.device)] = 1e6
    lm = float(ovmm_model.reduce_masked(losses, pad)[0])
    lm_c = float(ovmm_model.reduce_masked(bad, pad)[0])
    lu, lu_c = float(ovmm_model.reduce_masked(losses)[0]), float(ovmm_model.reduce_masked(bad)[0])
    lm_manual = float(losses[0, vm.to(losses.device)].mean())
    ok["F_pad_no_target_supervision"] = (bool(pm.any()) and lm == lm_c and abs(lm - lm_manual) <= 1e-6 * max(1.0, abs(lm))
                                         and abs(lu - lu_c) > 1.0)
    ok["F_pad_no_information_to_valid_steps"] = bool(pm.any()) and max(inv) <= 1e-6
    ok["F_pad_probe_detects_unmasked_leak"] = min(sens) > 1e-4
    ok["F_masked_forward_equals_lerobot_without_padding"] = (not bool(nopad.any())) and bool(torch.equal(lerobot_l, ours_l))
    ok["F_sft_loss_uses_padding_contract"] = abs(lm_api - lm_ref) <= 1e-6 * max(1.0, abs(lm_ref))
    if loss_cfg:                                                  # the run's configured loss (v2 steering terms)
        corrupt = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in raw.items()}
        corrupt["action"][0, pm] = 100.0
        with torch.no_grad():
            tm = torch.full((1,), 0.5, device=dev)
            cl, cinf = ovmm_model.configured_loss(policy, pre(dict(raw)), pad, loss_cfg, noise=nz, time=tm)
            cl_c, _ = ovmm_model.configured_loss(policy, pre(dict(corrupt)), pad, loss_cfg, noise=nz, time=tm)
        ok["F_configured_loss_padding_contract"] = bool(torch.isfinite(cl)) and abs(float(cl) - float(cl_c)) <= 1e-6 * max(1.0, abs(float(cl)))
        rep["F_configured"] = {"loss_cfg": loss_cfg, "loss": float(cl), "loss_corrupted_pads": float(cl_c), "terms": {k: cinf.get(k) for k in ("flow_weighted", "steer_aux")}}
    exp = window_exposure(epi, stg, sid, cfg.chunk_size)
    ok["F_settle_exposure_bounded"] = exp["max_settle_targets_per_window"] <= keep
    rep["F"] = {"keep_settle": keep, "padded_steps_in_probe": int(pm.sum()), "probe_times": [0.1, 0.5, 0.9],
                "valid_v_max_abs_change_masked": inv, "valid_v_max_abs_change_unmasked_lerobot": sens,
                "masked_loss": lm, "masked_loss_corrupted_pad_losses": lm_c, "masked_loss_manual_valid_mean": lm_manual,
                "unmasked_loss": lu, "unmasked_loss_corrupted_pad_losses": lu_c,
                "sft_masked_loss_same_seed": lm_api, "reference_same_seed": lm_ref,
                "lerobot_vs_masked_no_padding_max_abs": float((lerobot_l - ours_l).abs().max()),
                "residual_note": "valid steps cannot attend to padded tokens; the only boundary signal left is the attention "
                                 "support size (how many real steps remain, <= 50), which carries no tail action values",
                "window_exposure": exp,
                "settle_frames_dropped_total": int(sum(e.get("settle_frames_dropped", 0) for e in index["episodes"]))}

    # ---------------- one-frame visualization
    try:
        from PIL import Image, ImageDraw
        tiles = []
        for k in [c for c in IO.MODEL_CAMERA_ORDER]:
            if k in IO.CAMERA_MAP:
                a = (item[f"observation.images.{k}"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                label = f"{k} <- {IO.CAMERA_MAP[k]}"
            else:
                a = np.zeros((240, 320, 3), np.uint8)
                label = f"{k}: MASKED (absent)"
            im = Image.fromarray(a)
            ImageDraw.Draw(im).text((6, 6), label, fill=(255, 255, 0))
            tiles.append(im)
        canvas = Image.new("RGB", (320 * len(tiles), 240))
        for i, im in enumerate(tiles):
            canvas.paste(im, (320 * i, 0))
        canvas.save(out / "interface_frame.png")
        rep["visualization"] = str(out / "interface_frame.png")
    except Exception as e:                                                   # visualization is a report, not a check
        rep["visualization_error"] = repr(e)[:200]

    res = {"gate": "interface", "pass": all(ok.values()), "checks": ok, "report": rep, "policy_io": IO.describe(),
           "config": args.config}
    (out / "interface_gate.json").write_text(json.dumps(res, indent=1, default=str))
    print("###GATE###" + json.dumps({"gate": "interface", "pass": res["pass"], "checks": ok}), flush=True)
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
