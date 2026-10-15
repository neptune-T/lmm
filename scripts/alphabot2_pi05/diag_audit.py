"""Open-loop action audit of an AlphaBot 2 OVMM pi0.5 checkpoint on recorded expert frames (diagnostics).

Separates interface/normalization faults from learning faults before any retraining:
  normalization  the q01/q99 inside the checkpoint's saved normalizer == the training dataset's exact stats
  server         policy_server.py (the eval path, bf16) vs an in-process float32 reference on identical
                 inputs and noise
  actions        on sampled train (seen) and val (held-out SFT-val) frames, the server's chunk vs the expert
                 label chunk over the first H_exec steps (non-padded): per-dim MAE in raw and range-normalized
                 units vs a predict-the-train-mean baseline; base v / omega by stage group (navigation vs
                 manipulation): means, MAE, correlation, sign agreement, stall rate (expert drives, policy
                 does not); gripper command agreement.

    python diag_audit.py --ckpt BEST --train DS_TRAIN --val DS_VAL --port 5610 --out OUT.json [--n 400]
"""
import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wire import recv_msg, send_msg  # noqa: E402
from ovmm_model import IO  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

NAV = ("navigate_to_object", "navigate_to_receptacle")
LAB = list(IO.ACTION_LABELS)


def normalizer_stats(ckpt):
    """q01/q99 the checkpoint's saved preprocessor normalizes with."""
    from safetensors.torch import load_file
    f = next(os.path.join(ckpt, x) for x in os.listdir(ckpt) if x.startswith("policy_preprocessor") and x.endswith(".safetensors"))
    t = load_file(f)
    return {k: v.numpy() for k, v in t.items()}


def sample(ds_root, n, seed):
    fm = dict(np.load(os.path.join(ds_root, "frame_meta.npz")))
    names = [str(s) for s in fm["stage_names"]]
    st = np.asarray(fm["stage"])
    nav = np.isin(st, [i for i, s in enumerate(names) if s in NAV])
    rng = np.random.default_rng(seed)
    idx = np.r_[rng.choice(np.flatnonzero(nav), n // 2, replace=False), rng.choice(np.flatnonzero(~nav), n - n // 2, replace=False)]
    return idx, nav, names, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--port", type=int, default=5610)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--h-exec", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--save-npz", default=None, help="also save per-frame predictions/labels (OUT_<split>.npz)")
    a = ap.parse_args()
    rep = {"ckpt": a.ckpt, "h_exec": a.h_exec}

    # ---- normalization: checkpoint normalizer vs exact dataset stats
    stats = json.load(open(os.path.join(a.train, "meta", "stats.json")))
    ns = normalizer_stats(a.ckpt)
    chk = {}
    for key in ("observation.state", "action"):
        for q in ("q01", "q99"):
            k = next((x for x in ns if x.startswith(key + ".") and x.endswith(q)), None)
            if k is not None:
                chk[f"{key}.{q}"] = float(np.abs(ns[k] - np.asarray(stats[key][q])).max())
    rep["normalizer_vs_dataset_max_abs"] = chk
    rep["normalizer_keys"] = sorted(ns)[:12]
    q01, q99 = np.asarray(stats["action"]["q01"]), np.asarray(stats["action"]["q99"])
    rng_ = np.maximum(q99 - q01, 1e-6)
    mean = np.asarray(stats["action"]["mean"])

    sock = __import__("socket").create_connection(("127.0.0.1", a.port))
    send_msg(sock, {"cmd": "ping"})
    print("server", recv_msg(sock), flush=True)

    ref = None
    try:                                                         # float32 in-process reference (same code path)
        from policy_server import Server
        ref = Server(a.ckpt, device="cuda", dtype="float32")
    except Exception as e:
        rep["reference_error"] = repr(e)[:300]

    for split, root in [x for x in (("train", a.train), ("val", a.val)) if x[0] in a.splits.split(",")]:
        ds = LeRobotDataset(repo_id="local/audit", root=root, delta_timestamps={"action": [i / 10 for i in range(50)]})
        idx, nav, names, st = sample(root, min(a.n, len(ds) // 2), 7 if split == "train" else 8)
        P, E, PAD, NAVF, SRV_REF = [], [], [], [], []
        for j, i in enumerate(idx):
            it = ds[int(i)]
            imgs = {c: (it[f"observation.images.{c}"].permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
                    for c in IO.CAMERA_MAP}
            req = {"cmd": "infer", "images": imgs, "state": it["observation.state"].numpy().astype(np.float32),
                   "task": it["task"], "seed": 1000 + int(i)}
            send_msg(sock, req)
            r = recv_msg(sock)
            pa = np.asarray(r["actions"])
            if ref is not None and j < 32:
                ra, _ = ref.infer_batch([req])
                SRV_REF.append(float(np.abs(ra[0][: a.h_exec] - pa[: a.h_exec]).max() / rng_.max()))
            P.append(pa)                                         # full 50-step chunks (metrics use the first h_exec)
            E.append(it["action"].numpy())
            PAD.append(it["action_is_pad"].numpy())
            NAVF.append(bool(nav[int(i)]))
        PF, EF, PADF, NAVF = np.asarray(P), np.asarray(E), np.asarray(PAD, bool), np.asarray(NAVF)
        P, E, PAD = PF[:, : a.h_exec], EF[:, : a.h_exec], PADF[:, : a.h_exec]
        m = ~PAD
        err = np.abs(P - E)[m]                                   # (N, 18)
        base = np.abs(mean[None] - E[m])
        out = {"frames": int(len(idx)), "steps_scored": int(m.sum()),
               "mae_raw": dict(zip(LAB, err.mean(0).round(4).tolist())),
               "mae_norm": dict(zip(LAB, (err / rng_).mean(0).round(4).tolist())),
               "mean_baseline_mae_norm": dict(zip(LAB, (base / rng_).mean(0).round(4).tolist())),
               "server_vs_fp32_ref_max_norm": max(SRV_REF) if SRV_REF else None}
        for g, sel in (("navigation", NAVF), ("manipulation", ~NAVF)):
            mm = m & sel[:, None]
            gv = {}
            for d, name in ((16, "v"), (17, "omega")):
                pe, ee = P[..., d][mm], E[..., d][mm]
                thr = 0.05 if name == "v" else 0.05
                big = np.abs(ee) > thr
                gv[name] = {"expert_mean": round(float(ee.mean()), 4), "policy_mean": round(float(pe.mean()), 4),
                            "expert_abs_mean": round(float(np.abs(ee).mean()), 4), "policy_abs_mean": round(float(np.abs(pe).mean()), 4),
                            "mae": round(float(np.abs(pe - ee).mean()), 4),
                            "corr": round(float(np.corrcoef(pe, ee)[0, 1]), 3) if pe.std() > 1e-6 and ee.std() > 1e-6 else None,
                            "sign_agree_when_expert_moves": round(float((np.sign(pe[big]) == np.sign(ee[big])).mean()), 3) if big.any() else None}
            pv, ev = P[..., 16][mm], E[..., 16][mm]
            gv["stall_rate_expert_v_gt_0.1_policy_v_lt_0.05"] = round(float(((ev > 0.1) & (pv < 0.05)).sum() / max(1, (ev > 0.1).sum())), 3)
            for d, name in ((14, "left_gripper"), (15, "right_gripper")):
                gv[f"{name}_cmd_agree"] = round(float(((P[..., d][mm] > 0.5) == (E[..., d][mm] > 0.5)).mean()), 3)
            # dithering proxy: omega sign flips (|omega| > 0.05 steps) over the whole valid 50-step chunk
            def flips(seq):
                s_ = np.sign(seq[np.abs(seq) > 0.05])
                return int((s_[1:] != s_[:-1]).sum()) if len(s_) > 1 else 0
            fp = [flips(PF[k, ~PADF[k], 17]) for k in np.flatnonzero(sel)]
            fe = [flips(EF[k, ~PADF[k], 17]) for k in np.flatnonzero(sel)]
            gv["omega_chunk_flips_mean_pred"] = round(float(np.mean(fp)), 3) if fp else None
            gv["omega_chunk_flips_mean_expert"] = round(float(np.mean(fe)), 3) if fe else None
            gv["dither_ratio"] = round(float(np.mean(fp) / max(np.mean(fe), 0.05)), 3) if fp else None
            out[g] = gv
        rep[split] = out
        if a.save_npz:
            np.savez(f"{a.save_npz}_{split}.npz", idx=np.asarray(idx), P=PF, E=EF, PAD=PADF, NAV=NAVF)
        print(split, json.dumps({k: out[k] for k in ("frames", "server_vs_fp32_ref_max_norm")}), json.dumps(out["navigation"]), flush=True)
    json.dump(rep, open(a.out, "w"), indent=1)
    print("###AUDIT###" + json.dumps({"normalizer": rep["normalizer_vs_dataset_max_abs"]}), flush=True)


def stratify(npz, ds_root):
    """omega/v agreement on navigation frames by steps remaining until the navigation stage ends."""
    t = np.load(npz)
    fm = dict(np.load(os.path.join(ds_root, "frame_meta.npz")))
    ep, st = np.asarray(fm["episode_index"]), np.asarray(fm["stage"])
    left = np.zeros(len(st), int)
    for e in np.unique(ep):                                     # steps until this stage run ends
        ix = np.flatnonzero(ep == e)
        run_end = len(ix)
        for j in range(len(ix) - 1, -1, -1):
            if j < len(ix) - 1 and st[ix[j]] != st[ix[j + 1]]:
                run_end = j + 1
            left[ix[j]] = run_end - j
    out = {}
    for lo, hi in ((0, 25), (25, 75), (75, 150), (150, 10 ** 9)):
        sel = t["NAV"] & (left[t["idx"]] >= lo) & (left[t["idx"]] < hi)
        if sel.sum() < 5:
            continue
        P, E, m = t["P"][sel][:, :5], t["E"][sel][:, :5], ~t["PAD"][sel][:, :5]          # the executed H_exec steps
        pw, ew = P[..., 17][m], E[..., 17][m]
        big = np.abs(ew) > 0.05
        out[f"{lo}-{hi}"] = {"frames": int(sel.sum()), "omega_sign_agree": round(float((np.sign(pw[big]) == np.sign(ew[big])).mean()), 3) if big.any() else None,
                             "omega_corr": round(float(np.corrcoef(pw, ew)[0, 1]), 3), "omega_mae": round(float(np.abs(pw - ew).mean()), 3),
                             "v_mae": round(float(np.abs(P[..., 16][m] - E[..., 16][m]).mean()), 3)}
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--stratify":
        print("###STRATIFY###" + json.dumps(stratify(sys.argv[2], sys.argv[3])))
    else:
        main()
