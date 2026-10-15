"""Navigation observability probes for the OVMM pi0.5 policy (DIAGNOSTICS ONLY; decides the v2 architecture).

Question: is the base-omega failure a lack of data, of temporal memory, or of explicit goal/map context?
Inputs: nav_obs_features.py records (privileged per-frame visibility / bearings, expert v / omega) of
accepted episodes, their raw head-camera frames, the SFT checkpoint (its SigLIP vision tower = the policy's
own image features) and, optionally, the open-loop audit's per-frame pi0.5 predictions.

  oracle rule     sign(bearing to the next path waypoint / to the target) vs sign(expert omega)
  probes          2-layer MLP heads (left / straight / right at |omega| 0.05, + omega regression), trained
                  on probe-train episodes, scored on held-out episodes (split by episode):
                    current        head-camera embedding at t
                    hist4 / hist8  embeddings at t, t-5, ..., (4 / 8 frames, 0.5 s apart; 2 s / 4 s)
                    oracle         privileged bearings (waypoint, target), geodesic left, distance, visibility
                    current+oracle
                    current @ 25 / 50 / 100 % of the probe-train episodes (data scaling)
  visibility      every probe and the pi0.5 policy itself scored separately on navigate_to_object frames
                  with the target visible / not visible from the head camera
Metrics: turn-sign accuracy on frames where the expert turns (|omega| > 0.05), 3-class accuracy, omega
correlation and MAE.

    python nav_probe.py --navobs DIR --units UNITS.jsonl --ckpt BEST --out OUT.json [--audit-npz F --audit-ds DS]
"""
import argparse, glob, json, os, sys, zlib

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HIST = 5                                                      # history spacing (steps at 10 Hz)


def load_navobs(d):
    recs = {}
    for f in sorted(glob.glob(os.path.join(d, "navobs_w*.jsonl"))):
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                if r.get("frames"):
                    recs[r["unit"]] = r
    return recs


def cls(w):
    return np.where(w > 0.05, 2, np.where(w < -0.05, 0, 1))


def metrics(pred_w, pred_c, w):
    turn = np.abs(w) > 0.05
    out = {"n": int(len(w)), "n_turn": int(turn.sum())}
    if turn.any():
        out["turn_sign_acc"] = round(float((np.sign(pred_w[turn]) == np.sign(w[turn])).mean()), 3)
    if pred_c is not None:
        out["acc3"] = round(float((pred_c == cls(w)).mean()), 3)
    if pred_w.std() > 1e-6 and w.std() > 1e-6:
        out["corr"] = round(float(np.corrcoef(pred_w, w)[0, 1]), 3)
    out["mae"] = round(float(np.abs(pred_w - w).mean()), 3)
    return out


def train_probe(Xtr, ytr, Xte, dev, epochs=30, hidden=256, seed=0):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xtr_t = torch.as_tensor((Xtr - mu) / sd, dtype=torch.float32, device=dev)
    Xte_t = torch.as_tensor((Xte - mu) / sd, dtype=torch.float32, device=dev)
    yc = torch.as_tensor(cls(ytr), device=dev)
    yr = torch.as_tensor(ytr, dtype=torch.float32, device=dev)
    net = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], hidden), torch.nn.GELU(), torch.nn.Dropout(0.2),
                              torch.nn.Linear(hidden, 4)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    w = torch.bincount(yc, minlength=3).float()
    cw = (w.sum() / (3 * w.clamp_min(1)))
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr_t), device=dev)
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]
            o = net(Xtr_t[b])
            loss = torch.nn.functional.cross_entropy(o[:, :3], yc[b], weight=cw) + 4 * torch.nn.functional.mse_loss(o[:, 3], yr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        o = net(Xte_t)
    pc = o[:, :3].argmax(1).cpu().numpy()
    pw = o[:, 3].cpu().numpy()
    return pw, pc


@torch.no_grad()
def embed_frames(policy, imgs, dev, bs=128):
    """Mean-pooled SigLIP features of head-camera frames (N, H, W, 3 uint8) as the policy computes them."""
    out = []
    for i in range(0, len(imgs), bs):
        x = torch.as_tensor(imgs[i:i + bs], device=dev).permute(0, 3, 1, 2).float() / 255.0
        batch = {f"observation.images.{c}": (x if c == "camera_top" else torch.zeros_like(x)) for c in ("camera_left", "camera_right", "camera_top")}
        ims, _ = policy._preprocess_images(batch)
        keys = [k.split(".")[-1] for k in policy.config.image_features]
        e = policy.model.paligemma_with_expert.embed_image(ims[keys.index("camera_top")])
        out.append(e.float().mean(1).cpu().numpy().astype(np.float16))
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--navobs", required=True)
    ap.add_argument("--units", required=True, help="jsonl {unit, dir}")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--audit-npz", default=None)
    ap.add_argument("--audit-ds", default=None)
    a = ap.parse_args()
    dev = "cuda"
    recs = load_navobs(a.navobs)
    dirs = {json.loads(l)["unit"]: json.loads(l)["dir"] for l in open(a.units) if l.strip()}
    rep = {"episodes": len(recs), "errors": sum(1 for r in recs.values() if r.get("error"))}

    # ---- frame table
    rows = []
    for u, r in recs.items():
        test = zlib.crc32(f"navprobe:{u.split('#')[0]}".encode()) % 4 == 0   # 25 % held-out episodes
        for f in r["frames"]:
            if f["bearing_waypoint"] is None:
                continue
            rows.append((u, f["t"], f["stage"] == "navigate_to_object", -1 if f["visible"] is None else int(f["visible"]),
                         f["bearing_waypoint"], f["bearing_target"], f["geodesic_left_m"] or 0.0, f["dist_target_m"],
                         f["omega"], f["v"], int(test), f["steps_left"]))
    U = np.array([x[0] for x in rows])
    T = np.array([x[1] for x in rows])
    OBJ, VIS = np.array([x[2] for x in rows]), np.array([x[3] for x in rows])
    BW, BT, GEO, DST, W, V, TEST, LEFT = (np.array([x[i] for x in rows], float) for i in (4, 5, 6, 7, 8, 9, 10, 11))
    TEST = TEST.astype(bool)
    rep["frames"] = {"total": int(len(rows)), "test": int(TEST.sum()), "obj_nav": int(OBJ.sum()),
                     "obj_nav_visible_frac": round(float((VIS[OBJ] == 1).mean()), 3),
                     "visible_frac_last10_steps_of_obj_nav": round(float((VIS[OBJ & (LEFT <= 10)] == 1).mean()), 3),
                     "visible_frac_first_half": round(float((VIS[OBJ & (LEFT > 100)] == 1).mean()), 3)}

    # ---- oracle sign rule (no learning)
    turn = np.abs(W) > 0.05
    for name, B in (("waypoint", BW), ("target", BT)):
        sel = turn & (np.abs(B) > 0.1)
        rep[f"oracle_rule_{name}"] = {"turn_sign_acc": round(float((np.sign(B[sel]) == np.sign(W[sel])).mean()), 3),
                                      "corr_omega_vs_bearing": round(float(np.corrcoef(np.clip(B, -1, 1), W)[0, 1]), 3),
                                      "frames": int(sel.sum())}
    print("oracle rules", json.dumps({k: v for k, v in rep.items() if k.startswith("oracle")}), flush=True)

    # ---- image embeddings (current + history) from the SFT checkpoint's vision tower
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.configs.policies import PreTrainedConfig
    cfg = PreTrainedConfig.from_pretrained(a.ckpt)
    cfg.compile_model, cfg.gradient_checkpointing, cfg.device, cfg.dtype = False, False, dev, "bfloat16"
    policy = PI05Policy.from_pretrained(a.ckpt, config=cfg).to(dev).eval()
    emb = {}
    for u in sorted(set(U.tolist())):
        need = sorted({max(0, t - k * HIST) for t in T[U == u].astype(int) for k in range(8)})
        ims = np.load(os.path.join(dirs[u], "images.npz"))["head_rgb"]
        E = embed_frames(policy, ims[need], dev)
        emb[u] = dict(zip(need, E))
    del policy
    torch.cuda.empty_cache()
    D = len(next(iter(next(iter(emb.values())).values())))

    def feats(k):
        return np.stack([np.concatenate([emb[u][max(0, int(t) - j * HIST)] for j in range(k)]).astype(np.float32)
                         for u, t in zip(U, T)])
    cur, h4, h8 = feats(1), feats(4), feats(8)
    orc = np.stack([np.sin(BW), np.cos(BW), np.sin(BT), np.cos(BT), np.minimum(GEO, 20), np.minimum(DST, 20),
                    (VIS == 1).astype(float)], 1).astype(np.float32)
    rep["embedding_dim"] = int(D)

    def score(name, pw, pc):
        Wt, Ot, Vt = W[TEST], OBJ[TEST], VIS[TEST]
        rep.setdefault("probes", {})[name] = {"all": metrics(pw, pc, Wt),
                                              "obj_nav_visible": metrics(pw[Ot & (Vt == 1)], pc[Ot & (Vt == 1)], Wt[Ot & (Vt == 1)]),
                                              "obj_nav_invisible": metrics(pw[Ot & (Vt == 0)], pc[Ot & (Vt == 0)], Wt[Ot & (Vt == 0)])}
        print(name, json.dumps(rep["probes"][name]), flush=True)

    tr = ~TEST
    for name, X in (("current", cur), ("hist4", h4), ("hist8", h8), ("oracle", orc),
                    ("current+oracle", np.concatenate([cur, orc], 1))):
        pw, pc = train_probe(X[tr], W[tr], X[TEST], dev)
        score(name, pw, pc)
    eps = np.array(sorted(set(U[tr].tolist())))
    rng = np.random.default_rng(0)
    for frac in (0.25, 0.5):
        keep = set(rng.choice(eps, max(2, int(len(eps) * frac)), replace=False).tolist())
        sel = tr & np.isin(U, list(keep))
        pw, pc = train_probe(cur[sel], W[sel], cur[TEST], dev)
        score(f"current@{int(frac * 100)}%", pw, pc)
    rep["probe_train_episodes"] = int(len(eps))

    # ---- the pi0.5 policy itself, by visibility (open-loop audit frames)
    if a.audit_npz and a.audit_ds:
        t = np.load(a.audit_npz)
        fm = dict(np.load(os.path.join(a.audit_ds, "frame_meta.npz")))
        idx_map = json.load(open(os.path.join(a.audit_ds, "ovmm_episode_index.json")))["episodes"]
        units = [str(e["unit"]) for e in idx_map]
        look = {}
        for u, tt, vis, obj in zip(U, T.astype(int), VIS, OBJ):
            look[(u, tt)] = (vis, obj)
        pv, ev, gv = [], [], []
        for k, i in enumerate(t["idx"]):
            if not t["NAV"][k] or t["PAD"][k][0]:
                continue
            u, f = units[int(fm["episode_index"][i])], int(fm["frame_index"][i])
            hit = look.get((u, f)) or look.get((u, f - 1)) or look.get((u, f + 1))
            if hit is None or not hit[1]:
                continue
            pv.append(t["P"][k][0, 17])
            ev.append(t["E"][k][0, 17])
            gv.append(hit[0])
        pv, ev, gv = np.array(pv), np.array(ev), np.array(gv)
        rep["pi05_policy_by_visibility"] = {"visible": metrics(pv[gv == 1], None, ev[gv == 1]),
                                            "invisible": metrics(pv[gv == 0], None, ev[gv == 0])}
        print("pi05 policy", json.dumps(rep["pi05_policy_by_visibility"]), flush=True)
    json.dump(rep, open(a.out, "w"), indent=1)
    print("###NAVPROBE###" + json.dumps({k: rep[k] for k in ("frames",) if k in rep}), flush=True)


if __name__ == "__main__":
    main()
