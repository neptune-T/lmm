"""Navigation observability probes for the OVMM pi0.5 policy (DIAGNOSTICS ONLY; decides the v2 architecture).

Question: is the base-omega failure a lack of data, of temporal memory, or of explicit goal/map context?
Probes see what the POLICY observes, from the SFT checkpoint's own encoders (frozen):
  observables(t)   SigLIP embeddings of all 3 policy cameras (camera_top <- head, camera_left <- right wrist,
                   camera_right <- left wrist; mean-pooled tokens, PCA 256 each fitted on probe-train frames)
                   + the instruction's language embedding (PaliGemma input embeddings, masked mean, PCA 64)
                   + proprio16 in the policy's state convention
  obs_current      observables(t)
  obs_hist4/8      observables(t) + cameras and state at t-5, t-10, ... (4 / 8 frames, 0.5 s apart) -- history
                   adds ONLY past vision and state
  goal_context     the stage target's bearing / distance / visibility (navigate_to_object -> object,
                   navigate_to_receptacle -> the goal receptacle the expert went to): alone, and with obs_current
                   (what an explicit goal / detection / memory input could provide)
  waypoint_oracle  bearing to the next navmesh path waypoint + geodesic left: alone, and with obs_current --
                   an ORACLE UPPER BOUND on omega predictability only, not evidence for a map
  data scaling     obs_current trained on 25 / 50 / 100 % of the probe-train episodes
Heads: 2-layer MLP, left / straight / right at |omega| 0.05 + omega regression, trained on probe-train
episodes, scored on held-out episodes (split by episode). Every score is broken down by stage x target
visibility; the pi0.5 policy's own open-loop predictions (diag_audit --save-npz) are scored the same way.

    python nav_probe.py --navobs DIR --units UNITS.jsonl --ckpt BEST --out OUT.json [--audit-npz F --audit-ds DS]
"""
import argparse, glob, json, os, sys, zlib

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ovmm_model import IO  # noqa: E402

HIST = 5                                                      # history spacing (steps at 10 Hz)
ROW_EVERY = 5                                                 # every 5th navobs frame (navobs stride 2 -> ~10 steps)
CAMS = ("camera_top", "camera_left", "camera_right")


def load_navobs(d):
    recs = {}
    for f in sorted(glob.glob(os.path.join(d, "navobs_w*.jsonl"))) + [os.path.join(d, f"navobs_w{i:02d}.jsonl") for i in range(16)]:
        if os.path.exists(f):
            for l in open(f):
                if l.strip():
                    r = json.loads(l)
                    if r.get("frames"):
                        recs[r["unit"]] = r
    return recs


def cls(w):
    return np.where(w > 0.05, 2, np.where(w < -0.05, 0, 1))


def metrics(pred_w, pred_c, w):
    out = {"n": int(len(w))}
    if not len(w):
        return out
    turn = np.abs(w) > 0.05
    out["n_turn"] = int(turn.sum())
    out["majority_acc3"] = round(float(max((cls(w) == k).mean() for k in range(3))), 3)
    if turn.any():
        out["turn_sign_acc_reg"] = round(float((np.sign(pred_w[turn]) == np.sign(w[turn])).mean()), 3)
        if pred_c is not None:                                  # classifier: predicted turn class == true turn side
            out["turn_sign_acc_cls"] = round(float((pred_c[turn] == cls(w)[turn]).mean()), 3)
    if pred_c is not None:
        out["acc3"] = round(float((pred_c == cls(w)).mean()), 3)
    if len(w) > 2 and pred_w.std() > 1e-6 and w.std() > 1e-6:
        out["corr"] = round(float(np.corrcoef(pred_w, w)[0, 1]), 3)
    out["mae"] = round(float(np.abs(pred_w - w).mean()), 3)
    return out


def train_probe(Xtr, ytr, Xte, dev, epochs=40, hidden=256, seed=0):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-3
    # clipped standardization: near-constant train dims (e.g. per-episode language PCA) must not explode on
    # held-out episodes (the unclipped regression head diverged: omega MAE 30-150)
    Xtr_t = torch.as_tensor(np.clip((Xtr - mu) / sd, -5, 5), dtype=torch.float32, device=dev)
    Xte_t = torch.as_tensor(np.clip((Xte - mu) / sd, -5, 5), dtype=torch.float32, device=dev)
    yc = torch.as_tensor(cls(ytr), device=dev)
    yr = torch.as_tensor(ytr, dtype=torch.float32, device=dev)
    net = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], hidden), torch.nn.GELU(), torch.nn.Dropout(0.3),
                              torch.nn.Linear(hidden, 4)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=5e-2)
    cnt = torch.bincount(yc, minlength=3).float()
    cw = cnt.sum() / (3 * cnt.clamp_min(1))
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(Xtr_t), device=dev)
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]
            o = net(Xtr_t[b])
            loss = torch.nn.functional.cross_entropy(o[:, :3], yc[b], weight=cw) + 4 * torch.nn.functional.smooth_l1_loss(o[:, 3], yr[b], beta=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        o = net(Xte_t)
    return o[:, 3].cpu().numpy(), o[:, :3].argmax(1).cpu().numpy()


def pca_fit(X, k, dev):
    Xt = torch.as_tensor(X, dtype=torch.float32, device=dev)
    mu = Xt.mean(0, keepdim=True)
    _, _, V = torch.pca_lowrank(Xt - mu, q=k, center=False)
    return mu, V[:, :k]


def pca_apply(X, mv, dev, bs=65536):
    mu, V = mv
    out = []
    for i in range(0, len(X), bs):
        out.append(((torch.as_tensor(X[i:i + bs], dtype=torch.float32, device=dev) - mu) @ V).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


@torch.no_grad()
def embed_frames(policy, ims3, dev, bs=96):
    """{camera: (N, H, W, 3) uint8} -> {camera: (N, D) float16} mean-pooled SigLIP features, as the policy embeds."""
    keys = [k.split(".")[-1] for k in policy.config.image_features]
    out = {c: [] for c in CAMS}
    n = len(ims3[CAMS[0]])
    for i in range(0, n, bs):
        batch = {f"observation.images.{c}": torch.as_tensor(ims3[c][i:i + bs], device=dev).permute(0, 3, 1, 2).float() / 255.0
                 for c in CAMS}
        ims, _ = policy._preprocess_images(batch)
        for c in CAMS:
            e = policy.model.paligemma_with_expert.embed_image(ims[keys.index(c)])
            out[c].append(e.float().mean(1).cpu().numpy().astype(np.float16))
    return {c: np.concatenate(v) for c, v in out.items()}


@torch.no_grad()
def embed_text(policy, tok, text, dev):
    t = tok(text, padding="longest")
    ids, m = t["input_ids"].to(dev), t["attention_mask"].to(dev).float()
    e = policy.model.paligemma_with_expert.embed_language_tokens(ids).float()
    return ((e * m[..., None]).sum(1) / m.sum(1, keepdim=True)).cpu().numpy()[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--navobs", required=True)
    ap.add_argument("--units", required=True, help="jsonl {unit, dir}")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--audit-npz", default=None)
    ap.add_argument("--audit-ds", default=None)
    ap.add_argument("--cache", default=None, help="npz cache of the per-frame encoder features (reused if present)")
    a = ap.parse_args()
    dev = "cuda"
    recs = load_navobs(a.navobs)
    dirs = {json.loads(l)["unit"]: json.loads(l)["dir"] for l in open(a.units) if l.strip()}
    rep = {"episodes": len(recs), "errors": sum(1 for r in recs.values() if r.get("error"))}

    # ---- rows (subsampled navigation frames)
    rows = []
    for u, r in recs.items():
        test = zlib.crc32(f"navprobe:{u.split('#')[0]}".encode()) % 4 == 0      # 25 % held-out episodes
        for k, f in enumerate(r["frames"]):
            if k % ROW_EVERY or f["bearing_waypoint"] is None:
                continue
            rows.append((u, f["t"], f["stage"] == "navigate_to_object", int(bool(f["visible"])), f["bearing_waypoint"],
                         f["bearing_target"], f["geodesic_left_m"] or 0.0, f["dist_target_m"], f["omega"], int(test), f["steps_left"]))
    U = np.array([x[0] for x in rows])
    T = np.array([x[1] for x in rows], int)
    OBJ, VIS = np.array([x[2] for x in rows], bool), np.array([x[3] for x in rows], int)
    BW, BT, GEO, DST, W = (np.array([x[i] for x in rows], float) for i in (4, 5, 6, 7, 8))
    TEST, LEFT = np.array([x[9] for x in rows], bool), np.array([x[10] for x in rows], float)
    rep["frames"] = {"rows": int(len(rows)), "test_rows": int(TEST.sum()), "obj_nav_rows": int(OBJ.sum()), "rec_nav_rows": int((~OBJ).sum()),
                     "visible_frac_obj_nav": round(float(VIS[OBJ].mean()), 3), "visible_frac_rec_nav": round(float(VIS[~OBJ].mean()), 3),
                     "visible_frac_obj_nav_last_10_steps": round(float(VIS[OBJ & (LEFT <= 10)].mean()), 3) if (OBJ & (LEFT <= 10)).any() else None,
                     "visible_frac_rec_nav_last_10_steps": round(float(VIS[~OBJ & (LEFT <= 10)].mean()), 3) if (~OBJ & (LEFT <= 10)).any() else None}

    # ---- oracle sign rules (no learning): stage target, path waypoint (upper bound)
    turn = np.abs(W) > 0.05
    for name, B in (("stage_target", BT), ("waypoint_upper_bound", BW)):
        for g, sel0 in (("obj_nav", OBJ), ("rec_nav", ~OBJ)):
            sel = sel0 & turn & (np.abs(B) > 0.1)
            rep.setdefault("oracle_rule", {})[f"{name}/{g}"] = {
                "turn_sign_acc": round(float((np.sign(B[sel]) == np.sign(W[sel])).mean()), 3) if sel.any() else None,
                "frames": int(sel.sum())}
    print("oracle rules", json.dumps(rep["oracle_rule"]), flush=True)

    # ---- policy-observable features from the SFT checkpoint's own encoders
    import pickle
    if a.cache and os.path.exists(a.cache):
        img, state, lang = pickle.load(open(a.cache, "rb"))
    else:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from pi05_runtime import SPM, SentencePieceTokenizer
        cfg = PreTrainedConfig.from_pretrained(a.ckpt)
        cfg.compile_model, cfg.gradient_checkpointing, cfg.device, cfg.dtype = False, False, dev, "bfloat16"
        policy = PI05Policy.from_pretrained(a.ckpt, config=cfg).to(dev).eval()
        tok = SentencePieceTokenizer(SPM)
        img, state, lang = {}, {}, {}
        for u in sorted(set(U.tolist())):
            need = sorted({max(0, t - k * HIST) for t in T[U == u] for k in range(8)})
            z = np.load(os.path.join(dirs[u], "images.npz"))
            ims3 = {c: z[IO.CAMERA_MAP[c]][need] for c in CAMS}
            E = embed_frames(policy, ims3, dev)
            img[u] = {t: np.concatenate([E[c][i] for c in CAMS]) for i, t in enumerate(need)}
            prop = np.load(os.path.join(dirs[u], "lowdim.npz"))["proprio"]
            state[u] = {t: IO.state_from_proprio(prop[t]).astype(np.float32) for t in need}
            lang[u] = embed_text(policy, tok, json.load(open(os.path.join(dirs[u], "meta.json")))["instruction"], dev)
        del policy
        torch.cuda.empty_cache()
        if a.cache:
            pickle.dump((img, state, lang), open(a.cache, "wb"), protocol=4)
    Dc = len(next(iter(next(iter(img.values())).values()))) // 3
    tr = ~TEST
    tr_units = set(U[tr].tolist())
    Xi_tr = np.stack([v for u in tr_units for v in img[u].values()]).astype(np.float32)
    pcas = [pca_fit(Xi_tr[:, j * Dc:(j + 1) * Dc], 256, dev) for j in range(3)]
    Ltr = np.stack([lang[u] for u in tr_units])
    lp = pca_fit(Ltr, min(64, len(Ltr) - 1), dev)
    rimg = {}
    for u, d in img.items():                                           # batched per episode
        ts = list(d)
        M = np.stack([d[t] for t in ts]).astype(np.float32)
        R = np.concatenate([pca_apply(M[:, j * Dc:(j + 1) * Dc], pcas[j], dev) for j in range(3)], 1)
        rimg[u] = dict(zip(ts, R))
    rlang = {u: pca_apply(lang[u][None], lp, dev)[0] for u in lang}

    def obs(k):
        out = []
        for u, t in zip(U, T):
            parts = [rimg[u][max(0, t - j * HIST)] for j in range(k)] + [state[u][max(0, t - j * HIST)] for j in range(k)]
            out.append(np.concatenate(parts + [rlang[u]]))
        return np.stack(out).astype(np.float32)
    cur, h4, h8 = obs(1), obs(4), obs(8)
    goal = np.stack([np.sin(BT), np.cos(BT), np.minimum(DST, 20), VIS.astype(float), OBJ.astype(float)], 1).astype(np.float32)
    wpt = np.stack([np.sin(BW), np.cos(BW), np.minimum(GEO, 30), OBJ.astype(float)], 1).astype(np.float32)
    rep["feature_dims"] = {"obs_current": int(cur.shape[1]), "obs_hist4": int(h4.shape[1]), "obs_hist8": int(h8.shape[1])}

    def score(name, pw, pc):
        Wt, Ot, Vt = W[TEST], OBJ[TEST], VIS[TEST]
        g = {"all": metrics(pw, pc, Wt)}
        for sname, so in (("obj_nav", Ot), ("rec_nav", ~Ot)):
            for vname, sv in (("visible", Vt == 1), ("invisible", Vt == 0)):
                m = so & sv
                g[f"{sname}/{vname}"] = metrics(pw[m], pc[m], Wt[m])
        rep.setdefault("probes", {})[name] = g
        print(name, json.dumps(g), flush=True)

    for name, X in (("obs_current", cur), ("obs_hist4", h4), ("obs_hist8", h8), ("goal_context_only", goal),
                    ("obs_current+goal_context", np.concatenate([cur, goal], 1)),
                    ("waypoint_oracle_only(upper_bound)", wpt), ("obs_current+waypoint_oracle(upper_bound)", np.concatenate([cur, wpt], 1))):
        pw, pc = train_probe(X[tr], W[tr], X[TEST], dev)
        score(name, pw, pc)
    eps = np.array(sorted(tr_units))
    rng = np.random.default_rng(0)
    for frac in (0.25, 0.5):
        keep = set(rng.choice(eps, max(2, int(len(eps) * frac)), replace=False).tolist())
        sel = tr & np.isin(U, list(keep))
        pw, pc = train_probe(cur[sel], W[sel], cur[TEST], dev)
        score(f"obs_current@{int(frac * 100)}%_episodes", pw, pc)
    rep["probe_train_episodes"] = int(len(eps))

    # ---- the pi0.5 policy itself (open-loop audit frames), by stage x visibility
    if a.audit_npz and a.audit_ds:
        t = np.load(a.audit_npz)
        fm = dict(np.load(os.path.join(a.audit_ds, "frame_meta.npz")))
        units = [str(e["unit"]) for e in json.load(open(os.path.join(a.audit_ds, "ovmm_episode_index.json")))["episodes"]]
        look = {}
        for u, r in recs.items():
            for f in r["frames"]:
                look[(u, f["t"])] = (int(bool(f["visible"])), f["stage"] == "navigate_to_object")
        P, E, G = [], [], []
        for k, i in enumerate(t["idx"]):
            if not t["NAV"][k] or t["PAD"][k][0]:
                continue
            u, f = units[int(fm["episode_index"][i])], int(fm["frame_index"][i])
            hit = look.get((u, f)) or look.get((u, f - 1)) or look.get((u, f + 1))
            if hit is None:
                continue
            P.append(t["P"][k][0, 17])
            E.append(t["E"][k][0, 17])
            G.append(hit)
        P, E = np.array(P), np.array(E)
        vis, obj = np.array([g[0] for g in G]), np.array([g[1] for g in G], bool)
        out = {"all": metrics(P, None, E)}
        for sname, so in (("obj_nav", obj), ("rec_nav", ~obj)):
            for vname, sv in (("visible", vis == 1), ("invisible", vis == 0)):
                out[f"{sname}/{vname}"] = metrics(P[so & sv], None, E[so & sv])
        rep["pi05_policy_open_loop"] = out
        print("pi05 policy", json.dumps(out), flush=True)
    json.dump(rep, open(a.out, "w"), indent=1)
    print("###NAVPROBE###" + json.dumps(rep["frames"]), flush=True)


if __name__ == "__main__":
    main()
