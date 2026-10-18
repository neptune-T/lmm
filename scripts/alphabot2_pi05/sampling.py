"""Per-frame sampling weights for AlphaBot 2 OVMM SFT (training sampler only; never a policy input).

phase_balanced (frame_meta.npz from ovmm_to_lerobot.py):
  w = stage_weight[stage] * (cruise_weight if straight navigation cruise) * (transition_weight near a
      navigation <-> manipulation switch) * arm_balance * scene_balance
  * straight cruise: a navigation stage, |omega| <= cruise_omega, v >= cruise_v_frac * max v of that
    episode, and >= transition_window steps from any stage change;
  * transition: within transition_window steps of a stage change between a navigation stage and a
    non-navigation stage (approach -> reach, lift/carry -> navigate, navigate -> reach_place);
  * arm_balance: each arm's episodes get equal total weight;
  * scene_balance: each scene's total weight scaled by (mean scene mass / scene mass) ** scene_power.
  * nav_omega_balance (optional): within navigation frames, the anchor's omega class (left / straight /
    right at |omega| 0.05) gets equal total weight, navigation mass unchanged -- replaces the implicit
    turn oversampling of the cruise downweight (sanity_v1 sampled 35 / 26 / 38 % with straight
    under-taught); turn_weight (optional) multiplies turning navigation frames afterwards.
  * mixture (optional; recovery-data pilot): frames whose stage name starts with mixture.stage_prefix are zeroed
    BEFORE any balancing (so the other frames keep exactly the weights they would have without them); afterwards the
    mixture.anchor_stages (name -> relative weight, uniform within a name) get mixture.mass of the total probability
    mass; every other prefixed stage stays at 0 (history / chunk-continuation context only).
uniform: w = 1 (baseline config).
Weights are normalized to mean 1.
"""
import json

import numpy as np

NAV_STAGES = ("navigate_to_object", "navigate_to_receptacle")


def load_config(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def weights(frame_meta: dict, cfg: dict) -> tuple[np.ndarray, dict]:
    n = len(frame_meta["episode_index"])
    if cfg.get("mode", "uniform") == "uniform":
        return np.ones(n), {"mode": "uniform"}
    names = [str(s) for s in frame_meta["stage_names"]]
    st = np.asarray(frame_meta["stage"])
    ep = np.asarray(frame_meta["episode_index"])
    v, om = np.abs(np.asarray(frame_meta["v"])), np.abs(np.asarray(frame_meta["omega"]))
    arm, scene = np.asarray(frame_meta["arm"]), np.asarray(frame_meta["scene"])
    sw = cfg.get("stage_weight", {})
    w = np.array([float(sw.get(names[s], sw.get("default", 1.0))) for s in st])
    mx = cfg.get("mixture")
    is_mx = np.isin(st, [i for i, s in enumerate(names) if mx and s.startswith(mx.get("stage_prefix", "recovery_"))]) if mx \
        else np.zeros(n, bool)
    w[is_mx] = 0.0
    is_nav = np.isin(st, [i for i, s in enumerate(names) if s in NAV_STAGES])
    win = int(cfg.get("transition_window", 15))
    # steps to the nearest stage change (all changes / navigation<->manipulation changes), per episode
    d_change, d_navx = np.full(n, 10 ** 9), np.full(n, 10 ** 9)
    for e in np.unique(ep):
        idx = np.flatnonzero(ep == e)
        s_e, nav_e = st[idx], is_nav[idx]
        ch = np.r_[False, s_e[1:] != s_e[:-1]]
        nx = np.r_[False, ch[1:] & (nav_e[1:] != nav_e[:-1])]
        for mask, out in ((ch, d_change), (nx, d_navx)):
            pos = np.flatnonzero(mask)
            if len(pos):
                t = np.arange(len(idx))
                out[idx] = np.min(np.abs(t[:, None] - pos[None, :]), axis=1)
    vmax = np.zeros(n)
    for e in np.unique(ep):
        m = ep == e
        vmax[m] = v[m].max() if m.any() else 0.0
    cruise = is_nav & (om <= float(cfg.get("cruise_omega", 0.05))) & (v >= float(cfg.get("cruise_v_frac", 0.8)) * vmax) \
        & (vmax > 0) & (d_change > win)
    w = w * np.where(cruise, float(cfg.get("cruise_weight", 0.2)), 1.0)
    w = w * np.where(d_navx <= win, float(cfg.get("transition_weight", 3.0)), 1.0)
    if cfg.get("arm_balance", True):
        for a in (0, 1):
            m = arm == a
            if m.any():
                w[m] *= (w.sum() / 2) / w[m].sum()
    p = float(cfg.get("scene_power", 0.5))
    if p > 0:
        mass = {s: w[scene == s].sum() for s in np.unique(scene)}
        mean = np.mean(list(mass.values()))
        for s, ms in mass.items():
            w[scene == s] *= (mean / ms) ** p
    nav_om = {}
    if cfg.get("nav_omega_balance") or cfg.get("turn_weight"):
        om = np.asarray(frame_meta["omega"])
        oc = np.where(om > 0.05, 0, np.where(om < -0.05, 2, 1))           # 0 left, 1 straight, 2 right
        if cfg.get("nav_omega_balance"):
            tot = w[is_nav].sum()
            for c in range(3):
                m = is_nav & (oc == c)
                if m.any():
                    w[m] *= (tot / 3) / w[m].sum()
        tw = float(cfg.get("turn_weight", 1.0))
        if tw != 1.0:
            w[is_nav & (oc != 1)] *= tw
        nav_om = {k: float(w[is_nav & (oc == c)].sum() / max(1e-12, w[is_nav].sum())) for c, k in enumerate(("left", "straight", "right"))}
    mx_info = {}
    if mx:
        anchors = mx["anchor_stages"]
        aw = np.array([float(anchors.get(names[s], 0.0)) if is_mx[i] else 0.0 for i, s in enumerate(st)])
        if aw.sum() > 0:
            base = w[~is_mx].sum(); mass = float(mx["mass"])
            w = np.where(is_mx, aw * (mass / (1 - mass)) * base / aw.sum(), w)
        mx_info = {"mass_target": float(mx["mass"]), "mass_realized": float(w[is_mx].sum() / w.sum()), "frames": int(is_mx.sum()),
                   "anchor_frames": int((aw > 0).sum()),
                   "by_stage": {names[s]: float(w[is_mx & (st == s)].sum() / w.sum()) for s in np.unique(st[is_mx])}}
    w = w / w.mean()
    info = {"mode": cfg.get("mode"), "mixture": mx_info, "frames": int(n), "cruise_fraction": float(cruise.mean()),
            "nav_omega_class_mass": nav_om,
            "transition_fraction": float((d_navx <= win).mean()),
            "effective_mass_by_stage": {names[s]: float(w[st == s].sum() / w.sum()) for s in np.unique(st)},
            "frame_share_by_stage": {names[s]: float((st == s).mean()) for s in np.unique(st)},
            "mass_by_arm": {("left", "right")[a]: float(w[arm == a].sum() / w.sum()) for a in np.unique(arm)},
            "scene_mass_max_over_min": float(max(w[scene == s].sum() for s in np.unique(scene))
                                             / max(1e-9, min(w[scene == s].sum() for s in np.unique(scene))))}
    return w, info


if __name__ == "__main__":
    import sys
    fm = dict(np.load(sys.argv[1]))
    w, info = weights(fm, load_config(sys.argv[2]))
    print(json.dumps(info, indent=1))
