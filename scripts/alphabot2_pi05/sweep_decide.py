"""Steering-weight sanity sweep decision (pre-registered; stdlib only).

Per weight w (omega flow weight w, steering auxiliary weight w/4; 2k steps each, identical otherwise):
  open loop   v2-val navigation frames (diag_audit): omega turn-sign agreement (first H_exec steps) and
              dither_ratio = predicted / expert omega sign flips over the whole valid 50-step chunk
  closed loop 10 trained-on units x 1500 steps (diag_closed_loop): median omega flips / 100 steps, stall
Rule: maximize the open-loop turn-sign agreement subject to dither_ratio <= 1.5 and closed-loop median flip
rate <= 1.25 x that of w = 1; ties within 0.02 -> the smaller w; nothing eligible -> w = 1.

    python3 sweep_decide.py --root RUNS/v2/sweep --weights 1 2 4 --out DECISION.json
"""
import argparse, json, os, statistics as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--weights", nargs="+", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = {}
    for w in a.weights:
        tag = f"w{w:g}"
        au = json.load(open(os.path.join(a.root, tag, "audit.json")))
        nav = au["val"]["navigation"]
        cl = json.load(open(os.path.join(a.root, tag, "closed_loop.json")))
        P = [x["policy"] for x in cl.values()]
        rows[w] = {"turn_sign_agree": nav["omega"]["sign_agree_when_expert_moves"], "omega_corr": nav["omega"]["corr"],
                   "omega_mae": nav["omega"]["mae"], "dither_ratio": nav.get("dither_ratio"),
                   "closed_loop_flip_per100": round(S.median(p["omega_flip_rate_per100"] for p in P), 3),
                   "closed_loop_stall": round(S.median(p["stall_frac"] for p in P), 3),
                   "closed_loop_min_dist_m": round(S.median(p["min_dist_m"] for p in P), 3),
                   "closed_loop_within_1m": sum(p["first_step_within_1m"] is not None for p in P),
                   "val_loss_best": au.get("val_loss_best")}
    base_flip = rows[min(rows)]["closed_loop_flip_per100"]
    elig = [w for w, r in rows.items() if (r["dither_ratio"] or 9) <= 1.5 and r["closed_loop_flip_per100"] <= 1.25 * base_flip]
    if elig:
        best = max(rows[w]["turn_sign_agree"] for w in elig)
        chosen = min(w for w in elig if rows[w]["turn_sign_agree"] >= best - 0.02)
    else:
        chosen = min(rows)
    dec = {"rule": "max open-loop nav turn-sign agreement s.t. dither_ratio <= 1.5 and closed-loop flips <= 1.25 x (w=1); "
                   "ties within 0.02 -> smaller w; none eligible -> w=1",
           "results": {f"{w:g}": r for w, r in rows.items()}, "eligible": [f"{w:g}" for w in elig], "chosen_w": chosen,
           "loss": {"dim_weights": {"base.omega": chosen}, "steer_aux": {"weight": round(chosen / 4, 4), "steps": 10, "dims": ["base.v", "base.omega"]}}}
    json.dump(dec, open(a.out, "w"), indent=1)
    print(json.dumps(dec, indent=1))


if __name__ == "__main__":
    main()
