"""v2 A / B vs the sanity_v1 baseline: closed-loop NAV metrics and full-task success (stdlib only).

    python v2_compare.py --root RUNS/v2 --baseline RUNS/diag/sanity_v1 --out OUT.json
"""
import argparse, json, os, statistics as S

EVAL = "/workspace/runtime/runs/eval/sanity_v1_balanced"


def summ(d):
    f = os.path.join(d, "summary.json")
    if not os.path.exists(f):
        return None
    s = json.load(open(f))
    return {"success": f"{s['success_count']}/{s['eval_count']}", "funnel_counts": s["counts"],
            "engineering_errors": s["engineering_errors"], "by_hand": {k: v["counts"]["full"] for k, v in s["by_hand"].items()}}


def nav(f):
    if not f or not os.path.exists(f):
        return None
    r = json.load(open(f))
    P = [x["policy"] for x in r.values()]
    if not P:
        return None
    med = lambda k: round(S.median([p[k] for p in P]), 3)
    return {"units": len(P), "reached_within_1m": sum(p["first_step_within_1m"] is not None for p in P),
            "median_min_dist_m": med("min_dist_m"), "median_v": med("v_mean"), "median_stall_frac": med("stall_frac"),
            "median_omega_flips_per100": med("omega_flip_rate_per100"), "median_path_m": med("path_m"),
            "find": sum(x["funnel"]["find"] for x in r.values()), "full": sum(x["funnel"]["full"] for x in r.values())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rep = {}
    for v in ("BASE", "A", "B"):                                  # BASE = sanity_v1 re-evaluated under the same fp32 protocol
        E = os.path.join(a.root, f"eval_{v}")
        if not os.path.isdir(E):
            continue
        rep["sanity_v1(fp32)" if v == "BASE" else f"v2{v}"] = {"train_closed_loop": summ(os.path.join(E, "train_cl")), "train_nav": nav(os.path.join(E, "closed_loop_train.json")),
                         "val_pass0": summ(os.path.join(E, "val_pass0")), "val_pass1": summ(os.path.join(E, "val_pass1")),
                         "val_nav_pass0": nav(os.path.join(E, "closed_loop_val.json"))}
    json.dump(rep, open(a.out, "w"), indent=1)
    for k, x in rep.items():
        print(k)
        for kk, vv in x.items():
            print(f"  {kk:18s} {json.dumps(vv)}")


if __name__ == "__main__":
    main()
