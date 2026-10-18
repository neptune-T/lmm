# EXPLORATORY supplement: path-deviation-only version of perturb_drift's primary statistics (heading disabled because
# nearest-pose heading error is ill-defined where the expert rotates in place); same 0.5 m threshold (and 0.25 / 1.0).
import json, sys, glob, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import perturb_drift as D
F = "/workspace/runtime/runs/v2/perturb/full"
ref = {k: np.asarray(v) for k, v in json.load(open("/workspace/runtime/runs/v2/stall_diag/A_prelim.json"))["expert_nav_pose"].items() if k in ("left", "right")}
rows = [json.loads(l) for f in sorted(glob.glob(f"{F}/perturb_w*.jsonl")) for l in open(f)]
ok = {(r["unit"], r["t"], r["seed"], r["level"]): r for r in rows if "steps" in r and os.path.exists(f"{F}/traces/{r['job']}.npz")}
lv = ["none", "yaw15", "yaw30", "yaw60"]
keys = sorted({k[:3] for k in ok if all((*k[:3], L) in ok for L in lv)})
R, groups = [], []
for k in keys:
    r = ok[(*k, "none")]; o = D.analyse(f"{F}/traces/{r['job']}.npz", r["source"], ref); o["head1e+09"] = None; R.append(o); groups.append(k[0].split("#")[0])
rng = np.random.default_rng(0)
out = {}
for p in (0.25, 0.5, 1.0):
    S = D.stats(R, 1e9, p)                      # heading threshold 1e9 deg = disabled -> "either" == path only
    ci = D.cluster_ci(R, groups, 1e9, p, rng)
    e = S["either"]; hz = S["hazard"]
    on = hz["on_track"]
    out[p] = {"triggered": S["triggered"], "drifted_before": e["triggered_drifted_before"], "p_trig": e["p_triggered"],
              "p_time_matched_ctrl": e["p_time_matched_nontriggered"], "diff_ci95": ci["either_p_triggered_minus_time_matched_ci95"],
              "lead_med": e["lead_steps_median"], "lead_iqr": e["lead_steps_iqr"], "hazard": hz,
              "hazard_ratio": S["hazard_ratio_drifted_vs_on_track"], "hazard_ratio_ci95": ci["hazard_ratio_ci95"],
              "on_track_rate_upper95_rule_of_3": round(3 / on["eligible"], 5) if on["events"] == 0 and on["eligible"] else None}
    print(f"path>{p} m:", json.dumps(out[p]))
json.dump({"label": "EXPLORATORY supplement (path-only; heading metric artifact at in-place rotation states)", "results": out},
          open(f"{F}/drift_exploratory_pathonly.json", "w"), indent=1)
