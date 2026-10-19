"""Manipulation frontier pilot analysis (numpy only).
    frontier_analysis.py --source S --verify V --dup D --sweep W --plan PREFIX --gpus 3 --out OUT.json
Restore check: each noise -1 rollout must equal its source episode bitwise (event types + every executed joint action,
whole episode) and give the same OVMM funnel. Matched determinism: each dup rollout (other server / GPU) must equal the
sweep's noise-0 rollout of the same root bitwise. Frontier: per root, full-task successes over the valid noise rollouts;
mixed root = 0 < successes < n; reported overall and by subtask, with what each rollout reached after the root.
Yield: successes / rollouts (all), successes inside mixed roots / rollouts ("effective positives": rollouts whose root
also has a failure under another noise, i.e. a non-zero advantage); GPU-hours = allocated GPUs x wall-clock span of each
stage, plus client wall time and server inference seconds.
"""
import argparse, glob, json

import numpy as np


def rows(d, mode):
    return [json.loads(l) for f in sorted(glob.glob(f"{d}/{mode}_w*.jsonl")) for l in open(f)]


def same_trace(a, b):
    try:
        A, B = np.load(a), np.load(b)
        return bool(A["ev"].shape == B["ev"].shape and (A["ev"] == B["ev"]).all() and A["joint18"].shape == B["joint18"].shape
                    and (A["joint18"] == B["joint18"]).all())
    except Exception:
        return False


def first_diff(a, b):
    A, B = np.load(a)["joint18"], np.load(b)["joint18"]
    n = min(len(A), len(B))
    d = np.flatnonzero(np.abs(A[:n] - B[:n]).max(1) > 0) if n else []
    return {"len": [int(len(A)), int(len(B))], "first_diff_step": int(d[0]) if len(d) else (None if len(A) == len(B) else n)}


def span_h(rs):
    ts = [r["t_end"] for r in rs if "t_end" in r]
    if not ts:
        return 0.0
    start = min(r["t_end"] - r["wall_s"] for r in rs if "t_end" in r)
    return (max(ts) - start) / 3600


def main():
    ap = argparse.ArgumentParser()
    for k in ("source", "verify", "dup", "sweep", "plan", "out"):
        ap.add_argument(f"--{k}", required=k in ("source", "plan", "out"))
    ap.add_argument("--gpus", type=int, default=3)
    a = ap.parse_args()
    plan = json.load(open(f"{a.plan}_sweep.json"))
    roots = {r["root_id"]: r for r in plan["roots"]}
    src = {r["job"]: r for r in rows(a.source, "source")}
    out = {"roots_planned": len(roots), "candidates": plan["n_candidates"]}
    # ---- restore check
    ver = rows(a.verify, "rollout") if a.verify else []
    vres = []
    for r in ver:
        s = src.get(r["source_job"], {})
        ok = (not r.get("engineering_error")) and same_trace(r["trace_out"], s.get("trace_out")) and r.get("funnel") == s.get("funnel")
        vres.append({"root_id": r["root_id"], "subtask": r["subtask"], "bitwise_equal": ok,
                     **({} if ok or r.get("engineering_error") else first_diff(r["trace_out"], s["trace_out"])),
                     "error": r.get("engineering_error")})
    out["restore_check"] = {"n": len(vres), "pass": sum(v["bitwise_equal"] for v in vres), "rows": vres}
    passed = {v["root_id"] for v in vres if v["bitwise_equal"]}
    # ---- sweep
    sw = rows(a.sweep, "rollout") if a.sweep else []
    by_root = {}
    for r in sw:
        by_root.setdefault(r["root_id"], []).append(r)
    dup = rows(a.dup, "rollout") if a.dup else []
    sw0 = {r["root_id"]: r for r in sw if r["noise"] == 0}
    dres = [{"root_id": r["root_id"], "bitwise_equal": bool(r["root_id"] in sw0 and same_trace(r["trace_out"], sw0[r["root_id"]]["trace_out"]))}
            for r in dup if not r.get("engineering_error")]
    out["matched_determinism"] = {"n": len(dres), "pass": sum(d["bitwise_equal"] for d in dres), "rows": dres}
    per = []
    for rid, rs in sorted(by_root.items()):
        val = [r for r in rs if not r.get("engineering_error")]
        succ = sum(bool((r.get("funnel") or {}).get("full")) for r in val)
        reach = {k: sum(bool((r.get("after") or {}).get(k)) for r in val) for k in ("carry_ready", "manip2", "released_in_manip2")}
        place = sum(bool((r.get("funnel") or {}).get("place")) for r in val)
        per.append({"root_id": rid, "unit": roots[rid]["unit"], "subtask": roots[rid]["subtask"], "source_full": roots[rid]["source_full"],
                    "restore_ok": rid in passed, "n": len(val), "eng_errors": len(rs) - len(val), "full": succ, "place_phase": place,
                    **reach, "class": "mixed" if 0 < succ < len(val) else ("all_success" if succ == len(val) and val else "all_fail")})
    out["per_root"] = per
    usable = [p for p in per if p["restore_ok"]]

    def summ(ps):
        n = sum(p["n"] for p in ps); s = sum(p["full"] for p in ps)
        mixed = [p for p in ps if p["class"] == "mixed"]
        return {"roots": len(ps), "rollouts": n, "full": s, "full_rate": round(s / n, 3) if n else None,
                "mixed_roots": len(mixed), "all_success_roots": sum(p["class"] == "all_success" for p in ps),
                "all_fail_roots": sum(p["class"] == "all_fail" for p in ps),
                "effective_positives": sum(p["full"] for p in mixed), "effective_negatives": sum(p["n"] - p["full"] for p in mixed),
                "effective_positive_yield": round(sum(p["full"] for p in mixed) / n, 3) if n else None,
                "units_with_mixed_root": len({p["unit"] for p in mixed})}
    out["frontier"] = {"all": summ(usable), "by_subtask": {s: summ([p for p in usable if p["subtask"] == s]) for s in ("grasp", "lift", "tuck", "place")},
                       "by_source_outcome": {k: summ([p for p in usable if p["source_full"] == v]) for k, v in (("source_success", True), ("source_fail", False))}}
    # ---- cost
    stages = {"source": rows(a.source, "source"), "verify": ver, "dup": dup, "sweep": sw}
    cost = {}
    for k, rs in stages.items():
        h = span_h(rs)
        cost[k] = {"rollouts": len(rs), "wall_span_h": round(h, 2), "gpu_h_allocated": round(h * a.gpus, 2),
                   "client_h": round(sum(r.get("wall_s", 0) for r in rs) / 3600, 2),
                   "server_inference_gpu_h": round(sum(r.get("server_gpu_s", 0) for r in rs) / 3600, 2)}
    out["cost"] = cost
    sw_gpu = cost["sweep"]["gpu_h_allocated"]
    f = out["frontier"]["all"]
    out["yield_per_gpu_h"] = {"full_per_gpu_h": round(f["full"] / sw_gpu, 2) if sw_gpu else None,
                              "effective_positives_per_gpu_h": round(f["effective_positives"] / sw_gpu, 2) if sw_gpu else None}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("per_root",)} | {"restore_check": {k: v for k, v in out["restore_check"].items() if k != "rows"},
                                                                                  "matched_determinism": {k: v for k, v in out["matched_determinism"].items() if k != "rows"}}, indent=1))


if __name__ == "__main__":
    main()
