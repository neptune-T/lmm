"""Funnel summary of eval_policy.py results (stdlib only): overall, by grasping hand, by scene.

eval_count = every scheduled unit (UNITS, comma-separated); an engineering error or a unit with no
result row counts as a failure at every stage and is also reported separately. Funnel (OVMM's measures):
find/approach -> grasp -> carry -> place -> full success.

    python3 eval_summary.py EVAL_DIR [UNITS]   -> prints ###EVAL_SUMMARY### JSON and writes EVAL_DIR/summary.json
"""
import json, os, sys
from collections import defaultdict

STAGES = ("find", "grasp", "carry", "place", "full")
LABELS = {"find": "find/approach", "grasp": "grasp", "carry": "carry", "place": "place", "full": "full success"}


def load(d):
    rows = {}
    for i in range(100):                          # explicit names: bosfs directory listings lag
        f = f"{d}/eval_w{i:02d}.jsonl"
        if os.path.exists(f):
            for l in open(f):
                if l.strip():
                    r = json.loads(l)
                    rows[r["unit"]] = r
    return rows


def agg(rs):
    n = len(rs)
    counts = {s: int(sum(r.get("funnel", {}).get(s, 0) for r in rs)) for s in STAGES}
    out = {s: round(counts[s] / n, 4) if n else None for s in STAGES}
    out.update({"n": n, "counts": counts})
    return out


def summarize(d, units=None):
    rows = load(d)
    units = units or sorted(rows)
    rs = []
    for u in units:
        r = rows.get(u) or {"unit": u, "engineering_error": "no result row (worker crashed before this unit)"}
        if "funnel" not in r:
            r = {**r, "funnel": {s: 0 for s in STAGES}, "scene": r.get("scene") or "unknown"}
        rs.append(r)
    eng = [r["unit"] for r in rs if r.get("engineering_error")]
    groups = lambda key: {k: agg(v) for k, v in sorted(_group(rs, key).items())}
    ov = agg(rs)
    return {"eval_count": len(rs), "success_count": ov["counts"]["full"], "full_success_count": ov["counts"]["full"],
            "success_rate": ov["full"], "episodes": len(rs), "engineering_errors": len(eng), "engineering_error_units": eng,
            "funnel_labels": LABELS, "counts": ov["counts"], "overall": ov,
            "by_hand": groups("hand"), "by_scene": groups("scene")}


def _group(rs, key):
    g = defaultdict(list)
    for r in rs:
        g[str(r.get(key))].append(r)
    return g


if __name__ == "__main__":
    us = [u for u in sys.argv[2].split(",") if u] if len(sys.argv) > 2 else None
    s = summarize(sys.argv[1], us)
    json.dump(s, open(os.path.join(sys.argv[1], "summary.json"), "w"), indent=1)
    print("###EVAL_SUMMARY###" + json.dumps(s))
