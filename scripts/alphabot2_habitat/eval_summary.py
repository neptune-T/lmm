"""Funnel summary of eval_policy.py results (stdlib only): overall, by grasping hand, by scene.

    python3 eval_summary.py EVAL_DIR   -> prints ###EVAL_SUMMARY### JSON and writes EVAL_DIR/summary.json
"""
import json, os, sys
from collections import defaultdict

STAGES = ("find", "grasp", "carry", "place", "full")


def load(d):
    rows = []
    for i in range(100):                          # explicit names: bosfs directory listings lag
        f = f"{d}/eval_w{i:02d}.jsonl"
        if os.path.exists(f):
            rows += [json.loads(l) for l in open(f) if l.strip()]
    return rows


def agg(rs):
    out = {s: round(sum(r["funnel"][s] for r in rs) / len(rs), 4) if rs else None for s in STAGES}
    out["n"] = len(rs)
    return out


def summarize(d):
    rows = load(d)
    ok = [r for r in rows if "funnel" in r]
    groups = lambda key: {k: agg(v) for k, v in sorted(_group(ok, key).items())}
    return {"episodes": len(rows), "engineering_errors": len(rows) - len(ok), "overall": agg(ok),
            "full_success_count": int(sum(r["funnel"]["full"] for r in ok)),
            "by_hand": groups("hand"), "by_scene": groups("scene")}


def _group(rs, key):
    g = defaultdict(list)
    for r in rs:
        g[str(r.get(key))].append(r)
    return g


if __name__ == "__main__":
    s = summarize(sys.argv[1])
    json.dump(s, open(os.path.join(sys.argv[1], "summary.json"), "w"), indent=1)
    print("###EVAL_SUMMARY###" + json.dumps(s))
