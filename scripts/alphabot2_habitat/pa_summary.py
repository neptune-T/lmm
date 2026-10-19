"""Funnel summary of pa_eval.py rows (stdlib only): per (label, split, seed) pass, OVMM funnel counts at episode end
(find, grasp, carry, place, full = success_count) / eval_count, end reasons, engineering errors, and the successes.
    python pa_summary.py OUT_DIR [OUT_DIR ...]  -> first OUT_DIR/pa_summary.json
"""
import collections, glob, json, sys

FUN = ("find", "grasp", "carry", "place", "full")
rows = [json.loads(l) for O in sys.argv[1:] for f in sorted(glob.glob(f"{O}/pa_w*.jsonl")) for l in open(f)]
P = collections.OrderedDict()
for r in rows:
    k = f"{r['label']} | split={r['split']} | seed={r['seed_offset']}"
    p = P.setdefault(k, {"eval_count": 0, "engineering_errors": 0, "funnel": collections.Counter(), "end": collections.Counter(),
                         "successes": [], "units": []})
    if r.get("engineering_error"):
        p["engineering_errors"] += 1
        p["end"]["engineering_error"] += 1
        continue
    p["eval_count"] += 1
    p["units"].append(r["unit"])
    f = r.get("funnel") or {}
    for n in FUN:
        p["funnel"][n] += int(bool(f.get(n)))
    p["end"][r.get("end", "?")] += 1
    if f.get("full"):
        p["successes"].append({"unit": r["unit"], "trace": r.get("trace_out"), "steps": r.get("steps")})
out = {}
for k, p in P.items():
    out[k] = {"success": f"{p['funnel']['full']}/{p['eval_count']}", "funnel": {n: p["funnel"][n] for n in FUN},
              "eval_count": p["eval_count"], "engineering_errors": p["engineering_errors"], "end": dict(p["end"]),
              "successes": p["successes"]}
json.dump(out, open(f"{sys.argv[1]}/pa_summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))
