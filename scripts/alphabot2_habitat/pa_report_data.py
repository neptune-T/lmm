"""Aggregate every planner-assisted / arrival-diagnostic result into one JSON for the report (stdlib only).
    python pa_report_data.py > runs/v2/pa_report_data.json
Passes: every runs/v2/pa_*/pa_w*.jsonl row grouped by (dir, label, split, seed): funnel counts at episode end,
success_count / eval_count, end reasons, engineering errors, successful units (+ trace, steps). Replays:
runs/v2/pa_replays/replay.jsonl + the 10064#0 replay. Diagnostics: manip_arrival(_msft)/summary.json if present.
"""
import collections, glob, json, os

R = "/workspace/runtime/runs/v2"
FUN = ("find", "grasp", "carry", "place", "full")
passes = collections.OrderedDict()
for d in sorted(glob.glob(f"{R}/pa_*")):
    for f in sorted(glob.glob(f"{d}/pa_w*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            k = (os.path.basename(d), r["label"], r["split"], r["seed_offset"], str(r.get("policy_ckpt")))
            p = passes.setdefault(k, {"dir": k[0], "label": k[1], "split": k[2], "seed": k[3], "ckpt": k[4], "eval_count": 0,
                                      "engineering_errors": 0, "funnel": dict.fromkeys(FUN, 0), "end": collections.Counter(),
                                      "successes": [], "units": []})
            if r.get("engineering_error"):
                p["engineering_errors"] += 1
                continue
            p["eval_count"] += 1
            p["units"].append(r["unit"])
            fu = r.get("funnel") or {}
            for n in FUN:
                p["funnel"][n] += int(bool(fu.get(n)))
            p["end"][r.get("end", "?")] += 1
            if fu.get("full"):
                p["successes"].append({"unit": r["unit"], "steps": r.get("steps"), "trace": r.get("trace_out")})
replays = []
for f in (f"{R}/pa_replays/replay.jsonl", f"{R}/pa_repro/replay/replay.jsonl"):
    if os.path.exists(f):
        replays += [json.loads(l) for l in open(f)]
diag = {}
for name in ("manip_arrival", "manip_arrival_msft"):
    f = f"{R}/{name}/summary.json"
    if os.path.exists(f):
        diag[name] = json.load(open(f))
out = {"passes": [{**p, "end": dict(p["end"])} for p in passes.values()], "replays": replays, "diagnostics": diag}
print(json.dumps(out, indent=1))
