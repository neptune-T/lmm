"""Funnel summary of manip_from_arrival.py rows (stdlib only).
    python manip_summary.py OUT_DIR  -> OUT_DIR/summary.json + printed table
"""
import collections, glob, json, os, sys

O = sys.argv[1]
rows = [json.loads(l) for f in sorted(glob.glob(f"{O}/manip_w*.jsonl")) for l in open(f)]
S = collections.OrderedDict()
FAIL = collections.Counter()
for r in rows:
    key = (r["entry"], r["mode"], r["group"])
    s = S.setdefault(key, collections.Counter())
    s["jobs"] += 1
    if "skipped" in r or "engineering_error" in r:
        s["skipped" if "skipped" in r else "eng_error"] += 1
        continue
    s["valid"] += 1
    o = r["outcome"]
    if r["entry"] == "obj":
        for k in ("grasp", "lift", "carry_ready", "dropped", "held_end"):
            s[k] += int(bool(o.get(k)))
        s["ovmm_carry_ever"] += int(o["ever"].get("carry", False))
        s["ovmm_full_end"] += int(o["end"].get("full", False))
        if r["mode"] == "policy":
            FAIL[("obj", "no_grasp" if not o["grasp"] else "no_lift" if not o["lift"] else "no_tuck" if not o["carry_ready"]
                  else "dropped_later" if o["dropped"] else "ok")] += 1
    else:
        s["held_at_start"] += int(bool(o.get("held_at_start")))
        s["released"] += int(bool(o.get("released")))
        s["place_ever"] += int(o["ever"].get("place", False))
        s["full_ever"] += int(o["ever"].get("full", False))
        s["full_end"] += int(o["end"].get("full", False))
        if r["mode"] == "policy":
            FAIL[("recep", "no_release" if not o["released"] else "released_no_place" if not o["ever"].get("place") else
                  "placed_not_full_end" if not o["end"].get("full") else "ok")] += 1
out = {"rows": len(rows), "by": {" / ".join(k): dict(v) for k, v in S.items()},
       "policy_failure_causes": {" / ".join(k): v for k, v in FAIL.items()}}
json.dump(out, open(f"{O}/summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))
