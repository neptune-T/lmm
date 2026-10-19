"""Root selection for the manipulation frontier pilot (deterministic; uses only the frontier_pa.py SOURCE traces, written
before any frontier rollout). Every root is a policy query boundary inside a policy phase of a v2-B PA-OG source episode
on a v2 TRAIN unit (policy-visited state). Subtask label at the query (privileged per-step logs, selection only):
  grasp  MANIP1, target not held yet, nearest gripper pad within GRASP_NEAR_M of the target (first such query)
  lift   MANIP1, held, target < LIFT_M above its start height (first such query)
  tuck   MANIP1, held, lifted >= LIFT_M, an arm still > TUCK_RAD from the NAV tuck (first such query)
  place  MANIP2, held: the last query <= PRE_RELEASE steps before the source's release; if it never released, the query
         PLACE_FALLBACK steps into MANIP2
Selection: at most one root per (source episode, subtask) and MAX_PER_UNIT per unit; round-robin over units (sorted) and
source seeds so the roots spread over tasks; quotas per subtask; leftover quota is filled from the other subtasks.
Job files: OUT_verify.json = noise -1 for every root (restore check: must reproduce the source bitwise); OUT_dup.json =
noise 0 for N_DUP roots (one per subtask first), run on OTHER servers and compared bitwise with the sweep's noise 0
(matched-rollout determinism); OUT_sweep.json = noises 0..N-1 for every root.
    frontier_plan.py --source-dir DIR --out PREFIX [--quota grasp=8,lift=6,tuck=6,place=10] [--noises 8] [--n-dup 6]
"""
import argparse, glob, json
from pathlib import Path

import numpy as np

GRASP_NEAR_M, LIFT_M, TUCK_RAD, PRE_RELEASE, PLACE_FALLBACK, MAX_PER_UNIT = 0.12, 0.05, 0.30, 10, 15, 3


def candidates(row):
    d = np.load(row["trace_out"])
    snaps = json.loads(str(d["snaps"]))
    held, dh, te = d["held"].astype(bool), d["obj_dh"], d["tuck_err"].reshape(-1, 2).max(1)
    pad = d["pad_d"]; ph = [str(p) for p in d["phase"]]
    out = {}
    m2_steps = [i for i, p in enumerate(ph) if p == "manip2"]
    rel = next((i for i in m2_steps if not held[i]), None)
    for si, s in enumerate(snaps):
        t = s["t"]
        if t == 0:
            continue
        i = t - 1                                                    # state at the query = after step t-1
        if s["phase"] == "manip1":
            if not held[i] and pad[i] < GRASP_NEAR_M and not held[: i + 1].any():
                out.setdefault("grasp", (si, t))
            elif held[i] and dh[i] < LIFT_M:
                out.setdefault("lift", (si, t))
            elif held[i] and dh[i] >= LIFT_M and te[i] > TUCK_RAD:
                out.setdefault("tuck", (si, t))
        elif s["phase"] == "manip2" and held[i]:
            if rel is not None:
                if t <= rel - PRE_RELEASE:
                    out["place"] = (si, t)                           # keeps the LAST query before the cutoff
            elif m2_steps and t >= m2_steps[0] + PLACE_FALLBACK:
                out.setdefault("place", (si, t))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quota", default="grasp=8,lift=6,tuck=6,place=10")
    ap.add_argument("--noises", type=int, default=8)
    ap.add_argument("--n-dup", type=int, default=6)
    a = ap.parse_args()
    quota = {k: int(v) for k, v in (x.split("=") for x in a.quota.split(","))}
    rows = [json.loads(l) for f in sorted(glob.glob(f"{a.source_dir}/source_w*.jsonl")) for l in open(f)]
    rows = [r for r in rows if r.get("trace_out") and not r.get("engineering_error")]
    cands = []
    for r in sorted(rows, key=lambda r: (r["unit"], r["seed"])):
        for sub, (si, t) in candidates(r).items():
            cands.append({"unit": r["unit"], "source_job": r["job"], "source_seed": r["seed"], "source_trace": r["trace_out"],
                          "source_full": bool((r.get("funnel") or {}).get("full")), "subtask": sub, "snap": si, "t_root": t})
    picked, per_unit = [], {}
    order = sorted(quota, key=lambda s: -quota[s])
    for sub in order:                                                # round-robin over units within a subtask
        pool = [c for c in cands if c["subtask"] == sub]
        units = sorted({c["unit"] for c in pool})
        took = 0
        while took < quota[sub] and pool:
            progressed = False
            for u in units:
                if took >= quota[sub]:
                    break
                if per_unit.get(u, 0) >= MAX_PER_UNIT:
                    continue
                c = next((c for c in pool if c["unit"] == u), None)
                if c is None:
                    continue
                pool.remove(c); picked.append(c); per_unit[u] = per_unit.get(u, 0) + 1; took += 1; progressed = True
            if not progressed:
                break
    short = sum(quota.values()) - len(picked)
    if short > 0:                                                    # fill from any subtask, same caps
        for c in cands:
            if short <= 0:
                break
            if c in picked or per_unit.get(c["unit"], 0) >= MAX_PER_UNIT:
                continue
            if any(p["source_job"] == c["source_job"] and p["subtask"] == c["subtask"] for p in picked):
                continue
            picked.append(c); per_unit[c["unit"]] = per_unit.get(c["unit"], 0) + 1; short -= 1
    for i, c in enumerate(picked):
        c["root_id"] = f"R{i:02d}_{c['subtask']}_{c['unit'].replace('#', '_r')}_s{c['source_seed']}_t{c['t_root']}"
    mk = lambda c, nz: {**c, "noise": nz, "job": f"{c['root_id']}_n{nz}"}
    dup = []
    for sub in ("grasp", "lift", "tuck", "place"):                  # one per subtask first, then by root order
        dup += [c for c in picked if c["subtask"] == sub][:1]
    dup += [c for c in picked if c not in dup]
    files = {"verify": [mk(c, -1) for c in picked], "dup": [mk(c, 0) for c in dup[: a.n_dup]],
             "sweep": [mk(c, nz) for nz in range(a.noises) for c in picked]}
    meta = {"constants": {"GRASP_NEAR_M": GRASP_NEAR_M, "LIFT_M": LIFT_M, "TUCK_RAD": TUCK_RAD, "PRE_RELEASE": PRE_RELEASE,
                          "PLACE_FALLBACK": PLACE_FALLBACK, "MAX_PER_UNIT": MAX_PER_UNIT}, "quota": quota,
            "n_candidates": len(cands), "roots": picked}
    for name, jobs in files.items():
        Path(f"{a.out}_{name}.json").write_text(json.dumps({**meta, "jobs": jobs}, indent=1))
    by = {}
    for c in picked:
        by[c["subtask"]] = by.get(c["subtask"], 0) + 1
    print(json.dumps({"candidates": len(cands), "roots": len(picked), "by_subtask": by, "units": len(per_unit),
                      "source_full_roots": sum(c["source_full"] for c in picked), "jobs": {k: len(v) for k, v in files.items()}}))


if __name__ == "__main__":
    main()
