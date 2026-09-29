"""Summarize nav_qual.py results (one or more result directories)."""
import json, statistics as S, sys
from pathlib import Path


def leg(ok, k):
    L = [r.get(k, {}) for r in ok]
    pf = [l for l in L if l.get("path_found")]
    arr = [l for l in pf if l.get("arrived")]
    eff = [l["geodesic_m"] / max(l["travelled_m"], 1e-6) for l in arr if l.get("geodesic_m", 0) > 0.3]
    succ_key = "ovmm_nav_to_pick_succ" if k == "to_object" else "ovmm_nav_to_place_succ"
    return {"episodes": len(L), "path_found": len(pf), "arrived": len(arr),
            "collision_episodes": sum(l.get("collision_steps", 0) > 0 for l in pf),
            "collision_steps": sum(l.get("collision_steps", 0) for l in pf),
            "max_penetration_m": min([l.get("max_depth_m", 0.0) for l in pf] or [0.0]),
            "navmesh_filtered_median_m": round(S.median([l["navmesh_filtered_m"] for l in pf]), 3) if pf else None,
            "path_efficiency_median": round(S.median(eff), 3) if eff else None,
            "arrival_dist_median_m": round(S.median([l["arrival_dist_m"] for l in arr]), 3) if arr else None,
            "ovmm_nav_succ": sum(float(l.get("ovmm_metrics", {}).get(succ_key, 0) or 0) for l in pf),
            "collision_links": {k2: v for l in pf for k2, v in l.get("collision_links", {}).items()}}


def summarize(d):
    rows = [json.loads(l) for f in sorted(Path(d).glob("results_w*.jsonl")) for l in open(f)]
    err = [r for r in rows if r.get("engineering_error")]
    ok = [r for r in rows if not r.get("engineering_error")]
    moved = [r["start_fix"]["moved_m"] for r in ok if (r.get("start_fix") or {}).get("moved_m")]
    both = [r for r in ok if r.get("to_object", {}).get("arrived") and r.get("to_receptacle", {}).get("arrived")]
    clean = [r for r in both if r["to_object"].get("collision_steps", 0) == 0 and r["to_receptacle"].get("collision_steps", 0) == 0]
    vp = [r["viewpoints"] for r in ok if "viewpoints" in r]
    return {"dir": str(d), "episodes": len(rows), "scenes": len({r.get("scene") for r in ok}),
            "reset_goals_unreachable": sum("None of the goals" in r["engineering_error"] for r in err),
            "other_errors": [r["engineering_error"][:100] for r in err if "None of the goals" not in r["engineering_error"]],
            "start_snapped": len(moved), "start_snap_median_m": round(S.median(moved), 3) if moved else 0.0,
            "to_object": leg(ok, "to_object"), "to_receptacle": leg(ok, "to_receptacle"),
            "both_legs_arrived": len(both), "both_legs_collision_free": len(clean),
            "object_viewpoints_navigable": [sum(v["object_navigable"] for v in vp), sum(v["object_total"] for v in vp)],
            "recep_viewpoints_navigable": [sum(v["recep_navigable"] for v in vp), sum(v["recep_total"] for v in vp)]}


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(json.dumps(summarize(d), indent=1))
