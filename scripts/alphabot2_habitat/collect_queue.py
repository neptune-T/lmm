"""Priority queue of AlphaBot 2 OVMM collection units (stdlib only).

Orders every not-yet-processed unit (episode, seed-bank replica) of the given AlphaBot-feasible unit lists
so that the next trajectories are the most useful ones to add:

  score = p_accept(episode) * scene_weight(scene) * arm_weight(unit)

  p_accept      Beta(1, 1) posterior mean of the episode's accept rate over its processed replicas
                ((accepted + 1) / (processed + 2)); unseen episodes 0.5 -- yield
  scene_weight  ((median accepted per scene) + 1) / (accepted in this scene + 1), clipped to [0.25, 4] --
                scarce scenes first
  arm_weight    for a unit only the under-represented arm can do: (majority / minority accepted) clipped
                to [1, 3]; else 1 -- arm balance (the expert picks the arm, so only arm-exclusive units steer it)
  repeat decay  the k-th queued replica of the same episode (k = 0, 1, ...) is scaled by 0.7**k, so new
                episodes / scenes come before more start poses of the same task

Episodes listed in manual_exclusions.json (RESET_NONDETERMINISTIC) never enter the queue. Ties break
by replica, then unit id (deterministic). Writes the queue atomically (one unit per line) and a JSON
summary next to it. The collection supervisor rebuilds it periodically and workers re-read it, so the
order follows the accepted data as it grows.

    python3 collect_queue.py --units U1.json [U2.json ...] --results RUN --out QUEUE.txt [--exclusions X.json]
"""
import argparse, json, os, statistics, time
from collections import Counter, defaultdict


def results(src):
    rows = {}
    for i in range(100):                                       # bosfs readdir lags: probe worker files
        f = f"{src}/results_w{i:02d}.jsonl"
        if os.path.exists(f):
            for line in open(f):
                try:
                    r = json.loads(line)
                    rows[str(r["episode_id"])] = r
                except (ValueError, KeyError):
                    pass
    return rows


def build(unit_files, src, exclusions=None):
    units = {}
    for f in unit_files:
        for u in json.load(open(f)):
            units[str(u["unit"])] = u
    excl = set()
    if exclusions and os.path.exists(exclusions):
        x = json.load(open(exclusions))
        excl = {str(e) for e in (x.get("episodes") or x.get("reset_nondeterministic") or [])}
    rs = results(src)
    per_ep = defaultdict(lambda: [0, 0])                       # episode -> [processed, accepted]
    scene_acc, arm_acc = Counter(), Counter()
    for u, r in rs.items():
        ep = str(r.get("episode") or u.split("#")[0])
        per_ep[ep][0] += 1
        if r.get("accepted"):
            per_ep[ep][1] += 1
            scene_acc[r.get("scene")] += 1
            arm_acc[r.get("arm")] += 1
    scenes = {u["scene"] for u in units.values()}
    med = statistics.median([scene_acc.get(s, 0) for s in scenes]) if scenes else 0
    lo, hi = sorted(("left", "right"), key=lambda a: arm_acc.get(a, 0))
    ratio = min(3.0, max(1.0, (arm_acc.get(hi, 0) + 1) / (arm_acc.get(lo, 0) + 1)))
    todo, k_ep = [], Counter()
    for uid, u in sorted(units.items(), key=lambda kv: (int(kv[1].get("replica", 0)), kv[0])):
        ep = str(u["episode_id"])
        if uid in rs or ep in excl:
            continue
        n, a = per_ep.get(ep, (0, 0))
        p = (a + 1) / (n + 2)
        sw = min(4.0, max(0.25, (med + 1) / (scene_acc.get(u["scene"], 0) + 1)))
        aw = ratio if u.get("feasible_arms") == [lo] else 1.0
        decay = 0.7 ** k_ep[ep]
        k_ep[ep] += 1
        todo.append((-(p * sw * aw * decay), int(u.get("replica", 0)), uid, round(p, 3), round(sw, 3), aw))
    todo.sort()
    summary = {"time": time.strftime("%F %T"), "units_total": len(units), "processed": sum(1 for u in units if u in rs),
               "excluded_episodes": sorted(excl), "queued": len(todo), "accepted_by_arm": dict(arm_acc),
               "minority_arm": lo, "arm_weight": round(ratio, 3), "median_accepted_per_scene": med,
               "scenes": len(scenes), "head": [{"unit": t[2], "score": round(-t[0], 4), "p_accept": t[3], "scene_w": t[4],
                                                "arm_w": t[5]} for t in todo[:10]]}
    return [t[2] for t in todo], summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", nargs="+", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclusions", default=None)
    a = ap.parse_args()
    q, summary = build(a.units, a.results, a.exclusions)
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(q) + ("\n" if q else ""))
    os.replace(tmp, a.out)
    json.dump(summary, open(a.out + ".summary.json", "w"), indent=1)
    print("###QUEUE###" + json.dumps({k: summary[k] for k in ("queued", "processed", "units_total", "minority_arm",
                                                             "arm_weight", "scenes")}), flush=True)


if __name__ == "__main__":
    main()
