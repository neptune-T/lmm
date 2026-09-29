"""Stratified sample of geometrically feasible OVMM episodes (feasibility_screen.py output).

Round-robin over scenes; within a scene prefer object categories not yet used.
    python pick_episodes.py SCREEN_JSON N [SEED] > episode ids (comma separated)
"""
import json, random, sys
from collections import defaultdict

rows = [r for r in json.load(open(sys.argv[1])) if r["feasible"]]
n, seed = int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 0
rnd = random.Random(seed)
by_scene = defaultdict(list)
for r in rows:
    by_scene[r["scene"]].append(r)
for v in by_scene.values():
    rnd.shuffle(v)
scenes = sorted(by_scene)
rnd.shuffle(scenes)
out, used_cat = [], set()
while len(out) < n and any(by_scene.values()):
    for s in scenes:
        if not by_scene[s] or len(out) >= n:
            continue
        pool = by_scene[s]
        pick = next((r for r in pool if r["object"] not in used_cat), pool[0])
        pool.remove(pick)
        used_cat.add(pick["object"])
        out.append(pick["episode_id"])
print(",".join(out))
