"""Geometric feasibility screen of OVMM episodes for the AlphaBot 2 embodiment (no rollouts).

Per episode, from the dataset alone (+ one bare-sim instantiation per unique
object model for its bounding box):
  * target object's narrowest horizontal width in its episode orientation
    (graspable if 1.6-7.2 cm across the pads; EG2-4C2 pad gap 1.4-8.0 cm) and
    its height (>= 3 cm: fingertips reach 2.85 cm past the pad midpoint);
  * pick height and goal placement height above the start floor
    (reachable band with the canonical crouched torso: 0.35-1.15 m).
This is a PHYSICAL feasibility filter (gripper + reach), never an outcome
filter. Outputs per-episode flags and summary tables by split, scene,
object category and receptacle.

    SPLITS=train,val run_habitat.sh feasibility_screen.py
"""
import json, os, sys
from collections import Counter, defaultdict

import numpy as np

REPO = os.environ.get("LMM_REPO", "/workspace/lmm")
sys.path.insert(0, REPO)
import habitat  # noqa: E402
import habitat_sim  # noqa: E402
from habitat_baselines.config.default import get_config  # noqa: E402
from lmm.embodiments.alphabot2.habitat import env as HE  # noqa: E402

OUT = os.environ.get("OUT", "/workspace/runtime/runs/feasibility")
os.makedirs(OUT, exist_ok=True)
W_MIN, W_MAX = 0.016, 0.072
H_MIN, H_MAX = 0.35, 1.15

cfg = get_config(HE.HAB_CONFIG, overrides=[f"~{k}" for k in HE.STRETCH_ONLY])
sc = habitat_sim.SimulatorConfiguration()
sc.gpu_device_id = int(os.environ.get("GPU", "0"))
sc.scene_id = "NONE"
sc.enable_physics = False
sim = habitat_sim.Simulator(habitat_sim.Configuration(sc, [habitat_sim.agent.AgentConfiguration()]))
otm = sim.get_object_template_manager()
for p in cfg.habitat.simulator.additional_object_paths:
    otm.load_configs(os.path.join(os.getcwd(), p))   # run_habitat.sh runs from the data tree root
rom = sim.get_rigid_object_manager()
bbox_cache = {}


def local_extent(handle):
    name = handle.split("_:")[0]
    if name in bbox_cache:
        return bbox_cache[name]
    tmpl = otm.get_template_handles(name)
    ext = None
    if tmpl:
        o = rom.add_object_by_template_handle(tmpl[0])
        if o is not None:
            bb = o.root_scene_node.cumulative_bb
            ext = (np.array(bb.max) - np.array(bb.min)).tolist()
            rom.remove_object_by_id(o.object_id)
    bbox_cache[name] = ext
    return ext


summary = {}
for split in os.environ.get("SPLITS", "train,val").split(","):
    c = get_config(HE.HAB_CONFIG, overrides=[f"~{k}" for k in HE.STRETCH_ONLY] + [f"habitat.dataset.split={split}"])
    ds = habitat.make_dataset(c.habitat.dataset.type, config=c.habitat.dataset)
    TM = ds.transformations_matrix
    rows = []
    for e in ds.episodes:
        fy = float(e.start_position[1])
        # target object = the rigid object whose handle is the episode's target key
        tkey = list(e.targets.keys())[0]
        ro = next((r for r in e.rigid_objs if r[0].split(".")[0] == tkey.split("_:")[0]), e.rigid_objs[0])
        T = np.vstack([TM[ro[1]], [0, 0, 0, 1]]) if TM is not None else np.asarray(ro[1])
        ext = local_extent(ro[0])
        R = T[:3, :3]
        if ext is not None:
            wext = np.abs(R) @ (np.array(ext) / 2) * 2              # world-axis extents (Habitat y up)
            vert = int(np.argmax(np.abs(R[1, :])))                  # local axis closest to world up
            hz = [i for i in range(3) if i != vert]
            width = float(min(ext[i] for i in hz))
            height = float(ext[vert])
        else:
            width = height = None
        g = e.targets[tkey]
        GT = np.vstack([TM[g], [0, 0, 0, 1]]) if TM is not None and np.ndim(g) == 0 else np.asarray(g)
        pick_h = float(e.candidate_objects[0].position[1]) - fy
        place_h = float(GT[1, 3]) - fy
        rows.append({"episode_id": e.episode_id, "scene": e.scene_id.split("/")[-1].split(".")[0],
                     "object": e.object_category, "start_recep": e.start_recep_category,
                     "goal_recep": e.goal_recep_category, "width_m": width, "pick_h": pick_h, "place_h": place_h,
                     "height_m": height,
                     "grasp_ok": bool(width is not None and W_MIN <= width <= W_MAX and height >= 0.03),
                     "pick_h_ok": H_MIN <= pick_h <= H_MAX, "place_h_ok": H_MIN <= place_h <= H_MAX})
    for r in rows:
        r["feasible"] = r["grasp_ok"] and r["pick_h_ok"] and r["place_h_ok"]
    json.dump(rows, open(f"{OUT}/screen_{split}.json", "w"))
    f = [r for r in rows if r["feasible"]]
    summary[split] = {"episodes": len(rows), "grasp_ok": sum(r["grasp_ok"] for r in rows),
                      "pick_h_ok": sum(r["pick_h_ok"] for r in rows), "place_h_ok": sum(r["place_h_ok"] for r in rows),
                      "feasible": len(f), "feasible_scenes": len({r["scene"] for r in f}),
                      "scenes": len({r["scene"] for r in rows}),
                      "feasible_objects": dict(Counter(r["object"] for r in f).most_common(25)),
                      "feasible_start_receps": dict(Counter(r["start_recep"] for r in f).most_common(12)),
                      "feasible_goal_receps": dict(Counter(r["goal_recep"] for r in f).most_common(12)),
                      "widest_objects_rejected": dict(Counter(r["object"] for r in rows if not r["grasp_ok"]).most_common(12))}
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
print("###SCREEN###" + json.dumps({s: {k: v for k, v in d.items() if not isinstance(v, dict)} for s, d in summary.items()}))
sim.close()
