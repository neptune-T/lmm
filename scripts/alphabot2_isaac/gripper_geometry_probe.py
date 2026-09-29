"""Measure EG2-4C2 finger collision geometry in the gripper_base frame (open and closed).

Reports, along the approach axis (+z of gripper_base): where the pad link
origins sit and how far the finger collision geometry actually extends, so that
grasp targets can be expressed for the fingertips rather than the link origins.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("LMM_REPO", os.path.dirname(os.path.dirname(HERE)))
USDS = os.environ.get("ROBOT_USDS", "/nix/yanjun/zts/alphabot2/deps/robot_usds")
sys.path.insert(0, REPO)
from lmm.embodiments.alphabot2 import isaac_adapter as IA  # noqa: E402
from lmm.embodiments.alphabot2 import isaac_backend as IB  # noqa: E402
sys.argv.append(IB.RTX_DRIVER_CHECK_BYPASS)
from isaacsim import SimulationApp  # noqa: E402
app = SimulationApp({"headless": True})
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleArticulation  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402
from lmm.embodiments.alphabot2.contract import AlphaBot2Contract  # noqa: E402

world = World(stage_units_in_meters=1.0, physics_dt=1 / IA.PHYSICS_HZ, rendering_dt=1 / IA.PHYSICS_HZ)
stage = omni.usd.get_context().get_stage()
IB.build_stage(app, stage, USDS)
world.scene.add_default_ground_plane()
art = SingleArticulation(prim_path=IB.ROBOT_PRIM, name="bot2")
world.scene.add(art)
world.reset()
B = IB.IsaacArticulationBackend(world, art)
A = IA.AlphaBot2IsaacAdapter(AlphaBot2Contract(), backend=B)
bbc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"])
LINKS = ("left_pad", "right_pad", "left_inner_link", "right_inner_link", "left_outer_link", "right_outer_link")
out = {}
for g, label in ((0.0, "open"), (1.0, "closed")):
    a = A.home_action()
    a[7] = g
    for _ in range(12):
        A.apply_action(a)
    pg, rg = B.link_pose("Gripper_Left/gripper_base")
    res = {}
    for link in LINKS:
        lp = stage.GetPrimAtPath(f"{IB.ROBOT_PRIM}/Gripper_Left/{link}")
        pl, rl = B.link_pose(f"Gripper_Left/{link}")
        pts = []
        for p in Usd.PrimRange(lp, Usd.TraverseInstanceProxies()):
            if p.HasAPI(UsdPhysics.CollisionAPI):
                rng = bbc.ComputeRelativeBound(p, lp).ComputeAlignedRange()
                lo, hi = np.array(rng.GetMin()), np.array(rng.GetMax())
                for cx in (lo[0], hi[0]):
                    for cy in (lo[1], hi[1]):
                        for cz in (lo[2], hi[2]):
                            pts.append(rg.T @ (pl + rl @ np.array([cx, cy, cz]) - pg))
        pts = np.array(pts)
        origin = rg.T @ (pl - pg)
        res[link] = {"origin_gb": [round(float(v), 4) for v in origin],
                     "collision_min_gb": [round(float(v), 4) for v in pts.min(0)] if len(pts) else None,
                     "collision_max_gb": [round(float(v), 4) for v in pts.max(0)] if len(pts) else None}
    tips = [res[l]["collision_max_gb"][2] for l in LINKS if res[l]["collision_max_gb"]]
    res["fingertip_max_along_approach_m"] = round(float(max(tips)), 4)
    res["pad_origin_mid_along_approach_m"] = round(float((res["left_pad"]["origin_gb"][2] + res["right_pad"]["origin_gb"][2]) / 2), 4)
    out[label] = res
print("###GEOM###" + json.dumps(out), flush=True)
app.close()
