"""Which prim fills the black half of the wrist camera images?

Adds an instance-id segmentation annotator to both wrist cameras, renders the
home pose, and reports the prim paths behind the black (max channel < 8)
pixels, plus the camera placement relative to the d405 prim, link7 and the
gripper base.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import simworld as SW  # noqa: E402

app, ZD = SW.launch()
import numpy as np  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

W = SW.SimWorld(app, ZD)
B = W.B
WRISTS = ("cam_left_wrist", "cam_right_wrist")
for k in WRISTS:
    B.cameras[k].add_instance_id_segmentation_to_frame()
W.reset_robot((0.0, 0.0), 0.0)
for _ in range(4):
    W.world.step(render=True)
R = {}
xc = UsdGeom.XformCache()
for k in WRISTS:
    img = B.render(k)
    black = img.max(axis=2) < 8
    ann = B.cameras[k]._custom_annotators["instance_id_segmentation"]
    d = ann.get_data()
    seg, info = np.asarray(d["data"]), d.get("info", {})
    id2path = {int(i): p for i, p in info.get("idToLabels", {}).items()}
    ids, cnt = np.unique(seg[black], return_counts=True)
    top = sorted(zip(cnt.tolist(), ids.tolist()), reverse=True)[:8]
    ids_all, cnt_all = np.unique(seg, return_counts=True)
    top_all = sorted(zip(cnt_all.tolist(), ids_all.tolist()), reverse=True)[:10]
    side = "Left" if "left" in k else "Right"
    cam_p, cam_r = W.camera_world_pose(k)
    d405 = [p for p in Usd.PrimRange(W.stage.GetPrimAtPath(f"/World/Bot2/Arm_{side}"), Usd.TraverseInstanceProxies())
            if p.GetName() == "d405"][0]
    d405_m = np.array(xc.GetLocalToWorldTransform(d405)).T
    R[k] = {"black_frac": float(black.mean()),
            "black_pixels_by_prim": [{"pixels": c, "prim": id2path.get(i, str(i))} for c, i in top],
            "all_pixels_by_prim": [{"pixels": c, "prim": id2path.get(i, str(i))} for c, i in top_all],
            "camera_pos": cam_p.round(4).tolist(), "optical_axis": (-cam_r[:, 2]).round(3).tolist(),
            "d405_prim": str(d405.GetPath()), "d405_pos": d405_m[:3, 3].round(4).tolist(),
            "d405_axes_xyz": d405_m[:3, :3].T.round(3).tolist(),
            "gripper_base_pos": B.link_pose(f"Gripper_{side}/gripper_base")[0].round(4).tolist(),
            "d405_children": [str(c.GetPath()) for c in Usd.PrimRange(d405, Usd.TraverseInstanceProxies())][:25]}
    print(json.dumps({k: R[k]["black_pixels_by_prim"]}), flush=True)
print("###OCC###" + json.dumps(R), flush=True)
app.close()
