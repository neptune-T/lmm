"""Camera / state / action synchronisation check for the 10 Hz recording loop.

Uses the same loop as the pilot recorder: at step t the observation (images +
proprio) is read BEFORE action t is applied; cameras render on the last physics
substep of every action. Two stimuli: a block teleported into view (arm still)
and the right hand commanded 10 cm sideways. Measured: the camera annotators
return the PREVIOUS render, so images read before action t show the state one
action old. Reading images AFTER applying action t (``read_after``) pairs them
with the proprio read before action t -- the recorder does this.
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
from pxr import Usd, UsdGeom, UsdLux  # noqa: E402
from lmm.embodiments.alphabot2 import cartesian as C  # noqa: E402
from lmm.embodiments.alphabot2.contract import AlphaBot2Contract  # noqa: E402

world = World(stage_units_in_meters=1.0, physics_dt=1 / IA.PHYSICS_HZ, rendering_dt=1 / IA.PHYSICS_HZ)
stage = omni.usd.get_context().get_stage()
IB.build_stage(app, stage, USDS)
world.scene.add_default_ground_plane()
sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
sun.CreateIntensityAttr(3000.0)
from isaacsim.core.api.objects import FixedCuboid  # noqa: E402
block = FixedCuboid(prim_path="/World/sync_block", name="sync_block", position=np.array([0.0, 0.0, -5.0]),
                    scale=np.array([0.30, 0.30, 0.30]), color=np.array([1.0, 0.0, 1.0]))
art = SingleArticulation(prim_path=IB.ROBOT_PRIM, name="bot2")
world.scene.add(art)
world.reset()
B = IB.IsaacArticulationBackend(world, art)
A = IA.AlphaBot2IsaacAdapter(AlphaBot2Contract(), backend=B)
xc = UsdGeom.XformCache()
B.add_camera("cam_front", f"{IB.ROBOT_PRIM}/head_link2/head_camera", (0, 0, 0), IB.camera_rows((0, 0, 1), (0, -1, 0)), (320, 240), 69.0, 0.10)
S = "Right"
d405 = [p for p in Usd.PrimRange(stage.GetPrimAtPath(f"{IB.ROBOT_PRIM}/Arm_{S}"), Usd.TraverseInstanceProxies()) if p.GetName() == "d405"][0]
p7, r7 = B.link_pose(f"Arm_{S}/link7")
rel = np.array(xc.GetLocalToWorldTransform(d405) * xc.GetLocalToWorldTransform(stage.GetPrimAtPath(f"{IB.ROBOT_PRIM}/Arm_{S}/link7")).GetInverse()).T
cam_w = p7 + r7 @ rel[:3, 3]
pg, rg = B.link_pose(f"Gripper_{S}/gripper_base")
mid = (B.link_pose(f"Gripper_{S}/left_pad")[0] + B.link_pose(f"Gripper_{S}/right_pad")[0]) / 2
up = cam_w - (pg + rg[:, 2] * ((cam_w - pg) @ rg[:, 2]))
B.add_camera("cam_right_wrist", f"{IB.ROBOT_PRIM}/Arm_{S}/link7", r7.T @ (cam_w - p7),
             IB.camera_rows(r7.T @ (mid + 0.1 * rg[:, 2] - cam_w), r7.T @ up), (320, 240), 87.0, 0.02)
for _ in range(30):
    world.step(render=True)
B.render_on_step = True
home = A.home_action()
cart0 = C.cartesian_from_joint_action(home)
DT = 1.0 / IA.PHYSICS_HZ


def apply(ex, cart, mode):
    j = ex.lower(cart)[0]
    if mode == "last2":                      # render on the last two physics substeps
        B.render_on_step = False
        B.set_joint_targets(A.action_to_commands(j)["joint_targets"])
        c = A.action_to_commands(j)
        B.set_base_velocity(c["base_forward_mps"], c["base_yaw_radps"])
        for i in range(IA.PHYSICS_STEPS_PER_ACTION):
            world.step(render=i >= IA.PHYSICS_STEPS_PER_ACTION - 2)
        B.render_on_step = True
    else:
        A.apply_action(j)
        for _ in range({"none": 0, "render1": 1, "render2": 2}[mode]):
            world.render()


def trial(mode, stimulus="arm"):
    """mode: none | render1 | render2 (extra world.render() calls after each action) | last2."""
    ex = C.CartesianExecutor.from_proprio(A.observe_proprio())
    for _ in range(15):
        apply(ex, cart0, "none")
    K = 10
    rows, prev_img, prev_prop = [], None, None
    for t in range(16):
        prop = A.observe_proprio()
        a = cart0.copy()
        if stimulus == "arm" and t >= K:
            a[C.CART_RIGHT_XYZ.start + 1] -= 0.10          # right hand 10 cm to the robot's right
        if stimulus == "block" and t == K:                  # static world change, arm still
            block.set_world_pose(position=np.array([0.95, -0.25, 0.95]))
        if mode == "read_after":                            # images read after action t
            apply(ex, a, "none")
        imgs = {k: B.render(k) for k in ("cam_front", "cam_right_wrist")}
        rt = {k: B.cameras[k].get_current_frame().get("rendering_time") for k in imgs}
        row = {"t": t, "sim_time": round(float(world.current_time), 4), "render_time": {k: round(float(v), 4) for k, v in rt.items()}}
        # 16x16 block means suppress the RTX denoiser's per-pixel noise
        cur = {k: imgs[k].astype(float).reshape(15, 16, 20, 16, 3).mean(axis=(1, 3)) for k in imgs}
        if prev_img is not None:
            row["wrist_img_change"] = round(float(np.abs(cur["cam_right_wrist"] - prev_img["cam_right_wrist"]).mean()), 3)
            row["head_img_change"] = round(float(np.abs(cur["cam_front"] - prev_img["cam_front"]).mean()), 3)
            row["joint_change"] = round(float(np.abs(prop[8:15] - prev_prop[8:15]).max()), 4)
        rows.append(row)
        prev_img, prev_prop = cur, prop
        if mode != "read_after":
            apply(ex, a, mode)
    noise = float(np.median([r["wrist_img_change"] for r in rows[2:K + 1]]))
    hnoise = float(np.median([r["head_img_change"] for r in rows[2:K + 1]]))
    first_head = next((r["t"] for r in rows if r.get("head_img_change", 0) > 3 * hnoise + 0.5), None)
    first_prop = next((r["t"] for r in rows if r.get("joint_change", 0) > 0.01), None)
    first_img = next((r["t"] for r in rows if r.get("wrist_img_change", 0) > 3 * noise + 0.5), None)
    block.set_world_pose(position=np.array([0.0, 0.0, -5.0]))
    res = {"mode": mode, "stimulus": stimulus, "stimulus_at_step": K, "first_proprio_change_at_step": first_prop,
           "first_wrist_image_change_at_step": first_img, "first_head_image_change_at_step": first_head,
           "wrist_image_noise_level": round(noise, 3),
           "render_time_offset_s": sorted({round(r["sim_time"] - v, 4) for r in rows for v in r["render_time"].values()}),
           "all_cameras_same_render_time": all(len(set(r["render_time"].values())) == 1 for r in rows),
           "rows": [{k: r[k] for k in ("t", "wrist_img_change", "head_img_change", "joint_change") if k in r} for r in rows[K - 1:K + 4]]}
    # the state changes in the observation of step K+1 (block placed / hand moved during action K);
    # synchronized = the images of that observation already show it
    res["synchronized"] = bool(first_img == K + 1 and (stimulus == "block" or first_prop == K + 1)
                               and (first_head in (None, K + 1)) and res["all_cameras_same_render_time"])
    # back home for the next trial
    for _ in range(15):
        apply(ex, cart0, "none")
    return res


out = {"block_read_before": trial("none", "block"), "block_read_after": trial("read_after", "block"),
       "arm_read_before": trial("none", "arm"), "arm_read_after": trial("read_after", "arm")}
print("###SYNC###" + json.dumps(out), flush=True)
app.close()
