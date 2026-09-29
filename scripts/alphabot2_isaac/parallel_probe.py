"""Bounded feasibility probe: N AlphaBot 2 robots in one Isaac scene, CPU vs GPU PhysX.

One configuration per process: PROBE_MODE=cpu|gpu, PROBE_N=<robots>. Each robot
holds the vendor home pose with its wheels turning; 400 physics steps at 200 Hz
are timed without rendering. Prints one JSON line.
"""
import json, os, sys, time, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("LMM_REPO", os.path.dirname(os.path.dirname(HERE)))
USDS = os.environ.get("ROBOT_USDS", "/nix/yanjun/zts/alphabot2/deps/robot_usds")
MODE = os.environ.get("PROBE_MODE", "cpu")
N = int(os.environ.get("PROBE_N", "1"))
sys.path.insert(0, REPO)
from lmm.embodiments.alphabot2 import isaac_adapter as IA  # noqa: E402
from lmm.embodiments.alphabot2 import isaac_backend as IB  # noqa: E402
sys.argv.append(IB.RTX_DRIVER_CHECK_BYPASS)
from isaacsim import SimulationApp  # noqa: E402
app = SimulationApp({"headless": True})
out = {"mode": MODE, "n_robots": N}
try:
    import numpy as np
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.prims import Articulation
    from pxr import UsdGeom, Gf
    kw = dict(stage_units_in_meters=1.0, physics_dt=1 / IA.PHYSICS_HZ, rendering_dt=1 / IA.PHYSICS_HZ)
    if MODE == "gpu":
        kw.update(device="cuda:0", backend="torch")
    world = World(**kw)
    stage = omni.usd.get_context().get_stage()
    t = time.time()
    for i in range(N):
        env = f"/World/Env_{i}"
        UsdGeom.Xform.Define(stage, env).AddTranslateOp().Set(Gf.Vec3d(3.0 * i, 0, 0))
        IB.build_stage(app, stage, USDS, robot_prim=f"{env}/Bot2")
    out["build_s"] = round(time.time() - t, 1)
    world.scene.add_default_ground_plane()
    view = Articulation(prim_paths_expr="/World/Env_.*/Bot2", name="bots")
    world.scene.add(view)
    world.reset()
    pv = view._physics_view
    paths = [p.split("/Bot2/", 1)[1] for p in pv.dof_paths[0]]
    idx = {p: i for i, p in enumerate(paths)}
    for d in IA.WHEEL_DOFS:
        if d not in idx:
            idx[d] = [i for i, p in enumerate(paths) if p.endswith("/" + d)][0]
    home = np.zeros(len(paths))
    for side, dofs, hq in (("left", IA.LEFT_ARM_DOFS, IA.HOME_LEFT_ARM), ("right", IA.RIGHT_ARM_DOFS, IA.HOME_RIGHT_ARM)):
        m = IA.LEFT_ARM_MAP if side == "left" else IA.RIGHT_ARM_MAP
        for d, v in zip(dofs, m.to_sim(hq)):
            home[idx[d]] = v
    home[idx[IA.LEFT_GRIPPER_DOF]] = home[idx[IA.RIGHT_GRIPPER_DOF]] = 0.82
    q = np.tile(home, (N, 1)).astype(np.float32)
    wl, wr = IA.twist_to_wheel_radps(0.2, 0.3)
    vel = np.zeros_like(q)
    vel[:, idx[IA.WHEEL_DOFS[0]]], vel[:, idx[IA.WHEEL_DOFS[1]]] = wl, wr
    if MODE == "gpu":
        import torch
        q, vel = torch.tensor(q, device="cuda:0"), torch.tensor(vel, device="cuda:0")
    view.set_joint_positions(q)
    view.set_joint_position_targets(q)
    view.set_joint_velocity_targets(vel)
    for _ in range(20):
        world.step(render=False)
    t = time.perf_counter()
    steps = 400
    for _ in range(steps):
        world.step(render=False)
    wall = time.perf_counter() - t
    pos = view.get_world_poses()[0]
    pos = pos.cpu().numpy() if hasattr(pos, "cpu") else np.asarray(pos)
    out.update({"steps": steps, "wall_s": round(wall, 3), "physics_steps_per_s": round(steps / wall, 1),
                "robot_steps_per_s": round(N * steps / wall, 1),
                "realtime_factor_per_env": round((steps / IA.PHYSICS_HZ) / wall, 3),
                "aggregate_realtime_factor": round(N * (steps / IA.PHYSICS_HZ) / wall, 3),
                "robots_moved": int(np.sum(np.linalg.norm(pos[:, :2] - np.array([[3.0 * i, 0] for i in range(N)]), axis=1) > 0.05)),
                "num_dof": len(paths)})
except Exception as e:
    out["error"] = repr(e)
    out["tb"] = traceback.format_exc()[-1500:]
print("###PROBE###" + json.dumps(out), flush=True)
app.close()
