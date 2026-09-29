"""Stage-1 AlphaBot 2 Isaac world with ONE timestamped observation pipeline.

Every Stage-1 consumer goes through :class:`SimWorld`: the demonstration
recorder, replay validation, closed-loop policy inference and evaluation. None
of them read cameras or proprio directly. That makes the observation timing
contract (docs/ALPHABOT2_OBSERVATION_TIMING_CONTRACT.md) a property of this
file, not a convention each script must remember.

Timing model (all indices are counted from the last reset):

* ``physics_frame`` -- number of 200 Hz physics steps taken. One action is
  20 physics steps; only the last substep of an action renders.
* ``policy_step t`` -- the index of the action about to be decided.
  ``observe()`` returns the observation used to decide action ``t``;
  ``act()`` applies action ``t`` and increments ``t``.
* Each render is logged with the physics frame it was taken at. The camera
  annotators may deliver an older render than the newest one (measured:
  ``RENDER_LAG`` renders in the default Isaac app). ``observe()`` therefore
  returns images tagged with the physics frame of the render they actually
  come from, and pairs them with the proprio recorded at THAT frame (a
  ring buffer), never with a newer proprio.

Must be imported after ``SimulationApp`` exists (use :func:`launch`).
"""
from __future__ import annotations

import math
import os
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("LMM_REPO", os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

ISAAC_ROOT = os.environ.get("ISAAC_ROOT", "/nix/yanjun/zts/alphabot2/isaacsim-4.5")
USDS = os.environ.get("ROBOT_USDS", "/nix/yanjun/zts/alphabot2/deps/robot_usds")


def launch(zero_delay: bool | None = None, headless: bool = True):
    """Create the SimulationApp. ``zero_delay`` selects Isaac's
    ``isaacsim.exp.base.zero_delay.kit`` (renderer waits for completion).
    Default comes from ``$ALPHABOT2_ZERO_DELAY`` (unset = the canonical choice
    recorded in :data:`CANONICAL_ZERO_DELAY`)."""
    from lmm.embodiments.alphabot2 import isaac_backend as IB
    if IB.RTX_DRIVER_CHECK_BYPASS not in sys.argv:
        sys.argv.append(IB.RTX_DRIVER_CHECK_BYPASS)
    if zero_delay is None:
        env = os.environ.get("ALPHABOT2_ZERO_DELAY")
        zero_delay = CANONICAL_ZERO_DELAY if env is None else bool(int(env))
    from isaacsim import SimulationApp
    exp = f"{ISAAC_ROOT}/apps/isaacsim.exp.base.zero_delay.kit" if zero_delay else ""
    app = SimulationApp({"headless": headless}, experience=exp)
    return app, zero_delay


#: Chosen from the timing probe (docs/ALPHABOT2_OBSERVATION_TIMING_CONTRACT.md).
CANONICAL_ZERO_DELAY = True
#: Renders by which the camera annotators lag, for the app mode in use; the
#: probe measures it and SimWorld verifies it on every reset (see _verify_lag).
RENDER_LAG = {False: 1, True: 0}

#: Canonical fixed neck pose (head_joint1, head_joint2), radians. Chosen by the
#: bounded camera-posture sweep (docs/ALPHABOT2_CANONICAL_CAMERA_POSTURE.md).
CANONICAL_HEAD = (0.0, 0.20)       # 41.4 deg down; 0.18 rad inside the head_joint2 limit (0.384)
CANONICAL_BODY = (0.0, 0.0, 0.0, 0.0)

CAMS = ("cam_front", "cam_left_wrist", "cam_right_wrist", "cam_chest")
IMG_H, IMG_W = 240, 320
TABLE_TOP = 0.67
TABLE_SIZE = (0.50, 0.80)          # depth (along the table's x), width
BOX_INNER, BOX_WALL_H, BOX_T = 0.14, 0.07, 0.01
PARK = (-30.0, 0.0)                # where unused objects wait (on the ground, out of view)

#: Graspable objects: name -> (kind, size, colour, mass). size: cuboid xyz or (radius, height).
OBJECTS = {
    "red cube": ("cuboid", (0.05, 0.05, 0.05), (0.90, 0.15, 0.10), 0.05),
    "green cylinder": ("cylinder", (0.024, 0.07), (0.15, 0.75, 0.25), 0.05),
    "yellow block": ("cuboid", (0.04, 0.04, 0.08), (0.95, 0.85, 0.15), 0.05),
    "purple bar": ("cuboid", (0.09, 0.04, 0.04), (0.55, 0.25, 0.75), 0.05),
}


def object_height(name: str) -> float:
    kind, size, _, _ = OBJECTS[name]
    return float(size[2] if kind == "cuboid" else size[1])


@dataclass
class Obs:
    """One policy observation with its timing record."""
    images: dict
    proprio: Any                      # canonical 16-D, read at ``proprio_frame``
    policy_step: int                  # the action this observation decides
    image_frame: int                  # physics frame of the render the images show
    proprio_frame: int                # physics frame the proprio was read at
    image_time: float
    proprio_time: float
    obs_frame: int                    # physics frame when observe() was called
    isaac_render_time: float | None = None   # Isaac's own (unreliable) metadata

    def timing(self) -> dict:
        return {"policy_step": self.policy_step, "image_frame": self.image_frame,
                "proprio_frame": self.proprio_frame, "obs_frame": self.obs_frame,
                "image_time": self.image_time, "proprio_time": self.proprio_time,
                "isaac_render_time": self.isaac_render_time}


class SimWorld:
    """Robot + Stage-1 scene + cameras + the canonical observation pipeline."""

    def __init__(self, app, zero_delay: bool, head=CANONICAL_HEAD, with_scene: bool = True,
                 cams=CAMS, extra_camera_annotators: tuple = ()):
        import numpy as np
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from pxr import UsdGeom, UsdLux
        from lmm.embodiments.alphabot2 import isaac_adapter as IA
        from lmm.embodiments.alphabot2 import isaac_backend as IB
        from lmm.embodiments.alphabot2.contract import AlphaBot2Contract

        self.np, self.IA, self.IB = np, IA, IB
        self.app, self.zero_delay = app, zero_delay
        self.render_lag = RENDER_LAG[zero_delay]
        self.head = tuple(float(v) for v in head)
        self.world = World(stage_units_in_meters=1.0, physics_dt=1 / IA.PHYSICS_HZ, rendering_dt=1 / IA.PHYSICS_HZ)
        self.stage = omni.usd.get_context().get_stage()
        self.build_report = IB.build_stage(app, self.stage, USDS)
        self.world.scene.add_default_ground_plane()
        sun = UsdLux.DistantLight.Define(self.stage, "/World/Sun")
        sun.CreateIntensityAttr(3000.0)
        sun.CreateAngleAttr(1.0)
        UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set((-35.0, 20.0, 0.0))
        self.objects: dict[str, Any] = {}
        self.fixed: dict[str, Any] = {}
        if with_scene:
            self._build_scene()
        IB.enable_contact_reports(self.stage)
        self.art = SingleArticulation(prim_path=IB.ROBOT_PRIM, name="bot2")
        self.world.scene.add(self.art)
        self.world.reset()
        self.B = IB.IsaacArticulationBackend(self.world, self.art)
        self.A = IA.AlphaBot2IsaacAdapter(AlphaBot2Contract(), backend=self.B)
        for k, o in self.objects.items():
            self.B.register_object(k, o)
        self.cams = tuple(cams)
        self._add_cameras(extra_camera_annotators)
        for _ in range(30):
            self.world.step(render=True)
        # chassis -> articulation-root offset (constant with the torso fixed)
        self.world.reset()
        self.B.refresh()
        root_p, root_q = self.art.get_world_pose()
        cp0, cr0, _ = self.B.chassis_pose()
        self.chassis_z = float(cp0[2])
        tc = np.eye(4); tc[:3, :3], tc[:3, 3] = cr0, cp0
        tr = np.eye(4); tr[:3, :3] = _quat_wxyz_to_R(np.asarray(root_q)); tr[:3, 3] = np.asarray(root_p)
        self.T_CH_ROOT = np.linalg.inv(tc) @ tr
        self.policy_step = 0
        self.frame0 = 0
        self.render_log: deque = deque(maxlen=8)     # (physics_frame, sim_time) per render
        self.state_log: deque = deque(maxlen=8)      # (physics_frame, sim_time, proprio) at each render
        self.lag_checks: list = []

    # ------------------------------------------------------------------ scene
    def _build_scene(self):
        np = self.np
        from isaacsim.core.api.materials import PhysicsMaterial
        from isaacsim.core.api.objects import DynamicCuboid, DynamicCylinder, FixedCuboid

        for name in ("table_a", "table_b"):
            self.fixed[name] = FixedCuboid(prim_path=f"/World/{name}", name=name,
                                           position=np.array([0.0, 20.0 + len(self.fixed), TABLE_TOP / 2]),
                                           scale=np.array([*TABLE_SIZE, TABLE_TOP]), color=np.array([0.62, 0.52, 0.40]))
        self.box_parts = {}
        for part in ("floor", "wx0", "wx1", "wy0", "wy1"):
            self.box_parts[part] = FixedCuboid(prim_path=f"/World/box_{part}", name=f"box_{part}",
                                               position=np.array([5.0, 25.0, -1.0]), scale=np.array([0.1, 0.1, 0.1]),
                                               color=np.array([0.25, 0.35, 0.75]))
        mat = PhysicsMaterial(prim_path="/World/mat_obj", static_friction=1.2, dynamic_friction=1.0, restitution=0.0)
        for i, (name, (kind, size, col, mass)) in enumerate(OBJECTS.items()):
            path = "/World/obj_" + name.replace(" ", "_")
            park = np.array([PARK[0] - 1.5 * i, PARK[1], 0.2])
            if kind == "cuboid":
                o = DynamicCuboid(prim_path=path, name=name.replace(" ", "_"), position=park, scale=np.array(size),
                                  mass=mass, color=np.array(col), physics_material=mat)
            else:
                o = DynamicCylinder(prim_path=path, name=name.replace(" ", "_"), position=park, radius=size[0],
                                    height=size[1], mass=mass, color=np.array(col), physics_material=mat)
            self.world.scene.add(o)
            self.objects[name] = o
        self.obj_paths = {k: "/World/obj_" + k.replace(" ", "_") for k in OBJECTS}

    def place_table(self, name, x, y, yaw):
        np = self.np
        self.fixed[name].set_world_pose(position=np.array([x, y, TABLE_TOP / 2]), orientation=_yaw_quat_wxyz(yaw))

    def place_box(self, x, y, yaw, z=TABLE_TOP):
        np = self.np
        bw, bh, th = BOX_INNER, BOX_WALL_H, BOX_T
        spec = {"floor": ((0, 0, th / 2), (bw + 2 * th, bw + 2 * th, th)),
                "wx0": ((-(bw + th) / 2, 0, bh / 2), (th, bw + 2 * th, bh)),
                "wx1": (((bw + th) / 2, 0, bh / 2), (th, bw + 2 * th, bh)),
                "wy0": ((0, -(bw + th) / 2, bh / 2), (bw, th, bh)),
                "wy1": ((0, (bw + th) / 2, bh / 2), (bw, th, bh))}
        c, s = math.cos(yaw), math.sin(yaw)
        for part, (off, sc) in spec.items():
            p = np.array([x + c * off[0] - s * off[1], y + s * off[0] + c * off[1], z + off[2]])
            prim = self.box_parts[part]
            prim.set_world_pose(position=p, orientation=_yaw_quat_wxyz(yaw))
            prim.set_local_scale(np.array(sc))
        self.box_pose = (float(x), float(y), float(yaw), float(z))

    def hide_box(self):
        for i, prim in enumerate(self.box_parts.values()):
            prim.set_world_pose(position=self.np.array([5.0 + i, 25.0, -1.0]))
        self.box_pose = None

    def place_object(self, name, x, y, yaw, z=TABLE_TOP):
        np = self.np
        for i, (k, o) in enumerate(self.objects.items()):
            if k == name:
                o.set_world_pose(position=np.array([x, y, z + object_height(k) / 2 + 0.001]), orientation=_yaw_quat_wxyz(yaw))
            else:
                o.set_world_pose(position=np.array([PARK[0] - 1.5 * i, PARK[1], object_height(k) / 2 + 0.001]),
                                 orientation=np.array([1.0, 0, 0, 0]))
            o.set_linear_velocity(np.zeros(3))
            o.set_angular_velocity(np.zeros(3))
        self.active_object = name

    # ---------------------------------------------------------------- cameras
    def _add_cameras(self, extra):
        np, IB, B, stage = self.np, self.IB, None, self.stage
        from pxr import Usd, UsdGeom
        B = self.B
        xc = UsdGeom.XformCache()
        heads = {"cam_front": f"{IB.ROBOT_PRIM}/head_link2/head_camera",
                 "cam_chest": f"{IB.ROBOT_PRIM}/body_link4/front_camera"}
        for name in self.cams:
            if name in heads:
                B.add_camera(name, heads[name], (0, 0, 0), IB.camera_rows((0, 0, 1), (0, -1, 0)),
                             (IMG_W, IMG_H), hfov_deg=69.0, near_m=0.10)
                continue
            side = "left" if "left" in name else "right"
            S = side.capitalize()
            d405 = [p for p in Usd.PrimRange(stage.GetPrimAtPath(f"{IB.ROBOT_PRIM}/Arm_{S}"), Usd.TraverseInstanceProxies())
                    if p.GetName() == "d405"][0]
            link7 = f"Arm_{S}/link7"
            p7, r7 = B.link_pose(link7)
            rel = np.array(xc.GetLocalToWorldTransform(d405)
                           * xc.GetLocalToWorldTransform(stage.GetPrimAtPath(f"{IB.ROBOT_PRIM}/{link7}")).GetInverse()).T
            cam_w = p7 + r7 @ rel[:3, 3]
            pg, rg = B.link_pose(f"Gripper_{S}/gripper_base")
            a_, b_ = B.link_pose(f"Gripper_{S}/left_pad")[0], B.link_pose(f"Gripper_{S}/right_pad")[0]
            target = (a_ + b_) / 2 + 0.10 * rg[:, 2]
            up = cam_w - (pg + rg[:, 2] * ((cam_w - pg) @ rg[:, 2]))
            B.add_camera(name, f"{IB.ROBOT_PRIM}/{link7}", r7.T @ (cam_w - p7),
                         IB.camera_rows(r7.T @ (target - cam_w), r7.T @ up), (IMG_W, IMG_H), hfov_deg=87.0, near_m=0.02)
        for name in self.cams:
            if "depth" in extra:
                B.cameras[name].add_distance_to_image_plane_to_frame()

    def camera_world_pose(self, name):
        """(position, R) of the camera in the world; R columns = image-right, image-up, -optical axis."""
        from pxr import UsdGeom
        m = self.np.array(UsdGeom.XformCache().GetLocalToWorldTransform(self.B.cameras[name].prim)).T
        return m[:3, 3].copy(), m[:3, :3].copy()

    # ------------------------------------------------------------- stepping
    @property
    def physics_frame(self) -> int:
        return self.B.steps_taken - self.frame0

    def sim_time(self) -> float:
        return float(self.world.current_time)

    def _log_render(self):
        """Called right after a physics step that rendered: record frame + state."""
        f, t = self.physics_frame, self.sim_time()
        self.render_log.append((f, t))
        self.state_log.append((f, t, self.A.observe_proprio()))

    def set_render(self, on: bool):
        """Rendering on/off for subsequent actions (off only for image-free replay)."""
        self.render_enabled = bool(on)

    def apply_joint_action(self, joint18):
        """One policy step: 20 physics substeps, render on the last (via the backend)."""
        self.B.render_on_step = getattr(self, "render_enabled", True)
        self.A.apply_action(joint18)
        self._log_render()
        self.policy_step += 1

    def observe(self, need_images: bool = True) -> Obs:
        """The canonical observation for deciding action ``policy_step``."""
        imgs = {k: self.B.render(k) for k in self.cams} if need_images else {}
        obs_frame = self.physics_frame
        if not self.render_log:
            raise RuntimeError("observe() before any render; call reset() first")
        # the images come from the render RENDER_LAG renders before the newest one
        f_img, t_img, prop = self.state_log[-1 - self.render_lag]
        rt = None
        try:
            rt = float(self.B.cameras[self.cams[0]].get_current_frame().get("rendering_time"))
        except Exception:
            pass
        return Obs(images=imgs, proprio=prop.copy(), policy_step=self.policy_step, image_frame=f_img,
                   proprio_frame=f_img, image_time=t_img, proprio_time=t_img, obs_frame=obs_frame,
                   isaac_render_time=rt)

    # --------------------------------------------------------------- reset
    def reset_robot(self, base_xy, base_yaw, arm_left=None, arm_right=None, settle_actions: int = 8, scene=None):
        """World reset, scene placement (``scene()`` runs after the world reset,
        which returns scene objects to their spawn poses), base pose, home arms,
        canonical head/torso, then settle.

        Settling uses real actions (rendering each) so the render/state ring
        buffers are full and the images are current before policy step 0."""
        np, IA, A, B = self.np, self.IA, self.A, self.B
        from lmm.embodiments.alphabot2 import cartesian as C
        self.world.reset()
        B.refresh()
        if scene is not None:
            scene()
        tc = np.eye(4)
        tc[:3, :3] = _yaw_R(base_yaw)
        tc[:3, 3] = [base_xy[0], base_xy[1], self.chassis_z]
        tr = tc @ self.T_CH_ROOT
        rq = _R_to_quat_wxyz(tr[:3, :3])
        self.art.set_world_pose(position=tr[:3, 3], orientation=rq)
        ql = np.array(IA.HOME_LEFT_ARM if arm_left is None else arm_left, float)
        qr = np.array(IA.HOME_RIGHT_ARM if arm_right is None else arm_right, float)
        m = {}
        for d, v in zip(IA.LEFT_ARM_DOFS, A.arm_vendor_to_sim("left", ql)):
            m[d] = float(v)
        for d, v in zip(IA.RIGHT_ARM_DOFS, A.arm_vendor_to_sim("right", qr)):
            m[d] = float(v)
        m[IA.LEFT_GRIPPER_DOF] = m[IA.RIGHT_GRIPPER_DOF] = A.gripper_fraction_to_joint(0.0)
        for d, v in zip(IA.BODY_DOFS, CANONICAL_BODY):
            m[d] = float(v)
        for d, v in zip(IA.HEAD_DOFS, self.head):
            m[d] = float(v)
        B.teleport(m)
        self.executor = C.CartesianExecutor(q={"left": ql.copy(), "right": qr.copy()})
        hold = np.zeros(18)
        hold[0:7], hold[8:15] = ql, qr
        self.render_log.clear()
        self.state_log.clear()
        self.frame0 = B.steps_taken
        self.policy_step = 0
        for _ in range(max(settle_actions, self.render_lag + 2)):
            self.apply_joint_action(hold)
        self.last_sync_check = self._verify_sync(hold)
        # settle actions are not part of the episode
        self.policy_step = 0
        self.home_cart = C.cartesian_from_joint_action(hold)
        return self.home_cart.copy()

    def _verify_sync(self, hold):
        """Live per-reset synchronization check (cheap, 3 actions, not recorded).

        A visual-only magenta marker is put 0.40 m in front of the head camera during one
        action and removed in the next. With the configured RENDER_LAG the
        observation after the marker action must show it exactly when the
        pipeline says the image frame is the marker frame -- i.e. the images
        and the paired proprio come from the same physics state."""
        np = self.np
        if not hasattr(self, "_marker"):
            from isaacsim.core.api.objects import VisualCuboid      # no collider: cannot disturb the robot
            self._marker = VisualCuboid(prim_path="/World/sync_marker", name="sync_marker",
                                        position=np.array([0.0, 40.0, -2.0]), scale=np.array([0.20, 0.20, 0.20]),
                                        color=np.array([1.0, 0.0, 1.0]))
        park = np.array([0.0, 40.0, -2.0])
        p, r = self.camera_world_pose("cam_front")
        seen, frames = [], []
        for present in (True, False, False):
            self._marker.set_world_pose(position=(p - 0.40 * r[:, 2]) if present else park)
            self.apply_joint_action(hold)
            ob = self.observe()
            im = ob.images["cam_front"]
            m = (im[..., 0] > 150) & (im[..., 2] > 150) & (im[..., 1] < 90)
            seen.append(bool(m.mean() > 0.03))
            frames.append((ob.image_frame, ob.proprio_frame, ob.obs_frame))
        # marker present only in the state of the first check action
        want = [True, False, False] if self.render_lag == 0 else [False, True, False]
        ok = seen == want and all(a == b for a, b, _ in frames)
        res = {"marker_seen": seen, "expected": want, "frames": frames, "ok": ok, "render_lag": self.render_lag}
        if not ok:
            raise RuntimeError(f"observation synchronization check failed: {res}")
        return res

    def act(self, cart18):
        """Lower a CartesianMobileAction18 through the qualified executor and apply it."""
        joint, rep = self.executor.lower(cart18)
        self.apply_joint_action(joint)
        return joint, rep

    # ------------------------------------------------------------ helpers
    def chassis_pose(self):
        p, _, yaw = self.B.chassis_pose()
        return p, yaw

    def object_pose(self, name=None):
        p, q = self.objects[name or self.active_object].get_world_pose()
        return self.np.asarray(p, float), self.np.asarray(q, float)

    def contacts(self):
        return self.IB.contact_pairs()


def _yaw_R(y):
    import numpy as np
    return np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])


def _yaw_quat_wxyz(y):
    import numpy as np
    return np.array([math.cos(y / 2), 0.0, 0.0, math.sin(y / 2)])


def _quat_wxyz_to_R(q):
    from lmm.embodiments.alphabot2 import pi05_contract as P
    w, x, y, z = q
    return P.quat_xyzw_to_matrix((x, y, z, w))


def _R_to_quat_wxyz(r):
    import numpy as np
    from lmm.embodiments.alphabot2 import pi05_contract as P
    x, y, z, w = P.matrix_to_quat_xyzw(r)
    return np.array([w, x, y, z])
