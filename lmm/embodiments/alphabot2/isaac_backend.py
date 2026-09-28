"""Live Isaac Sim backend for :class:`AlphaBot2IsaacAdapter` (Isaac Sim 4.5).

Drives the public Ai2 Bot2 asset through the Isaac Python articulation API
(no ROS 2). Every Isaac/pxr import happens inside a function, so this module
byte-compiles and its pure helpers are testable on a machine without Isaac;
the generic LMM stack never imports it.

Launch requirements learned on the A800 host (see
``docs/ALPHABOT2_SIM_ASSET_PROVENANCE.md``):

* pass :data:`RTX_DRIVER_CHECK_BYPASS` on ``sys.argv`` before creating
  ``SimulationApp``: driver 535.261.03 has a minor version (261) that overflows
  the 8-bit Vulkan driverVersion field, reads as 535.5, and RTX refuses to
  start and then hangs;
* run headless, detached (``setsid nohup ... < /dev/null``) over the relay.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .isaac_adapter import (
    BODY_DOFS,
    HEAD_DOFS,
    LEFT_ARM_DOFS,
    LEFT_GRIPPER_DOF,
    RIGHT_ARM_DOFS,
    RIGHT_GRIPPER_DOF,
    WHEEL_DOFS,
    right_arm_limit_patch,
    twist_to_wheel_radps,
)

ROBOT_PRIM = "/World/Bot2"
BOT2_USD_REL = "humanoid/Ai2/Bot2/Bot2.usda"
GRIPPER_USD_REL = "grippers/Inspire/EG2_4C2/Inspire_EG2_4C2.usda"
SIDES = ("Left", "Right")
RTX_DRIVER_CHECK_BYPASS = "--/rtx/verifyDriverVersion/enabled=false"

#: DOFs that carry position drives the adapter commands. Mimic and caster
#: joints are deliberately absent: targets on them would fight the mimic
#: constraint or lock the casters.
CONTROLLED_DOFS: tuple[str, ...] = (
    *LEFT_ARM_DOFS, *RIGHT_ARM_DOFS, LEFT_GRIPPER_DOF, RIGHT_GRIPPER_DOF,
    *BODY_DOFS, *HEAD_DOFS,
)


# --------------------------------------------------------------------------
# pure helpers (no Isaac)
# --------------------------------------------------------------------------

def quat_xyzw_to_matrix(q: Any) -> np.ndarray:
    """PhysX link transforms carry quaternions as (x, y, z, w)."""
    x, y, z, w = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def camera_rows(forward: Any, up: Any) -> np.ndarray:
    """Rotation rows for a USD camera (looks along -Z, image-up +Y).

    Returns a 3x3 whose rows are the camera's X, Y, Z axes expressed in the
    parent frame -- the row-vector layout ``Gf.Matrix4d`` uses. ``forward`` is
    the optical axis, ``up`` the desired image-up (orthogonalised here).
    """
    f = np.asarray(forward, float); f = f / np.linalg.norm(f)
    u = np.asarray(up, float); u = u - f * (f @ u)
    n = np.linalg.norm(u)
    if n < 1e-6:
        raise ValueError("camera up vector is parallel to the optical axis")
    u = u / n
    z = -f
    x = np.cross(u, z)
    return np.stack([x, u, z])


# --------------------------------------------------------------------------
# stage construction
# --------------------------------------------------------------------------

def build_stage(app: Any, stage: Any, robot_usds_root: str) -> dict[str, Any]:
    """Load Ai2 Bot2 with two Inspire EG2-4C2 grippers at :data:`ROBOT_PRIM`.

    Applies, and reports, three fixes to the public asset:

    1. **Gripper re-attachment.** The arm's ``EE=EG2_4C2`` variant nests the
       gripper's rigid bodies under ``link7``; PhysX folds nested bodies into
       their ancestor, so the fingers never become DOFs. The EE variant is set
       to ``None`` and the gripper is referenced as a sibling
       (``Gripper_{Side}``) at the identical pose, welded to ``link7`` by a
       FixedJoint.
    2. **World anchor removal.** The standalone gripper ships ``root_joint``,
       a FixedJoint with an empty body0 -- a weld to the WORLD. Left in place it
       bolts the robot to the world through both hands (fixed_base=True, base
       cannot drive). Joints with an empty body are deactivated.
    3. **Right-arm joint limits.** The right arm reuses the left arm's joint
       definitions, so joints 3 and 5 ship left-arm ranges that, through the
       right-arm map, exclude the vendor home pose. They are widened to the
       mapped vendor range (:func:`isaac_adapter.right_arm_limit_patch`).
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    from isaacsim.core.utils.stage import add_reference_to_stage

    report: dict[str, Any] = {}
    bot2 = f"{robot_usds_root}/{BOT2_USD_REL}"
    grip = f"{robot_usds_root}/{GRIPPER_USD_REL}"
    add_reference_to_stage(usd_path=bot2, prim_path=ROBOT_PRIM)
    root = stage.GetPrimAtPath(ROBOT_PRIM)
    root.GetVariantSets().GetVariantSet("ROS").SetVariantSelection("none")
    for _ in range(3):
        app.update()

    xc = UsdGeom.XformCache()
    poses = {}
    for s in SIDES:
        g = f"{ROBOT_PRIM}/Arm_{s}/link7/link8/tcp/Inspire_EG2_4C2"
        poses[s] = {"root": xc.GetLocalToWorldTransform(stage.GetPrimAtPath(g)),
                    "base": xc.GetLocalToWorldTransform(stage.GetPrimAtPath(g + "/gripper_base"))}
    for s in SIDES:
        stage.GetPrimAtPath(f"{ROBOT_PRIM}/Arm_{s}").GetVariantSets().GetVariantSet("EE").SetVariantSelection("None")
    for _ in range(3):
        app.update()

    nested = {}
    for s in SIDES:
        t = stage.GetPrimAtPath(f"{ROBOT_PRIM}/Arm_{s}/link7/link8/tcp")
        nested[s] = [str(q.GetPath()) for q in Usd.PrimRange(t, Usd.TraverseInstanceProxies())
                     if q.HasAPI(UsdPhysics.RigidBodyAPI) or q.IsA(UsdPhysics.Joint)] if t else []
    report["nested_leftover_bodies_or_joints"] = nested

    anchored = []
    for s in SIDES:
        path = f"{ROBOT_PRIM}/Gripper_{s}"
        add_reference_to_stage(usd_path=grip, prim_path=path)
        gp = stage.GetPrimAtPath(path)
        gp.GetVariantSets().GetVariantSet("Physics").SetVariantSelection("physx")
        for q in Usd.PrimRange(gp):
            if q.HasAPI(UsdPhysics.ArticulationRootAPI):
                q.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        for q in Usd.PrimRange(gp):
            if q.IsA(UsdPhysics.Joint):
                j = UsdPhysics.Joint(q)
                if not j.GetBody0Rel().GetTargets() or not j.GetBody1Rel().GetTargets():
                    q.SetActive(False)
                    anchored.append(str(q.GetPath()))
        xf = UsdGeom.Xformable(gp)
        xf.ClearXformOpOrder()
        parent_w = xc.GetLocalToWorldTransform(root)
        xf.AddTransformOp().Set(poses[s]["root"] * parent_w.GetInverse())
    report["deactivated_world_anchors"] = anchored
    for _ in range(3):
        app.update()

    xc = UsdGeom.XformCache()
    err = {}
    for s in SIDES:
        link7 = stage.GetPrimAtPath(f"{ROBOT_PRIM}/Arm_{s}/link7")
        gbase = stage.GetPrimAtPath(f"{ROBOT_PRIM}/Gripper_{s}/gripper_base")
        rel = xc.GetLocalToWorldTransform(gbase) * xc.GetLocalToWorldTransform(link7).GetInverse()
        j = UsdPhysics.FixedJoint.Define(stage, f"{ROBOT_PRIM}/Gripper_{s}/attach_to_link7")
        j.CreateBody0Rel().SetTargets([link7.GetPath()])
        j.CreateBody1Rel().SetTargets([gbase.GetPath()])
        j.CreateLocalPos0Attr().Set(Gf.Vec3f(rel.ExtractTranslation()))
        j.CreateLocalRot0Attr().Set(Gf.Quatf(rel.ExtractRotationQuat()))
        j.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
        j.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
        err[s] = round(float((xc.GetLocalToWorldTransform(gbase).ExtractTranslation()
                              - poses[s]["base"].ExtractTranslation()).GetLength()), 6)
    report["gripper_base_reattach_err_m"] = err

    patched = {}
    for dof, (lo, hi) in right_arm_limit_patch().items():
        p = stage.GetPrimAtPath(f"{ROBOT_PRIM}/{dof}")
        if not p or p.IsInstanceProxy():
            raise RuntimeError(f"cannot patch limits on {dof}: prim missing or an instance proxy")
        j = UsdPhysics.RevoluteJoint(p)
        shipped = (float(j.GetLowerLimitAttr().Get()), float(j.GetUpperLimitAttr().Get()))
        j.GetLowerLimitAttr().Set(math.degrees(lo))
        j.GetUpperLimitAttr().Set(math.degrees(hi))
        patched[dof] = {"shipped_deg": [round(v, 4) for v in shipped],
                        "patched_deg": [round(math.degrees(lo), 4), round(math.degrees(hi), 4)]}
    report["right_arm_limit_patch"] = patched
    return report


def enable_contact_reports(stage: Any, root: str = "/World") -> int:
    """Apply PhysxContactReportAPI to every rigid body under ``root``.

    Must run before the simulation starts. Returns the number of bodies.
    """
    from pxr import PhysxSchema, Usd, UsdPhysics

    n = 0
    for p in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(p).CreateThresholdAttr().Set(0.0)
            n += 1
    return n


def contact_pairs() -> dict[str, int]:
    """Current contact pairs as ``"pathA <-> pathB" -> n_contacts``."""
    from omni.physx import get_physx_simulation_interface
    from pxr import PhysicsSchemaTools

    headers, _ = get_physx_simulation_interface().get_contact_report()
    out: dict[str, int] = {}
    for h in headers:
        a = str(PhysicsSchemaTools.intToSdfPath(h.actor0))
        b = str(PhysicsSchemaTools.intToSdfPath(h.actor1))
        key = " <-> ".join(sorted((a, b)))
        out[key] = out.get(key, 0) + int(h.num_contact_data)
    return out


def collider_owners(stage: Any, root: str = ROBOT_PRIM) -> dict[str, int]:
    """Rigid body -> number of collision prims it owns (instance proxies included)."""
    from pxr import Usd, UsdPhysics

    owners: dict[str, int] = {}
    for p in Usd.PrimRange(stage.GetPrimAtPath(root), Usd.TraverseInstanceProxies()):
        if not p.HasAPI(UsdPhysics.CollisionAPI):
            continue
        a = p
        while a and not a.HasAPI(UsdPhysics.RigidBodyAPI):
            a = a.GetParent()
        key = str(a.GetPath()) if a else "<no rigid body>"
        owners[key] = owners.get(key, 0) + 1
    return owners


# --------------------------------------------------------------------------
# live backend
# --------------------------------------------------------------------------

class IsaacArticulationBackend:
    """Implements :class:`isaac_adapter.IsaacBackend` on a live Isaac World.

    ``world`` is an ``isaacsim.core.api.World`` and ``art`` the robot's
    ``SingleArticulation``, both already reset. Call :meth:`refresh` after any
    ``world.reset()``: reset invalidates the PhysX view.
    """

    def __init__(self, world: Any, art: Any, *, render_on_step: bool = False) -> None:
        self.world = world
        self.art = art
        self.render_on_step = render_on_step
        self.cameras: dict[str, Any] = {}
        self.objects: dict[str, Any] = {}
        self.steps_taken = 0
        self.refresh()

    # -- plumbing ----------------------------------------------------------
    def _pv(self) -> Any:
        return self.art._articulation_view._physics_view

    @staticmethod
    def _np(x: Any) -> np.ndarray:
        return x.numpy() if hasattr(x, "numpy") else np.asarray(x)

    def refresh(self) -> None:
        pv = self._pv()
        pre = ROBOT_PRIM + "/"
        self.dof_paths = [p.replace(pre, "") for p in pv.dof_paths[0]]
        self.link_paths = [p.replace(pre, "") for p in pv.link_paths[0]]
        self.index = {p: i for i, p in enumerate(self.dof_paths)}
        # Contract names are exact paths or, for chassis joints that sit under
        # a payload prefix, a suffix that must match exactly one DOF.
        self.resolved: dict[str, str] = {}
        missing = []
        for d in (*CONTROLLED_DOFS, *WHEEL_DOFS):
            if d in self.index:
                self.resolved[d] = d
                continue
            hits = [p for p in self.dof_paths if p.endswith("/" + d)]
            if len(hits) == 1:
                self.index[d] = self.index[hits[0]]
                self.resolved[d] = hits[0]
            else:
                missing.append((d, hits))
        if missing:
            raise RuntimeError(f"asset lacks (or has ambiguous) DOFs the contract addresses: {missing}")
        chassis = [i for i, p in enumerate(self.link_paths)
                   if p.split("/")[-1] == "base_link" and not p.startswith(("Arm_", "Gripper_"))]
        if len(chassis) != 1:
            raise RuntimeError(f"chassis base_link ambiguous: {[self.link_paths[i] for i in chassis]}")
        self.chassis = chassis[0]
        self.ctrl_idx = np.array([self.index[d] for d in CONTROLLED_DOFS], dtype=np.int32)
        self.wheel_idx = np.array([self.index[d] for d in WHEEL_DOFS], dtype=np.int32)
        self.targets = self._np(self.art.get_joint_positions()).astype(np.float64)
        self.wheel_targets = np.zeros(2)

    def dof_limits(self) -> np.ndarray:
        return self._np(self._pv().get_dof_limits()).reshape(-1, 2)

    # -- IsaacBackend protocol --------------------------------------------
    def set_joint_targets(self, mapping: dict[str, float]) -> None:
        from isaacsim.core.utils.types import ArticulationAction

        for k, v in mapping.items():
            self.targets[self.index[k]] = float(v)
        self.art.apply_action(ArticulationAction(
            joint_positions=self.targets[self.ctrl_idx].astype(np.float32),
            joint_indices=self.ctrl_idx))

    def get_joint_positions(self) -> dict[str, float]:
        q = self._np(self.art.get_joint_positions())
        return {p: float(q[i]) for i, p in enumerate(self.dof_paths)}

    def get_joint_velocities(self) -> dict[str, float]:
        v = self._np(self.art.get_joint_velocities())
        return {p: float(v[i]) for i, p in enumerate(self.dof_paths)}

    def set_base_velocity(self, forward_mps: float, yaw_radps: float) -> None:
        self.wheel_targets = np.array(twist_to_wheel_radps(forward_mps, yaw_radps))
        self.art._articulation_view.set_joint_velocity_targets(
            self.wheel_targets.astype(np.float32)[None, :], joint_indices=self.wheel_idx)

    def step(self, n: int) -> None:
        """Advance ``n`` physics steps; with ``render_on_step`` only the last
        one renders, so every camera shows the state the action ended in."""
        n = int(n)
        for i in range(n):
            self.world.step(render=self.render_on_step and i == n - 1)
        self.steps_taken += n

    def render(self, camera: str) -> np.ndarray:
        rgba = np.asarray(self.cameras[camera].get_rgba())
        if rgba.ndim != 3 or rgba.shape[-1] < 3:
            raise RuntimeError(f"camera {camera!r} returned no frame (shape {rgba.shape})")
        return rgba[..., :3].astype(np.uint8)

    # -- extras used by the smoke -----------------------------------------
    def teleport(self, mapping: dict[str, float]) -> None:
        """Set DOF positions directly, zero all DOF velocities, match targets."""
        q = self._np(self.art.get_joint_positions()).astype(np.float64)
        for k, v in mapping.items():
            q[self.index[k]] = float(v)
        self.art.set_joint_positions(q)
        self.art.set_joint_velocities(np.zeros_like(q))
        self.set_joint_targets({k: q[self.index[k]] for k in CONTROLLED_DOFS})
        self.set_base_velocity(0.0, 0.0)

    def link_transforms(self) -> np.ndarray:
        """(n_links, 7) world poses: xyz + quaternion xyzw, straight from PhysX."""
        return self._np(self._pv().get_link_transforms()).reshape(-1, len(self.link_paths), 7)[0]

    def link_pose(self, path: str) -> tuple[np.ndarray, np.ndarray]:
        t = self.link_transforms()[self.link_paths.index(path)]
        return t[:3].copy(), quat_xyzw_to_matrix(t[3:7])

    def chassis_pose(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Chassis base_link: position, rotation, yaw (from its +x axis)."""
        t = self.link_transforms()[self.chassis]
        r = quat_xyzw_to_matrix(t[3:7])
        return t[:3].copy(), r, math.atan2(r[1, 0], r[0, 0])

    def in_chassis(self, p_world: np.ndarray, r_world: np.ndarray | None = None):
        pc, rc, _ = self.chassis_pose()
        p = rc.T @ (np.asarray(p_world) - pc)
        return p if r_world is None else (p, rc.T @ r_world)

    def add_camera(self, name: str, parent: str, translation: Any, rows: np.ndarray,
                   resolution: tuple[int, int], hfov_deg: float, near_m: float) -> Any:
        """Rigidly attach an RGB camera prim under ``parent`` (a link prim).

        ``resolution`` is (width, height). ``rows`` from :func:`camera_rows`.
        """
        import omni.usd
        from isaacsim.sensors.camera import Camera
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        path = f"{parent}/lmm_{name}"
        cam = Camera(prim_path=path, resolution=tuple(resolution))
        prim = stage.GetPrimAtPath(path)
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        r, t = np.asarray(rows, float), np.asarray(translation, float)
        xf.AddTransformOp().Set(Gf.Matrix4d(
            r[0, 0], r[0, 1], r[0, 2], 0.0,
            r[1, 0], r[1, 1], r[1, 2], 0.0,
            r[2, 0], r[2, 1], r[2, 2], 0.0,
            t[0], t[1], t[2], 1.0))
        g = UsdGeom.Camera(prim)
        ha = 20.955
        g.GetHorizontalApertureAttr().Set(ha)
        g.GetVerticalApertureAttr().Set(ha * resolution[1] / resolution[0])
        g.GetFocalLengthAttr().Set(ha / (2.0 * math.tan(math.radians(hfov_deg) / 2.0)))
        g.GetClippingRangeAttr().Set(Gf.Vec2f(near_m, 100.0))
        cam.initialize()
        self.cameras[name] = cam
        return cam

    # -- snapshot / restore -----------------------------------------------
    def register_object(self, name: str, rigid: Any) -> None:
        """Track a dynamic rigid object (pose + velocity) in snapshots."""
        self.objects[name] = rigid

    def snapshot(self) -> dict[str, Any]:
        pos, orient = self.art.get_world_pose()
        snap = {
            "root_pose": (self._np(pos).copy(), self._np(orient).copy()),
            "root_lin_vel": self._np(self.art.get_linear_velocity()).copy(),
            "root_ang_vel": self._np(self.art.get_angular_velocity()).copy(),
            "q": self._np(self.art.get_joint_positions()).astype(np.float64).copy(),
            "qd": self._np(self.art.get_joint_velocities()).astype(np.float64).copy(),
            "targets": self.targets.copy(),
            "wheel_targets": self.wheel_targets.copy(),
            "steps_taken": self.steps_taken,
            "sim_time_s": float(self.world.current_time),
            "objects": {},
        }
        for k, o in self.objects.items():
            p, r = o.get_world_pose()
            snap["objects"][k] = {"pose": (self._np(p).copy(), self._np(r).copy()),
                                  "lin_vel": self._np(o.get_linear_velocity()).copy(),
                                  "ang_vel": self._np(o.get_angular_velocity()).copy()}
        return snap

    def restore(self, snap: dict[str, Any]) -> None:
        """Write a snapshot back. PhysX solver caches are not restorable."""
        from isaacsim.core.utils.types import ArticulationAction

        self.art.set_world_pose(*snap["root_pose"])
        self.art.set_linear_velocity(snap["root_lin_vel"])
        self.art.set_angular_velocity(snap["root_ang_vel"])
        self.art.set_joint_positions(snap["q"])
        self.art.set_joint_velocities(snap["qd"])
        self.targets = snap["targets"].copy()
        self.art.apply_action(ArticulationAction(
            joint_positions=self.targets[self.ctrl_idx].astype(np.float32),
            joint_indices=self.ctrl_idx))
        self.wheel_targets = snap["wheel_targets"].copy()
        self.art._articulation_view.set_joint_velocity_targets(
            self.wheel_targets.astype(np.float32)[None, :], joint_indices=self.wheel_idx)
        for k, o in self.objects.items():
            s = snap["objects"][k]
            o.set_world_pose(*s["pose"])
            o.set_linear_velocity(s["lin_vel"])
            o.set_angular_velocity(s["ang_vel"])
        self.steps_taken = snap["steps_taken"]
