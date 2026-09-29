"""Habitat articulated-agent class for AlphaBot 2 (dual 7-DoF arms, two EG2-4C2 grippers).

Loaded from ``alphabot2.urdf`` built by ``build_urdf.py`` from the pinned public
Ai2 Bot2 xacro. Policy-facing conventions are exactly the Isaac-qualified ones
(``lmm.embodiments.alphabot2``):

* 16-D proprio ``[left 7 vendor joints, left closed fraction, right 7, right closed fraction]``.
  The URDF joints ARE the xacro (vendor) joints, so no per-joint sign/offset map
  is needed here (in Isaac it was, because the USD differs).
* gripper: joint 0 rad = closed, 0.82 rad = open (measured in Isaac, identical
  pad separations in this URDF); closed fraction ``f = (0.82 - q) / 0.82``.
* base: root frame +x forward, +y up (the URDF carries the ROS->Habitat
  rotation in its fixed ``habitat_yup_joint``; see build_urdf.py), yaw about +y.

Every Habitat link/joint index is resolved from names after loading and
checked against the table measured on the live runtime; any change raises.

Habitat places a link's scene node at the link's INERTIAL frame (measured:
error 0.0 after composing the URDF inertial origin, 0.144 m on base_link
without). Link frames are therefore recovered through the URDF inertial
origins (:meth:`link_frame`).
"""
from __future__ import annotations

import math
from pathlib import Path

import magnum as mn
import numpy as np

from habitat.articulated_agents.mobile_manipulator import (
    ArticulatedAgentCameraParams,
    MobileManipulator,
    MobileManipulatorParams,
)

from .urdf_kin import URDFKin

SIDES = ("left", "right")
ARM_JOINTS = {s: tuple(f"{s}_joint{i}" for i in range(1, 8)) for s in SIDES}
GRIPPER_JOINT = {s: f"{s}_gripper_joint" for s in SIDES}
GRIPPER_MIMICS = {s: tuple(f"{s}_{n}" for n in ("left_inner_joint", "right_inner_joint", "right_outer_joint",
                                                 "left_pad_joint", "right_pad_joint")) for s in SIDES}
EE_LINK = {s: f"{s}_gripper_base" for s in SIDES}
GRIPPER_OPEN_RAD, GRIPPER_CLOSED_RAD = 0.82, 0.0
#: gripper_base -> pad midpoint along the approach axis (+z of gripper_base), measured in Isaac.
REACH_M = 0.1196
#: Vendor home posture (the xacro's home_2), identical to the Isaac adapter's.
HOME = {"left": (0.8333, 0.0, 0.0, -1.7800, 0.0, -0.6410, 0.0),
        "right": (-0.8333, 0.0, 0.0, -1.7800, 0.0, -0.6410, 0.0)}
#: joint_positions offsets measured on the live runtime (2026-09-29, load_probe.py).
EXPECTED_POS_OFFSETS = {**{f"left_joint{i}": i - 1 for i in range(1, 8)},
                        **{f"right_joint{i}": 6 + i for i in range(1, 8)},
                        "left_gripper_joint": 14, "right_gripper_joint": 20}
EXPECTED_MOVABLE = 26
BASE_LINK_NAMES = {"habitat_base", "base_footprint", "base_link", "driving_wheel_left_link", "driving_wheel_right_link",
                   "driven_fl_link_v", "driven_fl_link_h", "driven_fr_link_v", "driven_fr_link_h",
                   "driven_rl_link_v", "driven_rl_link_h", "driven_rr_link_v", "driven_rr_link_h"}
#: Camera specs (same physical sensors as in Isaac): parent link, hfov.
CAMERA_HFOV = {"head": 69.0, "left_wrist": 87.0, "right_wrist": 87.0}
#: optical frame (+z along the view, +y image-down) -> Habitat camera frame (-z view, +y image-up)
_OPT_TO_HAB = np.diag([1.0, -1.0, -1.0, 1.0])


def _m4(T: np.ndarray) -> mn.Matrix4:
    return mn.Matrix4([[float(T[r, c]) for r in range(4)] for c in range(4)])      # magnum is column-major


def _np4(M: mn.Matrix4) -> np.ndarray:
    return np.array([[M[c][r] for c in range(4)] for r in range(4)])


def wrist_optical_in_link8(kin: URDFKin, side: str) -> np.ndarray:
    """Wrist D405 optical frame in ``{side}_link8``, constructed exactly as in Isaac.

    Position: the d405 visual origin of the xacro's link8. Optical axis: towards
    the pad midpoint + 0.10 m along the approach (gripper open). Image-up: away
    from the gripper axis (fingers at the image bottom)."""
    T8 = kin.pose(f"{side}_link8")
    inv8 = np.linalg.inv(T8)
    q_open = {GRIPPER_JOINT[side]: GRIPPER_OPEN_RAD}
    cam = np.array([-0.13322, -0.0005, 0.044651])            # xacro: <visual origin> of d405.glb in link8
    gb = inv8 @ kin.pose(EE_LINK[side], q_open)
    pads = [(inv8 @ kin.pose(f"{side}_{p}", q_open))[:3, 3] for p in ("left_pad", "right_pad")]
    approach = gb[:3, 2]
    target = (pads[0] + pads[1]) / 2 + 0.10 * approach
    fwd = target - cam
    fwd /= np.linalg.norm(fwd)
    up = cam - (gb[:3, 3] + approach * ((cam - gb[:3, 3]) @ approach))
    up -= fwd * (fwd @ up)
    up /= np.linalg.norm(up)
    y = -up                                                    # optical +y = image down
    x = np.cross(y, fwd)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, fwd], axis=1)
    T[:3, 3] = cam
    return T


class AlphaBot2Robot(MobileManipulator):
    """AlphaBot 2 in Habitat: both arms and both grippers always present and controllable."""

    cls_uuid = "alphabot2_robot"

    def __init__(self, urdf_path, sim, limit_robo_joints=True, fixed_base=True):
        home = np.array(HOME["left"] + HOME["right"], dtype=np.float32)
        params = MobileManipulatorParams(
            arm_joints=[], gripper_joints=[], wheel_joints=None,          # resolved by name in reconfigure()
            arm_init_params=home,
            gripper_init_params=np.full(12, GRIPPER_OPEN_RAD, dtype=np.float32),
            ee_offset=[mn.Vector3.zero_init(), mn.Vector3.zero_init()],
            ee_links=[],
            ee_constraint=np.array([[[0.0, 1.0], [-1.0, 1.0], [0.3, 1.8]]] * 2, dtype=np.float32),
            cameras={},
            gripper_closed_state=np.full(12, GRIPPER_CLOSED_RAD, dtype=np.float32),
            gripper_open_state=np.full(12, GRIPPER_OPEN_RAD, dtype=np.float32),
            gripper_state_eps=0.01,
            arm_mtr_pos_gain=0.3, arm_mtr_vel_gain=0.3, arm_mtr_max_impulse=100.0,
            wheel_mtr_pos_gain=None, wheel_mtr_vel_gain=None, wheel_mtr_max_impulse=None,
            base_offset=mn.Vector3.zero_init(),
            base_link_names=set(BASE_LINK_NAMES),
            ee_count=2,
        )
        super().__init__(params, urdf_path, sim, limit_robo_joints, fixed_base, maintain_link_order=True)
        self.kin = URDFKin(urdf_path)
        self.mimic = {s: [(n, self.kin.joints[n].mimic[1], self.kin.joints[n].mimic[2]) for n in GRIPPER_MIMICS[s]]
                      for s in SIDES}
        for s in SIDES:
            for n, _, _ in self.mimic[s]:
                if self.kin.joints[n].mimic[0] != GRIPPER_JOINT[s]:
                    raise RuntimeError(f"{n} does not mimic {GRIPPER_JOINT[s]}")

    # ------------------------------------------------------------ name resolution
    def _resolve(self):
        ao = self.sim_obj
        by_joint = {ao.get_link_joint_name(i): i for i in ao.get_link_ids()}
        by_link = {ao.get_link_name(i): i for i in ao.get_link_ids()}
        movable = [n for n, i in by_joint.items() if ao.get_link_num_joint_pos(i) > 0]
        if len(movable) != EXPECTED_MOVABLE or len(ao.joint_positions) != EXPECTED_MOVABLE:
            raise RuntimeError(f"AlphaBot URDF has {len(movable)} movable joints / {len(ao.joint_positions)} "
                               f"positions, expected {EXPECTED_MOVABLE}")
        need = [*ARM_JOINTS["left"], *ARM_JOINTS["right"], *GRIPPER_JOINT.values(),
                *GRIPPER_MIMICS["left"], *GRIPPER_MIMICS["right"]]
        missing = [n for n in need if n not in by_joint]
        if missing:
            raise RuntimeError(f"AlphaBot URDF lacks joints {missing}")
        for n, off in EXPECTED_POS_OFFSETS.items():
            got = ao.get_link_joint_pos_offset(by_joint[n])
            if got != off:
                raise RuntimeError(f"joint {n} sits at joint_positions[{got}], measured {off}: the articulation "
                                   "was reordered; re-run scripts/alphabot2_habitat/load_probe.py")
        self.joint_link = by_joint
        self.link_id = by_link
        self.pos_offset = {n: ao.get_link_joint_pos_offset(i) for n, i in by_joint.items()
                           if ao.get_link_num_joint_pos(i) > 0}
        self.arm_offsets = {s: [self.pos_offset[n] for n in ARM_JOINTS[s]] for s in SIDES}
        self.gripper_offsets = {s: [self.pos_offset[GRIPPER_JOINT[s]]] + [self.pos_offset[n] for n in GRIPPER_MIMICS[s]]
                                for s in SIDES}
        self.lower = {n: self.kin.joints[n].lower for n in need}
        self.upper = {n: self.kin.joints[n].upper for n in need}

    def reconfigure(self):
        if self.sim_obj is None or not self.sim_obj.is_alive:
            ao_mgr = self._sim.get_articulated_object_manager()
            self.sim_obj = ao_mgr.add_articulated_object_from_urdf(self.urdf_path, fixed_base=self._fixed_base,
                                                                   maintain_link_order=self._maintain_link_order)
            if self.sim_obj is None:
                raise RuntimeError(f"failed to load AlphaBot URDF {self.urdf_path}")
        self._resolve()
        self.params.arm_joints = [self.joint_link[n] for n in (*ARM_JOINTS["left"], *ARM_JOINTS["right"])]
        self.params.gripper_joints = [self.joint_link[n] for s in SIDES for n in (GRIPPER_JOINT[s], *GRIPPER_MIMICS[s])]
        self.params.ee_links = [self.link_id[EE_LINK[s]] for s in SIDES]
        self.params.cameras = self._camera_params()
        # habitat-lab binds sensors to camera prefixes in __init__, when params.cameras is still
        # empty (it can only be resolved after loading) -- without this rebind no camera would
        # ever follow its link.
        from collections import defaultdict
        self._cameras = defaultdict(list)
        for prefix in self.params.cameras:
            for sensor_name in self._sim._sensors:
                if sensor_name.startswith(prefix):
                    self._cameras[prefix].append(sensor_name)
        super().reconfigure()

    # ------------------------------------------------------------ frames
    def link_frame(self, link: str) -> np.ndarray:
        """World pose (4x4, Habitat world) of a URDF link frame."""
        node = _np4(self.sim_obj.get_link_scene_node(self.link_id[link]).transformation)
        return node @ np.linalg.inv(self.kin.inertial(link))

    def base_frame(self) -> np.ndarray:
        """World pose of the ROS chassis ``base_link`` frame (x fwd, y left, z up)."""
        return self.link_frame("base_link")

    def _camera_params(self):
        cams = {}
        # head: optical frame = the xacro's head_camera link (+z optical axis, +y image down)
        T_node = np.linalg.inv(self.kin.inertial("head_camera"))
        cams["head"] = ArticulatedAgentCameraParams(
            attached_link_id=self.link_id["head_camera"], relative_transform=_m4(T_node @ _OPT_TO_HAB))
        for s in SIDES:
            T_node = np.linalg.inv(self.kin.inertial(f"{s}_link8")) @ wrist_optical_in_link8(self.kin, s)
            cams[f"{s}_wrist"] = ArticulatedAgentCameraParams(
                attached_link_id=self.link_id[f"{s}_link8"], relative_transform=_m4(T_node @ _OPT_TO_HAB))
        return cams

    def ee_transform(self, ee_index: int = 0) -> mn.Matrix4:
        """Grasp point: pad midpoint (gripper_base + REACH along its approach axis), world frame."""
        T = self.link_frame(EE_LINK[SIDES[ee_index]])
        G = np.eye(4)
        G[2, 3] = REACH_M
        return _m4(T @ G)

    # ------------------------------------------------------------ state / commands
    def arm_q(self, side: str) -> np.ndarray:
        jp = np.asarray(self.sim_obj.joint_positions, np.float64)
        return jp[self.arm_offsets[side]]

    def gripper_q(self, side: str) -> float:
        return float(np.asarray(self.sim_obj.joint_positions)[self.gripper_offsets[side][0]])

    @staticmethod
    def fraction_from_q(q: float) -> float:
        return float(np.clip((GRIPPER_OPEN_RAD - q) / (GRIPPER_OPEN_RAD - GRIPPER_CLOSED_RAD), 0.0, 1.0))

    @staticmethod
    def q_from_fraction(f: float) -> float:
        return GRIPPER_OPEN_RAD - float(np.clip(f, 0.0, 1.0)) * (GRIPPER_OPEN_RAD - GRIPPER_CLOSED_RAD)

    def proprio16(self) -> np.ndarray:
        """Canonical 16-D proprio: [left 7, left closed fraction, right 7, right closed fraction]."""
        return np.concatenate([self.arm_q("left"), [self.fraction_from_q(self.gripper_q("left"))],
                               self.arm_q("right"), [self.fraction_from_q(self.gripper_q("right"))]]).astype(np.float64)

    def set_arm_q(self, side: str, q) -> dict:
        """Kinematic arm command (vendor joints). Clamps to the xacro limits and reports it."""
        q = np.asarray(q, np.float64).reshape(7)
        if not np.isfinite(q).all():
            raise ValueError(f"non-finite {side} arm target")
        lo = np.array([self.lower[n] for n in ARM_JOINTS[side]])
        hi = np.array([self.upper[n] for n in ARM_JOINTS[side]])
        qc = np.clip(q, lo, hi)
        jp = np.asarray(self.sim_obj.joint_positions, np.float64)
        jp[self.arm_offsets[side]] = qc
        self.sim_obj.joint_positions = jp.tolist()
        for n, v in zip(ARM_JOINTS[side], qc):
            self._set_motor_pos(self.joint_link[n], float(v))
        return {"clamped": bool(np.any(np.abs(qc - q) > 1e-9)), "max_clamp_rad": float(np.max(np.abs(qc - q)))}

    def set_gripper_q(self, side: str, q: float):
        """Kinematic gripper command: the actuated joint and its five mimic joints (Bullet ignores <mimic>)."""
        q = float(np.clip(q, GRIPPER_CLOSED_RAD, GRIPPER_OPEN_RAD))
        jp = np.asarray(self.sim_obj.joint_positions, np.float64)
        jp[self.gripper_offsets[side][0]] = q
        for (n, mult, off), o in zip(self.mimic[side], self.gripper_offsets[side][1:]):
            jp[o] = q * mult + off
        self.sim_obj.joint_positions = jp.tolist()

    # habitat-lab helpers (snap/desnap) call these on the WHOLE robot; the AlphaBot
    # action re-asserts both hands after them, so they must never be relied upon.
    def open_gripper(self):
        pass

    def close_gripper(self):
        pass

    def set_gripper_target_state(self, gripper_state: float) -> None:
        pass

    def reset(self) -> None:
        super().reset()
        for s in SIDES:
            self.set_arm_q(s, HOME[s])
            self.set_gripper_q(s, GRIPPER_OPEN_RAD)
