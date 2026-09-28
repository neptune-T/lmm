"""AlphaBot 2 Isaac Sim adapter — live-qualified on the public Ai2 Bot2 asset.

Architecture::

    LMM / pi0.5
        -> AlphaBot2 policy contract   (16-D proprio / 18-D action, physical units,
                                        vendor joint convention)
        -> AlphaBot2 Isaac adapter     (this module: naming, joint maps, gripper
                                        polarity, diff-drive, atomic stepping)
        -> Isaac backend               (isaac_backend.py: Isaac Python
                                        articulation API; imported lazily)

The generic LMM stack never imports Isaac. This module is numpy-only; the live
backend lives in :mod:`isaac_backend` and imports Isaac inside its functions.

What "vendor convention" means
------------------------------
Policy proprio/actions carry arm joints in the convention of the public Ai2
Bot2 xacro (``fiveages-sim/robot_descriptions``, the only public kinematic
description of this robot). The USD asset uses a *different* zero and sign per
joint, and — measured, not assumed — a different one per side:

* LEFT:  ``q_usd = q_vendor + (0, -pi/2, -pi/2, 0, +pi/2, 0, 0)``
* RIGHT: ``q_usd = (+,+,-,+,-,-,+) * q_vendor + (0, -pi/2, +pi/2, 0, -pi/2, 0, 0)``

The right arm differs because the USD builds it from the LEFT arm's joint
definitions (only the names change; ``Side/right.usda`` mirrors nothing but the
link1 visual), while the xacro right arm is a true mirror. Both maps were
identified from measured screw axes and then checked by forward kinematics
against the xacro on random poses (<= 7 mm, <= 0.8 deg including a constant
~4 mm frame offset). See ``docs/ALPHABOT2_SIM_KINEMATIC_AUDIT.md``.

Whether the xacro convention equals the real AlphaBot 2 SDK convention is NOT
established: the xacro is public third-party, not vendor-published.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import numpy as np

from .contract import (
    ACTION_DIM,
    BASE_FORWARD_INDEX,
    BASE_YAW_INDEX,
    LEFT_ARM_INDICES,
    LEFT_GRIPPER_INDEX,
    PROPRIO_DIM,
    RIGHT_ARM_INDICES,
    RIGHT_GRIPPER_INDEX,
    AlphaBot2Contract,
    ContractError,
)

_H = math.pi / 2

# --------------------------------------------------------------------------
# Asset naming — DOF paths relative to the articulation root, as reported by
# the PhysX articulation view. Paths, not names: the two grippers share joint
# names, and the PhysX DOF order is not the policy order.
# --------------------------------------------------------------------------

LEFT_ARM_DOFS: tuple[str, ...] = tuple(f"Arm_Left/joints/left_joint{i}" for i in range(1, 8))
RIGHT_ARM_DOFS: tuple[str, ...] = tuple(f"Arm_Right/joints/right_joint{i}" for i in range(1, 8))

#: The single actuated joint of each Inspire EG2-4C2 gripper. The gripper is
#: re-attached as a sibling of the arm (see isaac_backend.build_stage), so its
#: DOFs live under ``Gripper_{Side}``.
LEFT_GRIPPER_DOF = "Gripper_Left/joints/gripper_joint"
RIGHT_GRIPPER_DOF = "Gripper_Right/joints/gripper_joint"

#: Differential-drive wheels. Driven only through (v, omega); never exposed to
#: the policy.
WHEEL_DOFS: tuple[str, str] = ("joints/driving_left_joint", "joints/driving_right_joint")

#: Held fixed in v1. Present in the asset, outside the learned action space.
BODY_DOFS: tuple[str, ...] = tuple(f"joints/body_joint{i}" for i in range(1, 5))
HEAD_DOFS: tuple[str, ...] = ("joints/head_joint1", "joints/head_joint2")

#: Measured DOF census of the assembled robot (live, 2026-09-28):
#: 14 arm + 2 actuated gripper + 10 gripper mimic + 4 body + 2 head
#: + 2 drive wheels + 8 caster joints.
EXPECTED_DOF_COUNT = 42
GRIPPER_MIMIC_DOFS_PER_SIDE = 5

# --------------------------------------------------------------------------
# Arm joint maps (vendor/xacro convention <-> USD)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmJointMap:
    """``q_usd = sign * q_vendor + offset``, per joint. Signs are +-1."""

    sign: tuple[float, ...]
    offset: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.sign) != 7 or len(self.offset) != 7:
            raise ContractError("an arm map needs 7 signs and 7 offsets")
        if any(s not in (1.0, -1.0) for s in self.sign):
            raise ContractError(f"signs must be +-1, got {self.sign}")

    def to_sim(self, q_vendor: Sequence[float]) -> np.ndarray:
        return np.asarray(self.sign) * np.asarray(q_vendor, dtype=np.float64) + np.asarray(self.offset)

    def to_vendor(self, q_sim: Sequence[float]) -> np.ndarray:
        # sign is +-1, so it is its own inverse
        return np.asarray(self.sign) * (np.asarray(q_sim, dtype=np.float64) - np.asarray(self.offset))

    def sim_limits(self, vendor_limits: Sequence[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
        out = []
        for s, o, (lo, hi) in zip(self.sign, self.offset, vendor_limits):
            a, b = sorted((s * lo + o, s * hi + o))
            out.append((a, b))
        return tuple(out)


LEFT_ARM_MAP = ArmJointMap(sign=(1.0,) * 7, offset=(0.0, -_H, -_H, 0.0, _H, 0.0, 0.0))
RIGHT_ARM_MAP = ArmJointMap(sign=(1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0),
                            offset=(0.0, -_H, _H, 0.0, -_H, 0.0, 0.0))

#: Joint limits of the xacro, identical for both sides, vendor convention.
VENDOR_ARM_LIMITS: tuple[tuple[float, float], ...] = (
    (-3.1, 3.1), (-0.44, 3.1), (-3.1, 3.1), (-2.27, 0.0),
    (-3.1, 3.1), (-1.22, 1.22), (-2.09, 0.49),
)

#: Limits the USD ships, radians, identical for both sides (the right arm
#: reuses the left arm's joint definitions).
ASSET_SHIPPED_ARM_LIMITS: tuple[tuple[float, float], ...] = (
    (-3.1, 3.1), (-2.0108, 1.5292), (-4.6708, 1.5292), (-2.27, 0.0),
    (-1.5292, 4.6708), (-1.22, 1.22), (-2.09, 0.49),
)


def sim_arm_limits(side: str) -> tuple[tuple[float, float], ...]:
    """USD-side limits that admit exactly the vendor range through the map."""
    return (LEFT_ARM_MAP if side == "left" else RIGHT_ARM_MAP).sim_limits(VENDOR_ARM_LIMITS)


def right_arm_limit_patch(tol: float = 1e-3) -> dict[str, tuple[float, float]]:
    """DOFs whose shipped limits disagree with the mapped vendor range.

    On the right arm, joints 3 and 5 ship the LEFT arm's ranges. Through the
    right map those ranges exclude a 0.083 rad band around vendor q3 = 0 and
    q5 = 0 — which contains the vendor home pose — so the backend widens them to
    the mapped vendor range. The left arm needs no patch.
    """
    patch = {}
    for dof, want, have in zip(RIGHT_ARM_DOFS, sim_arm_limits("right"), ASSET_SHIPPED_ARM_LIMITS):
        if abs(want[0] - have[0]) > tol or abs(want[1] - have[1]) > tol:
            patch[dof] = want
    return patch


#: Safe posture, vendor convention: the xacro's shipped ``home_2``. It is a
#: mirror pose (gripper bases at y = +-0.227 m, both approaching forward).
#: In vendor convention only joint1 changes sign between sides, because the
#: xacro mirrors the axes of joints 3-6 itself.
HOME_LEFT_ARM = (0.8333, 0.0, 0.0, -1.7800, 0.0, -0.6410, 0.0)
HOME_RIGHT_ARM = (-0.8333, 0.0, 0.0, -1.7800, 0.0, -0.6410, 0.0)
HOME_BODY = (0.0, 0.0, 0.0, 0.0)
HOME_HEAD = (0.0, 0.0)

# --------------------------------------------------------------------------
# Gripper (Inspire EG2-4C2) — measured live
# --------------------------------------------------------------------------

GRIPPER_JOINT_LIMITS = (0.0, 0.82)

#: Measured live (200 Hz, 2026-09-28): joint 0.0 is CLOSED. Settled joint
#: value -> pad-link separation, identical on both sides and monotonic.
GRIPPER_ZERO_IS_OPEN = False
MEASURED_PAD_SEPARATION_M = {0.0107: 0.0362, 0.2157: 0.0517, 0.4207: 0.0684,
                             0.6252: 0.0855, 0.82: 0.1016}

# --------------------------------------------------------------------------
# Base (differential drive) — measured live
# --------------------------------------------------------------------------

WHEEL_RADIUS_M = 0.085
WHEEL_TRACK_M = 0.416

#: Measured live (200 Hz, 2 s per test) with chassis +x forward and +omega
#: counter-clockwise about world +z: achieved / commanded displacement. The
#: ~2.5 % yaw shortfall is caster scrub, not a sign or unit error.
MEASURED_BASE_RATIOS = {"forward": 1.007, "reverse": 0.986, "yaw_ccw": 0.974, "yaw_cw": 0.976}


def twist_to_wheel_radps(forward_mps: float, yaw_radps: float) -> tuple[float, float]:
    """(v, omega) -> (left, right) wheel angular velocity. Backend-internal."""
    half = 0.5 * WHEEL_TRACK_M
    return ((forward_mps - yaw_radps * half) / WHEEL_RADIUS_M,
            (forward_mps + yaw_radps * half) / WHEEL_RADIUS_M)


#: nav2 velocity bounds from the asset. vy is pinned to 0 => differential drive.
NAV2_MAX_V_MPS = 1.0
NAV2_MIN_V_MPS = 0.0          # nav2 forbids reverse; the contract permits it
NAV2_MAX_YAW_RADPS = 2.8

# --------------------------------------------------------------------------
# Timing — measured live
# --------------------------------------------------------------------------

#: Physics rate declared by the asset's own scene (``env/empty.usda``:
#: ``physxScene:timeStepsPerSecond = 200``, CPU dynamics, SAP broadphase).
#: Isaac's World default would silently be 60 Hz; the backend sets 200 Hz and
#: the smoke verifies dt = 0.005 s and one physics step per ``world.step``.
PHYSICS_HZ = 200.0
#: 20 physics steps per policy action -> 10 Hz, the fps of the dataset behind
#: the audited public AlphaBot pi0.5 checkpoint
#: (``alphabot2/aibot2_grab_bag_combined_over_under``, 10 fps). 15 Hz is not
#: an integer divisor of 200 Hz.
PHYSICS_STEPS_PER_ACTION = 20
POLICY_HZ = PHYSICS_HZ / PHYSICS_STEPS_PER_ACTION

#: Rates declared by the asset's ros2_controllers.yaml. NOT used on the Isaac
#: Python route (no ROS 2 on the host); kept for the record.
DECLARED_CONTROLLER_HZ = 50
DECLARED_MPC_HZ = 25


class IsaacBackend(Protocol):
    """Minimal surface the adapter drives. Kept tiny so it can be faked.

    ``set_joint_targets`` stages drive targets keyed by DOF path in USD units;
    nothing moves until ``step``.
    """

    def set_joint_targets(self, mapping: dict[str, float]) -> None: ...
    def get_joint_positions(self) -> dict[str, float]: ...
    def set_base_velocity(self, forward_mps: float, yaw_radps: float) -> None: ...
    def step(self, n: int) -> None: ...
    def render(self, camera: str) -> np.ndarray: ...


class IsaacUnavailable(RuntimeError):
    """Isaac Sim could not be reached or imported."""


@dataclass
class AlphaBot2IsaacAdapter:
    """Maps the policy contract onto the Ai2 Bot2 Isaac asset.

    ``backend`` is injected. With a fake backend the ordering, joint-map and
    unit story is unit-testable with no GPU; with
    :class:`isaac_backend.IsaacArticulationBackend` the same code drives the
    simulator.

    Defaults carry the values measured live on 2026-09-28. The gates remain so
    a caller that has NOT reproduced those measurements (for example, a
    different asset revision) can switch them off and have the adapter refuse.
    """

    contract: AlphaBot2Contract
    backend: IsaacBackend | None = None

    physics_steps_per_action: int | None = PHYSICS_STEPS_PER_ACTION
    arm_maps_verified: bool = True
    gripper_aperture_verified: bool = True
    gripper_zero_is_open: bool | None = GRIPPER_ZERO_IS_OPEN
    base_signs_verified: bool = True

    sim_step_count: int = field(default=0, init=False)

    # -- arm maps ----------------------------------------------------------
    def _require_arm_maps(self) -> None:
        if not self.arm_maps_verified:
            raise ContractError(
                "arm joint maps are unverified for this asset. The USD zero and "
                "sign differ per joint AND per side; identify them "
                "(scripts/alphabot2_isaac/identification/) before use."
            )

    def arm_vendor_to_sim(self, side: str, q_vendor: Sequence[float]) -> np.ndarray:
        self._require_arm_maps()
        return (LEFT_ARM_MAP if side == "left" else RIGHT_ARM_MAP).to_sim(q_vendor)

    def arm_sim_to_vendor(self, side: str, q_sim: Sequence[float]) -> np.ndarray:
        self._require_arm_maps()
        return (LEFT_ARM_MAP if side == "left" else RIGHT_ARM_MAP).to_vendor(q_sim)

    # -- state -------------------------------------------------------------
    def proprio_from_joint_map(self, joints: dict[str, float]) -> np.ndarray:
        """Assemble the 16-D policy proprio from an Isaac DOF-path dict.

        Ordering is explicit, never positional: each dimension is looked up by
        DOF path, so a reordering inside PhysX cannot silently permute the
        state vector. No base pose is included.
        """
        state = np.zeros(PROPRIO_DIM, dtype=np.float64)
        state[list(LEFT_ARM_INDICES)] = self.arm_sim_to_vendor(
            "left", [self._require(joints, n) for n in LEFT_ARM_DOFS])
        state[list(RIGHT_ARM_INDICES)] = self.arm_sim_to_vendor(
            "right", [self._require(joints, n) for n in RIGHT_ARM_DOFS])
        state[LEFT_GRIPPER_INDEX] = self.gripper_joint_to_fraction(
            self._require(joints, LEFT_GRIPPER_DOF))
        state[RIGHT_GRIPPER_INDEX] = self.gripper_joint_to_fraction(
            self._require(joints, RIGHT_GRIPPER_DOF))
        return state

    @staticmethod
    def _require(joints: dict[str, float], name: str) -> float:
        if name not in joints:
            raise ContractError(
                f"DOF {name!r} missing from the Isaac joint state. Present: "
                f"{sorted(joints)[:12]}... The asset must expose every DOF "
                "the contract addresses; a missing DOF would otherwise be "
                "read as zero."
            )
        return float(joints[name])

    # -- gripper -----------------------------------------------------------
    def gripper_fraction_to_joint(self, closed_fraction: float) -> float:
        """Policy [0=open, 1=closed] -> EG2-4C2 joint radians."""
        self._require_gripper_verified()
        f = float(np.clip(closed_fraction, 0.0, 1.0))
        lo, hi = GRIPPER_JOINT_LIMITS
        return lo + f * (hi - lo) if self.gripper_zero_is_open else hi - f * (hi - lo)

    def gripper_joint_to_fraction(self, joint_rad: float) -> float:
        """EG2-4C2 joint radians -> policy closed fraction."""
        self._require_gripper_verified()
        lo, hi = GRIPPER_JOINT_LIMITS
        span = hi - lo
        f = (joint_rad - lo) / span if self.gripper_zero_is_open else (hi - joint_rad) / span
        return float(np.clip(f, 0.0, 1.0))

    def _require_gripper_verified(self) -> None:
        if not self.gripper_aperture_verified or self.gripper_zero_is_open is None:
            raise ContractError(
                "gripper aperture direction is unverified. The EG2-4C2 asset "
                "declares limits [0, 0.82] rad but never states which end is "
                "open (measured on 2026-09-28: 0.0 is closed). Guessing here "
                "would invert every grasp in the dataset."
            )

    # -- base --------------------------------------------------------------
    def base_to_twist(self, forward_mps: float, yaw_radps: float,
                      *, enforce_nav2_bounds: bool = False) -> tuple[float, float]:
        """Policy (v, omega) -> backend twist, physical units unchanged.

        Deliberately unlike the WXAI bridge, where Habitat wanted normalized
        [-1, 1] and the missing conversion silently ran the base 4x too slow.
        Wheel velocities are the backend's business, never the policy's.
        """
        if not self.base_signs_verified:
            raise ContractError(
                "base sign convention is unverified. Measure that +forward "
                "moves the base along its own +x and +yaw rotates "
                "counter-clockwise, then set base_signs_verified=True."
            )
        v, w = float(forward_mps), float(yaw_radps)
        if enforce_nav2_bounds:
            if not (NAV2_MIN_V_MPS <= v <= NAV2_MAX_V_MPS):
                raise ContractError(
                    f"forward velocity {v} outside the nav2 bound "
                    f"[{NAV2_MIN_V_MPS}, {NAV2_MAX_V_MPS}] m/s. Note nav2 "
                    "forbids reverse while the policy contract permits it."
                )
            if abs(w) > NAV2_MAX_YAW_RADPS:
                raise ContractError(f"yaw rate {w} exceeds +-{NAV2_MAX_YAW_RADPS} rad/s")
        return v, w

    # -- action ------------------------------------------------------------
    def action_to_commands(self, action: Sequence[float]) -> dict[str, Any]:
        """Split one physical 18-D action into Isaac commands (USD units).

        Pure: no simulator contact, so ordering is testable without Isaac.
        """
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape != (ACTION_DIM,):
            raise ContractError(f"expected {ACTION_DIM}-D action, got {a.shape}")
        if not np.isfinite(a).all():
            raise ContractError("action contains non-finite values")

        joints: dict[str, float] = {}
        for name, v in zip(LEFT_ARM_DOFS, self.arm_vendor_to_sim("left", a[list(LEFT_ARM_INDICES)])):
            joints[name] = float(v)
        for name, v in zip(RIGHT_ARM_DOFS, self.arm_vendor_to_sim("right", a[list(RIGHT_ARM_INDICES)])):
            joints[name] = float(v)
        joints[LEFT_GRIPPER_DOF] = self.gripper_fraction_to_joint(a[LEFT_GRIPPER_INDEX])
        joints[RIGHT_GRIPPER_DOF] = self.gripper_fraction_to_joint(a[RIGHT_GRIPPER_INDEX])

        v, w = self.base_to_twist(a[BASE_FORWARD_INDEX], a[BASE_YAW_INDEX])
        return {"joint_targets": joints, "base_forward_mps": v, "base_yaw_radps": w}

    def home_action(self) -> np.ndarray:
        """18-D action holding the vendor home pose, grippers open, base still."""
        a = np.zeros(ACTION_DIM)
        a[list(LEFT_ARM_INDICES)] = HOME_LEFT_ARM
        a[list(RIGHT_ARM_INDICES)] = HOME_RIGHT_ARM
        return a

    def apply_action(self, action: Sequence[float]) -> dict[str, Any]:
        """Apply one 18-D action as ONE logical control transition.

        All targets -- both arms, both grippers and the base -- are staged
        before any physics advances, then physics is advanced once for the
        configured hold interval. Arms and base therefore begin executing under
        the same logical action.

        This is the defect that broke the previous WXAI embodiment, where arms
        and base were two separate environment transitions and one policy step
        silently consumed two simulator steps.
        """
        if self.backend is None:
            raise IsaacUnavailable("no Isaac backend attached")
        if self.physics_steps_per_action is None:
            raise ContractError(
                "physics_steps_per_action is unset. It must be MEASURED on a "
                "live simulator, not assumed -- the previous embodiment "
                "assumed 50 Hz and was actually 120 Hz."
            )
        commands = self.action_to_commands(action)
        before = self.sim_step_count
        # stage everything first
        self.backend.set_joint_targets(commands["joint_targets"])
        self.backend.set_base_velocity(
            commands["base_forward_mps"], commands["base_yaw_radps"])
        # then advance once
        self.backend.step(self.physics_steps_per_action)
        self.sim_step_count += self.physics_steps_per_action
        return {
            **commands,
            "sim_steps_before": before,
            "sim_steps_after": self.sim_step_count,
            "physics_steps_consumed": self.physics_steps_per_action,
        }

    def observe_proprio(self) -> np.ndarray:
        if self.backend is None:
            raise IsaacUnavailable("no Isaac backend attached")
        return self.proprio_from_joint_map(self.backend.get_joint_positions())

    @staticmethod
    def availability() -> dict[str, object]:
        return {
            "asset_located": True,
            "asset_classification": "PUBLIC_THIRD_PARTY_ASSET",
            "usd_root": "humanoid/Ai2/Bot2/Bot2.usda",
            "gripper_usd": "grippers/Inspire/EG2_4C2/Inspire_EG2_4C2.usda",
            "isaac_version_qualified": "4.5.0-rc.36 (headless, A800)",
            "backend_module": "lmm.embodiments.alphabot2.isaac_backend",
            "physics_hz": PHYSICS_HZ,
            "physics_steps_per_action": PHYSICS_STEPS_PER_ACTION,
            "policy_hz": POLICY_HZ,
            "declared_controller_hz": DECLARED_CONTROLLER_HZ,
            "arm_maps_verified": True,
            "gripper_aperture_verified": True,
            "base_signs_verified": True,
        }
