"""AlphaBot 2 Isaac Sim adapter — structure built, NOT live-validated.

Architecture::

    LMM / pi0.5
        -> AlphaBot2 policy contract   (16-D proprio / 18-D action, physical units)
        -> AlphaBot2 Isaac adapter     (this module: naming, ordering, unit mapping)
        -> Isaac robot / ros2_control  (joint position commands + /cmd_vel)

The generic LMM stack never imports Isaac. Isaac is imported lazily inside
:meth:`AlphaBot2IsaacAdapter.connect`, so this module is importable — and its
ordering logic testable — on a machine with no simulator.

Status
------
Every mapping below is derived from the pinned Ai2 Bot2 description files (see
``docs/ALPHABOT2_SIM_KINEMATIC_AUDIT.md``). **None of it has been executed
against a running simulator**, because no reachable host can run Isaac Sim
(``docs/ALPHABOT2_ISAAC_BLOCKER.md``). Two mappings are explicitly unverified
and guarded rather than assumed:

* ``GRIPPER_APERTURE_VERIFIED`` — whether ``gripper_joint = 0`` is open or closed.
* ``BASE_SIGNS_VERIFIED`` — whether +forward and +yaw match the policy convention.

Both default to ``False`` and make :meth:`apply_action` refuse, so an
unvalidated mapping cannot quietly produce a dataset.
"""

from __future__ import annotations

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

# --------------------------------------------------------------------------
# Asset naming, from the pinned Ai2 Bot2 description
# --------------------------------------------------------------------------

#: Arm joints in policy order. Index i of this tuple is policy proprio dim
#: LEFT_ARM_INDICES[i] / RIGHT_ARM_INDICES[i].
LEFT_ARM_JOINTS: tuple[str, ...] = tuple(f"left_joint{i}" for i in range(1, 8))
RIGHT_ARM_JOINTS: tuple[str, ...] = tuple(f"right_joint{i}" for i in range(1, 8))

#: The single actuated joint of each Inspire EG2-4C2 gripper. Five further
#: mimic joints follow it and must NOT be commanded independently.
LEFT_GRIPPER_JOINT = "left_gripper_joint"
RIGHT_GRIPPER_JOINT = "right_gripper_joint"

GRIPPER_MIMIC_JOINTS: tuple[str, ...] = (
    "left_inner_joint", "right_inner_joint", "right_outer_joint",
    "left_pad_joint", "right_pad_joint",
)

#: EG2-4C2 actuated-joint range, radians.
GRIPPER_JOINT_LIMITS = (0.0, 0.82)

#: Held fixed in v1. Present in the asset, outside the learned action space.
FIXED_BODY_JOINTS: tuple[str, ...] = tuple(f"body_joint{i}" for i in range(1, 5))
FIXED_HEAD_JOINTS: tuple[str, ...] = ("head_joint1", "head_joint2")

#: Safe posture. Taken verbatim from the asset's shipped ``home_2``, which is a
#: tucked arm pose suitable for navigation. Note joint1 is mirrored between
#: sides while joints 4 and 6 are NOT — the asset already mirrors those axes.
HOME_LEFT_ARM = (0.8333, 0.0, 0.0, -1.7800, 0.0, -0.6410, 0.0)
HOME_RIGHT_ARM = (-0.8333, 0.0, 0.0, -1.7800, 0.0, -0.6410, 0.0)
HOME_BODY = (0.0, 0.0, 0.0, 0.0)
HOME_HEAD = (0.0, 0.0)

#: ros2_control / Isaac topics for ``hardware:=isaac``.
ISAAC_JOINT_COMMAND_TOPIC = "/isaac/joint_command"
ISAAC_JOINT_STATE_TOPIC = "/isaac/joint_states"
BASE_CMD_TOPIC = "/cmd_vel"
ODOM_TOPIC = "/odom"

#: Controller rate declared by the asset's ros2_controllers.yaml.
DECLARED_CONTROLLER_HZ = 50
DECLARED_MPC_HZ = 25

#: nav2 velocity bounds. vy is pinned to 0 => differential drive.
NAV2_MAX_V_MPS = 1.0
NAV2_MIN_V_MPS = 0.0          # nav2 forbids reverse; the contract permits it
NAV2_MAX_YAW_RADPS = 2.8


class IsaacBackend(Protocol):
    """Minimal surface the adapter drives. Kept tiny so it can be faked."""

    def set_joint_positions(self, mapping: dict[str, float]) -> None: ...
    def get_joint_positions(self) -> dict[str, float]: ...
    def set_base_velocity(self, forward_mps: float, yaw_radps: float) -> None: ...
    def step(self, n: int) -> None: ...
    def render(self, camera: str) -> np.ndarray: ...


class IsaacUnavailable(RuntimeError):
    """Isaac Sim could not be reached or imported."""


@dataclass
class AlphaBot2IsaacAdapter:
    """Maps the policy contract onto the Ai2 Bot2 Isaac asset.

    ``backend`` is injected. With a fake backend the whole ordering and unit
    story is unit-testable with no GPU; with a real Isaac backend the same code
    drives the simulator.
    """

    contract: AlphaBot2Contract
    backend: IsaacBackend | None = None

    #: Number of physics steps advanced per policy action. MUST be measured on
    #: a live simulator before data collection -- see the temporal audit.
    physics_steps_per_action: int | None = None

    #: Gates. Both mappings are derived but unproven; see module docstring.
    gripper_aperture_verified: bool = False
    base_signs_verified: bool = False

    #: Set once measured: True if gripper_joint == 0 means fully OPEN.
    gripper_zero_is_open: bool | None = None

    sim_step_count: int = field(default=0, init=False)

    # -- naming ----------------------------------------------------------
    @property
    def policy_joint_order(self) -> tuple[str, ...]:
        """Asset joint names in policy proprio order, arms only."""
        return (*LEFT_ARM_JOINTS, *RIGHT_ARM_JOINTS)

    def proprio_from_joint_map(self, joints: dict[str, float]) -> np.ndarray:
        """Assemble the 16-D policy proprio from an Isaac joint dict.

        Ordering is explicit, never positional: each dimension is looked up by
        the asset's joint name, so a reordering inside Isaac cannot silently
        permute the state vector.
        """
        state = np.zeros(PROPRIO_DIM, dtype=np.float64)
        for i, name in zip(LEFT_ARM_INDICES, LEFT_ARM_JOINTS):
            state[i] = self._require(joints, name)
        for i, name in zip(RIGHT_ARM_INDICES, RIGHT_ARM_JOINTS):
            state[i] = self._require(joints, name)
        state[LEFT_GRIPPER_INDEX] = self.gripper_joint_to_fraction(
            self._require(joints, LEFT_GRIPPER_JOINT))
        state[RIGHT_GRIPPER_INDEX] = self.gripper_joint_to_fraction(
            self._require(joints, RIGHT_GRIPPER_JOINT))
        return state

    @staticmethod
    def _require(joints: dict[str, float], name: str) -> float:
        if name not in joints:
            raise ContractError(
                f"joint {name!r} missing from the Isaac joint state. Present: "
                f"{sorted(joints)[:12]}... The asset must expose every joint "
                "the contract addresses; a missing joint would otherwise be "
                "read as zero."
            )
        return float(joints[name])

    # -- gripper ---------------------------------------------------------
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
                "open. Measure it in simulation, then set "
                "gripper_aperture_verified=True and gripper_zero_is_open. "
                "Guessing here would invert every grasp in the dataset."
            )

    # -- base ------------------------------------------------------------
    def base_to_twist(self, forward_mps: float, yaw_radps: float,
                      *, enforce_nav2_bounds: bool = False) -> tuple[float, float]:
        """Policy (v, omega) -> /cmd_vel (linear.x, angular.z).

        Physical units pass through unchanged: the asset's diff-drive consumes
        m/s and rad/s directly, so there is no normalization step here. This is
        deliberately unlike the WXAI bridge, where Habitat wanted normalized
        [-1, 1] and the missing conversion silently ran the base 4x too slow.
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

    # -- action ----------------------------------------------------------
    def action_to_commands(self, action: Sequence[float]) -> dict[str, Any]:
        """Split one physical 18-D action into Isaac commands.

        Pure: no simulator contact, so ordering is testable without Isaac.
        """
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape != (ACTION_DIM,):
            raise ContractError(f"expected {ACTION_DIM}-D action, got {a.shape}")
        if not np.isfinite(a).all():
            raise ContractError("action contains non-finite values")

        joints: dict[str, float] = {}
        for i, name in zip(LEFT_ARM_INDICES, LEFT_ARM_JOINTS):
            joints[name] = float(a[i])
        for i, name in zip(RIGHT_ARM_INDICES, RIGHT_ARM_JOINTS):
            joints[name] = float(a[i])
        joints[LEFT_GRIPPER_JOINT] = self.gripper_fraction_to_joint(a[LEFT_GRIPPER_INDEX])
        joints[RIGHT_GRIPPER_JOINT] = self.gripper_fraction_to_joint(a[RIGHT_GRIPPER_INDEX])

        v, w = self.base_to_twist(a[BASE_FORWARD_INDEX], a[BASE_YAW_INDEX])
        return {"joint_positions": joints, "base_forward_mps": v, "base_yaw_radps": w}

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
        self.backend.set_joint_positions(commands["joint_positions"])
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

    # -- lifecycle -------------------------------------------------------
    def connect(self) -> None:
        """Attach a real Isaac backend. Lazy import; never at module scope."""
        try:  # pragma: no cover - requires Isaac Sim
            import isaacsim  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise IsaacUnavailable(
                "Isaac Sim is not importable here. No reachable host can run "
                "it: the local workstation has ~2.9 GB free (Isaac needs "
                "~30 GB) and the A800 jump host rejects our SSH key. See "
                "docs/ALPHABOT2_ISAAC_BLOCKER.md."
            ) from exc
        raise NotImplementedError(  # pragma: no cover
            "Bind the Isaac stage here: load humanoid/Ai2/Bot2/Bot2.usda, "
            "attach grippers/Inspire, then either drive ros2_control over "
            f"{ISAAC_JOINT_COMMAND_TOPIC} / {ISAAC_JOINT_STATE_TOPIC} or use "
            "the Isaac Python articulation API directly. The topic route needs "
            "Isaac Sim + ROS 2 + ros2_control + the fiveages controllers "
            "(ocs2_arm_controller, basic_joint_controller, "
            "adaptive_gripper_controller)."
        )

    @staticmethod
    def availability() -> dict[str, object]:
        return {
            "asset_located": True,
            "asset_classification": "PUBLIC_THIRD_PARTY_ASSET",
            "usd_root": "humanoid/Ai2/Bot2/Bot2.usda",
            "isaac_runnable_here": False,
            "blocker": "no reachable host can run Isaac Sim",
            "declared_controller_hz": DECLARED_CONTROLLER_HZ,
            "physics_hz_measured": None,
            "gripper_aperture_verified": False,
            "base_signs_verified": False,
        }
