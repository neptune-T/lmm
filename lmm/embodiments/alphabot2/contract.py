"""AlphaBot 2 v1 policy contract — two-finger gripper configuration.

Target hardware (confirmed by the project owner 2026-09-21):

    AlphaBot 2 + dual 7-DoF arms + two-finger grippers + mobile base

Frozen v1 shape
---------------
Proprio **16-D**::

     0:6   left arm joints   (7)
     7     left gripper      (1)
     8:14  right arm joints  (7)
     15    right gripper     (1)

Physical action **18-D**::

     0:6   left arm absolute joint targets
     7     left gripper
     8:14  right arm absolute joint targets
     15    right gripper
     16    base forward velocity
     17    base yaw velocity

Torso and head are **deliberately outside** the learned action space in v1.

Evidence and confidence
-----------------------
* **7-DoF arms** — high confidence. Two independent public dataset families
  agree: ``left_arm_joint_1..7_rad`` (RoboCOIN) and ``ZPF_left_Joint1..7.pos``
  (alphabot2 account, ``ZPF`` = 智平方).
* **Two-finger grippers, one controllable dimension per side** — high
  confidence. Public data names them ``2FEG_l_Joint1`` / ``2FEG_r_Joint1``
  (2-Finger Electric Gripper) and exposes a single scalar per side. This
  matches the confirmed hardware configuration.
* **Base forward/yaw** — **NOT VERIFIED.** No public SDK documents the base
  interface, and no public dataset contains a base action of any kind
  (including the one named ``move_base_pick``, whose name refers to the task,
  not the action space). Dimensions 16-17 are a *project decision*, not an
  observed fact.

Known tension with the hardware
-------------------------------
The vendor advertises a lifting waist with a 0-240 cm vertical work range. By
excluding the torso from the VLA action space, v1 requires the torso to be
held fixed or driven by something outside the policy. That is a defensible
simplification for a first contract but it does constrain reachable workspace,
and it should be revisited once the SDK is available.
"""

from __future__ import annotations

from dataclasses import dataclass

CONTRACT_VERSION = "alphabot2_two_finger_v1"

# --------------------------------------------------------------------------
# Dimensions
# --------------------------------------------------------------------------

ARM_DOF = 7
GRIPPER_DOF = 1
PROPRIO_DIM = 16
ACTION_DIM = 18

LEFT_ARM_INDICES = (0, 1, 2, 3, 4, 5, 6)
LEFT_GRIPPER_INDEX = 7
RIGHT_ARM_INDICES = (8, 9, 10, 11, 12, 13, 14)
RIGHT_GRIPPER_INDEX = 15
BASE_FORWARD_INDEX = 16
BASE_YAW_INDEX = 17

ARM_INDICES = (*LEFT_ARM_INDICES, *RIGHT_ARM_INDICES)
GRIPPER_INDICES = (LEFT_GRIPPER_INDEX, RIGHT_GRIPPER_INDEX)
BASE_INDICES = (BASE_FORWARD_INDEX, BASE_YAW_INDEX)

#: OpenPI pi0.5 padded width. Confirmed against a real AlphaBot pi0.5
#: checkpoint config (alphabot2/aibot2_grab_bag_combined_over_under_pi05):
#: max_state_dim = max_action_dim = 32.
MODEL_DIM = 32

PROPRIO_LABELS: tuple[str, ...] = (
    *(f"left_arm_joint_{i}_rad" for i in range(1, 8)),
    "left_gripper_closed_fraction",
    *(f"right_arm_joint_{i}_rad" for i in range(1, 8)),
    "right_gripper_closed_fraction",
)

ACTION_LABELS: tuple[str, ...] = (
    *(f"left_arm_joint_{i}_absolute_rad" for i in range(1, 8)),
    "left_gripper_closed_fraction",
    *(f"right_arm_joint_{i}_absolute_rad" for i in range(1, 8)),
    "right_gripper_closed_fraction",
    "base_forward_velocity_mps",
    "base_yaw_velocity_radps",
)

assert len(PROPRIO_LABELS) == PROPRIO_DIM
assert len(ACTION_LABELS) == ACTION_DIM


class ContractError(ValueError):
    """The AlphaBot 2 contract was violated or used unverified."""


# --------------------------------------------------------------------------
# Gripper semantics
# --------------------------------------------------------------------------

#: Policy-facing gripper convention. Unit-free on purpose: adapters translate
#: it to whatever the simulator or SDK wants (width, position, current).
GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 1.0

GRIPPER_SEMANTICS = {
    "policy_representation": "closed fraction in [0, 1]",
    "open": GRIPPER_OPEN,
    "closed": GRIPPER_CLOSED,
    "controllable_dims_per_side": GRIPPER_DOF,
    "hardware": "two-finger electric gripper (2FEG in public data)",
    "physical_unit": "UNKNOWN — no public SDK",
    "open_limit_physical": "UNKNOWN",
    "closed_limit_physical": "UNKNOWN",
    "sign_convention_physical": "UNKNOWN",
    "continuous_or_thresholded": "UNKNOWN on hardware; public data records a "
                                 "continuous scalar",
    "grasp_events": "simulator-specific; not part of the policy contract",
    "note": "Deliberately NOT inherited from WXAI. WXAI used a carriage "
            "position written kinematically with a 0.5 snap threshold; none "
            "of that is known to apply here.",
}


# --------------------------------------------------------------------------
# Base semantics
# --------------------------------------------------------------------------

BASE_SEMANTICS = {
    "policy_representation": "physical units: m/s forward, rad/s yaw",
    "forward_index": BASE_FORWARD_INDEX,
    "yaw_index": BASE_YAW_INDEX,
    "positive_forward": "UNVERIFIED — assumed robot-forward",
    "positive_yaw": "UNVERIFIED — assumed counter-clockwise viewed from above",
    "units_expected_by_real_api": "UNKNOWN — normalized vs physical unresolved",
    "limits": "UNKNOWN (vendor publishes 6 km/h ~ 1.67 m/s maximum)",
    "command_rate": "UNKNOWN",
    "kinematics": "UNKNOWN — vendor publishes 'six-wheeled' only; differential "
                  "vs omnidirectional unresolved",
    "verified": False,
    "note": "Policy semantics are physical and stay independent of any "
            "simulator normalization. Adapters own the conversion.",
}


# --------------------------------------------------------------------------
# Delta-action mask
# --------------------------------------------------------------------------


def delta_action_mask() -> tuple[bool, ...]:
    """Arms are deltas; grippers and base are absolute.

    **Length is 16 (proprio), not 18 (action).** OpenPI's ``DeltaActions``
    computes ``dims = len(mask)`` and evaluates
    ``np.where(mask, state[..., :dims], 0)``. The state is 16-D, so an
    18-element mask broadcasts ``(18,)`` against ``(16,)`` and raises. The
    prefix stopping at 16 is precisely what leaves the base channels absolute.

    This is the same failure mode found and fixed in the WXAI contract (where
    the numbers were 14 and 16). It is restated because the dimensions differ
    and the reasoning does not transfer by memory.
    """
    mask = [False] * PROPRIO_DIM
    for i in ARM_INDICES:
        mask[i] = True
    return tuple(mask)


def assert_contract_invariants() -> None:
    """Fail loudly if the frozen v1 shape has been edited into inconsistency."""
    mask = delta_action_mask()
    if len(mask) != PROPRIO_DIM:
        raise ContractError(
            f"delta mask must be {PROPRIO_DIM} elements so the prefix slice "
            f"stops before base dims {BASE_INDICES}; got {len(mask)}"
        )
    if any(mask[i] for i in GRIPPER_INDICES):
        raise ContractError("grippers must stay absolute")
    if sum(mask) != len(ARM_INDICES):
        raise ContractError("delta mask must select exactly the 14 arm joints")
    if PROPRIO_DIM + len(BASE_INDICES) != ACTION_DIM:
        raise ContractError("action dim must be proprio + base dims")
    for name in ACTION_LABELS:
        for banned in ("torso", "neck", "head", "waist"):
            if banned in name:
                raise ContractError(
                    f"v1 excludes torso/head from the action space; found {name!r}"
                )


@dataclass(frozen=True)
class AlphaBot2Contract:
    """Declared v1 contract with verification flags carried explicitly."""

    base_verified: bool = False
    gripper_physical_verified: bool = False
    version: str = CONTRACT_VERSION

    proprio_dim: int = PROPRIO_DIM
    action_dim: int = ACTION_DIM
    model_dim: int = MODEL_DIM

    def describe(self) -> dict[str, object]:
        return {
            "contract_version": self.version,
            "proprio_dim": self.proprio_dim,
            "action_dim": self.action_dim,
            "model_dim": self.model_dim,
            "proprio_labels": list(PROPRIO_LABELS),
            "action_labels": list(ACTION_LABELS),
            "arm_dof_per_side": ARM_DOF,
            "gripper_dims_per_side": GRIPPER_DOF,
            "torso_in_action_space": False,
            "head_in_action_space": False,
            "base_verified": self.base_verified,
            "gripper_physical_verified": self.gripper_physical_verified,
        }


def assert_ready_for_training(contract: AlphaBot2Contract) -> None:
    """Refuse a training run that rests on unverified embodiment facts."""
    problems = []
    if not contract.base_verified:
        problems.append(
            "base semantics unverified (no public SDK; no public dataset "
            "contains a base action)"
        )
    if not contract.gripper_physical_verified:
        problems.append(
            "gripper physical units/limits unverified (policy-side [0,1] "
            "convention is defined, hardware mapping is not)"
        )
    if problems:
        raise ContractError(
            "AlphaBot 2 contract not ready for training: " + "; ".join(problems)
            + ". See docs/ALPHABOT2_TWO_FINGER_HARDWARE_AUDIT.md."
        )
