"""Symmetric WXAI / differential-drive transform stack.

Replaces the ALOHA transform stack (``openpi.policies.aloha_policy``) for the
Trossen Mobile AI embodiment: two WXAI arms on a **differential-drive** base.

Why the ALOHA stack could not be reused
---------------------------------------
``AlohaInputs``/``AlohaOutputs`` encode three assumptions that are false here:

1. ``_joint_flip_mask()`` applies a ViperX-specific sign convention
   ``[+,-,-,+,+,+,+ | +,-,-,+,+,+,+]`` to the 12 arm joints. WXAI joint signs
   have not been shown equivalent, so an unconditional flip is unjustified.
2. ``_gripper_to_angular`` (state) and ``_gripper_from_angular`` (action) are
   **not inverses** — the first models an ALOHA linear encoder stage through
   ViperX four-bar geometry (``arm_length=0.036``, ``horn_radius=0.022``), the
   second uses ALOHA puppet *joint* limits. Habitat reports a linear
   displacement fraction of a 0.044 m WXAI carriage, which is neither.
3. ``AlohaOutputs`` slices ``actions[:, :14]``, discarding both base channels,
   and ``AlohaInputs`` multiplies a 14-element mask against the action array,
   which raises on a 16-D action.

Action-space definition
-----------------------
Dimensions 14 and 15 are **forward and yaw velocity**, not the Mobile ALOHA
holonomic ``vx``/``vy``. A differential-drive base has a structurally zero
instantaneous lateral velocity, so ``vy`` is not representable and yaw — which
the base does have — had no channel at all under the old contract.

Physical units on this boundary are metres per second and radians per second.
Conversion to Habitat's normalized ``[-1, 1]`` ``BaseVelAction`` happens in
exactly one place, :func:`lmm.env.wxai_habitat_bridge.to_habitat_base`.

Symmetry
--------
Training and inference share :func:`wxai_transform_group`. The input and output
transforms are constructed as a pair from the same calibration, so a change to
one cannot silently fail to apply to the other. :func:`encode_actions` and
:func:`decode_actions` are exact inverses; with the default (sign-only)
calibration the round trip is bit-exact.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, ClassVar, Sequence

import numpy as np

# --------------------------------------------------------------------------
# Dimension layout — the single source of truth
# --------------------------------------------------------------------------

STATE_LABELS: tuple[str, ...] = (
    *(f"left_joint_{i}_rad" for i in range(6)),
    "left_gripper_closed_fraction",
    *(f"right_joint_{i}_rad" for i in range(6)),
    "right_gripper_closed_fraction",
)

ACTION_LABELS: tuple[str, ...] = (
    *(f"left_joint_{i}_absolute_rad" for i in range(6)),
    "left_gripper_closed_fraction",
    *(f"right_joint_{i}_absolute_rad" for i in range(6)),
    "right_gripper_closed_fraction",
    "base_forward_velocity_mps",
    "base_yaw_velocity_radps",
)

STATE_DIM = len(STATE_LABELS)      # 14
ACTION_DIM = len(ACTION_LABELS)    # 16

#: Indices of the 12 arm joints inside a 14-D state or 16-D action.
ARM_INDICES: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)
#: Indices of the two gripper closed fractions.
GRIPPER_INDICES: tuple[int, ...] = (6, 13)
#: Indices of the two base velocity channels (actions only).
BASE_INDICES: tuple[int, ...] = (14, 15)

assert len(ARM_INDICES) + len(GRIPPER_INDICES) == STATE_DIM
assert STATE_DIM + len(BASE_INDICES) == ACTION_DIM


class ContractError(ValueError):
    """A state or action violated the WXAI 14-D/16-D contract."""


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class WXAICalibration:
    """Per-joint physical-to-model mapping.

    ``model = sign * (physical - offset_rad)``
    ``physical = sign * model + offset_rad``

    ``sign`` is restricted to ``+1``/``-1`` so the round trip is *bit-exact*
    when ``offset_rad`` is zero (multiplying by ``+-1`` is exact in IEEE 754,
    and ``sign * sign == 1``).

    The default is the identity. It is deliberately **not** the ALOHA/ViperX
    flip mask: WXAI joint signs and zero positions have not been verified
    against the URDF. When they are, record them here — this is the one place
    an embodiment sign convention may live, and
    ``tests/test_wxai_transforms.py`` will verify the round trip stays exact.

    Grippers and base channels are identity by definition: a closed fraction
    and a velocity have no embodiment-specific sign or zero to absorb.
    """

    joint_sign: tuple[float, ...] = (1.0,) * 12
    joint_offset_rad: tuple[float, ...] = (0.0,) * 12

    def __post_init__(self) -> None:
        if len(self.joint_sign) != 12:
            raise ContractError(f"joint_sign must have 12 entries, got {len(self.joint_sign)}")
        if len(self.joint_offset_rad) != 12:
            raise ContractError(
                f"joint_offset_rad must have 12 entries, got {len(self.joint_offset_rad)}"
            )
        if not all(s in (1.0, -1.0) for s in self.joint_sign):
            raise ContractError(
                f"joint_sign entries must be exactly +1.0 or -1.0, got {self.joint_sign}. "
                "Arbitrary scales would make the physical<->model round trip inexact."
            )

    @property
    def is_identity(self) -> bool:
        return all(s == 1.0 for s in self.joint_sign) and all(
            o == 0.0 for o in self.joint_offset_rad
        )

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "WXAICalibration":
        if not data:
            return cls()
        return cls(
            joint_sign=tuple(float(x) for x in data.get("joint_sign", (1.0,) * 12)),
            joint_offset_rad=tuple(float(x) for x in data.get("joint_offset_rad", (0.0,) * 12)),
        )

    def to_mapping(self) -> dict[str, list[float]]:
        return {
            "joint_sign": list(self.joint_sign),
            "joint_offset_rad": list(self.joint_offset_rad),
        }

    def _sign(self) -> np.ndarray:
        return np.asarray(self.joint_sign, dtype=np.float64)

    def _offset(self) -> np.ndarray:
        return np.asarray(self.joint_offset_rad, dtype=np.float64)


IDENTITY_CALIBRATION = WXAICalibration()


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def _as_2d(value: Any, dim: int, what: str) -> tuple[np.ndarray, bool]:
    """Return ``(array[..., dim], was_1d)`` in float64, validated."""
    array = np.asarray(value, dtype=np.float64)
    was_1d = array.ndim == 1
    if was_1d:
        array = array[None, :]
    if array.ndim != 2 or array.shape[-1] != dim:
        raise ContractError(f"{what} must have trailing dimension {dim}, got {np.shape(value)}")
    if not np.isfinite(array).all():
        raise ContractError(f"{what} contains non-finite values")
    return array, was_1d


def validate_gripper_range(array: np.ndarray, what: str) -> None:
    """Grippers are closed fractions and must lie in ``[0, 1]``."""
    grips = array[..., list(GRIPPER_INDICES)]
    if not np.logical_and(grips >= 0.0, grips <= 1.0).all():
        raise ContractError(
            f"{what} dimensions {GRIPPER_INDICES} are gripper closed fractions "
            "and must lie in [0, 1] (0 = open, 1 = closed)"
        )


# --------------------------------------------------------------------------
# Exact physical <-> model conversions
# --------------------------------------------------------------------------


def encode_state(state: Any, calibration: WXAICalibration = IDENTITY_CALIBRATION) -> np.ndarray:
    """Physical 14-D state -> model space. Inverse of :func:`decode_state`."""
    array, was_1d = _as_2d(state, STATE_DIM, "state")
    validate_gripper_range(array, "state")
    out = array.copy()
    idx = list(ARM_INDICES)
    out[:, idx] = calibration._sign() * (out[:, idx] - calibration._offset())
    return out[0] if was_1d else out


def decode_state(state: Any, calibration: WXAICalibration = IDENTITY_CALIBRATION) -> np.ndarray:
    """Model-space 14-D state -> physical. Inverse of :func:`encode_state`."""
    array, was_1d = _as_2d(state, STATE_DIM, "state")
    out = array.copy()
    idx = list(ARM_INDICES)
    out[:, idx] = calibration._sign() * out[:, idx] + calibration._offset()
    return out[0] if was_1d else out


def encode_actions(
    actions: Any, calibration: WXAICalibration = IDENTITY_CALIBRATION
) -> np.ndarray:
    """Physical 16-D actions -> model space. Inverse of :func:`decode_actions`.

    Arms are calibrated; grippers and the two base velocity channels pass
    through unchanged by definition.
    """
    array, was_1d = _as_2d(actions, ACTION_DIM, "actions")
    validate_gripper_range(array, "actions")
    out = array.copy()
    idx = list(ARM_INDICES)
    out[:, idx] = calibration._sign() * (out[:, idx] - calibration._offset())
    return out[0] if was_1d else out


def decode_actions(
    actions: Any, calibration: WXAICalibration = IDENTITY_CALIBRATION
) -> np.ndarray:
    """Model-space 16-D actions -> physical. Inverse of :func:`encode_actions`.

    Deliberately does **not** clip or rescale. Range enforcement belongs to the
    actuator boundary, where it can be reported, not hidden mid-transform.
    """
    array, was_1d = _as_2d(actions, ACTION_DIM, "actions")
    out = array.copy()
    idx = list(ARM_INDICES)
    out[:, idx] = calibration._sign() * out[:, idx] + calibration._offset()
    return out[0] if was_1d else out


def delta_action_mask() -> tuple[bool, ...]:
    """Delta mask for the 12 arm joints: 14 elements, not 16.

    ``[T x6, F, T x6, F]`` -- the 6 left joints, left gripper absolute, the 6
    right joints, right gripper absolute.

    **The length is 14 on purpose and must not be "corrected" to 16.**
    OpenPI's ``DeltaActions`` computes ``dims = len(mask)`` and then evaluates
    ``np.where(mask, state[..., :dims], 0)``. The state is 14-D at that point
    in the pipeline, so a 16-element mask broadcasts ``(16,)`` against
    ``(14,)`` and raises. A 14-element mask stops the prefix slice exactly
    before the base channels, which is what leaves dims 14-15 absolute.

    So the base channels are absolute because the mask ends before them --
    a structural consequence of the mask length, not an accident.
    :func:`assert_base_dims_absolute` pins this behaviour.
    """
    mask = [False] * STATE_DIM
    for i in ARM_INDICES:
        mask[i] = True
    return tuple(mask)


def assert_base_dims_absolute() -> None:
    """Fail loudly if the delta mask could ever touch the base channels."""
    mask = delta_action_mask()
    if len(mask) != STATE_DIM:
        raise ContractError(
            f"delta mask must be {STATE_DIM} elements so the prefix slice stops "
            f"before base dims {BASE_INDICES}; got {len(mask)}"
        )
    if any(mask[i] for i in GRIPPER_INDICES):
        raise ContractError("grippers must stay absolute in the delta mask")
    if sum(mask) != len(ARM_INDICES):
        raise ContractError(
            f"delta mask must select exactly the {len(ARM_INDICES)} arm joints"
        )


# --------------------------------------------------------------------------
# OpenPI DataTransformFn pair
# --------------------------------------------------------------------------

#: Habitat observation key -> model image key.
CAMERA_KEYS: dict[str, str] = {
    "cam_high": "base_0_rgb",
    "cam_left_wrist": "left_wrist_0_rgb",
    "cam_right_wrist": "right_wrist_0_rgb",
}


def _to_hwc_uint8(image: Any, name: str) -> np.ndarray:
    """Return HWC uint8 RGB.

    HWC is the contract, because that is what Habitat emits. CHW is accepted
    only when the last axis cannot be a channel axis, so the conversion is
    never applied to a valid HWC image.

    Residual ambiguity is acknowledged rather than hidden: a shape like
    ``(3, 4, 4)`` is readable both ways and is resolved as HWC. That is
    unreachable for real camera resolutions (a 3-pixel-tall frame), so the
    HWC preference is safe in practice; do not feed synthetic images whose
    leading and trailing axes are both channel-sized.
    """
    array = np.asarray(image)
    if array.ndim != 3:
        raise ContractError(f"{name} must be a 3-D image, got shape {array.shape}")

    if array.shape[-1] in (3, 4):
        pass                                        # HWC, the expected layout
    elif array.shape[0] in (3, 4):
        array = np.moveaxis(array, 0, -1)           # unambiguous CHW
    else:
        raise ContractError(
            f"{name} must have 3 or 4 channels on the first or last axis, "
            f"got shape {array.shape}"
        )

    array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        array = (255.0 * array).astype(np.uint8)
    return np.ascontiguousarray(array.astype(np.uint8, copy=False))


@dataclass(frozen=True)
class WXAIInputs:
    """Model-input transform. Used by **both** training and inference.

    During training ``data["actions"]`` is present and is encoded with exactly
    the calibration :class:`WXAIOutputs` will invert. During inference the key
    is absent and only the state is encoded.
    """

    calibration: WXAICalibration = IDENTITY_CALIBRATION

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = tuple(CAMERA_KEYS)

    def __call__(self, data: dict) -> dict:
        unexpected = set(data.get("images", {})) - set(self.EXPECTED_CAMERAS)
        if unexpected:
            raise ContractError(
                f"unexpected cameras {sorted(unexpected)}; expected {self.EXPECTED_CAMERAS}"
            )

        images: dict[str, np.ndarray] = {}
        image_masks: dict[str, np.bool_] = {}
        provided = data.get("images", {})
        if "cam_high" not in provided:
            raise ContractError("cam_high is required; it is the base view")
        base = _to_hwc_uint8(provided["cam_high"], "cam_high")
        for source, dest in CAMERA_KEYS.items():
            if source in provided:
                images[dest] = _to_hwc_uint8(provided[source], source)
                image_masks[dest] = np.True_
            else:
                # Absent wrist cameras are zero-filled and masked False, so the
                # model is told they are missing rather than shown black pixels
                # it cannot distinguish from a real observation.
                images[dest] = np.zeros_like(base)
                image_masks[dest] = np.False_

        inputs: dict[str, Any] = {
            "image": images,
            "image_mask": image_masks,
            "state": encode_state(data["state"], self.calibration),
        }
        if "actions" in data:
            inputs["actions"] = encode_actions(data["actions"], self.calibration)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclass(frozen=True)
class WXAIOutputs:
    """Model-output transform. Exact inverse of :class:`WXAIInputs` on actions.

    Slices the model's padded action dimension back to 16 and decodes to
    physical units. Both grippers and both base channels are preserved — the
    truncation to 14 that upstream ``AlohaOutputs`` performs is precisely the
    bug this class exists to avoid.
    """

    calibration: WXAICalibration = IDENTITY_CALIBRATION

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.ndim != 2:
            raise ContractError(f"expected [horizon, dim] actions, got {actions.shape}")
        if actions.shape[-1] < ACTION_DIM:
            raise ContractError(
                f"model produced {actions.shape[-1]} action dims, need at least {ACTION_DIM}"
            )
        return {"actions": decode_actions(actions[:, :ACTION_DIM], self.calibration)}


def wxai_transform_group(calibration: WXAICalibration = IDENTITY_CALIBRATION):
    """The input/output transform pair, built together from one calibration.

    Both the policy server and any future fine-tuning config must obtain their
    transforms here. Constructing the pair in a single call is what makes
    training/inference symmetry structural rather than a convention.

    Returns an ``openpi.transforms.Group`` when OpenPI is importable, and a
    dependency-free stand-in with the same ``inputs``/``outputs`` fields
    otherwise, so this module stays importable (and testable) without JAX.
    """
    inputs = [WXAIInputs(calibration)]
    outputs = [WXAIOutputs(calibration)]
    try:
        from openpi import transforms as _openpi_transforms
    except ImportError:
        return _TransformPair(inputs=tuple(inputs), outputs=tuple(outputs))
    return _openpi_transforms.Group(inputs=inputs, outputs=outputs)


def wxai_data_transform_group(
    calibration: WXAICalibration = IDENTITY_CALIBRATION,
    *,
    use_delta_joint_actions: bool = True,
):
    """Full data-transform pair, **including** the delta/absolute action pass.

    This is the single definition both the policy server and any future
    fine-tuning config must use. Returning inputs and outputs from one call is
    what makes training/inference symmetry structural:

    ``inputs  = [WXAIInputs, DeltaActions(mask)]``
    ``outputs = [AbsoluteActions(mask), WXAIOutputs]``

    The ordering mirrors OpenPI's own ``Group.push`` convention (input
    transforms appended at the end, output transforms prepended), so
    ``AbsoluteActions`` restores absolute joint targets in model space *before*
    :class:`WXAIOutputs` converts back to physical units.

    Requires OpenPI for the delta transforms. Use
    :func:`wxai_transform_group` when only the WXAI pair is needed.
    """
    from openpi import transforms as _openpi_transforms

    inputs: list[Any] = [WXAIInputs(calibration)]
    outputs: list[Any] = [WXAIOutputs(calibration)]
    if use_delta_joint_actions:
        mask = delta_action_mask()
        inputs.append(_openpi_transforms.DeltaActions(mask))
        outputs.insert(0, _openpi_transforms.AbsoluteActions(mask))
    return _openpi_transforms.Group(inputs=inputs, outputs=outputs)


@dataclass(frozen=True)
class _TransformPair:
    """Stand-in for ``openpi.transforms.Group`` when OpenPI is absent."""

    inputs: tuple[Any, ...] = ()
    outputs: tuple[Any, ...] = ()


def contract_description(calibration: WXAICalibration = IDENTITY_CALIBRATION) -> dict[str, Any]:
    """Machine-readable contract, for manifests and server health responses."""
    return {
        "schema": "wxai_differential_contract_v1",
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "state_labels": list(STATE_LABELS),
        "action_labels": list(ACTION_LABELS),
        "arm_indices": list(ARM_INDICES),
        "gripper_indices": list(GRIPPER_INDICES),
        "base_indices": list(BASE_INDICES),
        "base_semantics": "forward_yaw_velocity",
        "base_units": ["m/s", "rad/s"],
        "gripper_zero": "fully_open",
        "gripper_one": "fully_closed",
        "arm_units": "absolute_joint_position_rad",
        "delta_action_mask": list(delta_action_mask()),
        "calibration": calibration.to_mapping(),
        "calibration_is_identity": calibration.is_identity,
        "replaces": "openpi.policies.aloha_policy.AlohaInputs/AlohaOutputs",
    }


__all__ = [
    "ACTION_DIM", "ACTION_LABELS", "ARM_INDICES", "BASE_INDICES", "CAMERA_KEYS",
    "ContractError", "GRIPPER_INDICES", "IDENTITY_CALIBRATION", "STATE_DIM",
    "STATE_LABELS", "WXAICalibration", "WXAIInputs", "WXAIOutputs",
    "contract_description", "decode_actions", "decode_state", "delta_action_mask",
    "assert_base_dims_absolute", "encode_actions", "encode_state", "validate_gripper_range",
    "wxai_data_transform_group", "wxai_transform_group",
]
