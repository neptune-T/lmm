"""AlphaBot 2 model-input/output transforms.

One definition, used by **both** training and inference. Returning the input
and output halves from a single function is what makes the symmetry structural
rather than a review obligation.

Joint-space is canonical for v1: arm targets are absolute joint positions.
Cartesian/FK/IK APIs may still be used later by scripted oracles, but they do
not enter the learned policy contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from .contract import (
    ACTION_DIM,
    ARM_INDICES,
    GRIPPER_INDICES,
    MODEL_DIM,
    PROPRIO_DIM,
    ContractError,
    delta_action_mask,
)

#: Habitat/SDK camera key -> pi0.5 model image key.
CAMERA_KEYS: dict[str, str] = {
    "cam_front": "base_0_rgb",
    "cam_left_wrist": "left_wrist_0_rgb",
    "cam_right_wrist": "right_wrist_0_rgb",
}


def _validate_gripper_range(values: np.ndarray, what: str) -> None:
    g = values[..., list(GRIPPER_INDICES)]
    if g.size and (np.nanmin(g) < -1e-6 or np.nanmax(g) > 1.0 + 1e-6):
        raise ContractError(
            f"{what} gripper values must be closed fractions in [0, 1]; got "
            f"[{float(np.nanmin(g)):.4f}, {float(np.nanmax(g)):.4f}]"
        )


def encode_state(state: Any) -> np.ndarray:
    """Physical 16-D proprio -> model space. Identity, but validated."""
    s = np.asarray(state, dtype=np.float64).reshape(-1)
    if s.shape != (PROPRIO_DIM,):
        raise ContractError(f"expected {PROPRIO_DIM}-D proprio, got {s.shape}")
    if not np.isfinite(s).all():
        raise ContractError("proprio contains non-finite values")
    _validate_gripper_range(s, "state")
    return s.copy()


def encode_actions(actions: Any) -> np.ndarray:
    """Physical 18-D actions -> model space. Identity, but validated."""
    a = np.asarray(actions, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != ACTION_DIM:
        raise ContractError(f"expected [horizon, {ACTION_DIM}], got {a.shape}")
    if not np.isfinite(a).all():
        raise ContractError("actions contain non-finite values")
    _validate_gripper_range(a, "action")
    return a.copy()


def decode_actions(actions: Any) -> np.ndarray:
    """Model space -> physical 18-D actions.

    Rejects anything narrower than 18 so a truncated head can never be read as
    a valid action. Extra padded columns are dropped explicitly.
    """
    a = np.asarray(actions, dtype=np.float64)
    if a.ndim != 2:
        raise ContractError(f"expected 2-D [horizon, dim] actions, got {a.shape}")
    if a.shape[1] < ACTION_DIM:
        raise ContractError(
            f"model produced {a.shape[1]} action dims, need at least "
            f"{ACTION_DIM}. A narrower head would silently drop the grippers "
            "or the base channels."
        )
    return a[:, :ACTION_DIM].copy()


def pad_to_model(vector: np.ndarray, dim: int = MODEL_DIM) -> np.ndarray:
    """Zero-pad a physical vector up to the pi0.5 model width."""
    v = np.asarray(vector, dtype=np.float64)
    width = v.shape[-1]
    if width > dim:
        raise ContractError(f"cannot pad width {width} down to {dim}")
    pad = [(0, 0)] * (v.ndim - 1) + [(0, dim - width)]
    return np.pad(v, pad, mode="constant")


def unpad_from_model(vector: np.ndarray, width: int) -> np.ndarray:
    """Slice the physical prefix back out of a padded model vector."""
    v = np.asarray(vector, dtype=np.float64)
    if v.shape[-1] < width:
        raise ContractError(f"padded vector is narrower ({v.shape[-1]}) than {width}")
    return v[..., :width].copy()


def _to_hwc_uint8(image: Any, name: str) -> np.ndarray:
    """HWC uint8 RGB. HWC is preferred; CHW accepted only when unambiguous."""
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ContractError(f"{name} must be a 3-D image, got {arr.shape}")
    if arr.shape[-1] in (3, 4):
        pass
    elif arr.shape[0] in (3, 4):
        arr = np.moveaxis(arr, 0, -1)
    else:
        raise ContractError(f"{name} has no channel axis: {arr.shape}")
    arr = arr[..., :3]
    if np.issubdtype(arr.dtype, np.floating):
        arr = (255.0 * arr).astype(np.uint8)
    return np.ascontiguousarray(arr.astype(np.uint8, copy=False))


@dataclass(frozen=True)
class AlphaBot2Inputs:
    """Model-input transform, shared by training and inference."""

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = tuple(CAMERA_KEYS)

    def __call__(self, data: dict) -> dict:
        provided = data.get("images", {})
        unexpected = set(provided) - set(self.EXPECTED_CAMERAS)
        if unexpected:
            raise ContractError(
                f"unexpected cameras {sorted(unexpected)}; expected "
                f"{self.EXPECTED_CAMERAS}"
            )
        if "cam_front" not in provided:
            raise ContractError("cam_front is required; it is the base view")

        base = _to_hwc_uint8(provided["cam_front"], "cam_front")
        images, masks = {}, {}
        for src, dest in CAMERA_KEYS.items():
            if src in provided:
                images[dest] = _to_hwc_uint8(provided[src], src)
                masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base)
                masks[dest] = np.False_

        out: dict[str, Any] = {
            "image": images,
            "image_mask": masks,
            "state": encode_state(data["state"]),
        }
        if "actions" in data:
            out["actions"] = encode_actions(data["actions"])
        if "prompt" in data:
            out["prompt"] = data["prompt"]
        return out


@dataclass(frozen=True)
class AlphaBot2Outputs:
    """Model-output transform. Preserves all 18 physical dimensions."""

    def __call__(self, data: dict) -> dict:
        return {"actions": decode_actions(data["actions"])}


def alphabot2_transform_group(*, use_delta_joint_actions: bool = True):
    """The one transform pair both training and inference must use.

    ``inputs  = [AlphaBot2Inputs, DeltaActions(mask)]``
    ``outputs = [AbsoluteActions(mask), AlphaBot2Outputs]``

    Mirrors OpenPI's ``Group.push`` ordering, so ``AbsoluteActions`` restores
    absolute joint targets before :class:`AlphaBot2Outputs` converts back to
    physical units. Requires OpenPI for the delta transforms.
    """
    from openpi import transforms as _t

    inputs: list[Any] = [AlphaBot2Inputs()]
    outputs: list[Any] = [AlphaBot2Outputs()]
    if use_delta_joint_actions:
        mask = delta_action_mask()
        inputs.append(_t.DeltaActions(mask))
        outputs.insert(0, _t.AbsoluteActions(mask))
    return _t.Group(inputs=inputs, outputs=outputs)
