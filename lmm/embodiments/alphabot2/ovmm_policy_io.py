"""The ONE policy interface of AlphaBot 2 + OVMM <-> the public AlphaBot pi0.5 checkpoint (numpy only).

Converter (ovmm_to_lerobot.py), SFT (sft_ovmm.py), policy server and the Habitat eval all import this
module, so the mapping is decided in one place and checked by the policy-interface gate
(scripts/alphabot2_pi05/interface_gate.py) before any SFT.

State (16-D). Habitat canonical proprio16 = [left q1..q7 (vendor), left closed fraction,
right q1..q7 (vendor), right closed fraction]. The checkpoint state = [left q1..q7, right q1..q7 with
j3/j5/j7 sign-flipped, left 2FEG reading, right 2FEG reading] (pi05_contract.STATE_NAMES). The two
orderings differ; state_from_proprio / proprio_from_state are exact inverses.

Images. Checkpoint keys (pi05_contract.CAMERA_ROLES; the wrist names are CROSSED):
  camera_top   <- head_rgb          (head camera)
  camera_left  <- right_wrist_rgb   (the checkpoint's camera_left is the RIGHT wrist)
  camera_right <- left_wrist_rgb    (the checkpoint's camera_right is the LEFT wrist)
  camera_front <- (none)            chest camera; AlphaBot 2 in OVMM has none -> MASKED empty camera:
                                    kept in the model's image list in its pretrained position, filled with
                                    the model's empty-camera image, attention mask 0.

Action. CartesianMobileAction18 = pi05_contract.ACTION_NAMES (16, pretrained ordering) + base.v, base.omega.
All 18 dims are normalized with statistics of the OVMM TRAIN dataset (v/omega have no checkpoint stats).

Horizons. The model predicts H_PRED = 50 steps (10 Hz); execution is receding-horizon with an explicit
H_exec (schedule.json eval.exec_steps), never the checkpoint's default whole-chunk open loop.
"""
from __future__ import annotations

import numpy as np

from . import cartesian as C
from . import pi05_contract as P

CAMERA_MAP = {"camera_top": "head_rgb", "camera_left": "right_wrist_rgb", "camera_right": "left_wrist_rgb"}
MASKED_CAMERAS = ("camera_front",)
MODEL_CAMERA_ORDER = ("camera_front", "camera_left", "camera_right", "camera_top")   # pretrained image_features order
IMAGE_SHAPE = (240, 320, 3)
ACTION_LABELS = tuple(C.CARTESIAN_MOBILE_ACTION18_LABELS)
STATE_LABELS = tuple(P.STATE_NAMES)
H_PRED = P.CHUNK_SIZE
INSTRUCTION = "Move the {object} from the {start} to the {goal}."


def state_from_proprio(proprio16) -> np.ndarray:
    return P.dataset_state_from_proprio(proprio16)


def proprio_from_state(state16) -> np.ndarray:
    """Exact inverse of state_from_proprio."""
    s = np.asarray(state16, float)
    p = np.zeros(16)
    p[0:7] = s[0:7] / P.DATASET_JOINT_SIGN["left"]
    p[8:15] = s[7:14] / P.DATASET_JOINT_SIGN["right"]
    gl = s[14] * P.SIM_GRIPPER_OPEN_JOINT / P.DATASET_GRIPPER_OPEN_READING["left"]
    gr = s[15] * P.SIM_GRIPPER_OPEN_JOINT / P.DATASET_GRIPPER_OPEN_READING["right"]
    p[7] = 1.0 - gl / P.SIM_GRIPPER_OPEN_JOINT
    p[15] = 1.0 - gr / P.SIM_GRIPPER_OPEN_JOINT
    return p


def policy_images(obs: dict) -> dict:
    """Habitat sensor dict -> {checkpoint camera key: HxWx3 uint8} (masked cameras are absent)."""
    return {k: np.asarray(obs[src])[..., :3].astype(np.uint8) for k, src in CAMERA_MAP.items()}


def instruction(object_category, start_recep, goal_recep) -> str:
    nice = lambda s: str(s).replace("_", " ")
    return INSTRUCTION.format(object=nice(object_category), start=nice(start_recep), goal=nice(goal_recep))


def describe() -> dict:
    """Written into dataset metadata, SFT run_info and eval records."""
    return {"camera_map": dict(CAMERA_MAP), "masked_cameras": list(MASKED_CAMERAS),
            "model_camera_order": list(MODEL_CAMERA_ORDER), "image_shape": list(IMAGE_SHAPE),
            "state_labels": list(STATE_LABELS), "state_from": "canonical proprio16 via pi05_contract.dataset_state_from_proprio",
            "action_labels": list(ACTION_LABELS), "action_norm_stats": "OVMM TRAIN dataset (all 18 dims)",
            "h_pred": H_PRED, "checkpoint": f"{P.CHECKPOINT_REPO}@{P.CHECKPOINT_REVISION}"}
