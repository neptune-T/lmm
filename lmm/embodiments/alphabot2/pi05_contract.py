"""Execution contract of the public AlphaBot pi0.5 checkpoint, resolved from data.

Checkpoint ``alphabot2/aibot2_grab_bag_combined_over_under_pi05`` @
``cf43edfb58116c241b298a26342f800bf436d1d6`` (lerobot ``pi05``), trained on
dataset ``alphabot2/aibot2_grab_bag_combined_over_under`` @
``7f4eaaffb520757efd5144a3285698578f681bbc``. Everything below comes from the
checkpoint's own config/processors, the dataset metadata, or quantitative fits
of 21,050 recorded frames against the public xacro kinematics
(docs/ALPHABOT2_PI05_EXECUTION_CONTRACT.md). numpy only.

Frames used here:
* ``chassis`` — our xacro/Isaac chassis ``base_link`` with the torso at zero;
* ``B`` — the checkpoint's Cartesian base frame (one frame for both arms);
* ``gb`` — ``gripper_base`` of the EG2-4C2 (+z = approach);
* ``E`` — the checkpoint's end-effector frame (+x = approach, origin at the TCP).
"""

from __future__ import annotations

import numpy as np

from . import kinematics as K

CHECKPOINT_REPO = "alphabot2/aibot2_grab_bag_combined_over_under_pi05"
CHECKPOINT_REVISION = "cf43edfb58116c241b298a26342f800bf436d1d6"
DATASET_REPO = "alphabot2/aibot2_grab_bag_combined_over_under"
DATASET_REVISION = "7f4eaaffb520757efd5144a3285698578f681bbc"

FPS = 10
CHUNK_SIZE = 50                 # n_action_steps = 50: the whole chunk is executed by default
NUM_INFERENCE_STEPS = 10        # flow-matching steps
MAX_DIM = 32                    # state and action padded to 32
IMAGE_SIZE = (240, 320)         # dataset frames; resized-with-pad to 224x224 inside the model
TOKENIZER_MAX_LENGTH = 200
PROMPT_FORMAT = "Task: {task}, State: {state_bins};\nAction: "   # state: 256 bins of the normalized 16-D state

STATE_NAMES = (
    *(f"ZPF_left_Joint{i}.pos" for i in range(1, 8)),
    *(f"ZPF_right_Joint{i}.pos" for i in range(1, 8)),
    "2FEG_l_Joint1.pos", "2FEG_r_Joint1.pos",
)
ACTION_NAMES = (
    "left_arm.x", "left_arm.y", "left_arm.z", "left_arm.qx", "left_arm.qy", "left_arm.qz", "left_arm.qw",
    "right_arm.x", "right_arm.y", "right_arm.z", "right_arm.qx", "right_arm.qy", "right_arm.qz", "right_arm.qw",
    "left_gripper.pos", "right_gripper.pos",
)
TASKS = ("Grab bag from top", "Grab bag from bottom", "Close gripper and do nothing",
         "Pick object from table and put it inside the box")

#: checkpoint camera key -> what it shows (from decoded training frames).
#: NOTE the wrist names are crossed: ``camera_left`` is the wrist of the arm the
#: state/action call "right" (the only arm whose gripper ever actuates).
CAMERA_ROLES = {
    "camera_front": "chest camera, fixed forward view, no robot parts visible",
    "camera_top": "head camera, forward/down, sees the robot's right arm on the image right",
    "camera_left": "wrist camera of the RIGHT arm",
    "camera_right": "wrist camera of the LEFT arm",
}
#: checkpoint camera key -> simulator camera
SIM_CAMERA_FOR = {"camera_front": "cam_chest", "camera_top": "cam_front",
                  "camera_left": "cam_right_wrist", "camera_right": "cam_left_wrist"}

# --------------------------------------------------------------------------
# State: dataset joint convention and gripper readings
# --------------------------------------------------------------------------

#: q_vendor(xacro) = sign * q_dataset (offsets 0), per arm. Left: identity.
#: Right: joints 3, 5, 7 flipped — from the joint-limit coincidences
#: (j2, j4, j6 hit the xacro limits unflipped; j7 hits +0.49 only when flipped)
#: and the frame fit (12.7 mm / 1.9 deg median, shared base frame).
DATASET_JOINT_SIGN = {"left": np.ones(7), "right": np.array([1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])}

#: Real 2FEG readings in the dataset: same polarity and range as the sim EG2
#: joint (0 = closed). Open reading differs per gripper (left never moved).
DATASET_GRIPPER_OPEN_READING = {"left": 0.7959, "right": 0.8122}
SIM_GRIPPER_OPEN_JOINT = 0.82


def dataset_state_from_sim(q_left_vendor, q_right_vendor, g_left_joint: float, g_right_joint: float) -> np.ndarray:
    """16-D checkpoint state from simulator readings (vendor joints, EG2 joint rad)."""
    s = np.zeros(16)
    s[0:7] = DATASET_JOINT_SIGN["left"] * np.asarray(q_left_vendor, float)
    s[7:14] = DATASET_JOINT_SIGN["right"] * np.asarray(q_right_vendor, float)
    s[14] = g_left_joint * DATASET_GRIPPER_OPEN_READING["left"] / SIM_GRIPPER_OPEN_JOINT
    s[15] = g_right_joint * DATASET_GRIPPER_OPEN_READING["right"] / SIM_GRIPPER_OPEN_JOINT
    return s


def dataset_state_from_proprio(proprio16) -> np.ndarray:
    """Same, from the canonical 16-D proprio (vendor joints, closed fractions)."""
    p = np.asarray(proprio16, float)
    gl = SIM_GRIPPER_OPEN_JOINT * (1.0 - p[7])
    gr = SIM_GRIPPER_OPEN_JOINT * (1.0 - p[15])
    return dataset_state_from_sim(p[0:7], p[8:15], gl, gr)


# --------------------------------------------------------------------------
# Actions: absolute EE pose targets in frame B, tool frame E
# --------------------------------------------------------------------------

def _orthonormal(m) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.asarray(m, float))
    r = u @ vt
    return r if np.linalg.det(r) > 0 else -r


#: p_B = R_B_CHASSIS @ p_chassis + T_B_CHASSIS (shared by both arms; ~4.4 deg pitch).
R_B_CHASSIS = _orthonormal([[0.9971, 0.0071, 0.0763], [-0.0053, 0.9997, -0.0241], [-0.0764, 0.0236, 0.9968]])
T_B_CHASSIS = np.array([-0.1282, 0.0228, -0.3969])
#: R_E = R_gb @ R_GB_E; TCP = p_gb + R_gb @ T_GB_E. E's +x is the approach axis (gb +z).
R_GB_E = {"left": _orthonormal([[0.0126, 0.9997, 0.0211], [0.024, -0.0214, 0.9995], [0.9996, -0.0121, -0.0243]]),
          "right": _orthonormal([[0.0096, 0.9995, 0.0291], [0.0245, -0.0293, 0.9993], [0.9997, -0.0089, -0.0248]])}
T_GB_E = {"left": np.array([0.0144, -0.0123, 0.1332]), "right": np.array([0.0106, 0.0113, 0.1427])}

#: Fit quality behind the frames (task "Pick object from table...", 0.4 s lag):
FRAME_FIT_MEDIAN = {"left": {"pos_mm": 5.8, "rot_deg": 0.70}, "right": {"pos_mm": 12.7, "rot_deg": 1.86}}
#: Recorded actions lead the recorded joint state by ~0.4-0.8 s (teleop targets).
ACTION_LEADS_STATE_S = (0.4, 0.8)


def quat_xyzw_to_matrix(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def matrix_to_quat_xyzw(r) -> np.ndarray:
    r = np.asarray(r, float)
    t = np.trace(r)
    if t > 0:
        s = 2.0 * np.sqrt(t + 1.0)
        q = [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s]
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + r[i, i] - r[j, j] - r[k, k])
        q = [0.0, 0.0, 0.0, (r[k, j] - r[j, k]) / s]
        q[i] = 0.25 * s
        q[j] = (r[j, i] + r[i, j]) / s
        q[k] = (r[k, i] + r[i, k]) / s
    q = np.asarray(q)
    return q / np.linalg.norm(q)


def canonical_quat(q, prev=None) -> np.ndarray:
    """Unit quaternion (xyzw). Hemisphere: continuous with ``prev`` when given,
    else qw >= 0 — the training data satisfies both (0 negative qw, 0 flips)."""
    q = np.asarray(q, float)
    n = np.linalg.norm(q)
    if not np.isfinite(n) or n < 1e-6:
        raise ValueError(f"degenerate quaternion {q}")
    q = q / n
    ref = np.asarray(prev, float) if prev is not None else np.array([0.0, 0.0, 0.0, 1.0])
    return -q if float(q @ ref) < 0 else q


def gripper_base_from_action(side: str, xyz, quat_xyzw) -> np.ndarray:
    """Checkpoint EE target (frame B, tool E) -> gripper_base pose in the chassis frame."""
    r_be = quat_xyzw_to_matrix(canonical_quat(quat_xyzw))
    r_ce = R_B_CHASSIS.T @ r_be
    p_ce = R_B_CHASSIS.T @ (np.asarray(xyz, float) - T_B_CHASSIS)
    r_cg = r_ce @ R_GB_E[side].T
    m = np.eye(4)
    m[:3, :3] = r_cg
    m[:3, 3] = p_ce - r_cg @ T_GB_E[side]
    return m


def action_from_gripper_base(side: str, m_gb, prev_quat=None) -> tuple[np.ndarray, np.ndarray]:
    """gripper_base pose (chassis frame) -> checkpoint EE target (xyz in B, quat xyzw)."""
    m_gb = np.asarray(m_gb, float)
    r_ce = m_gb[:3, :3] @ R_GB_E[side]
    p_ce = m_gb[:3, 3] + m_gb[:3, :3] @ T_GB_E[side]
    xyz = R_B_CHASSIS @ p_ce + T_B_CHASSIS
    return xyz, canonical_quat(matrix_to_quat_xyzw(R_B_CHASSIS @ r_ce), prev_quat)


def action_from_vendor_joints(side: str, q_vendor, prev_quat=None):
    return action_from_gripper_base(side, K.fk(side, q_vendor), prev_quat)


#: Gripper action: 0 = open; recorded "close" values lie in [0.5, 1]. Our
#: closed-fraction convention is the same scale, so it passes through clipped.
def gripper_closed_fraction_from_action(a: float) -> float:
    return float(np.clip(a, 0.0, 1.0))
