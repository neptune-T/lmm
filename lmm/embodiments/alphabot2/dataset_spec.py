"""Public AlphaBot 2 data, assessed for bootstrap value against the v1 contract.

Audited 2026-09-21 from each dataset's ``meta/info.json``, not from READMEs.
The question answered here is narrow: can any public data supply joint-space
arm targets and two gripper actions compatible with the frozen 16-D/18-D
contract, before we collect our own mobile-manipulation demonstrations?
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .contract import ACTION_DIM, PROPRIO_DIM


@dataclass(frozen=True)
class PublicDatasetSpec:
    name: str
    attribution: str
    license: str
    fps: int
    state_dim: int
    action_dim: int
    state_space: str
    action_space: str
    has_joint_arm_targets: bool
    has_two_gripper_actions: bool
    has_base_action: bool
    has_language: bool
    usable_as_joint_action_labels: bool
    notes: str
    index_map: dict[str, tuple[int, ...]] = field(default_factory=dict)


ROBOCOIN = PublicDatasetSpec(
    name="RoboCOIN/AI2_Alphabot_2_* (~40 task repos)",
    attribution="third-party aggregation, BAAI FlagOpen; arXiv:2511.17441, "
                "83 authors, 15 platforms. Not stated to be a first-party AI² "
                "contribution.",
    license="apache-2.0 on repos; CC-BY-4.0 stated in the paper",
    fps=30,
    state_dim=34,
    action_dim=34,
    state_space="mixed joint + redundant Cartesian EE",
    action_space="mixed joint + redundant Cartesian EE, absolute",
    has_joint_arm_targets=True,
    has_two_gripper_actions=True,
    has_base_action=False,
    has_language=True,
    usable_as_joint_action_labels=True,
    notes=(
        "The ONLY public source carrying executed joint-space arm targets for "
        "both arms plus a scalar per gripper. Dims 0-6 and 13-19 of the action "
        "are the joint targets; dims 7-12 and 20-25 are a redundant Cartesian "
        "copy and must NOT be mistaken for them. 4 cameras at 480x640, en+zh "
        "language. Also carries 2 neck and 4 torso dims that v1 excludes."
    ),
    index_map={
        "left_arm_joints": tuple(range(0, 7)),
        "left_eef_pose_euler_REDUNDANT": tuple(range(7, 13)),
        "right_arm_joints": tuple(range(13, 20)),
        "right_eef_pose_euler_REDUNDANT": tuple(range(20, 26)),
        "left_gripper": (26,),
        "right_gripper": (27,),
        "neck_EXCLUDED_v1": (28, 29),
        "torso_EXCLUDED_v1": tuple(range(30, 34)),
    },
)

ALPHABOT2_ACCOUNT = PublicDatasetSpec(
    name="alphabot2/Aibot2_*, alphabot2/2026_07_30_move_base_* (115 datasets)",
    attribution="unattributed individual Hugging Face account; claims no AI² "
                "affiliation. The ZPF_ joint prefix (智平方) suggests genuine "
                "hardware access but is not attribution.",
    license="apache-2.0",
    fps=15,
    state_dim=16,
    action_dim=16,
    state_space="joint: 7+7 arm joints, 2 gripper joints (2FEG_l/r)",
    action_space="Cartesian: position + quaternion xyzw per arm, + 2 grippers",
    has_joint_arm_targets=False,
    has_two_gripper_actions=True,
    has_base_action=False,
    has_language=False,
    usable_as_joint_action_labels=False,
    notes=(
        "State matches our 16-D proprio in CONTENT but not in ORDER: it is "
        "L1..L7, R1..R7, Lgrip, Rgrip (grippers last), whereas the contract "
        "interleaves them at indices 7 and 15. A reindex is required. The "
        "ACTION is Cartesian and cannot be used as joint-action labels without "
        "IK proof, which is impossible without a kinematic model. Note the "
        "dataset named move_base_pick has NO base dimension -- the name "
        "describes the task, not the action space."
    ),
    index_map={
        "state_left_arm": tuple(range(0, 7)),
        "state_right_arm": tuple(range(7, 14)),
        "state_left_gripper": (14,),
        "state_right_gripper": (15,),
    },
)

PUBLIC_SOURCES = (ROBOCOIN, ALPHABOT2_ACCOUNT)

#: A real pi0.5 checkpoint trained on AlphaBot data exists publicly:
#: alphabot2/aibot2_grab_bag_combined_over_under_pi05. Its config confirms the
#: OpenPI padded width we assume.
PUBLIC_PI05_CHECKPOINT = {
    "repo": "alphabot2/aibot2_grab_bag_combined_over_under_pi05",
    "policy_type": "pi05",
    "chunk_size": 50,
    "state_dim": 16,
    "action_dim": 16,
    "max_state_dim": 32,
    "max_action_dim": 32,
    "cameras": 4,
    "action_space": "Cartesian (NOT our joint-space contract)",
    "value": "confirms 32-D model padding; NOT a usable starting checkpoint "
             "for a joint-space 18-D action head",
}


def bootstrap_assessment() -> dict[str, object]:
    """Can public data bootstrap pi0.5 before we collect our own demos?"""
    return {
        "contract_proprio_dim": PROPRIO_DIM,
        "contract_action_dim": ACTION_DIM,
        "sources_with_joint_arm_targets": [
            s.name for s in PUBLIC_SOURCES if s.has_joint_arm_targets
        ],
        "sources_usable_as_joint_action_labels": [
            s.name for s in PUBLIC_SOURCES if s.usable_as_joint_action_labels
        ],
        "sources_with_base_action": [
            s.name for s in PUBLIC_SOURCES if s.has_base_action
        ],
        "verdict": (
            "PARTIAL. RoboCOIN can supply 16 of the 18 action dimensions "
            "(14 arm joints + 2 grippers) and the full 16-D proprio, with "
            "language, at 30 fps. It cannot supply dims 16-17: no public "
            "AlphaBot dataset contains a base action. So public data can "
            "bootstrap the manipulation half of the policy but not the mobile "
            "half, which is the part this project is actually about."
        ),
        "caveats": [
            "RoboCOIN carries a redundant Cartesian copy; only the joint "
            "indices are executed targets",
            "the alphabot2 account's Cartesian actions must not be used as "
            "joint labels without IK proof",
            "the two sources disagree on fps, cameras, gripper ordering and "
            "action space, so they cannot be pooled without conversion",
            "no normalization statistics computed yet, by instruction",
        ],
    }
