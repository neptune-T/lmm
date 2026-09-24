"""AlphaBot 2 Isaac adapter tests — ordering, gates, atomicity.

No Isaac Sim required: the backend is a Protocol and these use a fake. What is
tested is the mapping logic between the policy contract and the Ai2 Bot2 asset,
plus the guards that stop unverified mappings from producing data.
"""

from __future__ import annotations

import unittest

import numpy as np

from _helpers import REPO_ROOT  # noqa: F401

from lmm.embodiments.alphabot2 import camera_spec as CS
from lmm.embodiments.alphabot2.contract import (
    ACTION_DIM, BASE_FORWARD_INDEX, BASE_YAW_INDEX, LEFT_ARM_INDICES,
    LEFT_GRIPPER_INDEX, PROPRIO_DIM, RIGHT_ARM_INDICES, RIGHT_GRIPPER_INDEX,
    AlphaBot2Contract, ContractError,
)
from lmm.embodiments.alphabot2.isaac_adapter import (
    GRIPPER_JOINT_LIMITS, GRIPPER_MIMIC_JOINTS, HOME_LEFT_ARM, HOME_RIGHT_ARM,
    LEFT_ARM_JOINTS, LEFT_GRIPPER_JOINT, NAV2_MAX_V_MPS, RIGHT_ARM_JOINTS,
    RIGHT_GRIPPER_JOINT, AlphaBot2IsaacAdapter, IsaacUnavailable,
)


class FakeIsaac:
    """Records commands and counts physics steps."""

    def __init__(self):
        self.joints: dict[str, float] = {
            **{n: 0.0 for n in LEFT_ARM_JOINTS},
            **{n: 0.0 for n in RIGHT_ARM_JOINTS},
            LEFT_GRIPPER_JOINT: 0.0, RIGHT_GRIPPER_JOINT: 0.0,
        }
        self.base = (0.0, 0.0)
        self.steps = 0
        self.order: list[str] = []

    def set_joint_positions(self, mapping): self.joints.update(mapping); self.order.append("joints")
    def get_joint_positions(self): return dict(self.joints)
    def set_base_velocity(self, f, y): self.base = (f, y); self.order.append("base")
    def step(self, n): self.steps += n; self.order.append(f"step:{n}")
    def render(self, camera): return np.zeros((8, 8, 3), np.uint8)


def _ready(backend=None, steps=4):
    a = AlphaBot2IsaacAdapter(AlphaBot2Contract(), backend=backend)
    a.gripper_aperture_verified = True
    a.gripper_zero_is_open = True
    a.base_signs_verified = True
    a.physics_steps_per_action = steps
    return a


class TestAssetNaming(unittest.TestCase):
    def test_seven_joints_per_arm(self):
        self.assertEqual(len(LEFT_ARM_JOINTS), 7)
        self.assertEqual(len(RIGHT_ARM_JOINTS), 7)
        self.assertEqual(LEFT_ARM_JOINTS[0], "left_joint1")
        self.assertEqual(RIGHT_ARM_JOINTS[6], "right_joint7")

    def test_policy_joint_order_is_14_left_then_right(self):
        order = _ready().policy_joint_order
        self.assertEqual(len(order), 14)
        self.assertEqual(order[:7], LEFT_ARM_JOINTS)
        self.assertEqual(order[7:], RIGHT_ARM_JOINTS)

    def test_single_actuated_gripper_joint_per_side(self):
        self.assertEqual(LEFT_GRIPPER_JOINT, "left_gripper_joint")
        self.assertEqual(RIGHT_GRIPPER_JOINT, "right_gripper_joint")
        self.assertEqual(len(GRIPPER_MIMIC_JOINTS), 5)

    def test_home_pose_mirrors_only_joint1(self):
        """The asset mirrors axes 3-6, so joints 4 and 6 must NOT be negated."""
        self.assertAlmostEqual(HOME_LEFT_ARM[0], -HOME_RIGHT_ARM[0])
        self.assertAlmostEqual(HOME_LEFT_ARM[3], HOME_RIGHT_ARM[3])
        self.assertAlmostEqual(HOME_LEFT_ARM[5], HOME_RIGHT_ARM[5])


class TestProprioAssembly(unittest.TestCase):
    def test_proprio_is_16d_and_name_indexed(self):
        a = _ready()
        joints = {n: 0.1 * (i + 1) for i, n in enumerate(LEFT_ARM_JOINTS)}
        joints.update({n: -0.1 * (i + 1) for i, n in enumerate(RIGHT_ARM_JOINTS)})
        joints[LEFT_GRIPPER_JOINT] = 0.0
        joints[RIGHT_GRIPPER_JOINT] = GRIPPER_JOINT_LIMITS[1]
        s = a.proprio_from_joint_map(joints)
        self.assertEqual(s.shape, (PROPRIO_DIM,))
        for k, i in enumerate(LEFT_ARM_INDICES):
            self.assertAlmostEqual(s[i], 0.1 * (k + 1))
        for k, i in enumerate(RIGHT_ARM_INDICES):
            self.assertAlmostEqual(s[i], -0.1 * (k + 1))
        self.assertAlmostEqual(s[LEFT_GRIPPER_INDEX], 0.0)
        self.assertAlmostEqual(s[RIGHT_GRIPPER_INDEX], 1.0)

    def test_missing_joint_raises_not_zero_fills(self):
        a = _ready()
        joints = {n: 0.0 for n in LEFT_ARM_JOINTS}
        with self.assertRaises(ContractError) as ctx:
            a.proprio_from_joint_map(joints)
        self.assertIn("missing", str(ctx.exception))

    def test_left_right_independent(self):
        a = _ready()
        joints = {n: 0.0 for n in (*LEFT_ARM_JOINTS, *RIGHT_ARM_JOINTS)}
        joints[LEFT_GRIPPER_JOINT] = 0.0
        joints[RIGHT_GRIPPER_JOINT] = 0.0
        joints["left_joint3"] = 1.234
        s = a.proprio_from_joint_map(joints)
        self.assertAlmostEqual(s[LEFT_ARM_INDICES[2]], 1.234)
        self.assertAlmostEqual(s[RIGHT_ARM_INDICES[2]], 0.0)


class TestGripperMapping(unittest.TestCase):
    def test_unverified_aperture_refuses(self):
        a = AlphaBot2IsaacAdapter(AlphaBot2Contract())
        with self.assertRaises(ContractError) as ctx:
            a.gripper_fraction_to_joint(0.5)
        self.assertIn("unverified", str(ctx.exception))

    def test_zero_is_open_mapping(self):
        a = _ready()
        lo, hi = GRIPPER_JOINT_LIMITS
        self.assertAlmostEqual(a.gripper_fraction_to_joint(0.0), lo)
        self.assertAlmostEqual(a.gripper_fraction_to_joint(1.0), hi)
        self.assertAlmostEqual(a.gripper_fraction_to_joint(0.5), (lo + hi) / 2)

    def test_zero_is_closed_mapping_inverts(self):
        a = _ready(); a.gripper_zero_is_open = False
        lo, hi = GRIPPER_JOINT_LIMITS
        self.assertAlmostEqual(a.gripper_fraction_to_joint(0.0), hi)
        self.assertAlmostEqual(a.gripper_fraction_to_joint(1.0), lo)

    def test_fraction_joint_round_trip(self):
        for zero_open in (True, False):
            a = _ready(); a.gripper_zero_is_open = zero_open
            for f in (0.0, 0.25, 0.5, 0.75, 1.0):
                back = a.gripper_joint_to_fraction(a.gripper_fraction_to_joint(f))
                self.assertAlmostEqual(back, f, places=9)


class TestBaseMapping(unittest.TestCase):
    def test_unverified_signs_refuse(self):
        a = AlphaBot2IsaacAdapter(AlphaBot2Contract())
        a.gripper_aperture_verified = True; a.gripper_zero_is_open = True
        with self.assertRaises(ContractError) as ctx:
            a.base_to_twist(0.1, 0.0)
        self.assertIn("sign convention", str(ctx.exception))

    def test_physical_units_pass_through_unscaled(self):
        """Unlike WXAI/Habitat, the diff-drive consumes m/s directly."""
        v, w = _ready().base_to_twist(0.42, -0.31)
        self.assertAlmostEqual(v, 0.42)
        self.assertAlmostEqual(w, -0.31)

    def test_nav2_bounds_enforced_only_on_request(self):
        a = _ready()
        a.base_to_twist(5.0, 0.0)                       # no bounds by default
        with self.assertRaises(ContractError):
            a.base_to_twist(NAV2_MAX_V_MPS + 0.5, 0.0, enforce_nav2_bounds=True)

    def test_nav2_forbids_reverse_but_contract_allows_it(self):
        a = _ready()
        a.base_to_twist(-0.2, 0.0)                      # contract permits
        with self.assertRaises(ContractError):
            a.base_to_twist(-0.2, 0.0, enforce_nav2_bounds=True)


class TestActionMappingAndAtomicity(unittest.TestCase):
    def test_action_splits_into_16_joints_plus_base(self):
        a = _ready()
        act = np.zeros(ACTION_DIM)
        for k, i in enumerate(LEFT_ARM_INDICES):
            act[i] = 0.01 * (k + 1)
        act[LEFT_GRIPPER_INDEX] = 0.0
        act[RIGHT_GRIPPER_INDEX] = 1.0
        act[BASE_FORWARD_INDEX] = 0.2
        act[BASE_YAW_INDEX] = -0.4
        c = a.action_to_commands(act)
        self.assertEqual(len(c["joint_positions"]), 16)
        self.assertAlmostEqual(c["joint_positions"]["left_joint1"], 0.01)
        self.assertAlmostEqual(c["base_forward_mps"], 0.2)
        self.assertAlmostEqual(c["base_yaw_radps"], -0.4)

    def test_mimic_joints_are_never_commanded(self):
        c = _ready().action_to_commands(np.zeros(ACTION_DIM))
        for m in GRIPPER_MIMIC_JOINTS:
            self.assertNotIn(m, c["joint_positions"])
            for side in ("left_", "right_"):
                self.assertNotIn(side + m, c["joint_positions"])

    def test_torso_and_head_never_commanded(self):
        c = _ready().action_to_commands(np.zeros(ACTION_DIM))
        for n in c["joint_positions"]:
            self.assertNotIn("body_joint", n)
            self.assertNotIn("head_joint", n)

    def test_16d_action_rejected(self):
        with self.assertRaises(ContractError):
            _ready().action_to_commands(np.zeros(16))

    def test_one_action_is_one_logical_transition(self):
        """Targets staged BEFORE physics advances, exactly once."""
        fake = FakeIsaac()
        a = _ready(backend=fake, steps=4)
        r = a.apply_action(np.zeros(ACTION_DIM))
        self.assertEqual(fake.order, ["joints", "base", "step:4"])
        self.assertEqual(r["sim_steps_before"], 0)
        self.assertEqual(r["sim_steps_after"], 4)
        self.assertEqual(r["physics_steps_consumed"], 4)
        self.assertEqual(fake.steps, 4)

    def test_arm_and_base_do_not_consume_two_transitions(self):
        """The WXAI defect: one policy action must not cost two env steps."""
        fake = FakeIsaac()
        a = _ready(backend=fake, steps=4)
        act = np.zeros(ACTION_DIM)
        act[LEFT_ARM_INDICES[0]] = 0.1
        act[BASE_FORWARD_INDEX] = 0.3
        a.apply_action(act)
        self.assertEqual(fake.order.count("step:4"), 1)
        self.assertEqual(sum(1 for o in fake.order if o.startswith("step")), 1)

    def test_unmeasured_timing_refuses_to_apply(self):
        a = _ready(backend=FakeIsaac())
        a.physics_steps_per_action = None
        with self.assertRaises(ContractError) as ctx:
            a.apply_action(np.zeros(ACTION_DIM))
        self.assertIn("MEASURED", str(ctx.exception))

    def test_no_backend_raises(self):
        a = _ready(backend=None)
        with self.assertRaises(IsaacUnavailable):
            a.apply_action(np.zeros(ACTION_DIM))


class TestAvailabilityAndCameras(unittest.TestCase):
    def test_availability_reports_asset_found_but_not_runnable(self):
        av = AlphaBot2IsaacAdapter.availability()
        self.assertTrue(av["asset_located"])
        self.assertEqual(av["asset_classification"], "PUBLIC_THIRD_PARTY_ASSET")
        self.assertFalse(av["isaac_runnable_here"])
        self.assertIsNone(av["physics_hz_measured"])

    def test_three_cameras_mapped_to_asset_frames(self):
        self.assertEqual(len(CS.CANONICAL_CAMERAS), 3)
        self.assertEqual(CS.ASSET_FRAMES["cam_front"], "head_camera")
        self.assertEqual(CS.ASSET_FRAMES["cam_left_wrist"], "left_link8")
        self.assertEqual(CS.MODEL_KEYS["cam_front"], "base_0_rgb")

    def test_camera_verification_flags_default_false(self):
        d = CS.CameraContract().describe()
        self.assertFalse(d["frames_verified"])
        self.assertFalse(d["orientation_verified"])
        self.assertFalse(d["channel_order_verified"])
        self.assertFalse(d["depth"])


if __name__ == "__main__":
    unittest.main()
