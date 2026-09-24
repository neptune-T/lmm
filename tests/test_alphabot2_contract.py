"""AlphaBot 2 v1 contract tests: 16-D proprio / 18-D action, two-finger grippers.

Covers per-dimension ordering, left/right independence, gripper polarity, base
ordering, transform symmetry, model padding, and the absence of torso/head and
of stale WXAI or ALOHA assumptions.
"""

from __future__ import annotations

import unittest

import numpy as np

from _helpers import REPO_ROOT  # noqa: F401

from lmm.embodiments.alphabot2 import camera_spec as CS
from lmm.embodiments.alphabot2.contract import (
    ACTION_DIM, ACTION_LABELS, ARM_DOF, ARM_INDICES, BASE_FORWARD_INDEX,
    BASE_INDICES, BASE_YAW_INDEX, GRIPPER_CLOSED, GRIPPER_INDICES, GRIPPER_OPEN,
    LEFT_ARM_INDICES, LEFT_GRIPPER_INDEX, MODEL_DIM, PROPRIO_DIM,
    PROPRIO_LABELS, RIGHT_ARM_INDICES, RIGHT_GRIPPER_INDEX, AlphaBot2Contract,
    ContractError, assert_contract_invariants, assert_ready_for_training,
    delta_action_mask,
)
from lmm.embodiments.alphabot2.dataset_spec import (
    PUBLIC_SOURCES, bootstrap_assessment,
)
from lmm.embodiments.alphabot2.real_adapter import (
    AlphaBot2RealAdapter, split_action,
)
from lmm.embodiments.alphabot2.sim_adapter import (
    AlphaBot2SimAdapter, SimulatorAssetUnavailable,
)
from lmm.embodiments.alphabot2.transforms import (
    AlphaBot2Inputs, AlphaBot2Outputs, decode_actions, encode_actions,
    encode_state, pad_to_model, unpad_from_model,
)


def _delta(data, mask):
    state, actions = data["state"], data["actions"].copy()
    m = np.asarray(mask); dims = m.shape[-1]
    actions[..., :dims] -= np.expand_dims(np.where(m, state[..., :dims], 0), -2)
    return {**data, "actions": actions}


def _absolute(data, mask):
    state, actions = data["state"], data["actions"].copy()
    m = np.asarray(mask); dims = m.shape[-1]
    actions[..., :dims] += np.expand_dims(np.where(m, state[..., :dims], 0), -2)
    return {**data, "actions": actions}


def _state(seed=0):
    rng = np.random.default_rng(seed)
    s = np.zeros(PROPRIO_DIM)
    s[list(ARM_INDICES)] = rng.uniform(-2, 2, len(ARM_INDICES))
    s[LEFT_GRIPPER_INDEX], s[RIGHT_GRIPPER_INDEX] = 0.25, 0.75
    return s


def _actions(h=3, seed=1):
    rng = np.random.default_rng(seed)
    a = np.zeros((h, ACTION_DIM))
    a[:, list(ARM_INDICES)] = rng.uniform(-2, 2, (h, len(ARM_INDICES)))
    a[:, LEFT_GRIPPER_INDEX] = rng.uniform(0, 1, h)
    a[:, RIGHT_GRIPPER_INDEX] = rng.uniform(0, 1, h)
    a[:, BASE_FORWARD_INDEX] = rng.uniform(-0.5, 0.5, h)
    a[:, BASE_YAW_INDEX] = rng.uniform(-1, 1, h)
    return a


def _img(h=8, w=6):
    return np.full((h, w, 3), 7, dtype=np.uint8)


class TestShapesAndOrdering(unittest.TestCase):
    def test_16d_proprio_18d_action(self):
        self.assertEqual((PROPRIO_DIM, ACTION_DIM), (16, 18))
        self.assertEqual(len(PROPRIO_LABELS), 16)
        self.assertEqual(len(ACTION_LABELS), 18)

    def test_exact_per_dimension_ordering(self):
        self.assertEqual(LEFT_ARM_INDICES, (0, 1, 2, 3, 4, 5, 6))
        self.assertEqual(LEFT_GRIPPER_INDEX, 7)
        self.assertEqual(RIGHT_ARM_INDICES, (8, 9, 10, 11, 12, 13, 14))
        self.assertEqual(RIGHT_GRIPPER_INDEX, 15)
        self.assertEqual((BASE_FORWARD_INDEX, BASE_YAW_INDEX), (16, 17))

    def test_seven_dof_arms(self):
        self.assertEqual(ARM_DOF, 7)
        self.assertEqual(len(LEFT_ARM_INDICES), 7)
        self.assertEqual(len(RIGHT_ARM_INDICES), 7)

    def test_base_ordering_forward_then_yaw(self):
        self.assertEqual(ACTION_LABELS[16], "base_forward_velocity_mps")
        self.assertEqual(ACTION_LABELS[17], "base_yaw_velocity_radps")

    def test_first_16_action_dims_match_proprio_actuators(self):
        for d in range(PROPRIO_DIM):
            a = ACTION_LABELS[d].replace("_absolute_rad", "").replace("_rad", "")
            s = PROPRIO_LABELS[d].replace("_rad", "")
            self.assertEqual(a, s, f"dim {d} names different actuators")

    def test_indices_partition_the_action_vector(self):
        self.assertEqual(sorted([*ARM_INDICES, *GRIPPER_INDICES, *BASE_INDICES]),
                         list(range(ACTION_DIM)))

    def test_invariants_hold(self):
        assert_contract_invariants()


class TestNoTorsoHeadOrStaleAssumptions(unittest.TestCase):
    def test_no_torso_or_head_dimensions(self):
        joined = " ".join(ACTION_LABELS + PROPRIO_LABELS)
        for banned in ("torso", "neck", "head", "waist", "lift"):
            self.assertNotIn(banned, joined)

    def test_no_stale_wxai_dimensions(self):
        """WXAI was 14-D/16-D with 6-DoF arms. None of that may survive."""
        self.assertNotEqual(PROPRIO_DIM, 14)
        self.assertNotEqual(ACTION_DIM, 16)
        self.assertNotEqual(ARM_DOF, 6)

    def test_no_aloha_assumptions(self):
        joined = " ".join(ACTION_LABELS)
        for banned in ("velocity_y", "lateral", "vx", "vy"):
            self.assertNotIn(banned, joined)

    def test_action_arm_labels_declare_absolute_targets(self):
        for i in ARM_INDICES:
            self.assertIn("absolute", ACTION_LABELS[i])


class TestDeltaMask(unittest.TestCase):
    def test_mask_is_16_not_18(self):
        m = delta_action_mask()
        self.assertEqual(len(m), PROPRIO_DIM)
        self.assertLess(len(m), ACTION_DIM)

    def test_mask_selects_only_arm_joints(self):
        m = delta_action_mask()
        self.assertEqual([i for i, b in enumerate(m) if b], list(ARM_INDICES))

    def test_grippers_excluded_from_delta(self):
        m = delta_action_mask()
        self.assertFalse(m[LEFT_GRIPPER_INDEX])
        self.assertFalse(m[RIGHT_GRIPPER_INDEX])

    def test_delta_absolute_identity_every_dimension(self):
        s, a = _state(), _actions()
        m = delta_action_mask()
        out = _absolute(_delta({"state": s, "actions": a}, m), m)
        for d in range(ACTION_DIM):
            np.testing.assert_allclose(
                out["actions"][:, d], a[:, d], rtol=0, atol=1e-12,
                err_msg=f"dim {d} ({ACTION_LABELS[d]}) broke symmetry")

    def test_arms_actually_become_deltas(self):
        s, a = _state(), _actions()
        d = _delta({"state": s, "actions": a}, delta_action_mask())
        for i in ARM_INDICES:
            np.testing.assert_allclose(d["actions"][:, i], a[:, i] - s[i], atol=1e-12)

    def test_base_excluded_from_delta(self):
        s, a = _state(), _actions()
        d = _delta({"state": s, "actions": a}, delta_action_mask())
        for i in BASE_INDICES:
            np.testing.assert_allclose(
                d["actions"][:, i], a[:, i], rtol=0, atol=1e-12,
                err_msg=f"base dim {i} was altered by the delta pass")

    def test_18_element_mask_would_raise(self):
        bad = tuple(i in ARM_INDICES for i in range(ACTION_DIM))
        with self.assertRaises(ValueError):
            _delta({"state": _state(), "actions": _actions()}, bad)


class TestTransformSymmetry(unittest.TestCase):
    def test_action_round_trip_every_dimension(self):
        a = _actions()
        back = decode_actions(encode_actions(a))
        for d in range(ACTION_DIM):
            np.testing.assert_allclose(back[:, d], a[:, d], rtol=0, atol=1e-12)

    def test_state_round_trip(self):
        s = _state()
        np.testing.assert_allclose(encode_state(s), s, rtol=0, atol=1e-12)

    def test_inputs_outputs_inverse(self):
        a = _actions()
        enc = AlphaBot2Inputs()({"state": _state(), "images": {"cam_front": _img()},
                                 "actions": a})["actions"]
        back = AlphaBot2Outputs()({"actions": enc})["actions"]
        np.testing.assert_allclose(back, a, rtol=0, atol=1e-12)

    def test_no_silent_truncation(self):
        with self.assertRaises(ContractError) as ctx:
            decode_actions(np.zeros((3, 16)))
        self.assertIn("at least 18", str(ctx.exception))

    def test_wrong_width_rejected(self):
        with self.assertRaises(ContractError):
            encode_actions(np.zeros((3, 16)))
        with self.assertRaises(ContractError):
            encode_state(np.zeros(18))

    def test_encode_does_not_mutate(self):
        a = _actions(); before = a.copy()
        encode_actions(a)
        np.testing.assert_array_equal(a, before)


class TestModelPadding(unittest.TestCase):
    def test_pads_to_32(self):
        self.assertEqual(MODEL_DIM, 32)
        self.assertEqual(pad_to_model(_state()).shape, (32,))
        self.assertEqual(pad_to_model(_actions()).shape, (3, 32))

    def test_padding_is_zero_and_recoverable(self):
        s = _state()
        p = pad_to_model(s)
        np.testing.assert_allclose(p[PROPRIO_DIM:], 0.0)
        np.testing.assert_allclose(unpad_from_model(p, PROPRIO_DIM), s, atol=1e-12)

    def test_action_padding_round_trip(self):
        a = _actions()
        np.testing.assert_allclose(
            unpad_from_model(pad_to_model(a), ACTION_DIM), a, atol=1e-12)

    def test_cannot_pad_down(self):
        with self.assertRaises(ContractError):
            pad_to_model(np.zeros(64))


class TestGripperPolarity(unittest.TestCase):
    def test_zero_open_one_closed(self):
        self.assertEqual((GRIPPER_OPEN, GRIPPER_CLOSED), (0.0, 1.0))

    def test_out_of_range_rejected(self):
        a = _actions(); a[0, LEFT_GRIPPER_INDEX] = 1.5
        with self.assertRaises(ContractError):
            encode_actions(a)
        a2 = _actions(); a2[0, RIGHT_GRIPPER_INDEX] = -0.2
        with self.assertRaises(ContractError):
            encode_actions(a2)

    def test_grippers_independent(self):
        a = np.zeros(ACTION_DIM)
        a[LEFT_GRIPPER_INDEX], a[RIGHT_GRIPPER_INDEX] = 1.0, 0.0
        d = split_action(a)
        self.assertEqual(d["left_gripper_closed_fraction"], 1.0)
        self.assertEqual(d["right_gripper_closed_fraction"], 0.0)


class TestLeftRightIndependence(unittest.TestCase):
    def test_arms_split_correctly(self):
        a = np.zeros(ACTION_DIM)
        for n, i in enumerate(LEFT_ARM_INDICES):
            a[i] = n + 1
        for n, i in enumerate(RIGHT_ARM_INDICES):
            a[i] = -(n + 1)
        d = split_action(a)
        np.testing.assert_allclose(d["left_arm_joint_targets_rad"], np.arange(1, 8))
        np.testing.assert_allclose(d["right_arm_joint_targets_rad"], -np.arange(1, 8))

    def test_base_split_correctly(self):
        a = np.zeros(ACTION_DIM)
        a[BASE_FORWARD_INDEX], a[BASE_YAW_INDEX] = 0.3, -0.7
        d = split_action(a)
        self.assertAlmostEqual(d["base_forward_mps"], 0.3)
        self.assertAlmostEqual(d["base_yaw_radps"], -0.7)

    def test_16d_action_rejected(self):
        with self.assertRaises(ContractError):
            split_action(np.zeros(16))


class TestCameraContract(unittest.TestCase):
    def test_exactly_three_cameras(self):
        self.assertEqual(len(CS.CANONICAL_CAMERAS), 3)
        self.assertEqual(CS.MODEL_KEYS["cam_front"], "base_0_rgb")

    def test_depth_and_extra_cameras_excluded(self):
        """Exclusion keys now use Ai2 Bot2 asset frame names."""
        self.assertFalse(CS.CameraContract().describe()["depth"])
        for k in ("back_camera", "chassis_camera", "wrist_depth"):
            self.assertIn(k, CS.EXCLUDED)

    def test_missing_base_view_rejected(self):
        with self.assertRaises(ContractError):
            AlphaBot2Inputs()({"state": _state(), "images": {}})

    def test_absent_wrists_are_masked_false(self):
        out = AlphaBot2Inputs()({"state": _state(), "images": {"cam_front": _img()}})
        self.assertTrue(bool(out["image_mask"]["base_0_rgb"]))
        self.assertFalse(bool(out["image_mask"]["left_wrist_0_rgb"]))

    def test_unexpected_camera_rejected(self):
        with self.assertRaises(ContractError):
            AlphaBot2Inputs()({"state": _state(),
                               "images": {"cam_front": _img(), "camera_top": _img()}})


class TestGuardsAndBlockers(unittest.TestCase):
    def test_training_blocked_on_unverified_base_and_gripper(self):
        """Both gaps must be named. Asserts the property, not the phrasing.

        The base *kinematics* are now established (differential drive, on the
        real robot and in the Ai2 Bot2 asset); what remains unverified is the
        live sign/unit mapping. The message wording changed accordingly, so
        this checks that each subject is still reported rather than matching
        an exact sentence.
        """
        with self.assertRaises(ContractError) as ctx:
            assert_ready_for_training(AlphaBot2Contract())
        msg = str(ctx.exception).lower()
        self.assertIn("base", msg)
        self.assertIn("gripper", msg)
        self.assertIn("verif", msg)

    def test_fully_verified_contract_passes(self):
        assert_ready_for_training(AlphaBot2Contract(
            base_verified=True, gripper_physical_verified=True))

    def test_sim_adapter_refuses_to_construct(self):
        with self.assertRaises(SimulatorAssetUnavailable):
            AlphaBot2SimAdapter(AlphaBot2Contract())

    def test_all_sim_asset_formats_not_found(self):
        a = AlphaBot2SimAdapter.availability()
        self.assertFalse(a["available"])
        for fmt in ("urdf_xacro", "usd_isaac", "mjcf_mujoco", "sdf_gazebo",
                    "ros_robot_description", "meshes", "collision_models"):
            self.assertEqual(a[fmt], "NOT_FOUND")

    def test_real_adapter_raises_not_stubs(self):
        r = AlphaBot2RealAdapter(AlphaBot2Contract())
        for call in (r.read_proprio, r.read_cameras):
            with self.assertRaises(NotImplementedError):
                call()
        with self.assertRaises(NotImplementedError):
            r.send_action(np.zeros(ACTION_DIM))


class TestPublicDataBootstrap(unittest.TestCase):
    def test_no_public_source_has_base_actions(self):
        self.assertEqual(bootstrap_assessment()["sources_with_base_action"], [])

    def test_only_robocoin_usable_as_joint_labels(self):
        usable = bootstrap_assessment()["sources_usable_as_joint_action_labels"]
        self.assertEqual(len(usable), 1)
        self.assertIn("RoboCOIN", usable[0])

    def test_cartesian_only_source_not_usable_as_joint_labels(self):
        """A Cartesian-ONLY action space cannot label joint targets.

        RoboCOIN is excluded here deliberately: its action space is *mixed*
        joint + redundant Cartesian, and its joint indices are executed
        targets. The disqualifying property is the absence of a joint
        component, not the presence of a Cartesian one.
        """
        cart_only = [
            s for s in PUBLIC_SOURCES
            if "Cartesian" in s.action_space and "joint" not in s.action_space
        ]
        self.assertTrue(cart_only, "expected at least one Cartesian-only source")
        for s in cart_only:
            self.assertFalse(s.usable_as_joint_action_labels)
            self.assertFalse(s.has_joint_arm_targets)

    def test_sources_are_attributed_not_assumed_official(self):
        for s in PUBLIC_SOURCES:
            self.assertNotIn("official", s.attribution.lower())


if __name__ == "__main__":
    unittest.main()
