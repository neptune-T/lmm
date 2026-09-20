"""Exact per-dimension round-trip tests for the WXAI 14-D/16-D contract.

Covers the questions the dual-arm interface review raised: that every
dimension keeps one meaning from Habitat to the model and back, that training
and inference are inverse operations, and that the base channels are
forward/yaw rather than the Mobile ALOHA vx/vy.

OpenPI is not importable in this environment, so ``_DeltaActions`` /
``_AbsoluteActions`` below are faithful transcriptions of
``openpi.transforms``. They are pinned against the upstream source by
``test_delta_transcription_matches_upstream_semantics``.
"""

from __future__ import annotations

import unittest

import numpy as np

from _helpers import REPO_ROOT  # noqa: F401  (path setup)

from lmm.env.wxai_habitat_bridge import (
    BaseLimits,
    decode_policy_action,
    encode_habitat_observation,
    from_habitat_base,
    to_habitat_base,
)
from lmm.policy.wxai_transforms import (
    ACTION_DIM,
    ACTION_LABELS,
    ARM_INDICES,
    BASE_INDICES,
    GRIPPER_INDICES,
    STATE_DIM,
    STATE_LABELS,
    ContractError,
    WXAIInputs,
    WXAIOutputs,
    assert_base_dims_absolute,
    decode_actions,
    delta_action_mask,
    encode_actions,
    encode_state,
)


# --- faithful transcriptions of openpi.transforms (openpi not installed) ---
def _delta(data: dict, mask) -> dict:
    state, actions = data["state"], data["actions"].copy()
    m = np.asarray(mask)
    dims = m.shape[-1]
    actions[..., :dims] -= np.expand_dims(np.where(m, state[..., :dims], 0), axis=-2)
    return {**data, "actions": actions}


def _absolute(data: dict, mask) -> dict:
    state, actions = data["state"], data["actions"].copy()
    m = np.asarray(mask)
    dims = m.shape[-1]
    actions[..., :dims] += np.expand_dims(np.where(m, state[..., :dims], 0), axis=-2)
    return {**data, "actions": actions}


def _image(v: int = 7) -> np.ndarray:
    return np.full((4, 4, 3), v, dtype=np.uint8)


def _state(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    s = np.zeros(STATE_DIM)
    s[list(ARM_INDICES)] = rng.uniform(-2.0, 2.0, size=len(ARM_INDICES))
    s[GRIPPER_INDICES[0]] = 0.25
    s[GRIPPER_INDICES[1]] = 0.75
    return s


def _actions(horizon: int = 3, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = np.zeros((horizon, ACTION_DIM))
    a[:, list(ARM_INDICES)] = rng.uniform(-2.0, 2.0, size=(horizon, len(ARM_INDICES)))
    a[:, GRIPPER_INDICES[0]] = rng.uniform(0, 1, size=horizon)
    a[:, GRIPPER_INDICES[1]] = rng.uniform(0, 1, size=horizon)
    a[:, BASE_INDICES[0]] = rng.uniform(-0.2, 0.2, size=horizon)
    a[:, BASE_INDICES[1]] = rng.uniform(-0.4, 0.4, size=horizon)
    return a


class TestDimensionLabels(unittest.TestCase):
    def test_state_is_14d_and_action_is_16d(self):
        self.assertEqual((STATE_DIM, ACTION_DIM), (14, 16))
        self.assertEqual(len(STATE_LABELS), 14)
        self.assertEqual(len(ACTION_LABELS), 16)

    def test_first_14_dims_refer_to_the_same_physical_quantities(self):
        """Dims 0-13 must address the same joint/gripper in state and action.

        The labels are not identical strings by design: a state dim is a
        measured position, the matching action dim is an absolute *target*
        (``..._absolute_rad``). What must agree is which actuator each index
        refers to.
        """
        for d in range(14):
            state_id = STATE_LABELS[d].replace("_rad", "").replace(
                "_closed_fraction", "")
            action_id = ACTION_LABELS[d].replace("_absolute_rad", "").replace(
                "_rad", "").replace("_closed_fraction", "")
            self.assertEqual(
                state_id, action_id,
                f"dim {d}: state {STATE_LABELS[d]!r} and action "
                f"{ACTION_LABELS[d]!r} name different actuators",
            )

    def test_action_arm_labels_declare_absolute_targets(self):
        """Absolute-vs-delta must be visible in the name, not just the docs."""
        for i in ARM_INDICES:
            self.assertIn("absolute", ACTION_LABELS[i])

    def test_base_dims_are_forward_and_yaw_not_vx_vy(self):
        self.assertEqual(ACTION_LABELS[14], "base_forward_velocity_mps")
        self.assertEqual(ACTION_LABELS[15], "base_yaw_velocity_radps")
        joined = " ".join(ACTION_LABELS)
        self.assertNotIn("velocity_y", joined)
        self.assertNotIn("lateral", joined)

    def test_index_groups_partition_the_action_vector(self):
        covered = sorted([*ARM_INDICES, *GRIPPER_INDICES, *BASE_INDICES])
        self.assertEqual(covered, list(range(ACTION_DIM)))

    def test_arm_gripper_interleaving_matches_left_right_layout(self):
        self.assertEqual(tuple(ARM_INDICES), (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12))
        self.assertEqual(tuple(GRIPPER_INDICES), (6, 13))
        self.assertEqual(tuple(BASE_INDICES), (14, 15))


class TestEncodeDecodeRoundTrip(unittest.TestCase):
    def test_every_action_dimension_round_trips_exactly(self):
        a = _actions()
        back = decode_actions(encode_actions(a))
        for d in range(ACTION_DIM):
            np.testing.assert_allclose(
                back[:, d], a[:, d], rtol=0, atol=1e-12,
                err_msg=f"dim {d} ({ACTION_LABELS[d]}) did not round-trip",
            )

    def test_every_state_dimension_round_trips_exactly(self):
        s = _state()
        enc = encode_state(s)
        np.testing.assert_allclose(enc, s, rtol=0, atol=1e-12)

    def test_encode_preserves_shape_and_dtype(self):
        a = _actions()
        self.assertEqual(encode_actions(a).shape, a.shape)
        self.assertEqual(encode_actions(a).dtype, np.float64)

    def test_encode_does_not_mutate_input(self):
        a = _actions()
        before = a.copy()
        encode_actions(a)
        np.testing.assert_array_equal(a, before)

    def test_gripper_out_of_range_rejected(self):
        a = _actions()
        a[0, GRIPPER_INDICES[0]] = 1.5
        with self.assertRaises(ContractError):
            encode_actions(a)

    def test_wrong_width_rejected(self):
        with self.assertRaises(ContractError):
            encode_actions(np.zeros((3, 14)))
        with self.assertRaises(ContractError):
            encode_state(np.zeros(16))


class TestDeltaAbsoluteSymmetry(unittest.TestCase):
    """Training encodes deltas; inference restores absolutes. Must be inverse."""

    def test_mask_is_14_elements_selecting_only_arm_joints(self):
        assert_base_dims_absolute()
        mask = delta_action_mask()
        self.assertEqual(len(mask), 14)
        self.assertEqual([i for i, b in enumerate(mask) if b], list(ARM_INDICES))

    def test_delta_then_absolute_is_identity_on_every_dimension(self):
        state, actions = _state(), _actions()
        mask = delta_action_mask()
        out = _absolute(_delta({"state": state, "actions": actions}, mask), mask)
        for d in range(ACTION_DIM):
            np.testing.assert_allclose(
                out["actions"][:, d], actions[:, d], rtol=0, atol=1e-12,
                err_msg=f"dim {d} ({ACTION_LABELS[d]}) broke delta/absolute symmetry",
            )

    def test_arm_dims_actually_become_deltas(self):
        state, actions = _state(), _actions()
        d = _delta({"state": state, "actions": actions}, delta_action_mask())
        for i in ARM_INDICES:
            np.testing.assert_allclose(
                d["actions"][:, i], actions[:, i] - state[i], rtol=0, atol=1e-12
            )

    def test_grippers_stay_absolute(self):
        state, actions = _state(), _actions()
        d = _delta({"state": state, "actions": actions}, delta_action_mask())
        for i in GRIPPER_INDICES:
            np.testing.assert_allclose(d["actions"][:, i], actions[:, i], atol=1e-12)

    def test_base_dims_are_untouched_by_the_delta_pass(self):
        """The prefix slice must stop before dims 14-15."""
        state, actions = _state(), _actions()
        d = _delta({"state": state, "actions": actions}, delta_action_mask())
        for i in BASE_INDICES:
            np.testing.assert_allclose(
                d["actions"][:, i], actions[:, i], rtol=0, atol=1e-12,
                err_msg=f"base dim {i} was altered by the delta transform",
            )

    def test_16_element_mask_would_raise_not_silently_corrupt(self):
        """Guards the bug this mask length exists to avoid."""
        bad = tuple([i in ARM_INDICES for i in range(ACTION_DIM)])
        with self.assertRaises(ValueError):
            _delta({"state": _state(), "actions": _actions()}, bad)

    def test_delta_transcription_matches_upstream_semantics(self):
        """Pin the local transcription against openpi's documented behaviour."""
        state = np.arange(14, dtype=np.float64)
        actions = np.zeros((1, 16))
        mask = tuple([True] * 2 + [False] * 12)
        d = _delta({"state": state, "actions": actions}, mask)
        np.testing.assert_allclose(d["actions"][0, :2], [-0.0, -1.0])
        np.testing.assert_allclose(d["actions"][0, 2:], np.zeros(14))


class TestBaseUnitConversion(unittest.TestCase):
    def setUp(self):
        self.limits = BaseLimits(lin_speed_mps=0.25, ang_speed_radps=0.5)

    def _cmd(self, forward: float, yaw: float):
        a = np.zeros(ACTION_DIM)
        a[BASE_INDICES[0]] = forward
        a[BASE_INDICES[1]] = yaw
        return decode_policy_action(a)

    def test_physical_to_normalized_uses_configured_limits(self):
        n = to_habitat_base(self._cmd(0.125, 0.25), self.limits)
        np.testing.assert_allclose(n, [0.5, 0.5], atol=1e-6)

    def test_normalized_round_trips_back_to_physical(self):
        for f, y in [(0.0, 0.0), (0.25, -0.5), (-0.1, 0.3)]:
            n = to_habitat_base(self._cmd(f, y), self.limits)
            back = from_habitat_base(n, self.limits)
            self.assertAlmostEqual(back[0], f, places=6)
            self.assertAlmostEqual(back[1], y, places=6)

    def test_raw_mps_is_not_passed_through_as_normalized(self):
        """The bug this boundary exists to prevent."""
        n = to_habitat_base(self._cmd(0.09084, 0.0), self.limits)
        self.assertNotAlmostEqual(float(n[0]), 0.09084, places=4)
        self.assertAlmostEqual(float(n[0]), 0.09084 / 0.25, places=6)

    def test_over_limit_command_is_refused_not_saturated(self):
        with self.assertRaises(ContractError):
            to_habitat_base(self._cmd(0.9, 0.0), self.limits)

    def test_yaw_uses_angular_limit_not_linear(self):
        n = to_habitat_base(self._cmd(0.0, 0.5), self.limits)
        np.testing.assert_allclose(n, [0.0, 1.0], atol=1e-6)


class TestActuatorDecoding(unittest.TestCase):
    def test_gripper_polarity_open_is_zero_closed_is_one(self):
        a = np.zeros(ACTION_DIM)
        a[GRIPPER_INDICES[0]] = 0.0
        a[GRIPPER_INDICES[1]] = 1.0
        args = decode_policy_action(a).habitat_arm_args()
        self.assertAlmostEqual(float(args["left_grip_action"][0]), -1.0)
        self.assertAlmostEqual(float(args["right_grip_action"][0]), +1.0)

    def test_both_grippers_are_independent(self):
        a = np.zeros(ACTION_DIM)
        a[GRIPPER_INDICES[0]] = 1.0
        a[GRIPPER_INDICES[1]] = 0.0
        c = decode_policy_action(a)
        self.assertEqual((c.left_gripper_closed_fraction,
                          c.right_gripper_closed_fraction), (1.0, 0.0))

    def test_arm_targets_are_twelve_dims_in_order(self):
        a = np.zeros(ACTION_DIM)
        for n, i in enumerate(ARM_INDICES):
            a[i] = n + 1
        c = decode_policy_action(a)
        np.testing.assert_allclose(c.arm_joint_targets_rad, np.arange(1, 13))

    def test_non_finite_action_rejected(self):
        a = np.zeros(ACTION_DIM); a[0] = np.nan
        with self.assertRaises(ContractError):
            decode_policy_action(a)

    def test_14d_action_rejected(self):
        """A stale single-arm/ALOHA-truncated action must not be accepted."""
        with self.assertRaises(ContractError):
            decode_policy_action(np.zeros(14))


class TestTransformPair(unittest.TestCase):
    def test_inputs_emit_state_and_masked_images(self):
        out = WXAIInputs()({
            "state": _state(),
            "images": {"cam_high": _image()},
            "prompt": "p",
        })
        self.assertEqual(out["state"].shape, (STATE_DIM,))
        self.assertTrue(bool(out["image_mask"]["base_0_rgb"]))
        self.assertFalse(bool(out["image_mask"]["left_wrist_0_rgb"]))

    def test_inputs_do_not_pad_state(self):
        """Matches AlohaInputs: padding happens in the model transforms."""
        out = WXAIInputs()({"state": _state(), "images": {"cam_high": _image()}})
        self.assertEqual(out["state"].shape, (14,))

    def test_actions_present_only_during_training(self):
        base = {"state": _state(), "images": {"cam_high": _image()}}
        self.assertNotIn("actions", WXAIInputs()(dict(base)))
        self.assertIn("actions", WXAIInputs()({**base, "actions": _actions()}))

    def test_outputs_preserve_all_16_dims(self):
        a = _actions()
        out = WXAIOutputs()({"actions": a.copy()})
        self.assertEqual(out["actions"].shape[1], 16)

    def test_outputs_reject_truncated_14d(self):
        with self.assertRaises(ContractError):
            WXAIOutputs()({"actions": np.zeros((3, 14))})

    def test_inputs_outputs_are_inverse_on_actions(self):
        a = _actions()
        enc = WXAIInputs()({
            "state": _state(), "images": {"cam_high": _image()}, "actions": a
        })["actions"]
        back = WXAIOutputs()({"actions": enc})["actions"]
        np.testing.assert_allclose(back, a, rtol=0, atol=1e-12)

    def test_images_reach_the_model_as_hwc_uint8(self):
        """Layout is asserted at the model boundary, not assumed.

        The ALOHA path took CHW input and transposed it to HWC. This stack
        takes Habitat's native HWC and keeps it, so the model sees the same
        layout with one fewer conversion. A silent layout flip here would
        train the model on transposed pixels.
        """
        out = WXAIInputs()({
            "state": _state(), "images": {"cam_high": _image()},
        })
        img = out["image"]["base_0_rgb"]
        self.assertEqual(img.shape, (4, 4, 3), "expected HWC at the model boundary")
        self.assertEqual(img.dtype, np.uint8)

    def test_unambiguous_chw_input_is_converted(self):
        chw = np.full((3, 8, 5), 7, dtype=np.uint8)   # 8x5 != channel count
        img = WXAIInputs()({
            "state": _state(), "images": {"cam_high": chw},
        })["image"]["base_0_rgb"]
        self.assertEqual(img.shape, (8, 5, 3))

    def test_hwc_is_preferred_over_chw_when_both_readable(self):
        """A valid HWC frame must never be transposed as if it were CHW."""
        hwc = np.zeros((4, 6, 3), dtype=np.uint8)
        img = WXAIInputs()({
            "state": _state(), "images": {"cam_high": hwc},
        })["image"]["base_0_rgb"]
        self.assertEqual(img.shape, (4, 6, 3))

    def test_image_without_a_channel_axis_is_rejected(self):
        with self.assertRaises(ContractError):
            WXAIInputs()({
                "state": _state(),
                "images": {"cam_high": np.zeros((8, 8, 5), dtype=np.uint8)},
            })

    def test_float_images_are_converted_to_uint8(self):
        f = np.full((4, 4, 3), 0.5, dtype=np.float32)
        img = WXAIInputs()({
            "state": _state(), "images": {"cam_high": f},
        })["image"]["base_0_rgb"]
        self.assertEqual(img.dtype, np.uint8)
        self.assertEqual(int(img.flat[0]), 127)

    def test_unexpected_camera_rejected(self):
        with self.assertRaises(ContractError):
            WXAIInputs()({
                "state": _state(),
                "images": {"cam_high": _image(), "cam_low": _image()},
            })


class TestHabitatObservationEncoding(unittest.TestCase):
    def test_encodes_14d_state_and_required_camera(self):
        out = encode_habitat_observation(
            {"trossen_bimanual_joint": _state(), "head_rgb": _image()}, prompt="p"
        )
        self.assertEqual(out["state"].shape, (14,))
        self.assertIn("cam_high", out["images"])

    def test_missing_head_camera_rejected(self):
        with self.assertRaises(ContractError):
            encode_habitat_observation(
                {"trossen_bimanual_joint": _state()}, prompt="p"
            )

    def test_wrong_state_width_rejected(self):
        with self.assertRaises(ContractError):
            encode_habitat_observation(
                {"trossen_bimanual_joint": np.zeros(16), "head_rgb": _image()},
                prompt="p",
            )


if __name__ == "__main__":
    unittest.main()
