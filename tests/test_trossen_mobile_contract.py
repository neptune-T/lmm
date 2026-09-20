from __future__ import annotations

import unittest

import numpy as np

from lmm.env.trossen_mobile_contract import (
    ACTION_LABELS,
    STATE_LABELS,
    assert_differential_base_executable,
    decode_openpi_action,
    encode_openpi_observation,
)


class TestTrossenMobileContract(unittest.TestCase):
    def test_every_dimension_has_one_stable_label(self):
        self.assertEqual(len(STATE_LABELS), 14)
        self.assertEqual(len(ACTION_LABELS), 16)
        self.assertEqual(ACTION_LABELS[6], "left_gripper_closed_fraction")
        self.assertEqual(ACTION_LABELS[13], "right_gripper_closed_fraction")
        self.assertEqual(ACTION_LABELS[14:], ("base_velocity_x", "base_velocity_y"))

    def test_decode_preserves_native_interleaved_order(self):
        a = np.arange(16, dtype=np.float32) / 20
        a[6], a[13] = 0.0, 1.0
        cmd = decode_openpi_action(a)
        np.testing.assert_array_equal(cmd.arm_joint_targets_rad, np.r_[a[:6], a[7:13]])
        np.testing.assert_array_equal(cmd.base_velocity_xy, a[14:16])
        args = cmd.habitat_arm_args()
        np.testing.assert_array_equal(args["left_grip_action"], [-1.0])
        np.testing.assert_array_equal(args["right_grip_action"], [1.0])

    def test_grippers_are_closed_fractions(self):
        a = np.zeros(16, dtype=np.float32)
        a[6] = 1.01
        with self.assertRaisesRegex(ValueError, "closed fractions"):
            decode_openpi_action(a)

    def test_lateral_velocity_is_never_reinterpreted_as_yaw(self):
        a = np.zeros(16, dtype=np.float32)
        a[14] = 0.12
        cmd = decode_openpi_action(a)
        np.testing.assert_allclose(assert_differential_base_executable(cmd), [0.12, 0.0])
        a[15] = 0.02
        with self.assertRaisesRegex(RuntimeError, "lateral base velocity"):
            assert_differential_base_executable(decode_openpi_action(a))

    def test_observation_maps_hwc_to_aloha_chw(self):
        obs = {
            "trossen_bimanual_joint": np.zeros(14, dtype=np.float32),
            "head_rgb": np.zeros((8, 9, 4), dtype=np.uint8),
            "left_wrist_rgb": np.zeros((4, 5, 3), dtype=np.uint8),
            "right_wrist_rgb": np.zeros((6, 7, 3), dtype=np.uint8),
        }
        got = encode_openpi_observation(obs, prompt="pick")
        self.assertEqual(got["images"]["cam_high"].shape, (3, 8, 9))
        self.assertEqual(got["images"]["cam_left_wrist"].shape, (3, 4, 5))
        self.assertEqual(got["prompt"], "pick")


if __name__ == "__main__":
    unittest.main()
