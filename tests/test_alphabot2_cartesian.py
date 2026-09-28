"""Cartesian mobile interface: kinematics, checkpoint frames, IK round trips. CPU only."""

from __future__ import annotations

import math
import unittest

import numpy as np

from _helpers import REPO_ROOT  # noqa: F401

from lmm.embodiments.alphabot2 import cartesian as C
from lmm.embodiments.alphabot2 import kinematics as K
from lmm.embodiments.alphabot2 import pi05_contract as P
from lmm.embodiments.alphabot2.contract import ACTION_LABELS, JOINT_MOBILE_ACTION18_LABELS, ContractError
from lmm.embodiments.alphabot2.isaac_adapter import HOME_LEFT_ARM, HOME_RIGHT_ARM

HOME = {"left": np.array(HOME_LEFT_ARM), "right": np.array(HOME_RIGHT_ARM)}


class TestKinematics(unittest.TestCase):
    def test_home_matches_live_verified_pose(self):
        """Isaac measured gripper_base at (0.2957, +-0.227, 1.0534); xacro FK 3.7 mm away."""
        for side, y in (("left", 0.227), ("right", -0.227)):
            np.testing.assert_allclose(K.fk(side, HOME[side])[:3, 3], (0.292, y, 1.0532), atol=5e-4)
            np.testing.assert_allclose(K.fk(side, HOME[side])[:3, 2], (0.9999, 0.0, 0.0169), atol=2e-3)

    def test_arms_mirror_at_mirrored_home(self):
        ml, mr = K.fk("left", HOME["left"]), K.fk("right", HOME["right"])
        np.testing.assert_allclose(ml[:3, 3] * [1, -1, 1], mr[:3, 3], atol=1e-5)   # xacro +-0.01 offsets: 3 um

    def test_fk_ik_fk_round_trip(self):
        rng = np.random.default_rng(0)
        fails = 0
        for side in ("left", "right"):
            for _ in range(40):
                q = rng.uniform(K.JOINT_LIMITS[:, 0] + 0.15, K.JOINT_LIMITS[:, 1] - 0.15)
                target = K.fk(side, q)
                r = K.ik(side, target, q + rng.normal(0, 0.1, 7))
                pe, re = K.pose_error(K.fk(side, r.q), target)
                fails += not (pe < 1e-3 and re < 1e-2)
        self.assertLessEqual(fails, 2)       # near-singular draws may miss; never more

    def test_ik_respects_limits(self):
        far = K.fk("left", HOME["left"])
        far[:3, 3] += (2.0, 0.0, 0.0)        # unreachable
        r = K.ik("left", far, HOME["left"])
        self.assertFalse(r.converged)
        self.assertTrue(np.all(r.q >= K.JOINT_LIMITS[:, 0]) and np.all(r.q <= K.JOINT_LIMITS[:, 1]))

    def test_smooth_path_is_continuous(self):
        """Seeding with the previous solution: no joint jumps along a 20 cm line."""
        q = HOME["right"].copy()
        m = K.fk("right", q)
        for i in range(40):
            t = m.copy()
            t[:3, 3] += (0.005 * i, -0.0025 * i, 0.0)
            r = K.ik("right", t, q)
            self.assertTrue(r.converged)
            self.assertLess(r.max_step_rad, C.MAX_JOINT_STEP_RAD)
            q = r.q


class TestCheckpointContract(unittest.TestCase):
    def test_action_layout_is_the_public_one(self):
        self.assertEqual(P.ACTION_NAMES[0:3], ("left_arm.x", "left_arm.y", "left_arm.z"))
        self.assertEqual(P.ACTION_NAMES[3:7], ("left_arm.qx", "left_arm.qy", "left_arm.qz", "left_arm.qw"))
        self.assertEqual(P.ACTION_NAMES[14:], ("left_gripper.pos", "right_gripper.pos"))
        self.assertEqual(P.STATE_NAMES[14:], ("2FEG_l_Joint1.pos", "2FEG_r_Joint1.pos"))

    def test_frames_are_rotations(self):
        for r in (P.R_B_CHASSIS, P.R_GB_E["left"], P.R_GB_E["right"]):
            np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-9)
            self.assertAlmostEqual(np.linalg.det(r), 1.0, places=9)

    def test_tool_x_is_the_approach(self):
        for side in ("left", "right"):
            self.assertGreater(P.R_GB_E[side][:, 0] @ (0, 0, 1), 0.99)
            self.assertGreater(P.T_GB_E[side][2], 0.12)          # TCP ~13-14 cm out

    def test_action_gripper_base_round_trip(self):
        rng = np.random.default_rng(1)
        for side in ("left", "right"):
            for _ in range(20):
                q = rng.uniform(K.JOINT_LIMITS[:, 0] + 0.1, K.JOINT_LIMITS[:, 1] - 0.1)
                m = K.fk(side, q)
                xyz, quat = P.action_from_gripper_base(side, m)
                np.testing.assert_allclose(P.gripper_base_from_action(side, xyz, quat), m, atol=1e-9)
                self.assertGreaterEqual(quat[3], 0.0)

    def test_home_lies_in_the_training_action_range(self):
        """Home poses map inside the dataset's action range (a frame sanity check)."""
        xl, _ = P.action_from_vendor_joints("left", HOME["left"])
        xr, _ = P.action_from_vendor_joints("right", HOME["right"])
        self.assertTrue(0.3 < xl[0] < 0.8 and xl[1] > 0.1 and 0.5 < xl[2] < 1.0)
        self.assertTrue(0.3 < xr[0] < 0.8 and xr[1] < -0.1 and 0.5 < xr[2] < 1.0)

    def test_quaternion_canonicalisation(self):
        q = np.array([0.1, 0.2, 0.3, -0.9])
        self.assertGreaterEqual(P.canonical_quat(q)[3], 0)
        prev = -P.canonical_quat(q)
        np.testing.assert_allclose(P.canonical_quat(q, prev), prev, atol=1e-12)
        self.assertAlmostEqual(np.linalg.norm(P.canonical_quat(2 * q)), 1.0)
        with self.assertRaises(ValueError):
            P.canonical_quat(np.zeros(4))

    def test_dataset_state_conversion(self):
        p = np.zeros(16)
        p[0:7] = HOME["left"]; p[8:15] = HOME["right"]
        s = P.dataset_state_from_proprio(p)                 # grippers open (fraction 0)
        np.testing.assert_allclose(s[0:7], HOME["left"])
        np.testing.assert_allclose(s[7:14], P.DATASET_JOINT_SIGN["right"] * HOME["right"])
        self.assertAlmostEqual(s[14], 0.7959, places=4)     # dataset "open" readings
        self.assertAlmostEqual(s[15], 0.8122, places=4)
        p[15] = 1.0
        self.assertAlmostEqual(P.dataset_state_from_proprio(p)[15], 0.0)

    def test_right_sign_map_respects_the_xacro_limits(self):
        """The dataset's right j7 range [-0.489, 0.914] fits the xacro [-2.09, 0.49] only flipped."""
        self.assertEqual(P.DATASET_JOINT_SIGN["right"][6], -1.0)
        self.assertLessEqual(-(-0.4888), K.JOINT_LIMITS[6, 1] + 1e-3)

    def test_camera_mapping_crosses_the_wrists(self):
        self.assertEqual(P.SIM_CAMERA_FOR["camera_left"], "cam_right_wrist")
        self.assertEqual(P.SIM_CAMERA_FOR["camera_right"], "cam_left_wrist")


class TestCartesianMobileAction18(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(len(C.CARTESIAN_MOBILE_ACTION18_LABELS), 18)
        self.assertEqual(C.CARTESIAN_MOBILE_ACTION18_LABELS[16:], ("base.v", "base.omega"))
        self.assertEqual(JOINT_MOBILE_ACTION18_LABELS, ACTION_LABELS)
        self.assertNotEqual(C.CARTESIAN_MOBILE_ACTION18_LABELS, JOINT_MOBILE_ACTION18_LABELS)

    def test_lowering_round_trip_from_home(self):
        j = np.zeros(18)
        j[0:7], j[8:15] = HOME["left"], HOME["right"]
        j[15], j[16], j[17] = 1.0, 0.2, -0.3
        cart = C.cartesian_from_joint_action(j)
        ex = C.CartesianExecutor(q={"left": HOME["left"] + 0.05, "right": HOME["right"] - 0.05})
        out, rep = ex.lower(cart)
        self.assertTrue(rep["left"]["ik_valid"] and rep["right"]["ik_valid"])
        for sl in (slice(0, 7), slice(8, 15)):
            side = "left" if sl.start == 0 else "right"
            np.testing.assert_allclose(K.fk(side, out[sl]), K.fk(side, j[sl]), atol=2e-4)
        self.assertEqual((out[15], out[16], out[17]), (1.0, 0.2, -0.3))

    def test_base_dims_pass_through_untouched(self):
        cart = C.cartesian_from_joint_action(np.r_[HOME["left"], 0.0, HOME["right"], 0.0, 0.0, 0.0])
        cart[16], cart[17] = -0.15, 0.4
        out, _ = C.CartesianExecutor(q={"left": HOME["left"], "right": HOME["right"]}).lower(cart)
        self.assertEqual((out[16], out[17]), (-0.15, 0.4))

    def test_unnormalised_and_flipped_quaternions_execute_identically(self):
        cart = C.cartesian_from_joint_action(np.r_[HOME["left"], 0.0, HOME["right"], 0.0, 0.0, 0.0])
        alt = cart.copy()
        alt[3:7] *= -1.7                                    # q == -q, any scale
        o1, _ = C.CartesianExecutor(q={"left": HOME["left"], "right": HOME["right"]}).lower(cart)
        o2, r2 = C.CartesianExecutor(q={"left": HOME["left"], "right": HOME["right"]}).lower(alt)
        np.testing.assert_allclose(o1, o2, atol=1e-9)
        self.assertAlmostEqual(r2["left"]["quat_norm_in"], 1.7, places=6)

    def test_unreachable_target_is_reported_not_hidden(self):
        cart = C.cartesian_from_joint_action(np.r_[HOME["left"], 0.0, HOME["right"], 0.0, 0.0, 0.0])
        cart[7] += 1.5                                      # right target 1.5 m further out
        _, rep = C.CartesianExecutor(q={"left": HOME["left"], "right": HOME["right"]}).lower(cart)
        self.assertFalse(rep["right"]["ik_valid"])
        self.assertTrue(rep["left"]["ik_valid"])

    def test_bad_input_rejected(self):
        ex = C.CartesianExecutor(q={"left": HOME["left"], "right": HOME["right"]})
        with self.assertRaises(ContractError):
            ex.lower(np.zeros(16))
        bad = np.zeros(18)
        bad[0] = math.nan
        with self.assertRaises(ContractError):
            ex.lower(bad)


if __name__ == "__main__":
    unittest.main()
