"""Habitat AlphaBot 2 URDF vs the Isaac-qualified reference (CPU only, no Habitat)."""
import unittest

import numpy as np

from lmm.embodiments.alphabot2 import isaac_adapter as IA
from lmm.embodiments.alphabot2 import kinematics as K
from lmm.embodiments.alphabot2.habitat.constants import TORSO
from lmm.embodiments.alphabot2.habitat.urdf_kin import URDFKin


class TestHabitatURDF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.u = URDFKin()

    def test_structure(self):
        u = self.u
        self.assertEqual(u.root, "habitat_base")
        mov = u.movable()
        self.assertEqual(len(mov), 26)
        for s in ("left", "right"):
            for i in range(1, 8):
                self.assertIn(f"{s}_joint{i}", mov)
            self.assertIn(f"{s}_gripper_joint", mov)
        for fixed in ("body_joint1", "body_joint2", "body_joint3", "body_joint4", "head_joint1", "head_joint2"):
            self.assertEqual(u.joints[fixed].type, "fixed")

    def test_yup_root(self):
        # ROS z-up base mapped to Habitat y-up; forward stays +x
        T = self.u.pose("base_footprint")
        np.testing.assert_allclose(T[:3, :3] @ [0, 0, 1], [0, 1, 0], atol=1e-9)
        np.testing.assert_allclose(T[:3, :3] @ [1, 0, 0], [1, 0, 0], atol=1e-9)

    def test_arm_fk_equals_reference(self):
        rng = np.random.default_rng(0)
        for side in ("left", "right"):
            for _ in range(100):
                q = rng.uniform(K.JOINT_LIMITS[:, 0], K.JOINT_LIMITS[:, 1])
                T = self.u.pose(f"{side}_gripper_base", {f"{side}_joint{i + 1}": q[i] for i in range(7)}, base="base_link")
                R = K.fk_chassis(side, q, TORSO)          # arm frame A (reference) composed with the fixed torso
                self.assertLess(np.linalg.norm(T[:3, 3] - R[:3, 3]), 1e-9)
                self.assertLess(np.linalg.norm(K.rotvec(T[:3, :3].T @ R[:3, :3])), 1e-9)

    def test_arm_frame_is_torso_attached(self):
        # the crouched torso stays upright: arm frame A is a pure translation of base_link
        A = K.arm_frame_in_chassis(TORSO)
        np.testing.assert_allclose(A[:3, :3], np.eye(3), atol=1e-4)
        self.assertLess(A[2, 3], -0.3)
        np.testing.assert_allclose(K.arm_frame_in_chassis((0, 0, 0, 0)), np.eye(4), atol=1e-12)
        for b in range(4):
            self.assertTrue(K.BODY_LIMITS[b, 0] <= TORSO[b] <= K.BODY_LIMITS[b, 1])

    def test_joint_limits_equal_vendor(self):
        for side in ("left", "right"):
            for i, (lo, hi) in enumerate(K.JOINT_LIMITS):
                j = self.u.joints[f"{side}_joint{i + 1}"]
                self.assertAlmostEqual(j.lower, lo)
                self.assertAlmostEqual(j.upper, hi)

    def test_gripper_polarity_equals_isaac_measurement(self):
        for side in ("left", "right"):
            for q, sep in IA.MEASURED_PAD_SEPARATION_M.items():
                qq = {f"{side}_gripper_joint": q}
                a = self.u.pose(f"{side}_left_pad", qq, base=f"{side}_gripper_base")
                b = self.u.pose(f"{side}_right_pad", qq, base=f"{side}_gripper_base")
                self.assertAlmostEqual(float(np.linalg.norm(a[:3, 3] - b[:3, 3])), sep, places=3)

    def test_head_camera_posture(self):
        # canonical neck pitch 0.20 rad baked in: optical axis ~41.4 deg below horizontal (ROS base frame)
        T = self.u.pose("head_camera", base="base_link")
        z = T[:3, 2]
        self.assertAlmostEqual(np.degrees(np.arcsin(-z[2])), 41.4, delta=0.3)
        self.assertAlmostEqual(z[1], 0.0, places=4)      # the xacro writes pi/2 as 1.5708


if __name__ == "__main__":
    unittest.main()
