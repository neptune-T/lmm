"""AlphaBot 2 OVMM <-> public pi0.5 interface: orderings, crossed wrists, inverses (CPU, numpy only)."""
import unittest

import numpy as np

from lmm.embodiments.alphabot2 import cartesian as C
from lmm.embodiments.alphabot2 import ovmm_policy_io as IO
from lmm.embodiments.alphabot2 import pi05_contract as P


class TestOVMMPolicyIO(unittest.TestCase):
    def test_state_round_trip(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            p = np.r_[rng.uniform(-2, 2, 7), rng.uniform(0, 1), rng.uniform(-2, 2, 7), rng.uniform(0, 1)]
            np.testing.assert_allclose(IO.proprio_from_state(IO.state_from_proprio(p)), p, atol=1e-12)
            s = rng.uniform(-2, 2, 16)
            np.testing.assert_allclose(IO.state_from_proprio(IO.proprio_from_state(s)), s, atol=1e-12)

    def test_state_ordering_unit_vectors(self):
        """Each proprio slot lands in the documented checkpoint slot (with the right-arm sign flips)."""
        expect = {}
        for i in range(7):
            expect[i] = (i, 1.0)                                          # left joints -> 0..6, same sign
            expect[8 + i] = (7 + i, P.DATASET_JOINT_SIGN["right"][i])     # right joints -> 7..13, j3/5/7 flipped
        base = np.zeros(16)
        base[7] = base[15] = 1.0                                          # both grippers closed -> readings 0
        for src, (dst, sign) in expect.items():
            p = base.copy()
            p[src] = 0.5
            s = IO.state_from_proprio(p)
            self.assertAlmostEqual(s[dst], 0.5 * sign, places=12, msg=f"proprio[{src}] -> state[{dst}]")
            self.assertEqual(int(np.count_nonzero(np.abs(s) > 1e-12)), 1, msg=f"proprio[{src}] leaked")
        self.assertEqual(P.STATE_NAMES[2], "ZPF_left_Joint3.pos")
        self.assertEqual(P.STATE_NAMES[9], "ZPF_right_Joint3.pos")
        # grippers: open (fraction 0) -> the dataset's open readings in slots 14 / 15
        s = IO.state_from_proprio(np.zeros(16))
        self.assertAlmostEqual(s[14], P.DATASET_GRIPPER_OPEN_READING["left"], places=9)
        self.assertAlmostEqual(s[15], P.DATASET_GRIPPER_OPEN_READING["right"], places=9)

    def test_cameras_crossed_and_masked(self):
        self.assertEqual(IO.CAMERA_MAP["camera_left"], "right_wrist_rgb")
        self.assertEqual(IO.CAMERA_MAP["camera_right"], "left_wrist_rgb")
        self.assertEqual(IO.CAMERA_MAP["camera_top"], "head_rgb")
        self.assertIn("RIGHT", P.CAMERA_ROLES["camera_left"])
        self.assertIn("LEFT", P.CAMERA_ROLES["camera_right"])
        self.assertEqual(set(IO.CAMERA_MAP) | set(IO.MASKED_CAMERAS), set(IO.MODEL_CAMERA_ORDER))
        obs = {k: np.full((240, 320, 4), i, np.uint8) for i, k in enumerate(("head_rgb", "left_wrist_rgb", "right_wrist_rgb"))}
        im = IO.policy_images(obs)
        self.assertEqual(int(im["camera_left"][0, 0, 0]), 2)               # right wrist
        self.assertEqual(int(im["camera_right"][0, 0, 0]), 1)              # left wrist
        self.assertEqual(im["camera_top"].shape, (240, 320, 3))

    def test_action_labels(self):
        self.assertEqual(IO.ACTION_LABELS[:16], P.ACTION_NAMES)
        self.assertEqual(IO.ACTION_LABELS[16:], ("base.v", "base.omega"))
        self.assertEqual((C.CART_BASE_V, C.CART_BASE_OMEGA), (16, 17))
        self.assertEqual(IO.H_PRED, 50)

    def test_instruction_has_no_arm_or_stage_token(self):
        t = IO.instruction("toy_fire_truck", "cabinet", "stool")
        self.assertEqual(t, "Move the toy fire truck from the cabinet to the stool.")
        for w in ("left", "right", "arm", "hand", "grasp", "stage"):
            self.assertNotIn(w, t.lower().split())


if __name__ == "__main__":
    unittest.main()
