"""Stage-1 task distribution, held-out splits and quality gates (CPU only)."""
import math
import unittest

import numpy as np

from lmm.embodiments.alphabot2 import stage1_gates as G
from lmm.embodiments.alphabot2 import stage1_tasks as T


class TestSplits(unittest.TestCase):
    def test_heldout_disjoint_and_every_level_in_train(self):
        train, val, test = (set(T.cells_of(s)) for s in T.SPLITS)
        self.assertFalse(train & val or train & test or val & test)
        self.assertEqual(len(train) + len(val) + len(test), 36)
        for axis, n in enumerate((4, 3, 3)):
            self.assertEqual({c[axis] for c in train}, set(range(n)), f"factor {axis} level missing from train")

    def test_generated_configs_respect_split_and_balance(self):
        n = {f: 40 for f in T.FAMILIES}
        cfgs = T.generate("train", n, seed=7)
        self.assertEqual(len(cfgs), 160)
        self.assertTrue(all(c.split == "train" and T.split_of(c.cells) == "train" for c in cfgs))
        for fam in T.FAMILIES:
            arms = [c.arm for c in cfgs if c.family == fam]
            self.assertEqual(arms.count("left"), arms.count("right"))
            objs = {c.object for c in cfgs if c.family == fam}
            self.assertEqual(objs, set(T.OBJECT_NAMES))
        ids = [c.config_id for c in cfgs]
        self.assertEqual(len(ids), len(set(ids)))
        test = T.generate("test", n, seed=7)
        self.assertTrue(all(T.split_of(c.cells) == "test" for c in test))
        self.assertFalse({c.cells for c in test} & {c.cells for c in cfgs})

    def test_deterministic(self):
        a = [c.to_dict() for c in T.generate("val", {"nav_pick_lift": 5}, seed=3)]
        b = [c.to_dict() for c in T.generate("val", {"nav_pick_lift": 5}, seed=3)]
        self.assertEqual(a, b)

    def test_geometry(self):
        rng = np.random.default_rng(0)
        for fam in T.FAMILIES:
            for cells in T.all_cells():
                c = T.sample_config(rng, fam, cells, "left" if cells[0] % 2 else "right", "red cube", "x", 0)
                edge = T.TABLE_A[0] - T.TABLE_A_DEPTH / 2
                self.assertTrue(edge < c.object_xy[0] < edge + 0.15)
                self.assertTrue(abs(c.object_xy[1]) < T.TABLE_A_WIDTH / 2 - 0.1)
                # the robot never starts inside or touching table A
                self.assertLess(c.base_xy[0], edge - 0.35, (fam, cells, c.base_xy))
                if fam == "reorient_pick_lift":
                    self.assertGreater(abs(c.base_yaw), 1.5)
                    self.assertLess(abs(c.base_yaw), math.pi)
                if fam == "short_pick_place":
                    self.assertEqual(c.box[3], "a")
                    self.assertGreaterEqual(abs(c.box[1] - c.object_xy[1]), 0.15)
                else:
                    self.assertEqual(c.box[3], "b")
                self.assertIn(c.instruction.split()[2], ("left", "right"))

    def test_instructions(self):
        self.assertIn("put it in the blue box", T.instruction("short_pick_place", "left", "red cube"))
        self.assertIn("lift it", T.instruction("reorient_pick_lift", "right", "purple bar"))


def _episode(T_=12, lag=0):
    f = np.arange(T_) * 20 + 200
    low = {"proprio": np.zeros((T_, 16)), "action": np.tile(np.r_[np.zeros(3), [0, 0, 0, 1.0], np.zeros(3), [0, 0, 0, 1.0], 0, 0, 0, 0], (T_, 1)),
           "image_frame": f - 20 * lag, "proprio_frame": f - 20 * lag, "obs_frame": f, "action_frame": f,
           "policy_step": np.arange(T_), "ik_valid": np.ones((T_, 2), bool), "collision": np.zeros(T_, bool)}
    rng = np.random.default_rng(0)
    imgs = {k: rng.integers(0, 255, (T_, 240, 320, 3), dtype=np.uint8) for k in G.CAMERAS}
    meta = {"outcome": {"success": True}, "sync_check": {"ok": True}, "render_lag": lag,
            "policy_inputs": ["cam_front", "proprio", "instruction"]}
    rp = {"success": True, "proprio_max_abs_diff": 0.0}
    return low, imgs, meta, rp


class TestGates(unittest.TestCase):
    def test_clean_episode_passes(self):
        for lag in (0, 1):
            g = G.gate_episode(*_episode(lag=lag))
            self.assertTrue(G.passed(g), G.failing(g))

    def test_each_defect_is_caught(self):
        low, imgs, meta, rp = _episode()
        cases = {
            "temporal_sync": lambda: low.__setitem__("image_frame", low["image_frame"] - 20),
            "collision_free": lambda: low["collision"].__setitem__(3, True),
            "ik_valid": lambda: low["ik_valid"].__setitem__((2, 1), False),
            "finite": lambda: low["action"].__setitem__((1, 5), np.nan),
            "cameras": lambda: imgs["cam_chest"].__setitem__(4, 0),
            "replay": lambda: rp.__setitem__("proprio_max_abs_diff", 1e-3),
            "task_success": lambda: meta["outcome"].__setitem__("success", False),
            "no_privileged_inputs": lambda: meta["policy_inputs"].append("stage_id"),
        }
        for gate, corrupt in cases.items():
            low, imgs, meta, rp = _episode()
            corrupt()
            g = G.gate_episode(low, imgs, meta, rp)
            self.assertIn(gate, G.failing(g), gate)

    def test_unnormalized_quaternion_fails(self):
        low, imgs, meta, rp = _episode()
        low["action"][:, 3:7] *= 1.01
        self.assertIn("finite", G.failing(G.gate_episode(low, imgs, meta, rp)))


if __name__ == "__main__":
    unittest.main()
