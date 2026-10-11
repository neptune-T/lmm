"""Action-chunk window exposure to the post-release settle interval (numpy)."""
import unittest

import numpy as np

from lmm.embodiments.alphabot2.ovmm_sft_windows import expected_pad, window_exposure

SETTLE = 9


def episodes(settle_lens, body=120):
    ep, st = [], []
    for i, n in enumerate(settle_lens):
        ep += [i] * (body + n)
        st += [0] * body + [SETTLE] * n
    return np.array(ep), np.array(st)


class TestWindows(unittest.TestCase):
    def test_pad_pattern(self):
        self.assertFalse(expected_pad(120, 0).any())
        p = expected_pad(120, 100)
        self.assertEqual(int(p.sum()), 30)
        self.assertTrue(p[20:].all() and not p[:20].any())

    def test_truncated_settle_bounds_exposure(self):
        ep, st = episodes([5, 5, 5])
        r = window_exposure(ep, st, SETTLE)
        self.assertLessEqual(r["max_settle_targets_per_window"], 5)

    def test_untruncated_settle_is_visible(self):
        ep, st = episodes([240, 72])                                      # the measured raw settle lengths
        r = window_exposure(ep, st, SETTLE)
        self.assertEqual(r["max_settle_targets_per_window"], 50)          # a full chunk of idle targets
        self.assertGreater(r["non_settle_windows_touching_settle_fraction"], 0.3)


if __name__ == "__main__":
    unittest.main()
