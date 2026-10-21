import unittest
from lmm.embodiments.alphabot2.timing_verdict import classify, summarize


class TimingVerdict(unittest.TestCase):
    def test_ambiguous_is_not_a_success(self):
        self.assertEqual(classify({-1: 0., 0: 0., 1: 1.053}, 0.), "ambiguous")
        rows = [{"status": "ambiguous", "negative_controls": {"-1": "ambiguous", "1": "shifted"}}] * 6
        self.assertFalse(summarize(rows)["pass"])

    def test_delayed_and_early_images_rejected(self):
        for offset in (-1, 1):
            d = {-1: 2., 0: 2., 1: 2.}
            d[offset] = 0.
            self.assertEqual(classify(d, 2.), "shifted")

    def test_absolute_mismatch_and_nonfinite_rejected(self):
        self.assertEqual(classify({-1: 4., 0: 1., 1: 3.}, 2.), "mismatch")
        self.assertEqual(classify({-1: 4., 0: float('nan'), 1: 3.}, 2.), "invalid")

    def test_minimum_evidence_and_controls(self):
        row = {"status": "aligned", "negative_controls": {"-1": "shifted", "1": "shifted"}}
        self.assertFalse(summarize([row] * 4)["pass"])
        self.assertTrue(summarize([row] * 5)["pass"])
        bad = {"status": "aligned", "negative_controls": {"-1": "aligned", "1": "shifted"}}
        self.assertFalse(summarize([row] * 5 + [bad])["pass"])
        shifted = {"status": "shifted", "negative_controls": {"-1": "shifted", "1": "shifted"}}
        self.assertFalse(summarize([row] * 5 + [shifted])["pass"])

if __name__ == '__main__':
    unittest.main()
