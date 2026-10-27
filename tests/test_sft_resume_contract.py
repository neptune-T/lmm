"""CPU-only tests in the pi0.5 environment; no model weights or GPU allocation."""
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/alphabot2_pi05"))
from sft_ovmm import WeightedBatches, capture_rng, restore_rng, lr_at, checked_rank_state


class ResumeTests(unittest.TestCase):
    def test_ddp_rank_snapshot_rejects_wrong_lineage_or_missing_rng(self):
        good = dict(rank=1, world=2, step=2, config_sha256='frozen', sampler_next_micro_step=2,
                    rng={'torch': torch.Generator().get_state()})
        expected = dict(rank=1, world=2, step=2, config_sha='frozen', accum=1)
        self.assertIs(checked_rank_state(good, **expected), good)
        for key, value in [('rank', 0), ('world', 1), ('step', 1), ('config_sha256', 'other'),
                           ('sampler_next_micro_step', 0), ('rng', None)]:
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                checked_rank_state({**good, key: value}, **expected)

    def test_sampler_resume_matches_next_accumulated_batches(self):
        stream = iter(WeightedBatches([1., 2., 8., 3.], 8, 1000, 0, 0))
        for _ in range(14):  # seven optimizer steps, two microbatches per step
            next(stream)
        expected = [next(stream) for _ in range(6)]
        resumed = iter(WeightedBatches([1., 2., 8., 3.], 8, 1000, 0, 14))
        self.assertEqual(expected, [next(resumed) for _ in range(6)])

    def test_rng_reload_restores_next_draws(self):
        # CUDA API is mocked so this test remains CPU-only on a shared host.
        fake_cuda = torch.Generator().manual_seed(42).get_state()
        with patch("torch.cuda.get_rng_state", return_value=fake_cuda), \
                patch("torch.cuda.set_rng_state") as set_cuda:
            state = capture_rng()
            expected = (torch.rand(8), np.random.random(8), [random.random() for _ in range(8)])
            torch.rand(100); np.random.random(100); random.random()
            restore_rng(state)
            self.assertTrue(torch.equal(expected[0], torch.rand(8)))
            np.testing.assert_array_equal(expected[1], np.random.random(8))
            self.assertEqual(expected[2], [random.random() for _ in range(8)])
            self.assertTrue(torch.equal(set_cuda.call_args.args[0], fake_cuda))

    def test_checkpoint_boundary_keeps_full_lr_horizon(self):
        full = [lr_at(i, 2.5e-5, 1000, 10000, 2.5e-6) for i in range(2001)]
        resumed = [lr_at(i, 2.5e-5, 1000, 10000, 2.5e-6) for i in range(1000, 2001)]
        self.assertEqual(full[1000:], resumed)
        self.assertNotEqual(full[2000], lr_at(2000, 2.5e-5, 1000, 2000, 2.5e-6))


if __name__ == "__main__":
    unittest.main()
