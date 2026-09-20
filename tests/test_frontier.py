"""Frontier discovery: proposal, V_hat estimation, selection, stochasticity."""

from __future__ import annotations

import unittest

from _helpers import FakeContinuationBackend  # noqa: E402
from lmm.frontier.candidate_proposer import (  # noqa: E402
    CandidateProposerConfig,
    propose,
    refine_around,
)
from lmm.frontier.continuation_probe import (  # noqa: E402
    ContinuationProbeRunner,
    ProbeConfig,
)
from lmm.frontier.frontier_estimator import (  # noqa: E402
    EstimationError,
    estimate_all,
    estimate_root,
    wilson_interval,
)
from lmm.frontier.frontier_selector import (  # noqa: E402
    FrontierSelectorConfig,
    classify,
    select,
)
from lmm.schemas import ProbeRecord, SnapshotRecord  # noqa: E402


def _snapshot(root_id: str, step: int = 30) -> SnapshotRecord:
    return SnapshotRecord(
        root_id=root_id,
        episode_id="ep1",
        scene_id="sc1",
        snapshot_step=step,
        policy_version_that_generated_it="q0",
        candidate_reason="fixed_interval:30",
    )


def _probes(root_id: str, n: int, n_success: int) -> list[ProbeRecord]:
    return [
        ProbeRecord(
            root_id=root_id,
            probe_id=f"{root_id}::probe::{i}",
            precision_backend="bf16",
            seed=i,
            policy_version="q0",
            terminal_success=i < n_success,
        )
        for i in range(n)
    ]


class TestCandidateProposal(unittest.TestCase):
    def test_fixed_interval(self):
        cands = propose(150, CandidateProposerConfig(interval=30))
        self.assertEqual([c.step for c in cands], [0, 30, 60, 90, 120])
        self.assertTrue(all(c.reason == "fixed_interval:30" for c in cands))

    def test_max_candidates_thins_evenly(self):
        cands = propose(1000, CandidateProposerConfig(interval=10, max_candidates=8))
        self.assertEqual(len(cands), 8)

    def test_semantic_event_strategy(self):
        events = [[] for _ in range(10)]
        events[3] = ["object_held"]
        events[7] = ["object_released"]
        cands = propose(
            10,
            CandidateProposerConfig(strategy="semantic_event", event_names=("object_held",)),
            step_events=events,
        )
        self.assertEqual([c.step for c in cands], [3])
        self.assertEqual(cands[0].reason, "semantic_event:object_held")

    def test_coarse_to_fine_refinement(self):
        fine = refine_around(90, coarse_interval=30, trajectory_length=200, fine_interval=10)
        steps = [c.step for c in fine]
        self.assertTrue(set([60, 70, 80, 90, 100, 110]).issubset(set(steps)))
        self.assertTrue(all(c.reason == "refine:90" for c in fine))

    def test_unknown_strategy_rejected(self):
        with self.assertRaises(ValueError):
            CandidateProposerConfig(strategy="whatever")


class TestVHatEstimation(unittest.TestCase):
    def test_success_rate(self):
        s = estimate_root("r1", _probes("r1", 8, 4))
        self.assertEqual(s.n_probes, 8)
        self.assertEqual(s.n_success, 4)
        self.assertAlmostEqual(s.success_rate, 0.5)

    def test_wilson_interval_is_non_degenerate_at_zero(self):
        lo, hi = wilson_interval(0, 4)
        self.assertEqual(lo, 0.0)
        self.assertGreater(hi, 0.0, "a 0/4 estimate must not claim certainty")

    def test_single_probe_is_refused(self):
        with self.assertRaises(EstimationError) as ctx:
            estimate_root("r1", _probes("r1", 1, 0))
        self.assertIn("single continuation", str(ctx.exception))

    def test_mixed_backends_refused(self):
        probes = _probes("r1", 4, 2)
        probes[0].precision_backend = "fp8_storage_bf16_compute"
        with self.assertRaises(EstimationError) as ctx:
            estimate_root("r1", probes)
        self.assertIn("mix precision backends", str(ctx.exception))

    def test_mixed_policy_versions_refused(self):
        probes = _probes("r1", 4, 2)
        probes[0].policy_version = "q1"
        with self.assertRaises(EstimationError):
            estimate_root("r1", probes)

    def test_foreign_root_probes_refused(self):
        probes = _probes("r1", 4, 2) + _probes("r2", 2, 1)
        with self.assertRaises(EstimationError):
            estimate_root("r1", probes)

    def test_estimate_all_groups_by_record_field(self):
        summaries = estimate_all(_probes("r1", 4, 2) + _probes("r2", 4, 0))
        self.assertEqual([s.root_id for s in summaries], ["r1", "r2"])
        self.assertAlmostEqual(summaries[0].success_rate, 0.5)
        self.assertAlmostEqual(summaries[1].success_rate, 0.0)


class TestFrontierSelection(unittest.TestCase):
    def setUp(self):
        self.config = FrontierSelectorConfig(frontier_low=0.2, frontier_high=0.8)

    def test_solved_root_excluded(self):
        self.assertEqual(classify(estimate_root("r", _probes("r", 8, 8)), self.config),
                         "no_outcome_variance")

    def test_impossible_root_excluded(self):
        self.assertEqual(classify(estimate_root("r", _probes("r", 8, 0)), self.config),
                         "no_outcome_variance")

    def test_intermediate_root_is_frontier(self):
        self.assertEqual(classify(estimate_root("r", _probes("r", 8, 4)), self.config),
                         "frontier")

    def test_mostly_solved_is_classified_solved(self):
        # 7/8 = 0.875 > frontier_high, but still has outcome variance
        self.assertEqual(classify(estimate_root("r", _probes("r", 8, 7)), self.config),
                         "solved")

    def test_thresholds_are_configurable_not_hardcoded(self):
        s = estimate_root("r", _probes("r", 10, 9))  # 0.9
        self.assertEqual(classify(s, FrontierSelectorConfig(0.2, 0.8)), "solved")
        self.assertEqual(classify(s, FrontierSelectorConfig(0.2, 0.95)), "frontier")

    def test_selection_ranks_by_band_centre(self):
        summaries = [
            estimate_root("balanced", _probes("balanced", 8, 4)),
            estimate_root("skewed", _probes("skewed", 8, 6)),
        ]
        ranked = select(summaries, self.config)
        self.assertEqual(ranked[0].root_id, "balanced")
        self.assertTrue(all(s.selection_reason for s in ranked))

    def test_invalid_band_rejected(self):
        with self.assertRaises(ValueError):
            FrontierSelectorConfig(frontier_low=0.9, frontier_high=0.1)


class TestControlledStochasticity(unittest.TestCase):
    def test_probes_use_distinct_seeds(self):
        runner = ContinuationProbeRunner(
            FakeContinuationBackend({"r1": 0.5}), ProbeConfig(k_probe=8)
        )
        records = runner.probe(_snapshot("r1"), "q0")
        self.assertEqual(len({r.seed for r in records}), 8)
        self.assertEqual(len({r.probe_id for r in records}), 8)

    def test_continuations_differ_when_intended(self):
        runner = ContinuationProbeRunner(
            FakeContinuationBackend({"r1": 0.5}), ProbeConfig(k_probe=16)
        )
        outcomes = {r.terminal_success for r in runner.probe(_snapshot("r1"), "q0")}
        self.assertEqual(outcomes, {True, False}, "stochastic probes must vary")

    def test_v_hat_recovers_ground_truth_rate(self):
        runner = ContinuationProbeRunner(
            FakeContinuationBackend({"r1": 0.5}), ProbeConfig(k_probe=200)
        )
        s = estimate_root("r1", runner.probe(_snapshot("r1"), "q0"))
        self.assertTrue(0.4 < s.success_rate < 0.6, f"V_hat={s.success_rate}")

    def test_probe_and_explore_share_one_backend(self):
        backend = FakeContinuationBackend({"r1": 0.5})
        runner = ContinuationProbeRunner(backend, ProbeConfig(k_probe=4, k_cheap=16))
        p = runner.probe(_snapshot("r1"), "q0")
        e = runner.explore(_snapshot("r1"), "q0")
        self.assertEqual([r.mode for r in p], ["probe"] * 4)
        self.assertEqual([r.mode for r in e], ["explore"] * 16)
        self.assertEqual(len(backend.calls), 20, "one backend served both roles")

    def test_k_probe_below_two_rejected(self):
        with self.assertRaises(ValueError):
            ProbeConfig(k_probe=1)


if __name__ == "__main__":
    unittest.main()
