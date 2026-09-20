"""Root grouping, cheap/BF16 lineage, and checkpoint lineage."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from _helpers import FakeVerificationBackend  # noqa: E402
from lmm.rl.advantages import (  # noqa: E402
    AdvantageConfig,
    GroupingError,
    degenerate_fraction,
    group_samples,
    normalize_groups,
)
from lmm.rl.checkpoint_manager import CheckpointManager, CheckpointRecord  # noqa: E402
from lmm.rl.dataset_builder import (  # noqa: E402
    DatasetConfig,
    audit_grouping,
    build_dataset,
    group_id_for,
)
from lmm.rl.grpo import GRPOConfig, evaluate_batch  # noqa: E402
from lmm.rl.reward import RewardConfig  # noqa: E402
from lmm.rollout.precision_backend import PrecisionError  # noqa: E402
from lmm.schemas import (  # noqa: E402
    CheapExplorationRecord,
    SnapshotRecord,
    TrainingSampleRecord,
    VerificationRecord,
)
from lmm.verification.bf16_verifier import (  # noqa: E402
    BF16Verifier,
    VerifierConfig,
    rank_candidates,
)


def _sample(root: str, group: str, rid: str, reward: float) -> TrainingSampleRecord:
    return TrainingSampleRecord(
        root_id=root, group_id=group, rollout_id=rid,
        policy_version="q0", precision_backend="bf16", reward=reward,
    )


def _verification(root: str, i: int, cheap: float, verified: bool) -> VerificationRecord:
    return VerificationRecord(
        source_root=root, cheap_rollout_id=f"cheap::{root}::{i}", cheap_score=cheap,
        cheap_outcome=cheap > 0.5, selection_rank=i,
        verification_rollout_id=f"bf16::{root}::{i}", verification_backend="bf16",
        verified_success=verified, policy_version="q0",
    )


class TestRootGrouping(unittest.TestCase):
    def test_siblings_share_a_group(self):
        groups = group_samples([
            _sample("r1", "g1", "a", 1.0), _sample("r1", "g1", "b", 0.0)
        ])
        self.assertEqual(len(groups["g1"]), 2)

    def test_group_spanning_two_roots_is_rejected(self):
        with self.assertRaises(GroupingError) as ctx:
            group_samples([_sample("r1", "g1", "a", 1.0), _sample("r2", "g1", "b", 0.0)])
        self.assertIn("spans multiple roots", str(ctx.exception))

    def test_missing_group_id_is_rejected_at_construction(self):
        """Group identity is mandatory in the schema, not merely checked later."""
        from lmm.schemas import SchemaError

        with self.assertRaises(SchemaError):
            _sample("r1", "", "a", 1.0)

    def test_grouping_rejects_blank_group_id_defensively(self):
        """group_samples re-checks, for records built by other means."""
        sample = _sample("r1", "g1", "a", 1.0)
        object.__setattr__(sample, "group_id", "")
        with self.assertRaises(GroupingError):
            group_samples([sample])

    def test_group_id_is_explicit_not_filename_derived(self):
        gid = group_id_for("root::ep1::s00030::q0", "q0")
        self.assertIn("root::ep1::s00030::q0", gid)
        self.assertTrue(gid.startswith("grp::"))

    def test_advantages_are_group_relative(self):
        samples = [
            _sample("r1", "g1", "a", 1.0), _sample("r1", "g1", "b", 0.0),
            _sample("r2", "g2", "c", 5.0), _sample("r2", "g2", "d", 4.0),
        ]
        out, stats = normalize_groups(samples, AdvantageConfig())
        by_id = {s.rollout_id: s.advantage for s in out}
        # Same within-group spread => same advantages despite different scales.
        self.assertAlmostEqual(by_id["a"], by_id["c"], places=5)
        self.assertGreater(by_id["a"], 0)
        self.assertLess(by_id["b"], 0)
        self.assertEqual({s.root_id for s in stats}, {"r1", "r2"})

    def test_degenerate_group_is_reported_not_silently_zeroed(self):
        """Reproduces BRANCH_SUPPORT_NOT_OBSERVED: all siblings fail."""
        samples = [_sample("r1", "g1", f"s{i}", 0.0) for i in range(8)]
        out, stats = normalize_groups(samples, AdvantageConfig())
        self.assertEqual(out, [], "degenerate groups must be dropped")
        self.assertTrue(stats[0].degenerate)
        self.assertEqual(degenerate_fraction(stats), 1.0)

    def test_cross_root_batching_preserves_per_root_stats(self):
        samples = [
            _sample("r1", "g1", "a", 1.0), _sample("r1", "g1", "b", 0.0),
            _sample("r2", "g2", "c", 1.0), _sample("r2", "g2", "d", 0.0),
        ]
        _, stats = normalize_groups(samples, AdvantageConfig())
        self.assertEqual(len(stats), 2)
        self.assertEqual({s.group_id: s.root_id for s in stats},
                         {"g1": "r1", "g2": "r2"})


class TestCheapToBF16Lineage(unittest.TestCase):
    def setUp(self):
        self.snapshot = SnapshotRecord(
            root_id="r1", episode_id="ep1", scene_id="sc1", snapshot_step=30,
            policy_version_that_generated_it="q0", candidate_reason="fixed_interval:30",
        )
        self.candidates = [
            CheapExplorationRecord(
                root_id="r1", group_id="g1", rollout_id=f"cheap::{i}",
                precision_backend="fp8_storage_bf16_compute", seed=i,
                policy_version="q0", cheap_score=i / 10.0, cheap_outcome=i > 5,
            )
            for i in range(10)
        ]

    def test_ranking_selects_top_k(self):
        top = rank_candidates(self.candidates, VerifierConfig(k_verify=3))
        self.assertEqual([c.rollout_id for c in top],
                         ["cheap::9", "cheap::8", "cheap::7"])

    def test_verification_preserves_full_lineage(self):
        verifier = BF16Verifier(FakeVerificationBackend({}), VerifierConfig(k_verify=3))
        records = verifier.verify(self.snapshot, self.candidates, "q0")
        self.assertEqual(len(records), 3)
        for rank, rec in enumerate(records):
            self.assertEqual(rec.source_root, "r1")
            self.assertEqual(rec.selection_rank, rank)
            self.assertTrue(rec.cheap_rollout_id.startswith("cheap::"))
            self.assertTrue(rec.verification_rollout_id.startswith("bf16::"))
            self.assertEqual(rec.verification_backend, "bf16")

    def test_cheap_and_verified_outcomes_stay_separate(self):
        verifier = BF16Verifier(FakeVerificationBackend({}), VerifierConfig(k_verify=4))
        for rec in verifier.verify(self.snapshot, self.candidates, "q0"):
            d = rec.to_dict()
            self.assertIn("cheap_outcome", d)
            self.assertIn("verified_success", d)

    def test_cheap_run_cannot_be_relabelled_as_its_own_verification(self):
        with self.assertRaises(Exception) as ctx:
            VerificationRecord(
                source_root="r1", cheap_rollout_id="x", verification_rollout_id="x",
                verification_backend="bf16", policy_version="q0",
            )
        self.assertIn("independent execution", str(ctx.exception))

    def test_cheap_backend_cannot_be_used_for_verification(self):
        with self.assertRaises(PrecisionError):
            VerifierConfig(backend="fp8_storage_bf16_compute")

    def test_foreign_root_candidates_rejected(self):
        alien = CheapExplorationRecord(
            root_id="OTHER", group_id="g1", rollout_id="x",
            precision_backend="fp8_storage_bf16_compute", policy_version="q0",
        )
        verifier = BF16Verifier(FakeVerificationBackend({}), VerifierConfig())
        with self.assertRaises(ValueError):
            verifier.verify(self.snapshot, [alien], "q0")


class TestDatasetBuilding(unittest.TestCase):
    def test_dataset_groups_by_root(self):
        verifications = (
            [_verification("r1", i, 0.9, i % 2 == 0) for i in range(4)]
            + [_verification("r2", i, 0.8, i == 0) for i in range(4)]
        )
        samples = build_dataset(verifications, RewardConfig(), DatasetConfig())
        audit = audit_grouping(samples)
        self.assertEqual(audit["n_roots"], 2)
        self.assertEqual(audit["n_groups"], 2)
        self.assertTrue(audit["all_groups_single_root"])

    def test_reward_uses_verified_not_cheap_outcome(self):
        # cheap_score high but verification failed => reward must be 0
        samples = build_dataset(
            [_verification("r1", i, 0.99, False) for i in range(2)],
            RewardConfig(), DatasetConfig(),
        )
        self.assertTrue(all(s.reward == 0.0 for s in samples))

    def test_cheap_backend_verification_rejected(self):
        bad = _verification("r1", 0, 0.9, True)
        bad.verification_backend = "fp8_storage_bf16_compute"
        with self.assertRaises(PrecisionError):
            build_dataset([bad, _verification("r1", 1, 0.9, True)],
                          RewardConfig(), DatasetConfig())


class TestGRPOObjective(unittest.TestCase):
    def test_zero_log_ratio_gives_zero_kl(self):
        samples = [_sample("r1", "g1", "a", 1.0), _sample("r1", "g1", "b", 0.0)]
        res = evaluate_batch(samples, [0.0, 0.0], GRPOConfig())
        self.assertAlmostEqual(res.approx_kl, 0.0)
        self.assertFalse(res.guard_tripped)

    def test_trust_region_guard_trips(self):
        samples = [_sample("r1", "g1", "a", 1.0)]
        res = evaluate_batch(samples, [1.5], GRPOConfig(max_approx_kl=0.05))
        self.assertTrue(res.guard_tripped)
        self.assertIn("rolled back", res.guard_reason)

    def test_approx_kl_is_non_negative(self):
        for lr in (-2.0, -0.5, 0.0, 0.5, 2.0):
            res = evaluate_batch([_sample("r", "g", "a", 1.0)], [lr], GRPOConfig())
            self.assertGreaterEqual(res.approx_kl, 0.0)

    def test_misaligned_log_ratios_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_batch([_sample("r", "g", "a", 1.0)], [0.0, 0.0], GRPOConfig())


class TestCheckpointLineage(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.mgr = CheckpointManager(Path(self._tmp.name) / "ledger.json")

    def tearDown(self):
        self._tmp.cleanup()

    def _reg(self, version: str, parent: str | None, step: int):
        return self.mgr.register(CheckpointRecord(
            policy_version=version, path=f"/ckpt/{version}", parent_version=parent,
            optimizer_step=step, created_utc="2026-09-19T00:00:00Z",
        ))

    def test_parent_child_chain(self):
        self._reg("q0", None, 0)
        self._reg("q1", "q0", 1)
        self._reg("q2", "q1", 2)
        self.assertEqual(self.mgr.lineage("q2"), ["q0", "q1", "q2"])

    def test_orphan_parent_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            self._reg("q5", "q4", 5)
        self.assertIn("not registered", str(ctx.exception))

    def test_duplicate_version_rejected(self):
        self._reg("q0", None, 0)
        with self.assertRaises(ValueError):
            self._reg("q0", None, 0)

    def test_ledger_persists_across_instances(self):
        self._reg("q0", None, 0)
        self._reg("q1", "q0", 1)
        reloaded = CheckpointManager(self.mgr.ledger_path)
        self.assertEqual(reloaded.lineage("q1"), ["q0", "q1"])


if __name__ == "__main__":
    unittest.main()
