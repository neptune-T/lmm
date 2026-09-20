"""Whole canonical pipeline on fakes, plus schema/manifest/backend contracts.

Exercises: rollout -> snapshot -> probe -> frontier -> cheap scale -> BF16
verify -> RL update -> evaluation, with no GPU, simulator or policy server.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from _helpers import (  # noqa: E402
    FakeContinuationBackend,
    FakePolicy,
    FakeSimulator,
    FakeVerificationBackend,
)
from lmm.env.behavior_env import BehaviorEnv, BehaviorEnvConfig  # noqa: E402
from lmm.env.snapshot import SnapshotStore  # noqa: E402
from lmm.eval.fidelity_eval import evaluate as fidelity_evaluate  # noqa: E402
from lmm.eval.frontier_eval import evaluate as frontier_evaluate  # noqa: E402
from lmm.eval.frontier_eval import held_out_comparison  # noqa: E402
from lmm.eval.full_task_eval import EvalContractError  # noqa: E402
from lmm.eval.full_task_eval import evaluate as full_task_evaluate  # noqa: E402
from lmm.eval.full_task_eval import paired_comparison  # noqa: E402
from lmm.frontier.candidate_proposer import CandidateProposerConfig, propose  # noqa: E402
from lmm.frontier.continuation_probe import ContinuationProbeRunner, ProbeConfig  # noqa: E402
from lmm.frontier.frontier_estimator import estimate_all  # noqa: E402
from lmm.frontier.frontier_selector import FrontierSelectorConfig, frontier_only  # noqa: E402
from lmm.instrumentation.event_detector import EventDetector  # noqa: E402
from lmm.manifest import ExperimentManifest, ManifestError  # noqa: E402
from lmm.policy.pi05_policy import OVMM_ACTION_SPEC  # noqa: E402
from lmm.rl.advantages import AdvantageConfig  # noqa: E402
from lmm.rl.dataset_builder import DatasetConfig  # noqa: E402
from lmm.rl.grpo import GRPOConfig, GRPOStepResult  # noqa: E402
from lmm.rl.reward import RewardConfig  # noqa: E402
from lmm.rl.trainer import OnlineTrainer, TrainerConfig  # noqa: E402
from lmm.rollout.runner import RolloutConfig, RolloutRunner  # noqa: E402
from lmm.schemas import (  # noqa: E402
    CheapExplorationRecord,
    FullRolloutRecord,
    StepRecord,
    read_jsonl,
    write_jsonl,
)
from lmm.verification.bf16_verifier import BF16Verifier, VerifierConfig  # noqa: E402


class _StubBackend:
    """Policy backend for the trainer: constant ratios, version bump."""

    def __init__(self, log_ratio: float = 0.0) -> None:
        self.log_ratio = log_ratio
        self.applied = 0

    def log_ratios(self, samples):
        return [self.log_ratio] * len(samples)

    def apply_update(self, samples, result: GRPOStepResult) -> str:
        self.applied += 1
        return f"q{self.applied}"

    def rollback(self, policy_version: str) -> None:
        pass


def _trainer_config(**kw):
    return TrainerConfig(
        reward=RewardConfig(), dataset=DatasetConfig(),
        advantage=AdvantageConfig(), grpo=GRPOConfig(**kw),
    )


class TestFullPipeline(unittest.TestCase):
    def test_pipeline_runs_end_to_end(self):
        # 1. full-task rollout with passive instrumentation
        sim = FakeSimulator(steps_to_done=100)
        env = BehaviorEnv(
            BehaviorEnvConfig(benchmark_root="/x", max_steps=100),
            OVMM_ACTION_SPEC, backend=sim, detector=EventDetector(),
        )
        runner = RolloutRunner(env, FakePolicy(OVMM_ACTION_SPEC),
                               RolloutConfig(max_steps=100, task="pick_and_place"))
        rollout = runner.run_full_task(
            rollout_id="full::1", episode_id="ep1", scene_id="sc1", seed=0)
        self.assertGreater(rollout.trajectory_length, 0)

        # 2. candidate snapshots at fixed intervals
        cands = propose(rollout.trajectory_length,
                        CandidateProposerConfig(interval=20, max_candidates=5))
        self.assertGreaterEqual(len(cands), 4)

        with tempfile.TemporaryDirectory() as tmp:
            store = SnapshotStore(Path(tmp))
            snapshots = [
                store.save(sim, episode_id="ep1", scene_id="sc1", step=c.step,
                           policy_version="q0", candidate_reason=c.reason,
                           source_rollout_id=rollout.rollout_id)
                for c in cands
            ]

            # 3. cheap continuation probes -> V_hat
            rates = {snapshots[0].root_id: 0.0, snapshots[1].root_id: 0.5,
                     snapshots[2].root_id: 0.5, snapshots[3].root_id: 1.0}
            probe_runner = ContinuationProbeRunner(
                FakeContinuationBackend(rates),
                ProbeConfig(k_probe=8, k_cheap=16,
                            probe_backend="fp8_storage_bf16_compute"))
            probes = probe_runner.probe_many(snapshots[:4], "q0")
            summaries = estimate_all(probes)
            self.assertEqual(len(summaries), 4)

            # 4. frontier selection
            frontier = frontier_only(summaries, FrontierSelectorConfig(0.2, 0.8))
            self.assertTrue(frontier, "expected at least one frontier root")
            chosen = next(s for s in snapshots if s.root_id == frontier[0].root_id)

            # 5. cheap scaling from the frontier root
            explored = probe_runner.explore(chosen, "q0")
            self.assertEqual(len(explored), 16)
            candidates = [
                CheapExplorationRecord(
                    root_id=chosen.root_id,
                    group_id=f"grp::{chosen.root_id}::q0",
                    rollout_id=p.probe_id,
                    precision_backend=p.precision_backend, seed=p.seed,
                    policy_version="q0",
                    cheap_score=p.terminal_metrics.get("progress", 0.0),
                    cheap_outcome=p.terminal_success,
                )
                for p in explored
            ]

            # 6. BF16 verification with lineage
            verifier = BF16Verifier(FakeVerificationBackend({}),
                                    VerifierConfig(k_verify=6))
            verifications = verifier.verify(chosen, candidates, "q0")
            self.assertEqual(len(verifications), 6)
            self.assertTrue(all(v.verification_backend == "bf16" for v in verifications))

            # 7. online RL update
            backend = _StubBackend(log_ratio=0.01)
            outcome = OnlineTrainer(backend, _trainer_config()).update(
                verifications, policy_before="q0", optimizer_step=1)
            self.assertTrue(outcome.applied, outcome.reason)
            self.assertEqual(outcome.record.policy_before, "q0")
            self.assertIn(chosen.root_id, outcome.record.root_ids)
            self.assertEqual(len(outcome.record.group_ids), 1)

            # 8. evaluation
            fr = frontier_evaluate(probes, split="train")
            self.assertEqual(fr.n_roots, 4)
            fid = fidelity_evaluate(verifications, top_k=3)
            self.assertEqual(fid.n, 6)

    def test_all_failure_bank_blocks_the_update(self):
        """The BRANCH_SUPPORT_NOT_OBSERVED case must refuse to train."""
        from test_rl_and_lineage import _verification

        verifications = [_verification("r1", i, 0.5, False) for i in range(8)]
        outcome = OnlineTrainer(_StubBackend(), _trainer_config()).update(
            verifications, policy_before="q0", optimizer_step=1)
        self.assertFalse(outcome.applied)
        self.assertIn("all_groups_degenerate", outcome.reason)
        self.assertEqual(outcome.degenerate_group_fraction, 1.0)

    def test_trust_region_guard_blocks_the_update(self):
        from test_rl_and_lineage import _verification

        verifications = [_verification("r1", i, 0.5, i % 2 == 0) for i in range(8)]
        outcome = OnlineTrainer(
            _StubBackend(log_ratio=2.0), _trainer_config(max_approx_kl=0.05)
        ).update(verifications, policy_before="q0", optimizer_step=1)
        self.assertFalse(outcome.applied)
        self.assertIn("trust_region_guard", outcome.reason)


class TestEvaluationContracts(unittest.TestCase):
    def _rollouts(self, version: str, successes: list[bool]):
        return [
            FullRolloutRecord(
                rollout_id=f"{version}::{i}", episode_id=f"ep{i}", scene_id="sc1",
                task="t", policy_version=version, precision_backend="bf16",
                terminal_success=s,
            )
            for i, s in enumerate(successes)
        ]

    def test_full_task_eval_rejects_skill_sequencer(self):
        rec = self._rollouts("q0", [True])[0]
        rec.steps = [StepRecord(step=0, action=[0.0],
                                observation_meta={"skill_sequencer_active": True})]
        with self.assertRaises(EvalContractError) as ctx:
            full_task_evaluate([rec])
        self.assertIn("end-to-end VLA", str(ctx.exception))

    def test_full_task_eval_rejects_mixed_policy_versions(self):
        with self.assertRaises(EvalContractError):
            full_task_evaluate(self._rollouts("q0", [True]) + self._rollouts("q1", [False]))

    def test_paired_comparison_detects_no_improvement(self):
        before = self._rollouts("q0", [True, False, False, False])
        after = self._rollouts("q20", [False, False, False, False])
        result = paired_comparison(before, after, n_bootstrap=200)
        self.assertEqual(result["interpretation"], "NO_RETAINED_IMPROVEMENT")
        self.assertLess(result["mean_delta"], 0)

    def test_paired_comparison_detects_improvement(self):
        before = self._rollouts("q0", [False] * 8)
        after = self._rollouts("q20", [True] * 8)
        result = paired_comparison(before, after, n_bootstrap=200)
        self.assertEqual(result["interpretation"], "RETAINED_IMPROVEMENT")

    def test_frontier_and_full_task_results_are_labelled_distinctly(self):
        full = full_task_evaluate(self._rollouts("q0", [True, False]))
        self.assertEqual(full.to_dict()["eval_type"], "full_task_from_initial_state")

    def test_held_out_comparison_reports_gap(self):
        from test_frontier import _probes

        train = frontier_evaluate(_probes("r1", 8, 4), split="train")
        held = frontier_evaluate(_probes("r2", 8, 1), split="held_out")
        cmp = held_out_comparison(train, held)
        self.assertLess(cmp["delta"], 0)


class TestFidelityMetrics(unittest.TestCase):
    def test_perfect_agreement_gives_full_retention(self):
        from test_rl_and_lineage import _verification

        records = [_verification("r", i, i / 10.0, i >= 6) for i in range(10)]
        res = fidelity_evaluate(records, top_k=4)
        self.assertEqual(res.top_k_retention, 1.0)
        self.assertGreater(res.spearman_rho, 0.8)

    def test_inverted_ranking_is_detected(self):
        from test_rl_and_lineage import _verification

        records = [_verification("r", i, i / 10.0, i < 4) for i in range(10)]
        res = fidelity_evaluate(records, top_k=4)
        self.assertLess(res.spearman_rho, 0.0)
        self.assertEqual(res.top_k_retention, 0.0)


class TestSchemaRoundTrip(unittest.TestCase):
    def test_jsonl_round_trip_preserves_type_and_identity(self):
        rec = FullRolloutRecord(
            rollout_id="r1", episode_id="ep1", scene_id="sc1", task="t",
            policy_version="q0", precision_backend="bf16",
            steps=[StepRecord(step=0, action=[0.1, 0.2], events=["object_held"])],
            terminal_success=True,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            write_jsonl(path, [rec])
            back = list(read_jsonl(path))
        self.assertEqual(len(back), 1)
        self.assertIsInstance(back[0], FullRolloutRecord)
        self.assertEqual(back[0].rollout_id, "r1")
        self.assertTrue(back[0].terminal_success)


class TestManifest(unittest.TestCase):
    def _manifest(self, **kw):
        base = dict(experiment_name="frontier_probe_smoke_v1",
                    policy_checkpoint="/ckpt/q0", output_directory="/tmp/out",
                    git_commit="abc123")
        base.update(kw)
        return ExperimentManifest(**base)

    def test_valid_manifest_passes(self):
        self._manifest().validate()

    def test_missing_git_commit_rejected(self):
        with self.assertRaises(ManifestError) as ctx:
            self._manifest(git_commit="").validate()
        self.assertIn("cannot be reproduced", str(ctx.exception))

    def test_conclusion_asserting_name_rejected(self):
        for name in ("successful_frontier_v1", "optimal_scaling", "stable_rl_run"):
            with self.assertRaises(ManifestError) as ctx:
                self._manifest(experiment_name=name).validate()
            self.assertIn("asserts a conclusion", str(ctx.exception))

    def test_neutral_names_accepted(self):
        for name in ("frontier_estimator_v1", "candidate_selector_sweep",
                     "online_trainer_pilot"):
            self._manifest(experiment_name=name).validate()

    def test_unknown_backend_rejected(self):
        from lmm.rollout.precision_backend import PrecisionError

        with self.assertRaises(PrecisionError):
            self._manifest(precision_backend="fp8_magic").validate()

    def test_manifest_written_to_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._manifest(output_directory=tmp).write()
            self.assertTrue(path.is_file())
            self.assertEqual(
                ExperimentManifest.load(path).experiment_name,
                "frontier_probe_smoke_v1")


if __name__ == "__main__":
    unittest.main()
