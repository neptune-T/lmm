"""Snapshot, adapter, and action-contract tests (cleanup requirement 28)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from _helpers import FakeSimulator  # noqa: E402
from lmm.env.simulator_adapter import (  # noqa: E402
    AdapterConfig,
    GripperEvent,
    SimulatorAdapter,
)
from lmm.env.snapshot import SnapshotStore, root_id_for, state_digest  # noqa: E402
from lmm.policy.base import ActionSpec  # noqa: E402
from lmm.policy.pi05_policy import (  # noqa: E402
    BIMANUAL_OVMM_ACTION_SPEC,
    OVMM_ACTION_SPEC,
)


class TestActionContract(unittest.TestCase):
    def test_action_dimensionality(self):
        self.assertEqual(OVMM_ACTION_SPEC.total_dim, 11)
        self.assertEqual(OVMM_ACTION_SPEC.base_dim, 3)
        self.assertEqual(OVMM_ACTION_SPEC.arm_dim, 7)
        self.assertEqual(OVMM_ACTION_SPEC.gripper_dim, 1)

    def test_bimanual_ovmm_action_dimensionality(self):
        self.assertEqual(BIMANUAL_OVMM_ACTION_SPEC.total_dim, 16)
        self.assertEqual(BIMANUAL_OVMM_ACTION_SPEC.base_dim, 2)
        self.assertEqual(BIMANUAL_OVMM_ACTION_SPEC.arm_dim, 12)
        self.assertEqual(BIMANUAL_OVMM_ACTION_SPEC.gripper_dim, 2)

    def test_base_arm_gripper_mapping_is_positional(self):
        spec = ActionSpec(base_dim=3, arm_dim=7, gripper_dim=1)
        action = [1, 2, 3] + [10, 11, 12, 13, 14, 15, 16] + [0.9]
        parts = spec.split(action)
        self.assertEqual(list(parts["base"]), [1, 2, 3])
        self.assertEqual(list(parts["arm"]), [10, 11, 12, 13, 14, 15, 16])
        self.assertEqual(list(parts["gripper"]), [0.9])

    def test_wrong_dimensionality_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            OVMM_ACTION_SPEC.split([0.0] * 10)
        self.assertIn("expected 11", str(ctx.exception))

    def test_generic_adapter_fails_closed_for_two_grippers(self):
        adapter = SimulatorAdapter(action_spec=BIMANUAL_OVMM_ACTION_SPEC)
        with self.assertRaisesRegex(NotImplementedError, "dual-gripper Habitat backend"):
            adapter.to_command([0.0] * 16)


class TestSnapDesnapSemantics(unittest.TestCase):
    def setUp(self):
        self.sim = FakeSimulator()
        self.adapter = SimulatorAdapter(
            action_spec=OVMM_ACTION_SPEC,
            config=AdapterConfig(snap_threshold=0.5, desnap_threshold=0.3),
            backend=self.sim,
        )

    def _action(self, gripper: float):
        return [0.0] * 10 + [gripper]

    def test_open_gripper_produces_no_event(self):
        self.assertEqual(self.adapter.gripper_event(0.0), GripperEvent.NONE)

    def test_closing_gripper_snaps(self):
        self.adapter.apply(self._action(0.9))
        self.assertEqual(self.sim.snap_calls, 1)
        self.assertTrue(self.adapter.held)

    def test_holding_closed_does_not_resnap(self):
        self.adapter.apply(self._action(0.9))
        self.adapter.apply(self._action(0.9))
        self.assertEqual(self.sim.snap_calls, 1, "SNAP must fire on transition only")

    def test_opening_while_held_desnaps(self):
        self.adapter.apply(self._action(0.9))
        self.adapter.apply(self._action(0.0))
        self.assertEqual(self.sim.desnap_calls, 1)
        self.assertFalse(self.adapter.held)

    def test_hysteresis_band_holds_state(self):
        self.adapter.apply(self._action(0.9))
        self.adapter.apply(self._action(0.4))  # between desnap(0.3) and snap(0.5)
        self.assertEqual(self.sim.desnap_calls, 0)
        self.assertTrue(self.adapter.held)

    def test_inverted_thresholds_are_rejected(self):
        with self.assertRaises(ValueError):
            AdapterConfig(snap_threshold=0.2, desnap_threshold=0.8)

    def test_adapter_does_no_task_planning(self):
        """The adapter's command must depend only on the action and held-state."""
        a = SimulatorAdapter(action_spec=OVMM_ACTION_SPEC, backend=FakeSimulator())
        b = SimulatorAdapter(action_spec=OVMM_ACTION_SPEC, backend=FakeSimulator())
        act = [0.5] * 10 + [0.1]
        self.assertEqual(
            {k: str(v) for k, v in a.to_command(act).items()},
            {k: str(v) for k, v in b.to_command(act).items()},
        )


class TestSnapshotRestore(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = SnapshotStore(Path(self._tmp.name))
        self.sim = FakeSimulator()

    def tearDown(self):
        self._tmp.cleanup()

    def _snapshot(self, step: int = 7):
        for _ in range(step):
            self.sim.step({})
        return self.store.save(
            self.sim,
            episode_id="ep1",
            scene_id="sc1",
            step=step,
            policy_version="q0",
            candidate_reason="fixed_interval:30",
        )

    def test_save_then_restore_round_trips(self):
        rec = self._snapshot()
        self.assertTrue(Path(rec.snapshot_path).is_file())
        self.sim.state["t"] = 9999
        self.store.restore(self.sim, rec)
        self.assertEqual(self.sim.state["t"], 7)

    def test_same_root_restores_identically(self):
        """Repeated restores of one root must yield an identical digest."""
        rec = self._snapshot()
        digests = set()
        for _ in range(5):
            self.sim.state = {"t": -1}
            digests.add(self.store.restore(self.sim, rec))
            digests.add(state_digest(self.sim.capture_state()))
        self.assertEqual(len(digests), 1, f"restore is not deterministic: {digests}")

    def test_distinct_roots_have_distinct_ids(self):
        self.assertNotEqual(
            root_id_for("ep1", 30, "q0"), root_id_for("ep1", 60, "q0")
        )
        self.assertNotEqual(
            root_id_for("ep1", 30, "q0"), root_id_for("ep1", 30, "q1")
        )

    def test_root_id_carries_identity_without_filename_parsing(self):
        rec = self._snapshot(30)
        self.assertEqual(rec.episode_id, "ep1")
        self.assertEqual(rec.snapshot_step, 30)
        self.assertEqual(rec.policy_version_that_generated_it, "q0")
        self.assertEqual(rec.candidate_reason, "fixed_interval:30")


if __name__ == "__main__":
    unittest.main()
