"""The event detector observes; it must never control.

    event_detector == observation/instrumentation only
    event_detector != policy controller

These tests assert that property three ways: by API surface, by behavioural
equivalence (identical actions with and without a detector attached), and by
source inspection.
"""

from __future__ import annotations

import inspect
import unittest

from _helpers import FakePolicy, FakeSimulator  # noqa: E402
from lmm.env.behavior_env import BehaviorEnv, BehaviorEnvConfig  # noqa: E402
from lmm.instrumentation import event_detector as ed_module  # noqa: E402
from lmm.instrumentation.event_detector import (  # noqa: E402
    EVENTS,
    EventDetector,
    EventDetectorConfig,
)
from lmm.policy.pi05_policy import OVMM_ACTION_SPEC  # noqa: E402
from lmm.rollout.runner import RolloutConfig, RolloutRunner  # noqa: E402


class TestDetectorApiSurface(unittest.TestCase):
    def test_detector_exposes_no_action_producing_method(self):
        forbidden = ("act", "action", "step", "control", "override",
                     "set_mode", "transition", "next_skill", "select_skill")
        public = {n for n in dir(EventDetector) if not n.startswith("_")}
        self.assertEqual(public & set(forbidden), set(),
                         f"detector exposes control-like API: {public & set(forbidden)}")

    def test_observe_returns_only_event_names(self):
        det = EventDetector()
        out = det.observe({"distance_to_object": 0.1, "held": True})
        self.assertTrue(all(isinstance(e, str) for e in out))
        self.assertTrue(set(out).issubset(set(EVENTS)))

    def test_event_vocabulary_is_descriptive_not_imperative(self):
        """Names describe what happened, never what to do next."""
        imperative = ("nav_to", "goto", "start_", "begin_", "switch_", "enter_")
        for name in EVENTS:
            for token in imperative:
                self.assertNotIn(token, name, f"{name} reads as a command")

    def test_module_code_does_not_use_a_skill_controller(self):
        """Executable code must not touch controller symbols.

        Docstrings may *name* the legacy controller to contrast with it, so
        prose is stripped before the check — only real code is inspected.
        """
        import ast

        tree = ast.parse(inspect.getsource(ed_module))
        for node in ast.walk(tree):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                    and isinstance(node.value.value, str):
                node.value.value = ""  # blank out docstrings
        code = ast.unparse(tree)
        for banned in ("semantic_mode_event_controller", "SKILL_TO_MODE",
                       "curr_skill", "VLA_MODES", "PHASES"):
            self.assertNotIn(banned, code,
                             f"observer code references controller symbol {banned}")


class TestDetectorDoesNotAlterBehaviour(unittest.TestCase):
    def _run(self, with_detector: bool):
        sim = FakeSimulator(steps_to_done=12)
        det = EventDetector(EventDetectorConfig()) if with_detector else None
        env = BehaviorEnv(
            BehaviorEnvConfig(benchmark_root="/nonexistent", max_steps=12),
            OVMM_ACTION_SPEC, backend=sim, detector=det,
        )
        policy = FakePolicy(OVMM_ACTION_SPEC)
        runner = RolloutRunner(env, policy, RolloutConfig(max_steps=12, task="t"))
        rec = runner.run_full_task(
            rollout_id="r", episode_id="ep1", scene_id="sc1", seed=7
        )
        actions = [tuple(round(x, 12) for x in s.action) for s in rec.steps]
        return actions, rec, sim

    def test_actions_are_identical_with_and_without_detector(self):
        without, rec_wo, sim_wo = self._run(False)
        with_, rec_w, sim_w = self._run(True)
        self.assertEqual(without, with_,
                         "attaching an observer changed the policy's actions")
        self.assertEqual(rec_wo.terminal_success, rec_w.terminal_success)
        self.assertEqual(len(sim_wo.commands), len(sim_w.commands))

    def test_events_are_recorded_when_detector_present(self):
        _, rec_w, _ = self._run(True)
        self.assertTrue(any(s.events for s in rec_w.steps),
                        "detector attached but no events were annotated")

    def test_no_events_recorded_without_detector(self):
        _, rec_wo, _ = self._run(False)
        self.assertTrue(all(not s.events for s in rec_wo.steps))


class TestEventSemantics(unittest.TestCase):
    def test_events_fire_once(self):
        det = EventDetector()
        first = det.observe({"held": True})
        second = det.observe({"held": True})
        self.assertIn("object_held", first)
        self.assertNotIn("object_held", second)

    def test_release_requires_prior_hold(self):
        det = EventDetector()
        self.assertNotIn("object_released", det.observe({"held": False}))
        det.observe({"held": True})
        self.assertIn("object_released", det.observe({"held": False}))

    def test_grasp_attempt_on_gripper_closing_edge(self):
        det = EventDetector()
        self.assertNotIn("grasp_attempted", det.observe({"gripper_command": 0.0}))
        self.assertIn("grasp_attempted", det.observe({"gripper_command": 0.9}))

    def test_reset_clears_memory(self):
        det = EventDetector()
        det.observe({"held": True})
        det.reset()
        self.assertEqual(det.seen, frozenset())
        self.assertIn("object_held", det.observe({"held": True}))


if __name__ == "__main__":
    unittest.main()
