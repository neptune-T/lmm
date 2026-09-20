"""Semantic event detection — OBSERVATION ONLY.

    event_detector == observation / instrumentation only
    event_detector != policy controller

This module detects semantic events (object approached, grasp attempted,
object held, released, receptacle approached, object on goal, task success) for
logging, debugging, candidate-snapshot proposal and trajectory analysis.

It must never influence what the policy does. The end-to-end VLA decides when
navigation ends and manipulation begins; nothing here forces a state
transition, gates an action, or switches a mode. That is the difference between
this module and the legacy
``scripts/rollout/semantic_mode_event_controller_v2.py``, which *did* drive a
NAV/MANIP mode and is retained only for historical reproduction.

The non-interference property is structural, not a convention:
:class:`EventDetector` has no method that returns or modifies an action, and
:meth:`EventDetector.observe` deep-copies nothing back into the caller. It is
asserted by ``tests/test_event_detector_noninterference.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

#: Canonical event vocabulary. Descriptive of what happened, never imperative.
EVENTS = (
    "object_approached",
    "grasp_attempted",
    "object_held",
    "object_released",
    "receptacle_approached",
    "object_on_goal",
    "object_at_rest",
    "task_success",
    "task_failure",
)


@dataclass
class EventDetectorConfig:
    """Thresholds for predicate-based event detection.

    These are *observation* thresholds. Changing them changes what gets
    logged and which steps become snapshot candidates; it cannot change the
    policy's behaviour.
    """

    approach_distance: float = 1.0
    receptacle_distance: float = 1.0
    rest_speed: float = 0.01
    gripper_close_threshold: float = 0.5


@dataclass
class EventDetector:
    """Stateless-per-step observer with monotonic event memory."""

    config: EventDetectorConfig = field(default_factory=EventDetectorConfig)
    _seen: set[str] = field(default_factory=set, init=False)
    _prev_held: bool = field(default=False, init=False)
    _prev_gripper: float = field(default=0.0, init=False)

    def reset(self) -> None:
        self._seen.clear()
        self._prev_held = False
        self._prev_gripper = 0.0

    @property
    def seen(self) -> frozenset[str]:
        """Events observed so far this episode."""
        return frozenset(self._seen)

    def observe(self, obs: Mapping[str, Any]) -> list[str]:
        """Return the events newly true at this step.

        ``obs`` is a read-only mapping of observable predicates. The return
        value is an annotation for the trajectory log. No caller in the
        canonical pipeline feeds it back into action selection.
        """
        cfg = self.config
        fired: list[str] = []

        def fire(name: str) -> None:
            if name not in self._seen:
                self._seen.add(name)
                fired.append(name)

        dist_obj = obs.get("distance_to_object")
        if dist_obj is not None and dist_obj <= cfg.approach_distance:
            fire("object_approached")

        gripper = float(obs.get("gripper_command", 0.0))
        if gripper >= cfg.gripper_close_threshold > self._prev_gripper:
            fire("grasp_attempted")

        held = bool(obs.get("held", False))
        if held:
            fire("object_held")
        if self._prev_held and not held:
            fire("object_released")

        dist_rec = obs.get("distance_to_receptacle")
        if dist_rec is not None and dist_rec <= cfg.receptacle_distance:
            fire("receptacle_approached")

        if obs.get("object_on_goal"):
            fire("object_on_goal")

        speed = obs.get("object_speed")
        if speed is not None and obs.get("object_on_goal") and speed <= cfg.rest_speed:
            fire("object_at_rest")

        if obs.get("task_success"):
            fire("task_success")
        elif obs.get("task_failure"):
            fire("task_failure")

        self._prev_held = held
        self._prev_gripper = gripper
        return fired


def event_steps(step_events: Sequence[Sequence[str]], wanted: Sequence[str]) -> list[int]:
    """Steps at which any of ``wanted`` fired — used for candidate proposal."""
    want = set(wanted)
    return [i for i, evs in enumerate(step_events) if want & set(evs)]
