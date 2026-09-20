"""Environment binding for BEHAVIOR / Habitat OVMM.

Thin by design. The environment exposes observations and accepts actions; it
does not sequence the task. All simulator-specific translation lives in
:mod:`lmm.env.simulator_adapter`, and all semantic interpretation lives in
:mod:`lmm.instrumentation` as passive observation.

Heavy dependencies (``habitat``, ``home_robot``) are imported lazily inside
:meth:`BehaviorEnv.make`, so this module can be imported for type-checking and
unit tests on a machine with no simulator installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..instrumentation.event_detector import EventDetector
from ..policy.base import ActionSpec
from .simulator_adapter import AdapterConfig, SimulatorAdapter


@dataclass
class BehaviorEnvConfig:
    """Environment selection and episode bounds."""

    benchmark_root: str
    scene_id: str = ""
    episode_id: str = ""
    task: str = ""
    max_steps: int = 1500
    adapter: AdapterConfig = field(default_factory=AdapterConfig)


class BehaviorEnv:
    """End-to-end environment wrapper.

    Holds an adapter (plumbing) and an optional detector (observation). The
    detector's output is attached to the returned observation as the
    ``events`` annotation and is never consulted when computing the next
    action.
    """

    def __init__(
        self,
        config: BehaviorEnvConfig,
        action_spec: ActionSpec,
        backend: Any = None,
        detector: EventDetector | None = None,
    ) -> None:
        self.config = config
        self.adapter = SimulatorAdapter(
            action_spec=action_spec, config=config.adapter, backend=backend
        )
        self.detector = detector
        self._steps = 0

    @classmethod
    def make(cls, config: BehaviorEnvConfig, action_spec: ActionSpec) -> "BehaviorEnv":
        """Construct against the real simulator.

        Imported lazily; raises a clear error when the simulator stack is
        absent rather than failing at module import time.
        """
        try:  # pragma: no cover - requires the simulator stack
            import habitat  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "BehaviorEnv.make requires the Habitat/HomeRobot stack, which is "
                "not installed in this interpreter. Unit tests should construct "
                "BehaviorEnv directly with a fake backend instead."
            ) from exc
        raise NotImplementedError(
            "Bind the concrete Habitat OVMM backend here. The historical "
            "construction path is scripts/rollout/pi0_ovmm_suffix_rank_probe.py"
            ":_make_env_and_agent_config, which must be ported behind the "
            "SimulatorBackend protocol without carrying its skill sequencer."
        )

    def reset(self, *, held: bool = False) -> dict[str, Any]:
        self._steps = 0
        self.adapter.reset(held=held)
        if self.detector is not None:
            self.detector.reset()
        obs, _ = self.adapter.apply([0.0] * self.adapter.action_spec.total_dim)
        return self._annotate(obs)

    def step(self, action: Any) -> tuple[dict[str, Any], bool]:
        """Apply one end-to-end action."""
        obs, done = self.adapter.apply(action)
        self._steps += 1
        if self._steps >= self.config.max_steps:
            done = True
        return self._annotate(obs), done

    def _annotate(self, obs: Mapping[str, Any]) -> dict[str, Any]:
        """Attach passive event annotations. Does not alter the observation."""
        out = dict(obs)
        out["events"] = list(self.detector.observe(obs)) if self.detector else []
        return out
