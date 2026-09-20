"""Structured trajectory logging.

Writes :class:`~lmm.schemas.FullRolloutRecord` and per-step annotations to
JSONL. The logger is a sink: it observes the rollout and never returns control
information to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..schemas import FullRolloutRecord, StepRecord, write_jsonl


@dataclass
class TrajectoryLogger:
    """Accumulates steps and emits one rollout record."""

    rollout_id: str
    episode_id: str
    scene_id: str
    task: str
    policy_version: str
    precision_backend: str
    seed: int = 0
    _steps: list[StepRecord] = field(default_factory=list, init=False)

    def log_step(
        self,
        step: int,
        action: Sequence[float],
        *,
        observation_meta: Mapping[str, Any] | None = None,
        events: Sequence[str] = (),
        reward: float | None = None,
    ) -> None:
        self._steps.append(
            StepRecord(
                step=step,
                action=list(action),
                observation_meta=dict(observation_meta or {}),
                events=list(events),
                reward=reward,
            )
        )

    @property
    def step_events(self) -> list[list[str]]:
        """Per-step event annotations, for candidate proposal."""
        return [s.events for s in self._steps]

    def finish(
        self, *, terminal_success: bool, terminal_metrics: Mapping[str, float] | None = None
    ) -> FullRolloutRecord:
        return FullRolloutRecord(
            rollout_id=self.rollout_id,
            episode_id=self.episode_id,
            scene_id=self.scene_id,
            task=self.task,
            policy_version=self.policy_version,
            precision_backend=self.precision_backend,
            seed=self.seed,
            steps=list(self._steps),
            terminal_success=terminal_success,
            terminal_metrics=dict(terminal_metrics or {}),
        )

    def write(self, path: str | Path, record: FullRolloutRecord) -> int:
        return write_jsonl(path, [record])
