"""Rollout execution: full-task and continuation-from-root.

One runner serves every stage of the pipeline, so a discovery probe, a scaled
exploration draw and a verification re-run all traverse identical code and
differ only in root, seed, backend and budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ..instrumentation.trajectory_logger import TrajectoryLogger
from ..schemas import FullRolloutRecord, SnapshotRecord
from .precision_backend import resolve


class _EnvLike(Protocol):
    def reset(self, **kwargs: Any) -> dict[str, Any]: ...
    def step(self, action: Any) -> tuple[dict[str, Any], bool]: ...


class _PolicyLike(Protocol):
    @property
    def policy_version(self) -> str: ...
    def reset(self, *, seed: int | None = None) -> None: ...
    def act(self, observation: Any) -> Any: ...


@dataclass
class RolloutConfig:
    """Execution budget and precision for one rollout."""

    max_steps: int = 1500
    precision_backend: str = "bf16"
    task: str = ""

    def __post_init__(self) -> None:
        resolve(self.precision_backend)


def _terminal(obs: dict[str, Any]) -> tuple[bool, dict[str, float]]:
    success = bool(obs.get("task_success", False))
    metrics = {
        k: float(v)
        for k, v in obs.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    return success, metrics


class RolloutRunner:
    """Executes end-to-end and continuation rollouts."""

    def __init__(self, env: _EnvLike, policy: _PolicyLike, config: RolloutConfig) -> None:
        self.env = env
        self.policy = policy
        self.config = config

    def run_full_task(
        self, *, rollout_id: str, episode_id: str, scene_id: str, seed: int
    ) -> FullRolloutRecord:
        """Execute the whole task end-to-end from the episode initial state.

        No skill sequencer is consulted. The loop is: observe, act, step.
        """
        logger = TrajectoryLogger(
            rollout_id=rollout_id,
            episode_id=episode_id,
            scene_id=scene_id,
            task=self.config.task,
            policy_version=self.policy.policy_version,
            precision_backend=self.config.precision_backend,
            seed=seed,
        )
        self.policy.reset(seed=seed)
        obs = self.env.reset()
        success, metrics, done = False, {}, False

        for step in range(self.config.max_steps):
            action = self.policy.act(obs)
            obs, done = self.env.step(action)
            logger.log_step(
                step,
                action,
                events=obs.get("events", ()),
                observation_meta={"held": obs.get("held")},
            )
            if done:
                success, metrics = _terminal(obs)
                break

        return logger.finish(terminal_success=success, terminal_metrics=metrics)

    def run_continuation(
        self,
        *,
        snapshot: SnapshotRecord,
        seed: int,
        precision_backend: str,
        policy_version: str,
        restore_fn: Any,
    ) -> tuple[bool, dict[str, float], int, str]:
        """Continue from a restored root.

        Satisfies :class:`~lmm.frontier.continuation_probe.ContinuationBackend`.
        ``restore_fn(snapshot)`` puts the simulator back at the root; the
        policy process is *not* restarted, per the determinism contract.
        """
        resolve(precision_backend)
        restore_fn(snapshot)
        self.policy.reset(seed=seed)
        obs = self.env.reset()
        success, metrics, length = False, {}, 0

        for step in range(self.config.max_steps):
            action = self.policy.act(obs)
            obs, done = self.env.step(action)
            length = step + 1
            if done:
                success, metrics = _terminal(obs)
                break

        path = f"{snapshot.root_id}::seed{seed}::{precision_backend}"
        return success, metrics, length, path
