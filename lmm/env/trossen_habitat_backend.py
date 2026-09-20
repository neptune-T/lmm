"""Concrete Habitat backend for the installed Trossen Mobile AI overlay.

No planner, skill sequencer, frontier logic, or RL lives here. The class only
constructs Habitat, converts observations, and transmits audited actuator
commands.

Action semantics come from :mod:`lmm.policy.wxai_transforms` (16-D, with
dimensions 14-15 as forward/yaw velocity) and reach Habitat through
:mod:`lmm.env.wxai_habitat_bridge`, which owns the single physical-to-
normalized unit conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .wxai_habitat_bridge import (
    BaseLimits,
    WXAICommand,
    decode_policy_action,
    encode_habitat_observation,
    to_habitat_base,
)


@dataclass(frozen=True)
class TrossenHabitatConfig:
    habitat_config: str
    prompt: str
    gpu_device_id: int = 0
    max_episode_steps: int = 1500
    base_limits: BaseLimits = BaseLimits()


@dataclass
class StepResult:
    """One actuator transmission and what it actually cost.

    ``sim_steps`` is recorded because arms and base are separate Habitat task
    actions: a policy command that moves both consumes two simulator steps.
    Callers that reason about control rate must read this rather than assume
    one policy step equals one simulator step.
    """

    observation: dict[str, Any]
    sim_steps: int
    base_normalized: tuple[float, float] | None = None


class TrossenHabitatBackend:
    """Real ``habitat.Env`` binding for the dual-arm overlay."""

    def __init__(self, config: TrossenHabitatConfig, env: Any) -> None:
        self.config = config
        self.env = env
        self.last_raw_observation: dict[str, Any] | None = None
        self.sim_step_count = 0

    @classmethod
    def make(cls, config: TrossenHabitatConfig) -> "TrossenHabitatBackend":
        import habitat
        from habitat_baselines.config.default import get_config

        overrides = [
            f"habitat.simulator.habitat_sim_v0.gpu_device_id={config.gpu_device_id}",
            f"habitat.environment.max_episode_steps={config.max_episode_steps}",
            # OVMMSim in this restored Habitat revision has no get_collisions(),
            # which BaseVelAction calls only when dynamic sliding is disabled.
            # The navmesh step_filter remains active in either mode.
            "habitat.task.actions.base_velocity.allow_dyn_slide=True",
        ]
        habitat_config = get_config(config.habitat_config, overrides=overrides)
        return cls(config, habitat.Env(config=habitat_config))

    def reset(self) -> dict[str, Any]:
        raw = self.env.reset()
        self.last_raw_observation = dict(raw)
        self.sim_step_count = 0
        return encode_habitat_observation(raw, prompt=self.config.prompt)

    @property
    def episode_over(self) -> bool:
        return bool(self.env.episode_over)

    @property
    def episode_id(self) -> str:
        return str(self.env.current_episode.episode_id)

    def observe(self) -> dict[str, Any]:
        """Re-encode the most recent raw observation for the policy."""
        if self.last_raw_observation is None:
            raise RuntimeError("no observation yet; call reset() first")
        return encode_habitat_observation(
            self.last_raw_observation, prompt=self.config.prompt
        )

    def step_arm_grippers(
        self,
        arm_joint_targets_rad: Sequence[float],
        left_closed_fraction: float,
        right_closed_fraction: float,
    ) -> dict[str, Any]:
        arm = np.asarray(arm_joint_targets_rad, dtype=np.float32)
        if arm.shape != (12,) or not np.isfinite(arm).all():
            raise ValueError(f"arm targets must be finite [12], got {arm.shape}")
        if not 0 <= left_closed_fraction <= 1 or not 0 <= right_closed_fraction <= 1:
            raise ValueError("gripper closed fractions must be in [0, 1]")
        raw = self.env.step(
            {
                "action": "arm_action",
                "action_args": {
                    "arm_action": arm,
                    "left_grip_action": np.asarray(
                        [2 * left_closed_fraction - 1], dtype=np.float32
                    ),
                    "right_grip_action": np.asarray(
                        [2 * right_closed_fraction - 1], dtype=np.float32
                    ),
                },
            }
        )
        self.last_raw_observation = dict(raw)
        self.sim_step_count += 1
        return dict(raw)

    def step_base_normalized(self, forward: float, yaw: float) -> dict[str, Any]:
        """Drive Habitat's differential base with **normalized** controls.

        Inputs are the ``[-1, 1]`` values ``BaseVelAction`` consumes directly.
        Use :func:`lmm.env.wxai_habitat_bridge.to_habitat_base` to obtain them
        from physical m/s and rad/s; do not scale by hand at the call site.
        """
        value = np.asarray([forward, yaw], dtype=np.float32)
        if not np.isfinite(value).all() or np.max(np.abs(value)) > 1:
            raise ValueError("normalized base controls must be finite and in [-1, 1]")
        raw = self.env.step(
            {"action": "base_velocity", "action_args": {"base_vel": value}}
        )
        self.last_raw_observation = dict(raw)
        self.sim_step_count += 1
        return dict(raw)

    def step_policy_action(
        self, action: Sequence[float], *, allow_split_base_step: bool = False
    ) -> StepResult:
        """Execute one physical 16-D policy action.

        Arms and base are separate Habitat task actions in the installed
        revision, so a command that moves both cannot be applied atomically.
        Rather than silently spending two simulator steps on one policy step,
        this refuses unless the caller opts in, and then reports ``sim_steps``
        so the rate discrepancy stays in the record.
        """
        command = decode_policy_action(action)
        if command.base_is_stationary:
            obs = self.step_arm_grippers(
                command.arm_joint_targets_rad,
                command.left_gripper_closed_fraction,
                command.right_gripper_closed_fraction,
            )
            return StepResult(observation=obs, sim_steps=1, base_normalized=(0.0, 0.0))

        normalized = to_habitat_base(command, self.config.base_limits)
        if not allow_split_base_step:
            raise RuntimeError(
                "arm+base execution is not atomic in the installed Habitat action "
                "space: applying this command would consume two simulator steps "
                "for one policy step. Pass allow_split_base_step=True to accept "
                "that explicitly; the cost is reported in StepResult.sim_steps."
            )
        self.step_arm_grippers(
            command.arm_joint_targets_rad,
            command.left_gripper_closed_fraction,
            command.right_gripper_closed_fraction,
        )
        obs = self.step_base_normalized(float(normalized[0]), float(normalized[1]))
        return StepResult(
            observation=obs,
            sim_steps=2,
            base_normalized=(float(normalized[0]), float(normalized[1])),
        )

    def close(self) -> None:
        self.env.close()
