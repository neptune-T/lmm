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
    CARRIAGE_JP_INDICES,
    CONTROL_HZ,
    GRIPPER_OPEN_M,
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
    control_period_s: float | None = None


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
            # CANONICAL, not a preference: allow_dyn_slide=False raises
            # AttributeError: 'OVMMSim' object has no attribute 'get_collisions'
            # on this Habitat revision, because BaseVelAction calls it only when
            # dynamic sliding is disabled. Verified 2026-09-20. The navmesh
            # step_filter remains active either way.
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

    def set_gripper_kinematic(self, ee_index: int, closed_fraction: float) -> None:
        """Actuate one gripper by writing carriage joint positions.

        The overlay's ``set_gripper_target_state_for_ee`` sets a joint *motor*
        target. Measured on the live runtime (2026-09-20): the target updates
        correctly (0.044 -> 0.0) but the carriages never move, because the
        articulation is driven kinematically by ``ArmAbsPosKinematicAction``
        and physics never integrates the motor. Two seconds of
        ``step_physics`` left the carriages at 0.044 m.

        Writing ``joint_positions`` directly actuates them, reproduces the
        requested fraction exactly, keeps the two hands independent, and
        survives ``env.step``.
        """
        import numpy as _np

        robot = self.env.sim.get_agent_data(0).articulated_agent
        target = GRIPPER_OPEN_M * (1.0 - float(min(max(closed_fraction, 0.0), 1.0)))
        jp = _np.asarray(robot.sim_obj.joint_positions, dtype=_np.float64).copy()
        for index in CARRIAGE_JP_INDICES[ee_index]:
            jp[index] = target
        robot.sim_obj.joint_positions = jp

    def step_policy_action(self, action: Sequence[float]) -> StepResult:
        """Execute one physical 16-D policy action as ONE simulator transition.

        Habitat's ``embodied_task`` accepts a tuple of action names, so arms and
        base are applied within a single ``env.step``. Verified on the live
        runtime: all 12 arm joints and both grippers round-trip exactly, and the
        base advances by exactly one control period (``ang_speed/120`` rad,
        ``lin_speed/120`` m) for one call.

        This replaces the earlier split path, which spent two simulator steps on
        one policy action.
        """
        import numpy as _np

        command = decode_policy_action(action)
        normalized = to_habitat_base(command, self.config.base_limits)

        # Grippers are written before and after the step: the magic-grasp
        # helpers inside BimanualArmAction call whole-robot open/close, which
        # would otherwise clobber the per-hand targets.
        self.set_gripper_kinematic(0, command.left_gripper_closed_fraction)
        self.set_gripper_kinematic(1, command.right_gripper_closed_fraction)

        raw = self.env.step({
            "action": ("arm_action", "base_velocity"),
            "action_args": {
                "arm_action": command.arm_joint_targets_rad.astype(_np.float32),
                "left_grip_action": _np.asarray(
                    [2.0 * command.left_gripper_closed_fraction - 1.0], dtype=_np.float32),
                "right_grip_action": _np.asarray(
                    [2.0 * command.right_gripper_closed_fraction - 1.0], dtype=_np.float32),
                "base_vel": normalized.astype(_np.float32),
            },
        })
        self.set_gripper_kinematic(0, command.left_gripper_closed_fraction)
        self.set_gripper_kinematic(1, command.right_gripper_closed_fraction)

        self.last_raw_observation = dict(raw)
        self.sim_step_count += 1
        return StepResult(
            observation=dict(raw),
            sim_steps=1,
            base_normalized=(float(normalized[0]), float(normalized[1])),
            control_period_s=1.0 / CONTROL_HZ,
        )

    def close(self) -> None:
        self.env.close()
