"""Concrete Habitat backend for the installed Trossen Mobile AI overlay.

No planner, skill sequencer, frontier logic, or RL lives here.  The class
only constructs Habitat, converts observations, and transmits audited
actuator commands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .trossen_mobile_contract import (
    assert_differential_base_executable,
    decode_openpi_action,
    encode_openpi_observation,
)


@dataclass(frozen=True)
class TrossenHabitatConfig:
    habitat_config: str
    prompt: str
    gpu_device_id: int = 0
    max_episode_steps: int = 1500


class TrossenHabitatBackend:
    """Real ``habitat.Env`` binding for the dual-arm overlay."""

    def __init__(self, config: TrossenHabitatConfig, env: Any) -> None:
        self.config = config
        self.env = env
        self.last_raw_observation: dict[str, Any] | None = None

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
        return encode_openpi_observation(raw, prompt=self.config.prompt)

    @property
    def episode_over(self) -> bool:
        return bool(self.env.episode_over)

    @property
    def episode_id(self) -> str:
        return str(self.env.current_episode.episode_id)

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
        return dict(raw)

    def step_base_forward_yaw(self, forward: float, yaw: float) -> dict[str, Any]:
        """Exercise Habitat's native differential-drive actuator directly.

        Inputs are normalized controls consumed by ``BaseVelAction``.  This is
        deliberately not called x/y, and is not used for OpenPI policy output.
        """
        value = np.asarray([forward, yaw], dtype=np.float32)
        if not np.isfinite(value).all() or np.max(np.abs(value)) > 1:
            raise ValueError("native base controls must be finite and in [-1, 1]")
        raw = self.env.step(
            {"action": "base_velocity", "action_args": {"base_vel": value}}
        )
        self.last_raw_observation = dict(raw)
        return dict(raw)

    def step_openpi_action(self, action: Sequence[float]) -> dict[str, Any]:
        """Execute only the proven differential-drive-compatible subset.

        Even when ``vy`` is zero, arms and base are separate Habitat actions,
        so this method rejects nonzero base commands instead of silently using
        two simulator timesteps for one 50 Hz policy command.
        """
        command = decode_openpi_action(action)
        native_base = assert_differential_base_executable(command)
        if np.max(np.abs(native_base)) > 1e-6:
            raise RuntimeError(
                "simultaneous arm+base execution is not yet atomic in the installed "
                "Habitat action space; refusing a two-timestep reinterpretation"
            )
        return self.step_arm_grippers(
            command.arm_joint_targets_rad,
            command.left_gripper_closed_fraction,
            command.right_gripper_closed_fraction,
        )

    def close(self) -> None:
        self.env.close()
