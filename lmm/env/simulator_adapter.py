"""Simulator plumbing — NO task planning.

This is the one place allowed to know simulator-specific details:

    continuous gripper command
            |
            v
    Habitat / BEHAVIOR-specific SNAP or DESNAP API

Permitted here
--------------
action scaling, coordinate transforms, robot control interfaces, Habitat /
BEHAVIOR API calls, gripper event translation, simulator stepping, snapshot
save/restore, collision and environment bookkeeping.

Forbidden here
--------------
Deciding *what the robot should do next*. No skill sequencing, no phase
machine, no "navigation is finished, start reaching", no goal selection. If a
change to this file would alter which behaviour the robot attempts — as
opposed to how a command is transmitted — it belongs somewhere else, and
probably nowhere in the canonical pipeline.

The legacy OVMM skill sequencer deliberately does not live here. See
``legacy/README.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from ..policy.base import ActionSpec


class GripperEvent:
    """Discrete simulator gripper transitions."""

    NONE = "none"
    SNAP = "snap"
    DESNAP = "desnap"


@dataclass
class AdapterConfig:
    """Action scaling and gripper thresholds.

    ``snap_threshold``/``desnap_threshold`` differ to create hysteresis, so a
    command hovering at the boundary does not chatter between SNAP and DESNAP.
    """

    base_scale: float = 1.0
    arm_scale: float = 1.0
    snap_threshold: float = 0.5
    desnap_threshold: float = 0.3
    clip_actions: bool = True
    action_low: float = -1.0
    action_high: float = 1.0

    def __post_init__(self) -> None:
        if self.desnap_threshold > self.snap_threshold:
            raise ValueError(
                "desnap_threshold must be <= snap_threshold for stable "
                f"hysteresis; got {self.desnap_threshold} > {self.snap_threshold}"
            )


class SimulatorBackend(Protocol):
    """The raw simulator surface the adapter drives."""

    def step(self, command: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
        """Apply a low-level command; return ``(observation, done)``."""

    def snap(self) -> bool:
        """Attempt to attach the in-range object. Return whether it attached."""

    def desnap(self) -> bool:
        """Release any held object. Return whether something was released."""


@dataclass
class SimulatorAdapter:
    """Translates end-to-end policy actions into simulator commands.

    Pure plumbing: given the same action and the same held-state, it always
    emits the same command and the same gripper event.
    """

    action_spec: ActionSpec
    config: AdapterConfig = field(default_factory=AdapterConfig)
    backend: SimulatorBackend | None = None
    _held: bool = field(default=False, init=False)

    @property
    def held(self) -> bool:
        return self._held

    def reset(self, held: bool = False) -> None:
        self._held = held

    def gripper_event(self, gripper_command: float) -> str:
        """Map a continuous gripper command to a discrete SNAP/DESNAP event.

        Hysteresis, and only a *change* of held-state produces an event:
        holding the gripper closed while already holding emits ``NONE``.
        """
        g = float(gripper_command)
        if not self._held and g >= self.config.snap_threshold:
            return GripperEvent.SNAP
        if self._held and g <= self.config.desnap_threshold:
            return GripperEvent.DESNAP
        return GripperEvent.NONE

    def to_command(self, action: Sequence[float]) -> dict[str, Any]:
        """Scale and split a flat policy action into a simulator command."""
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if self.config.clip_actions:
            a = np.clip(a, self.config.action_low, self.config.action_high)
        parts = self.action_spec.split(a)
        if parts["gripper"].size != 1:
            raise NotImplementedError(
                "the generic SimulatorAdapter only implements one gripper; "
                "bind the Trossen dual-gripper Habitat backend before running "
                f"a {parts['gripper'].size}-gripper action contract"
            )
        return {
            "base": parts["base"] * self.config.base_scale,
            "arm": parts["arm"] * self.config.arm_scale,
            "gripper_raw": float(parts["gripper"][0]),
            "gripper_event": self.gripper_event(float(parts["gripper"][0])),
        }

    def apply(self, action: Sequence[float]) -> tuple[dict[str, Any], bool]:
        """Send one action to the simulator and update held-state."""
        if self.backend is None:
            raise RuntimeError("SimulatorAdapter has no backend attached")
        command = self.to_command(action)
        event = command["gripper_event"]

        if event == GripperEvent.SNAP:
            self._held = bool(self.backend.snap())
        elif event == GripperEvent.DESNAP:
            self._held = not bool(self.backend.desnap()) and self._held

        obs, done = self.backend.step(command)
        obs = dict(obs)
        obs.setdefault("held", self._held)
        obs.setdefault("gripper_command", command["gripper_raw"])
        return obs, done
