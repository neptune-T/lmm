"""End-to-end policy interface.

The canonical policy is end-to-end:

    RGB / proprioception / language / policy history
                    |
                  pi0.5
                    v
           base + arm + gripper actions

A conforming policy receives an observation and returns an action. It is NOT
told which phase of the task it is in, and nothing in the canonical pipeline
may decide on its behalf when navigation ends, when reaching begins, or when
to switch to placing. Those decisions are the policy's output, not its input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class ActionSpec:
    """Layout of the flat action vector a policy emits.

    The canonical long-horizon action is the concatenation of base, arm and
    gripper commands. Keeping the layout explicit prevents the silent index
    drift that made historical action-contract audits necessary.
    """

    base_dim: int
    arm_dim: int
    gripper_dim: int

    @property
    def total_dim(self) -> int:
        return self.base_dim + self.arm_dim + self.gripper_dim

    def split(self, action: Sequence[float]) -> dict[str, np.ndarray]:
        """Split a flat action into its base/arm/gripper components."""
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.size != self.total_dim:
            raise ValueError(
                f"action has {a.size} dims, expected {self.total_dim} "
                f"(base={self.base_dim}, arm={self.arm_dim}, "
                f"gripper={self.gripper_dim})"
            )
        i = self.base_dim
        j = i + self.arm_dim
        return {"base": a[:i], "arm": a[i:j], "gripper": a[j:]}


class Policy(Protocol):
    """Minimal end-to-end policy contract."""

    @property
    def policy_version(self) -> str:
        """Identity of the weights currently served."""

    @property
    def action_spec(self) -> ActionSpec:
        """Action layout this policy emits."""

    def reset(self, *, seed: int | None = None) -> None:
        """Reset per-episode policy state (history, sampler)."""

    def act(self, observation: Mapping[str, Any]) -> np.ndarray:
        """Return one flat action for the given observation."""
