"""pi0.5 end-to-end policy.

Runs out-of-process. The simulator side (HomeRobot / Habitat, Python 3.9) and
the policy side (JAX / OpenPI, Python 3.11+) cannot share an interpreter, so
the policy is reached over the local HTTP transport in ``lmm.rollout.client``.
This mirrors the arrangement the historical
``scripts/rollout/pi0_http_policy_server.py`` already used.

Determinism: ``docs/FRESH_POLICY_PROCESS_FIRST_DIVERGENCE_AUDIT_V1.md`` found
that two cold policy processes produce different raw forwards from identical
inputs (274/320 float32 values differed). Continuations are therefore only
comparable when served by **one persistent process**. This class holds a client
to an already-running server and never spawns one per continuation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .base import ActionSpec, Policy

#: Layout for the OVMM/HomeRobot mobile-manipulation action used by pi0.5:
#: 3 base (x, y, yaw) + 7 arm joints + 1 continuous gripper.
OVMM_ACTION_SPEC = ActionSpec(base_dim=3, arm_dim=7, gripper_dim=1)

#: Dimension counts for Mobile ALOHA. The native OpenPI order is interleaved,
#: not base-first: left6, left-grip, right6, right-grip, base-vx, base-vy.
#: Use :mod:`lmm.env.trossen_mobile_contract` to decode it; ActionSpec records
#: counts only and its ``split`` method must not be used for this embodiment.
BIMANUAL_OVMM_ACTION_SPEC = ActionSpec(base_dim=2, arm_dim=12, gripper_dim=2)


@dataclass
class Pi05PolicyConfig:
    """Connection and sampling parameters for a served pi0.5 policy."""

    server_url: str
    policy_version: str
    precision_backend: str = "bf16"
    action_horizon: int = 1
    num_steps: int = 10
    action_spec: ActionSpec = OVMM_ACTION_SPEC


class Pi05Policy(Policy):
    """Client-side handle to a persistent pi0.5 policy server."""

    def __init__(self, config: Pi05PolicyConfig, client: Any) -> None:
        """``client`` must expose ``infer(observation, seed) -> array``.

        It is injected rather than constructed so the policy layer can be
        exercised without JAX, OpenPI or a GPU present.
        """
        self.config = config
        self._client = client
        self._seed: int | None = None

    @property
    def policy_version(self) -> str:
        return self.config.policy_version

    @property
    def action_spec(self) -> ActionSpec:
        return self.config.action_spec

    def reset(self, *, seed: int | None = None) -> None:
        self._seed = seed
        reset = getattr(self._client, "reset", None)
        if callable(reset):
            reset(seed=seed)

    def act(self, observation: Mapping[str, Any]) -> np.ndarray:
        """One end-to-end action. No phase or skill is supplied as input."""
        action = np.asarray(
            self._client.infer(observation, seed=self._seed), dtype=np.float64
        ).reshape(-1)
        expected = self.action_spec.total_dim
        if action.size != expected:
            raise ValueError(
                f"policy returned {action.size} action dims, expected {expected}"
            )
        return action
