"""Client for a persistent policy server.

Transport mirrors the historical ``scripts/rollout/pi0_http_policy_server.py``:
pickle over localhost HTTP, chosen because the simulator (Python 3.9) and the
policy (Python 3.11+/JAX) cannot share an interpreter.

One client instance corresponds to one persistent server process. Do not create
a client per continuation — see the determinism note in
:mod:`lmm.policy.pi05_policy`.
"""

from __future__ import annotations

import pickle
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass
class PolicyClientConfig:
    """Connection parameters."""

    url: str
    timeout_s: float = 120.0


class PolicyClient:
    """Minimal pickle-over-HTTP inference client."""

    def __init__(self, config: PolicyClientConfig) -> None:
        self.config = config
        self._calls = 0

    @property
    def calls(self) -> int:
        """Inference count, for auditing rollout cost."""
        return self._calls

    def reset(self, *, seed: int | None = None) -> None:
        self._post({"op": "reset", "seed": seed})

    def infer(self, observation: Mapping[str, Any], *, seed: int | None = None) -> Any:
        self._calls += 1
        return self._post({"op": "infer", "observation": dict(observation), "seed": seed})

    def health(self) -> dict[str, Any]:
        """Server identity: policy_version, backend, checkpoint hash."""
        return self._post({"op": "health"})

    def _post(self, payload: dict[str, Any]) -> Any:
        body = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        req = urllib.request.Request(
            self.config.url,
            data=body,
            headers={"Content-Type": "application/octet-stream"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.config.timeout_s) as resp:
            return pickle.loads(resp.read())
