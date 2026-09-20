"""Policy server contract.

Defines the request/response protocol a persistent policy server must honour
for :class:`lmm.rollout.client.PolicyClient`. The concrete pi0.5/JAX server
implementation is not reproduced here — the working historical servers are
``scripts/rollout/pi0_http_policy_server.py`` (BF16) and
``scripts/fp8_rollout_scaling/pi0_http_policy_server_fp8_q4_v3.py`` (quantised),
which should be ported behind this contract rather than rewritten from scratch.

Operations
----------
``health``  -> ``{"policy_version", "precision_backend", "checkpoint_sha256"}``
``reset``   -> ``{"ok": True}``
``infer``   -> flat action array

Why the contract is separate from the implementation: ``health`` is what lets
every downstream record state *which* weights and *which* precision produced
an outcome. A server that cannot answer ``health`` must not be used for a
scientific run.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol

HEALTH_FIELDS = ("policy_version", "precision_backend", "checkpoint_sha256")


class PolicyServerHandler(Protocol):
    """Server-side operations."""

    def health(self) -> dict[str, Any]: ...
    def reset(self, seed: int | None) -> dict[str, Any]: ...
    def infer(self, observation: Mapping[str, Any], seed: int | None) -> Any: ...


def validate_health(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Assert a health response identifies the served policy completely."""
    missing = [f for f in HEALTH_FIELDS if not payload.get(f)]
    if missing:
        raise ValueError(
            f"policy server health response is missing {missing}. A run whose "
            "policy identity cannot be established must not be recorded."
        )
    return dict(payload)
