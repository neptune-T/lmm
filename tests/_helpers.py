"""Shared fakes for the canonical test suite.

No GPU, simulator or policy server is required. Every heavy dependency in the
canonical pipeline sits behind a Protocol, and these fakes implement them.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class FakeSimulator:
    """Deterministic stand-in for a Habitat/BEHAVIOR backend."""

    def __init__(self, *, snap_succeeds: bool = True, steps_to_done: int = 5) -> None:
        self.snap_succeeds = snap_succeeds
        self.steps_to_done = steps_to_done
        self.state: dict[str, Any] = {"t": 0, "pos": [0.0, 0.0], "held": False}
        self.snap_calls = 0
        self.desnap_calls = 0
        self.commands: list[dict[str, Any]] = []

    def step(self, command: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
        self.commands.append(dict(command))
        self.state["t"] += 1
        done = self.state["t"] >= self.steps_to_done
        obs = {
            "t": self.state["t"],
            "held": self.state["held"],
            "task_success": done and self.state["held"],
        }
        return obs, done

    def snap(self) -> bool:
        self.snap_calls += 1
        if self.snap_succeeds:
            self.state["held"] = True
        return self.snap_succeeds

    def desnap(self) -> bool:
        self.desnap_calls += 1
        released = self.state["held"]
        self.state["held"] = False
        return released

    # snapshot backend
    def capture_state(self) -> dict[str, Any]:
        import copy

        return copy.deepcopy(self.state)

    def restore_state(self, state: dict[str, Any]) -> None:
        import copy

        self.state = copy.deepcopy(state)


class FakeContinuationBackend:
    """Continuation backend with seed-controlled stochastic outcomes.

    ``success_rate_by_root`` sets the ground-truth continuation probability per
    root, so an estimator test can assert that ``V_hat`` recovers it.
    """

    def __init__(self, success_rate_by_root: Mapping[str, float]) -> None:
        self.success_rate_by_root = dict(success_rate_by_root)
        self.calls: list[tuple[str, int]] = []

    def run_continuation(
        self, *, snapshot, seed: int, precision_backend: str, policy_version: str
    ) -> tuple[bool, dict[str, float], int, str]:
        self.calls.append((snapshot.root_id, seed))
        rng = random.Random(f"{snapshot.root_id}:{seed}")
        p = self.success_rate_by_root.get(snapshot.root_id, 0.0)
        success = rng.random() < p
        return success, {"progress": rng.random()}, 10, f"traj/{snapshot.root_id}/{seed}"


class FakeVerificationBackend:
    """High-fidelity re-run backend.

    ``agreement`` controls how often the verified outcome matches the cheap
    prediction, which is what the fidelity test varies.
    """

    def __init__(self, verified_by_rollout: Mapping[str, bool]) -> None:
        self.verified_by_rollout = dict(verified_by_rollout)
        self.calls = 0

    def rerun(
        self, *, snapshot, seed: int, precision_backend: str, policy_version: str
    ) -> tuple[str, bool, dict[str, float]]:
        self.calls += 1
        rid = f"bf16::{snapshot.root_id}::seed{seed}::{self.calls}"
        return rid, self.verified_by_rollout.get(rid, self.calls % 2 == 0), {"iou": 0.5}


class FakePolicy:
    """End-to-end policy emitting seed-dependent actions."""

    def __init__(self, action_spec, policy_version: str = "q0") -> None:
        self._spec = action_spec
        self._version = policy_version
        self._rng = random.Random(0)
        self.observations: list[Any] = []

    @property
    def policy_version(self) -> str:
        return self._version

    @property
    def action_spec(self):
        return self._spec

    def reset(self, *, seed: int | None = None) -> None:
        self._rng = random.Random(seed or 0)

    def act(self, observation):
        self.observations.append(observation)
        return [self._rng.uniform(-1, 1) for _ in range(self._spec.total_dim)]
