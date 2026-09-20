"""Simulator snapshot save / restore.

A snapshot is the unit that makes branching possible: restore it and every
continuation starts from an identical physical state. Snapshot *content* is
simulator-specific and produced by the backend; this module owns identity,
storage layout and the :class:`~lmm.schemas.SnapshotRecord` that travels with
it.

State-field selection for Habitat OVMM was established empirically by
``scripts/rollout/ovmm_snapshot_restore_probe.py``; the constants below are
carried over from that probe so restored roots stay comparable with the
historical banks.
"""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..schemas import SnapshotRecord

#: Task-measure attribute prefixes that must be captured for a faithful restore.
MEASURE_STATE_PREFIXES = ("_accum", "_add", "_cur", "_last", "_prev")

#: Explicit task-state fields captured alongside the physics state.
TASK_STATE_FIELDS = (
    "_cur_episode_step",
    "_done",
    "_in_manip_mode",
    "_is_episode_active",
    "_is_navmesh_violated",
    "_picked_object_idx",
    "_should_end",
    "_targ_idx",
    "coll_accum",
    "is_stop_called",
    "prev_coll_accum",
    "prev_measures",
)


class SnapshotBackend(Protocol):
    """Simulator-side snapshot capability."""

    def capture_state(self) -> dict[str, Any]:
        """Return a picklable dict fully describing the simulator state."""

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore a previously captured state."""


def state_digest(state: dict[str, Any]) -> str:
    """Stable digest of a captured state, for restore-fidelity checks."""
    return hashlib.sha256(
        pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
    ).hexdigest()


def root_id_for(episode_id: str, step: int, policy_version: str) -> str:
    """Canonical root identity. Never parsed back out of a filename."""
    return f"root::{episode_id}::s{step:05d}::{policy_version}"


@dataclass
class SnapshotStore:
    """Filesystem-backed snapshot storage."""

    root_dir: Path

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)

    def path_for(self, root_id: str) -> Path:
        safe = root_id.replace("::", "__").replace("/", "_")
        return self.root_dir / f"{safe}.pkl"

    def save(
        self,
        backend: SnapshotBackend,
        *,
        episode_id: str,
        scene_id: str,
        step: int,
        policy_version: str,
        candidate_reason: str,
        source_rollout_id: str = "",
    ) -> SnapshotRecord:
        """Capture and persist a snapshot, returning its record."""
        state = backend.capture_state()
        root_id = root_id_for(episode_id, step, policy_version)
        path = self.path_for(root_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return SnapshotRecord(
            root_id=root_id,
            episode_id=episode_id,
            scene_id=scene_id,
            snapshot_step=step,
            snapshot_path=str(path),
            policy_version_that_generated_it=policy_version,
            candidate_reason=candidate_reason,
            source_rollout_id=source_rollout_id,
        )

    def load_state(self, record: SnapshotRecord) -> dict[str, Any]:
        with Path(record.snapshot_path).open("rb") as fh:
            return pickle.load(fh)

    def restore(self, backend: SnapshotBackend, record: SnapshotRecord) -> str:
        """Restore a snapshot; returns the digest of the restored state.

        Callers compare the returned digest across repeated restores of the
        same root to assert restoration is deterministic.
        """
        state = self.load_state(record)
        backend.restore_state(state)
        return state_digest(state)
