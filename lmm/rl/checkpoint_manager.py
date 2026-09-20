"""Checkpoint identity and parent/child lineage.

The A800 loss on 2026-09-14 recovered only q20, the q20 optimizer state and
the q4 model; exact q0/q5/q10 weights are gone, so the original 20-update
trajectory can no longer be replayed exactly. This module exists so that
lineage is recorded *as it is produced*, not reconstructed afterwards from
directory names.

Note on naming: ``q0``/``q4``/``q20`` are **training round indices**. They are
not precisions, and in particular ``q4`` here is round 4, unrelated to
``int4_weight_only``. See ``docs/QUANTIZATION_BACKENDS.md``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class CheckpointRecord:
    """One policy checkpoint and its provenance."""

    policy_version: str
    path: str
    parent_version: str | None
    optimizer_step: int
    created_utc: str
    sha256: str | None = None
    notes: str = ""
    metadata: dict[str, object] = field(default_factory=dict)


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """Content hash of a checkpoint file, for identity that survives renames."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


class CheckpointManager:
    """Append-only checkpoint lineage ledger backed by a JSON file."""

    def __init__(self, ledger_path: str | Path) -> None:
        self.ledger_path = Path(ledger_path)
        self._records: list[CheckpointRecord] = []
        if self.ledger_path.exists():
            data = json.loads(self.ledger_path.read_text(encoding="utf-8"))
            self._records = [CheckpointRecord(**r) for r in data.get("records", [])]

    @property
    def records(self) -> list[CheckpointRecord]:
        return list(self._records)

    def register(self, record: CheckpointRecord) -> CheckpointRecord:
        """Record a new checkpoint, rejecting duplicate or orphan versions."""
        if any(r.policy_version == record.policy_version for r in self._records):
            raise ValueError(
                f"policy_version {record.policy_version!r} already registered; "
                "checkpoint versions are immutable"
            )
        if record.parent_version is not None and not any(
            r.policy_version == record.parent_version for r in self._records
        ):
            raise ValueError(
                f"parent_version {record.parent_version!r} is not registered; "
                "lineage must be recorded as it is produced"
            )
        self._records.append(record)
        self._flush()
        return record

    def lineage(self, policy_version: str) -> list[str]:
        """Ancestry chain from the root checkpoint to ``policy_version``."""
        index = {r.policy_version: r for r in self._records}
        if policy_version not in index:
            raise KeyError(f"unknown policy_version {policy_version!r}")
        chain: list[str] = []
        seen: set[str] = set()
        cur: str | None = policy_version
        while cur is not None:
            if cur in seen:
                raise ValueError(f"cycle in checkpoint lineage at {cur!r}")
            seen.add(cur)
            chain.append(cur)
            cur = index[cur].parent_version if cur in index else None
        return list(reversed(chain))

    def _flush(self) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "lmm_checkpoint_ledger_v1",
            "records": [asdict(r) for r in self._records],
        }
        self.ledger_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
