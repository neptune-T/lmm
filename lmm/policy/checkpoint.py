"""Checkpoint identity for served policies.

``policy_version`` is the single identifier threaded through every record in
``lmm.schemas``. It must be derivable from the checkpoint content, not from a
directory name, because historical checkpoints carry overloaded names
(``q0``/``q4``/``q20`` rounds, ``champion``, ``servable_selected``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..rl.checkpoint_manager import sha256_file


@dataclass(frozen=True)
class PolicyCheckpoint:
    """A resolved, identified policy checkpoint."""

    policy_version: str
    path: Path
    sha256: str | None = None
    norm_stats_sha256: str | None = None
    notes: str = ""

    @classmethod
    def from_dir(
        cls, path: str | Path, policy_version: str, *, manifest_name: str = "manifest.json"
    ) -> "PolicyCheckpoint":
        """Load a checkpoint directory, reading a manifest hash when present."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"checkpoint path does not exist: {p}")
        sha = None
        manifest = p / manifest_name
        if manifest.is_file():
            data = json.loads(manifest.read_text(encoding="utf-8"))
            sha = data.get("sha256") or sha256_file(manifest)
        return cls(policy_version=policy_version, path=p, sha256=sha)

    def describe(self) -> dict[str, str | None]:
        return {
            "policy_version": self.policy_version,
            "path": str(self.path),
            "sha256": self.sha256,
            "norm_stats_sha256": self.norm_stats_sha256,
        }
