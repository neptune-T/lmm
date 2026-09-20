"""Canonical experiment manifest.

Every run records what it was, what it ran on, and what produced it — before
it starts. The historical codebase carried these facts implicitly (in
directory names, in shell scripts, in the operator's head), which is why
reconstructing old experiments required filename archaeology.

Rules:

* a manifest is resolved once, then written into the output directory;
* ``git_commit`` and ``policy_checkpoint`` are mandatory — a run whose code or
  weights cannot be identified is not a scientific record;
* the resolved manifest is what downstream analysis reads, never the launch
  command.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .rollout.precision_backend import resolve

MANIFEST_VERSION = "lmm_manifest_v1"
MANIFEST_FILENAME = "resolved_manifest.json"


class ManifestError(ValueError):
    """The manifest is incomplete or internally inconsistent."""


@dataclass
class ExperimentManifest:
    """Complete description of one experiment run."""

    experiment_name: str
    policy_checkpoint: str
    output_directory: str
    task: str = ""
    environment: str = "behavior_ovmm"
    parent_checkpoint: str | None = None
    timestamp: str = ""
    git_commit: str = ""
    git_dirty: bool = False

    scene_ids: list[str] = field(default_factory=list)
    episode_ids: list[str] = field(default_factory=list)

    precision_backend: str = "bf16"
    rollout_mode: str = "full_task"  # full_task | probe | explore | verify
    candidate_strategy: str = "fixed_interval"

    k_probe: int = 4
    k_cheap: int = 32
    k_verify: int = 8
    number_of_roots: int = 0

    rl_algorithm: str = "grpo"
    learning_rate: float = 1e-5
    update_epochs: int = 1

    random_seeds: list[int] = field(default_factory=list)
    wandb_run: str = ""
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Assert the manifest is runnable and identifiable."""
        for name in ("experiment_name", "policy_checkpoint", "output_directory"):
            if not getattr(self, name):
                raise ManifestError(f"manifest field {name!r} is required")
        if not self.git_commit:
            raise ManifestError(
                "git_commit is required: a run whose code state is unknown "
                "cannot be reproduced"
            )
        resolve(self.precision_backend)
        if self.rollout_mode not in ("full_task", "probe", "explore", "verify"):
            raise ManifestError(f"unknown rollout_mode {self.rollout_mode!r}")
        if self.k_probe < 2:
            raise ManifestError("k_probe must be >= 2")
        if _looks_like_a_conclusion(self.experiment_name):
            raise ManifestError(
                f"experiment_name {self.experiment_name!r} asserts a conclusion. "
                "Name the experiment after what it tests, not after the result "
                "you hope for."
            )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["manifest_version"] = MANIFEST_VERSION
        return d

    def write(self, directory: str | Path | None = None) -> Path:
        """Write the resolved manifest into the output directory."""
        self.validate()
        out = Path(directory or self.output_directory)
        out.mkdir(parents=True, exist_ok=True)
        path = out / MANIFEST_FILENAME
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentManifest":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], *, repo_root: str | Path | None = None
    ) -> "ExperimentManifest":
        """Build a manifest from a config mapping, stamping git and time."""
        known = {f for f in cls.__dataclass_fields__}
        fields_ = {k: v for k, v in config.items() if k in known}
        fields_.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        commit, dirty = git_state(repo_root)
        fields_.setdefault("git_commit", commit)
        fields_.setdefault("git_dirty", dirty)
        m = cls(**fields_)
        m.validate()
        return m


#: Words that assert an outcome rather than describe a test.
_CONCLUSION_WORDS = (
    "successful", "optimal", "stable", "proven", "best", "winning",
    "breakthrough", "confirmed", "works",
)


def _looks_like_a_conclusion(name: str) -> bool:
    tokens = name.lower().replace("-", "_").split("_")
    return any(w in tokens for w in _CONCLUSION_WORDS)


def git_state(repo_root: str | Path | None = None) -> tuple[str, bool]:
    """Current commit and whether the tree is dirty. Empty on failure."""
    root = str(repo_root or Path(__file__).resolve().parent.parent)
    try:
        commit = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", root, "status", "--porcelain"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
        return commit, bool(status)
    except (subprocess.SubprocessError, OSError):
        return "", False
