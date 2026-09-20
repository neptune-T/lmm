"""Shared helpers for the canonical entry scripts.

Every canonical script resolves a manifest first and writes it into the output
directory before doing any work, so a run is identifiable from its own output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lmm.manifest import ExperimentManifest  # noqa: E402


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON config, dropping ``_comment`` keys recursively."""
    def strip(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: strip(v) for k, v in obj.items() if not k.startswith("_")}
        if isinstance(obj, list):
            return [strip(v) for v in obj]
        return obj

    return strip(json.loads(Path(path).read_text(encoding="utf-8")))


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--experiment-name", required=True,
                        help="Name what the run TESTS, not the hoped-for result")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--policy-checkpoint", required=True)
    parser.add_argument("--policy-version", required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="Resolve and write the manifest, then stop")
    return parser


def resolve_manifest(args: argparse.Namespace, **overrides: Any) -> ExperimentManifest:
    """Build, validate and persist the run manifest."""
    manifest = ExperimentManifest.from_config(
        {
            "experiment_name": args.experiment_name,
            "policy_checkpoint": args.policy_checkpoint,
            "output_directory": str(args.output_dir),
            **overrides,
        },
        repo_root=REPO_ROOT,
    )
    path = manifest.write()
    print(f"[manifest] {path}")
    return manifest


def require_simulator() -> None:
    """Fail early and clearly when the simulator stack is absent."""
    try:
        import habitat  # noqa: F401
    except ImportError:
        sys.exit(
            "ERROR: the Habitat/HomeRobot stack is not importable in this "
            "interpreter.\nThis script needs a GPU host with the simulator "
            "environment installed.\nSee docs/NEXT_EXPERIMENT.md for the "
            "required environment."
        )
