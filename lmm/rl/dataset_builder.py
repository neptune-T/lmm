"""Turn verified continuations into grouped training samples.

    root
      |-- continuation 1
      |-- continuation 2
      |-- ...
      `-- continuation K

Group identity is assigned here, once, and written into every record. Later
stages read it back from the record — never from a path.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

from ..rollout.precision_backend import require_high_fidelity
from ..schemas import TrainingSampleRecord, VerificationRecord
from .reward import RewardConfig, reward_from_verification


class DatasetError(ValueError):
    """The verified set cannot form a valid training dataset."""


@dataclass
class DatasetConfig:
    """Dataset assembly rules."""

    min_group_size: int = 2
    require_verified_backend: bool = True
    policy_version: str = ""


def group_id_for(root_id: str, policy_version: str) -> str:
    """Canonical group identity: one root, one policy version, one group."""
    return f"grp::{root_id}::{policy_version}"


def build_dataset(
    verifications: Iterable[VerificationRecord],
    reward_config: RewardConfig,
    config: DatasetConfig,
) -> list[TrainingSampleRecord]:
    """Build grouped, rewarded training samples from verification records.

    Only high-fidelity verified outcomes enter the dataset. Cheap outcomes
    selected the candidates; they never become training targets.
    """
    by_root: dict[str, list[VerificationRecord]] = defaultdict(list)
    for v in verifications:
        if config.require_verified_backend:
            require_high_fidelity(v.verification_backend)
        by_root[v.source_root].append(v)

    samples: list[TrainingSampleRecord] = []
    for root_id in sorted(by_root):
        members = by_root[root_id]
        if len(members) < config.min_group_size:
            continue

        versions = {m.policy_version for m in members}
        if len(versions) > 1:
            raise DatasetError(
                f"root {root_id!r}: verifications mix policy versions "
                f"{sorted(versions)}; a group must share one policy version"
            )
        policy_version = config.policy_version or next(iter(versions))
        gid = group_id_for(root_id, policy_version)

        for m in members:
            samples.append(
                TrainingSampleRecord(
                    root_id=root_id,
                    group_id=gid,
                    rollout_id=m.verification_rollout_id,
                    policy_version=policy_version,
                    precision_backend=m.verification_backend,
                    reward=reward_from_verification(m, reward_config),
                    advantage=0.0,  # filled by lmm.rl.advantages
                )
            )
    return samples


def audit_grouping(samples: Sequence[TrainingSampleRecord]) -> dict[str, object]:
    """Summary used by tests and by the training record.

    Reports whether every group maps to exactly one root — the invariant that
    makes root-level normalisation auditable.
    """
    per_group: dict[str, set[str]] = defaultdict(set)
    sizes: dict[str, int] = defaultdict(int)
    for s in samples:
        per_group[s.group_id].add(s.root_id)
        sizes[s.group_id] += 1
    return {
        "n_samples": len(samples),
        "n_groups": len(per_group),
        "n_roots": len({s.root_id for s in samples}),
        "group_sizes": dict(sizes),
        "all_groups_single_root": all(len(v) == 1 for v in per_group.values()),
    }
