"""Estimate empirical continuation success ``V_hat`` for candidate roots.

    V_hat(s) = successes among branched stochastic continuations
               / total continuation probes

    V_hat(s) ~= P_pi(task success | s)

This is the **empirical** definition of the competence frontier. It replaces
the historical definition, where a frontier was a predefined OVMM skill
boundary (``E5_PRE_PLACE_MINUS_ONE``, ``first_official_place_step``).

Why the change matters here: the frozen TRAIN K=8 pilot branched 8 semantic
pre-place roots x 8 siblings and observed 64/64 failures — zero outcome
variance, therefore zero learning signal (``BRANCH_SUPPORT_NOT_OBSERVED``).
A single trajectory outcome cannot tell you that. Only branched continuation
statistics can, which is why this module refuses to estimate from one sample.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Sequence

from ..schemas import FrontierSummary, ProbeRecord


class EstimationError(ValueError):
    """Probes were inconsistent or insufficient to estimate a frontier."""


def wilson_interval(
    n_success: int, n_total: int, z: float = 1.96
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because probe counts are small
    (``K_probe`` is typically 4-8), where the normal interval is badly wrong
    and degenerate at 0/N and N/N.
    """
    if n_total <= 0:
        return (0.0, 1.0)
    p = n_success / n_total
    z2 = z * z
    denom = 1.0 + z2 / n_total
    center = (p + z2 / (2 * n_total)) / denom
    margin = (
        z
        * math.sqrt(p * (1.0 - p) / n_total + z2 / (4 * n_total * n_total))
        / denom
    )
    return (max(0.0, center - margin), min(1.0, center + margin))


def estimate_root(
    root_id: str,
    probes: Sequence[ProbeRecord],
    *,
    min_probes: int = 2,
) -> FrontierSummary:
    """Compute ``V_hat`` and its uncertainty for a single root.

    All probes must share the same ``root_id``, ``precision_backend`` and
    ``policy_version``: ``V_hat`` is only meaningful for one policy at one
    fidelity. Mixing them is rejected rather than silently averaged.
    """
    if not probes:
        raise EstimationError(f"root {root_id!r}: no probes supplied")
    if len(probes) < min_probes:
        raise EstimationError(
            f"root {root_id!r}: {len(probes)} probe(s) < min_probes={min_probes}. "
            "A frontier must not be inferred from a single continuation."
        )

    bad = [p.root_id for p in probes if p.root_id != root_id]
    if bad:
        raise EstimationError(f"root {root_id!r}: probes from other roots: {set(bad)}")

    backends = {p.precision_backend for p in probes}
    if len(backends) > 1:
        raise EstimationError(
            f"root {root_id!r}: probes mix precision backends {sorted(backends)}. "
            "Cheap and high-fidelity outcomes must not be pooled."
        )

    versions = {p.policy_version for p in probes}
    if len(versions) > 1:
        raise EstimationError(
            f"root {root_id!r}: probes mix policy versions {sorted(versions)}. "
            "V_hat is defined for a single policy."
        )

    n_total = len(probes)
    n_success = sum(1 for p in probes if p.terminal_success)
    rate = n_success / n_total
    lo, hi = wilson_interval(n_success, n_total)

    return FrontierSummary(
        root_id=root_id,
        n_probes=n_total,
        n_success=n_success,
        success_rate=rate,
        uncertainty=(hi - lo) / 2.0,
        ci_low=lo,
        ci_high=hi,
        precision_backend=next(iter(backends)),
        policy_version=next(iter(versions)),
    )


def estimate_all(
    probes: Iterable[ProbeRecord], *, min_probes: int = 2
) -> list[FrontierSummary]:
    """Group probes by ``root_id`` and estimate every root.

    Grouping uses the record field, never the filename.
    """
    by_root: dict[str, list[ProbeRecord]] = defaultdict(list)
    for p in probes:
        by_root[p.root_id].append(p)
    return [
        estimate_root(rid, rs, min_probes=min_probes)
        for rid, rs in sorted(by_root.items())
    ]
