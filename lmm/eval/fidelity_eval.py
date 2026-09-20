"""Cheap-vs-high-fidelity agreement (question Q3).

Cheap rollout is useful only if it *orders* candidates well enough that
high-fidelity verification recovers genuinely good trajectories. This module
measures that, and nothing here ever overwrites a cheap outcome with a
verified one — they stay in separate fields.

Metrics
-------
``ranking``      Spearman rho and Kendall tau between cheap score and verified
                 outcome
``retention``    fraction of the true top-k (by verified outcome) that the
                 cheap ranking placed in its own top-k
``agreement``    per-candidate confusion between cheap and verified outcomes
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..schemas import VerificationRecord


@dataclass
class FidelityResult:
    """Cheap/high-fidelity agreement over one candidate set."""

    n: int
    spearman_rho: float
    kendall_tau: float
    top_k: int
    top_k_retention: float
    cheap_success_rate: float
    verified_success_rate: float
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int

    def to_dict(self) -> dict[str, object]:
        return {
            "eval_type": "cheap_vs_high_fidelity",
            "n": self.n,
            "spearman_rho": self.spearman_rho,
            "kendall_tau": self.kendall_tau,
            "top_k": self.top_k,
            "top_k_retention": self.top_k_retention,
            "cheap_success_rate": self.cheap_success_rate,
            "verified_success_rate": self.verified_success_rate,
            "confusion": {
                "true_positive": self.true_positive,
                "false_positive": self.false_positive,
                "false_negative": self.false_negative,
                "true_negative": self.true_negative,
            },
        }


def _ranks(values: Sequence[float]) -> list[float]:
    """Average ranks, handling ties."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation (Pearson on ranks; tie-safe)."""
    if len(x) < 2:
        return 0.0
    rx, ry = _ranks(x), _ranks(y)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


def kendall_tau(x: Sequence[float], y: Sequence[float]) -> float:
    """Kendall tau-b, tolerant of the ties that binary outcomes produce."""
    n = len(x)
    if n < 2:
        return 0.0
    conc = disc = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx == 0 and dy == 0:
                tx += 1; ty += 1
            elif dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif (dx > 0) == (dy > 0):
                conc += 1
            else:
                disc += 1
    n0 = conc + disc
    denom = ((n0 + tx) * (n0 + ty)) ** 0.5
    return (conc - disc) / denom if denom else 0.0


def evaluate(
    records: Sequence[VerificationRecord], *, top_k: int = 4
) -> FidelityResult:
    """Measure how well cheap scores predicted verified outcomes."""
    if not records:
        raise ValueError("no verification records to evaluate")

    cheap = [r.cheap_score for r in records]
    verified = [1.0 if r.verified_success else 0.0 for r in records]
    n = len(records)
    k = min(top_k, n)

    by_cheap = sorted(range(n), key=lambda i: -cheap[i])[:k]
    by_verified = sorted(range(n), key=lambda i: -verified[i])[:k]
    retention = len(set(by_cheap) & set(by_verified)) / k if k else 0.0

    tp = sum(1 for r in records if r.cheap_outcome and r.verified_success)
    fp = sum(1 for r in records if r.cheap_outcome and not r.verified_success)
    fn = sum(1 for r in records if not r.cheap_outcome and r.verified_success)
    tn = sum(1 for r in records if not r.cheap_outcome and not r.verified_success)

    return FidelityResult(
        n=n,
        spearman_rho=spearman(cheap, verified),
        kendall_tau=kendall_tau(cheap, verified),
        top_k=k,
        top_k_retention=retention,
        cheap_success_rate=sum(1 for r in records if r.cheap_outcome) / n,
        verified_success_rate=sum(verified) / n,
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        true_negative=tn,
    )
