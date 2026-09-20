"""Select frontier roots from empirical continuation statistics.

A root is on the frontier when its estimated continuation success lies in an
intermediate band:

    V_hat ~ 0   -> too hard, no signal to learn from
    V_hat ~ 1   -> already solved, nothing to learn
    in between  -> useful learning signal

The band is **configuration, not a conclusion**. ``frontier_low`` and
``frontier_high`` default to 0.2/0.8 but must be supplied from
``configs/frontier/*.json`` for any real run, so a threshold can never harden
into an unexamined scientific claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from ..schemas import FrontierSummary


@dataclass
class FrontierSelectorConfig:
    """Selection band and confidence policy."""

    frontier_low: float = 0.2
    frontier_high: float = 0.8
    min_probes: int = 2
    max_uncertainty: float | None = None
    require_outcome_variance: bool = True
    max_selected: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.frontier_low <= self.frontier_high <= 1.0:
            raise ValueError(
                "require 0 <= frontier_low <= frontier_high <= 1, got "
                f"{self.frontier_low} / {self.frontier_high}"
            )


def classify(summary: FrontierSummary, config: FrontierSelectorConfig) -> str:
    """Classify one root: ``too_hard`` / ``frontier`` / ``solved`` / rejection."""
    if summary.n_probes < config.min_probes:
        return "insufficient_probes"
    if (
        config.max_uncertainty is not None
        and summary.uncertainty > config.max_uncertainty
    ):
        return "too_uncertain"
    if config.require_outcome_variance and summary.n_success in (0, summary.n_probes):
        # No sibling disagreement => no contrastive signal for a group-relative
        # optimizer, regardless of where the point estimate sits.
        return "no_outcome_variance"
    if summary.success_rate < config.frontier_low:
        return "too_hard"
    if summary.success_rate > config.frontier_high:
        return "solved"
    return "frontier"


def select(
    summaries: Iterable[FrontierSummary], config: FrontierSelectorConfig
) -> list[FrontierSummary]:
    """Mark and return the frontier roots.

    Returns new summaries with ``selected_as_frontier`` and
    ``selection_reason`` set, ranked by proximity to the band centre (most
    balanced outcome diversity first).
    """
    centre = (config.frontier_low + config.frontier_high) / 2.0
    out: list[FrontierSummary] = []
    for s in summaries:
        verdict = classify(s, config)
        marked = FrontierSummary.from_dict(
            {
                **s.to_dict(),
                "selected_as_frontier": verdict == "frontier",
                "selection_reason": verdict,
            }
        )
        out.append(marked)

    out.sort(key=lambda s: (not s.selected_as_frontier, abs(s.success_rate - centre)))
    if config.max_selected is not None:
        for s in out[config.max_selected :]:
            if s.selected_as_frontier:
                s.selected_as_frontier = False
                s.selection_reason = "over_budget"
    return out


def frontier_only(
    summaries: Iterable[FrontierSummary], config: FrontierSelectorConfig
) -> list[FrontierSummary]:
    """Convenience: just the selected roots, ranked."""
    return [s for s in select(summaries, config) if s.selected_as_frontier]


def uncertain_regions(
    summaries: Sequence[FrontierSummary],
    steps_by_root: dict[str, int],
    config: FrontierSelectorConfig,
) -> list[int]:
    """Trajectory steps worth a denser second pass.

    Feeds :func:`lmm.frontier.candidate_proposer.refine_around`. A region is
    worth refining when a coarse root either sits in the band or is too
    uncertain to classify.
    """
    steps: list[int] = []
    for s in summaries:
        verdict = classify(s, config)
        if verdict in ("frontier", "too_uncertain") and s.root_id in steps_by_root:
            steps.append(steps_by_root[s.root_id])
    return sorted(set(steps))
