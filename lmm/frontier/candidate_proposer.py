"""Propose candidate states from a full-task trajectory.

A candidate is a trajectory step worth snapshotting and probing. Snapshotting
*every* simulator step is expensive, so proposal is a configurable, auditable
policy — and every proposal carries the reason it was made.

Strategies
----------
``fixed_interval``  every N steps (the default coarse scan)
``semantic_event``  steps where an observer flagged an event
``manual_debug``    an explicit, hand-supplied list of steps

Coarse-to-fine search is supported by :func:`refine_around`, which proposes a
denser scan inside an uncertain region identified by a coarse pass. Nothing
here decides what is *interesting* — that is the estimator's and selector's
job.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

VALID_STRATEGIES = ("fixed_interval", "semantic_event", "manual_debug")


@dataclass(frozen=True)
class Candidate:
    """A proposed snapshot step and the reason it was proposed."""

    step: int
    reason: str


@dataclass
class CandidateProposerConfig:
    """Config for candidate proposal. Values come from ``configs/frontier/``."""

    strategy: str = "fixed_interval"
    interval: int = 30
    max_candidates: int = 16
    skip_first_steps: int = 0
    event_names: tuple[str, ...] = ()
    manual_steps: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"unknown strategy {self.strategy!r}; valid: {VALID_STRATEGIES}"
            )
        if self.interval < 1:
            raise ValueError("interval must be >= 1")


def propose(
    trajectory_length: int,
    config: CandidateProposerConfig,
    step_events: Sequence[Sequence[str]] | None = None,
) -> list[Candidate]:
    """Propose candidate steps from a trajectory.

    ``step_events`` is the per-step event annotation produced by observers. It
    is read only to *locate* candidates; it never alters the trajectory.
    """
    if trajectory_length <= 0:
        return []

    if config.strategy == "fixed_interval":
        steps = [
            s
            for s in range(config.skip_first_steps, trajectory_length, config.interval)
        ]
        reason = f"fixed_interval:{config.interval}"
        out = [Candidate(s, reason) for s in steps]

    elif config.strategy == "semantic_event":
        if step_events is None:
            raise ValueError("semantic_event strategy requires step_events")
        wanted = set(config.event_names)
        out = []
        for step, events in enumerate(step_events):
            if step < config.skip_first_steps or step >= trajectory_length:
                continue
            hit = [e for e in events if not wanted or e in wanted]
            if hit:
                out.append(Candidate(step, f"semantic_event:{hit[0]}"))

    else:  # manual_debug
        out = [
            Candidate(s, "manual_debug")
            for s in config.manual_steps
            if 0 <= s < trajectory_length
        ]

    return _cap(out, config.max_candidates)


def refine_around(
    center_step: int,
    coarse_interval: int,
    trajectory_length: int,
    fine_interval: int = 10,
    max_candidates: int = 8,
) -> list[Candidate]:
    """Second pass of a coarse-to-fine search.

    Proposes a denser scan spanning one coarse interval either side of
    ``center_step`` — the region a coarse pass flagged as uncertain. Returns
    candidates tagged ``refine:<center>`` so the two passes stay separable in
    the records.
    """
    if fine_interval < 1:
        raise ValueError("fine_interval must be >= 1")
    lo = max(0, center_step - coarse_interval)
    hi = min(trajectory_length, center_step + coarse_interval + 1)
    steps = list(range(lo, hi, fine_interval))
    if center_step not in steps and lo <= center_step < hi:
        steps.append(center_step)
    reason = f"refine:{center_step}"
    return _cap([Candidate(s, reason) for s in sorted(set(steps))], max_candidates)


def _cap(cands: list[Candidate], limit: int) -> list[Candidate]:
    """Evenly thin a candidate list to at most ``limit`` entries."""
    if limit <= 0 or len(cands) <= limit:
        return cands
    stride = len(cands) / limit
    return [cands[int(i * stride)] for i in range(limit)]
