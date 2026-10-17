# v2 NAV decision rules (written 2026-10-17 ~19:30, before B's closed-loop eval and before any B/A paired result)

Analyses: `scripts/alphabot2_pi05/nav_causal.py` (paired NAV, unified onset, drift order), `rescore_trigger.py`,
`stall_diag.py` (15 cm reach-like arm threshold; 10/15/20 cm sensitivity). Inputs are the frozen v2 eval
(A, B, sanity_v1; H_exec 5, fp32) and the teacher-forced probe (`runs/v2/trigger`). No model / config changes.

## 1. B vs A NAV (preset effect-size targets, NOT significance tests)
Same units and noise seeds (train set + OVMM-val seed 0 traces; find from all eval rows). B "clearly better" only if
it beats A on >= 3 of 4 and is not worse on the 4th:
- closed-loop early-reach onset rate (far NAV, 15 cm) <= 1/2 of A's
- median stall fraction >= 0.15 lower
- median closest distance >= 1.0 m smaller
- reached-within-1 m or find: strictly more episodes, and >= 3
Always reported with the paired per-episode statistics (n, better/worse/tie, median diff, bootstrap 95 % CI of the
mean diff, exact sign-test p). A target met with a CI spanning 0 / sign test p > 0.05 is reported as "target met,
not statistically supported"; small counts (e.g. 3/30 approach) are never sufficient on their own.

## 2. Task-progress representation vs recovery data (only if B is not clearly better)
Unified onset statistic: first early-reach per eligible far-NAV query, teacher-forced (TF, expert observations)
and closed loop (CL) identically; CL exposure split by path state at each query; drift ordering per episode.
- Task-progress first: TF rate (training units) >= 1/2 of the CL on-path rate, AND >= half of the CL first triggers
  have no drift before them (no >= 1 m path deviation before the trigger).
- Recovery data first: CL off-path rate >= 3x both the TF rate and the CL on-path rate, AND >= 2/3 of CL first
  triggers are preceded by >= 1 m drift at least 50 steps earlier.
- Must hold at 15 cm and not flip at 10 or 20 cm. Otherwise: INCONCLUSIVE, reported with the numbers; no
  attribution forced, and never both interventions at once.
