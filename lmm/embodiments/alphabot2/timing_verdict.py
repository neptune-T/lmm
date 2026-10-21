"""Classify timing evidence without treating unobservable frames as successes.

Errors/separations are mean absolute RGB differences on the 0..255 scale.
Constants are fixed before re-running the simulator, not fitted to a result.
"""
import math

MATCH_TOL = 0.05
SEPARATION_MIN = 0.05
MIN_VALID = 5


def classify(errors, separation):
    if not all(math.isfinite(v) for v in (*errors.values(), separation)):
        return "invalid"
    if separation < SEPARATION_MIN:
        return "ambiguous"
    best = min(errors, key=errors.get)
    other = min(v for k, v in errors.items() if k != best)
    if other - errors[best] <= MATCH_TOL:
        return "ambiguous"
    if errors[best] > MATCH_TOL:
        return "mismatch"
    return "aligned" if best == 0 else "shifted"


def summarize(rows):
    counts = {s: sum(r["status"] == s for r in rows)
              for s in ("aligned", "ambiguous", "shifted", "mismatch", "invalid")}
    controls = {str(o): sum(r["negative_controls"][str(o)] == "shifted" for r in rows
                            if r["status"] == "aligned") for o in (-1, 1)}
    passed = (counts["aligned"] >= MIN_VALID
              and not any(counts[s] for s in ("shifted", "mismatch", "invalid"))
              and all(n == counts["aligned"] for n in controls.values()))
    return {"pass": passed, "counts": counts, "negative_control_rejections": controls,
            "min_valid": MIN_VALID, "match_tolerance": MATCH_TOL,
            "separation_min": SEPARATION_MIN}
