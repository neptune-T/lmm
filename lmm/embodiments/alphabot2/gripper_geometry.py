"""Measured pad-gap calibration shared by the simulator and CPU data audit."""
import numpy as np

GAP_Q = np.array([0.0, 0.12, 0.24, 0.36, 0.48, 0.60, 0.72, 0.82])
GAP_M = np.array([0.0142, 0.0231, 0.0324, 0.0422, 0.0521, 0.0622, 0.0722, 0.0804])
MIN_GRASP_WIDTH_M, MAX_GRASP_WIDTH_M = 0.015, 0.078


def q_for_aperture(width_m):
    return float(np.interp(width_m, GAP_M, GAP_Q))


def audit_gripper(cmd, nxt, held, widths):
    """Use post-action grasp-manager evidence, not closure alone, to identify holding.

    Empty hands must follow the commanded fraction. Held hands must match the
    measured object's aperture. Tolerance covers float32 joint readback only.
    """
    cmd, nxt, held, widths = map(np.asarray, (cmd, nxt, held, widths))
    if not (cmd.shape == nxt.shape == held.shape == widths.shape and cmd.ndim == 1):
        raise ValueError("Gripper audit arrays must have equal one-dimensional shapes")
    if not all(np.isfinite(x).all() for x in (cmd, nxt, held, widths)):
        raise ValueError("Nonfinite gripper audit evidence")
    if not (np.equal(held, np.floor(held)).all() and (held >= -1).all()):
        raise ValueError("Invalid held-object index")
    holding = held >= 0
    expected = cmd.copy()
    expected[holding] = 1.0 - np.interp(widths[holding], GAP_M, GAP_Q) / 0.82
    physical = ((cmd >= 0) & (cmd <= 1) & (nxt >= -1e-6) & (nxt <= 1 + 1e-6))
    physical &= np.where(holding, (cmd >= .5) & (widths >= MIN_GRASP_WIDTH_M)
                         & (widths <= MAX_GRASP_WIDTH_M), widths == 0)
    error = np.abs(nxt - expected)
    bad = ~(physical & (error <= 1e-5))
    return {"pass": bool(not bad.any()), "mode": "held_object_aperture",
            "holding_steps": int(holding.sum()), "empty_steps": int((~holding).sum()),
            "max_abs_fraction_error": float(error.max()) if len(error) else None,
            "bad_steps": np.flatnonzero(bad).tolist()}
