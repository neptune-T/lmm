"""Replay ordered expert simulator events, retaining contact-query side effects."""
import numpy as np


def replay_events(events, step, contacts, after_step=None):
    n = 0
    for event in events:
        kind = event["kind"]
        if kind == "step":
            action = np.asarray(event["action"], dtype=np.float64)
            if action.shape != (18,) or not np.isfinite(action).all():
                raise ValueError("Invalid replay action")
            step(action)
            n += 1
            if after_step is not None:
                after_step()
        elif kind == "contacts":
            contacts()
        else:
            raise ValueError(f"Unknown replay event: {kind}")
    return n
