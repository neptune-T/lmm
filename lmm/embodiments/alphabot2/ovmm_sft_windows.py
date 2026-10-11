"""SFT action-chunk windows of AlphaBot 2 OVMM datasets (numpy only; interface gate + tests).

pi0.5 predicts H_PRED = 50 steps. LeRobot builds each training target from the CONVERTED dataset at load
time: chunk index t+k beyond the episode end is clamped to the episode's last frame and flagged
action_is_pad (lerobot 0.6.1 dataset_reader._get_query_indices). The post-release settle interval is
truncated at conversion (first keep_settle frames kept), i.e. BEFORE any chunk exists; padded steps are
excluded from the SFT loss (ovmm_model.masked_loss). Hence no training window may contain more than
keep_settle real settle targets.
"""
import numpy as np


def expected_pad(T: int, t: int, chunk: int = 50) -> np.ndarray:
    """action_is_pad of the window starting at frame t of a T-frame episode."""
    return np.array([t + k >= T for k in range(chunk)])


def window_exposure(episode_index, stage, settle_id, chunk: int = 50) -> dict:
    """Per training window (every frame t): number of VALID (non-pad) targets that are settle frames."""
    ep, st = np.asarray(episode_index), np.asarray(stage)
    counts, pre_release_with_settle, n_pre = [], 0, 0
    for e in np.unique(ep):
        s = st[ep == e] == settle_id
        T = len(s)
        cs = np.r_[0, np.cumsum(s)]
        for t in range(T):
            c = int(cs[min(T, t + chunk)] - cs[t])                      # valid targets only: [t, min(T, t+chunk))
            counts.append(c)
            if not s[t]:
                n_pre += 1
                pre_release_with_settle += int(c > 0)
    counts = np.asarray(counts)
    return {"windows": int(len(counts)), "max_settle_targets_per_window": int(counts.max()) if len(counts) else 0,
            "mean_settle_targets_per_window": float(counts.mean()) if len(counts) else 0.0,
            "non_settle_windows_touching_settle_fraction": pre_release_with_settle / max(1, n_pre)}
