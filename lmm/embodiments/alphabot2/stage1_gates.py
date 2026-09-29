"""Stage-1 demonstration quality gates (numpy only).

An episode enters the Stage-1 SFT TRAIN set only if EVERY gate passes;
otherwise it is quarantined with the failing gates recorded. Nothing is
silently mixed in. Stage labels and other privileged fields are metadata and
are checked to be absent from the policy-input keys.
"""
from __future__ import annotations

import numpy as np

PHYSICS_STEPS_PER_ACTION = 20
CAMERAS = ("cam_front", "cam_left_wrist", "cam_right_wrist", "cam_chest")
POLICY_INPUT_KEYS = ("images", "proprio", "instruction")
PRIVILEGED_KEYS = ("stage_id", "events", "obj_pose", "chassis", "grasp", "config", "cells")
REPLAY_PROPRIO_TOL = 1e-6
QUAT_NORM_TOL = 1e-3


def gate_episode(low: dict, images: dict | None, meta: dict, replay: dict | None) -> dict:
    """Return {gate_name: (passed, detail)} for one recorded episode.

    ``low``: lowdim arrays (proprio, action, timing, ik_valid, collision...).
    ``images``: {camera: (T, H, W, 3) uint8} or None if not loaded.
    ``meta``: the episode metadata (outcome, per-reset sync check, policy inputs).
    ``replay``: replay-validation result or None if not run.
    """
    g: dict[str, tuple[bool, object]] = {}
    prop, act = np.asarray(low["proprio"]), np.asarray(low["action"])
    T = len(act)
    g["task_success"] = (bool(meta.get("outcome", {}).get("success")), meta.get("outcome", {}).get("reason"))
    ncol = int(np.sum(np.asarray(low["collision"]))) if "collision" in low else None
    g["collision_free"] = (ncol == 0, {"collision_steps": ncol})
    # --- temporal synchronization (timing contract)
    try:
        f_img, f_prop = np.asarray(low["image_frame"]), np.asarray(low["proprio_frame"])
        f_obs, f_act = np.asarray(low["obs_frame"]), np.asarray(low["action_frame"])
        ps = np.asarray(low["policy_step"])
        same = bool(np.array_equal(f_img, f_prop))
        lag = np.unique(f_obs - f_img).tolist()
        acts_at_obs = bool(np.array_equal(f_act, f_obs))
        spacing = np.unique(np.diff(f_obs)).tolist() if T > 1 else [PHYSICS_STEPS_PER_ACTION]
        steps_ok = bool(np.array_equal(ps, np.arange(T)))
        reset_ok = bool(meta.get("sync_check", {}).get("ok"))
        want_lag = [int(meta.get("render_lag", 0)) * PHYSICS_STEPS_PER_ACTION]
        ok = same and acts_at_obs and steps_ok and reset_ok and lag == want_lag and spacing == [PHYSICS_STEPS_PER_ACTION]
        g["temporal_sync"] = (ok, {"image_eq_proprio_frame": same, "obs_minus_image_frames": lag,
                                   "action_at_obs_frame": acts_at_obs, "obs_frame_spacing": spacing,
                                   "policy_steps_contiguous": steps_ok, "per_reset_sync_check": reset_ok})
    except KeyError as e:
        g["temporal_sync"] = (False, f"missing timing field {e}")
    ikv = np.asarray(low.get("ik_valid", np.zeros((0, 2), bool)))
    g["ik_valid"] = (bool(ikv.shape == (T, 2) and ikv.all()), {"invalid_steps": int((~ikv.all(axis=1)).sum()) if ikv.size else None})
    shapes_ok = prop.shape == (T, 16) and act.shape == (T, 18) and T > 0
    if images is not None:
        shapes_ok = shapes_ok and all(k in images and images[k].shape == (T, 240, 320, 3) and images[k].dtype == np.uint8
                                      for k in CAMERAS)
    g["shapes"] = (bool(shapes_ok), {"T": T, "proprio": list(prop.shape), "action": list(act.shape)})
    qn = np.concatenate([np.linalg.norm(act[:, 3:7], axis=1), np.linalg.norm(act[:, 10:14], axis=1)]) if T else np.ones(1)
    fin = bool(np.isfinite(prop).all() and np.isfinite(act).all())
    g["finite"] = (fin and bool(np.all(np.abs(qn - 1) < QUAT_NORM_TOL)),
                   {"finite": fin, "max_quat_norm_dev": float(np.max(np.abs(qn - 1)))})
    if images is not None:
        stds = {k: float(images[k][:: max(1, T // 8)].reshape(-1, 3).std()) for k in CAMERAS if k in images}
        black = {k: int(np.sum(images[k].reshape(T, -1).max(axis=1) == 0)) for k in CAMERAS if k in images}
        g["cameras"] = (len(stds) == len(CAMERAS) and all(v > 1.0 for v in stds.values()) and not any(black.values()),
                        {"pixel_std": stds, "all_black_frames": black})
    else:
        g["cameras"] = (False, "images not checked")
    if replay is None:
        g["replay"] = (False, "not run")
    else:
        ok = bool(replay.get("success")) and float(replay.get("proprio_max_abs_diff", np.inf)) <= REPLAY_PROPRIO_TOL
        g["replay"] = (ok, replay)
    leaked = [k for k in PRIVILEGED_KEYS if k in meta.get("policy_inputs", [])]
    g["no_privileged_inputs"] = (not leaked, {"leaked": leaked})
    return g


def passed(gates: dict) -> bool:
    return all(v[0] for v in gates.values())


def failing(gates: dict) -> list[str]:
    return [k for k, v in gates.items() if not v[0]]
