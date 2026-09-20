#!/usr/bin/env python3
"""Two bounded actuator episodes plus one pi0.5 inference contract check."""

from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import time
import urllib.request

import numpy as np

from lmm.env.trossen_habitat_backend import TrossenHabitatBackend, TrossenHabitatConfig
from lmm.env.trossen_mobile_contract import decode_openpi_action


def post(url: str, payload: dict):
    request = urllib.request.Request(
        url,
        data=pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL),
        headers={"Content-Type": "application/octet-stream"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return pickle.loads(response.read())


def frame(raw: dict) -> np.ndarray:
    import cv2

    head = np.asarray(raw["head_rgb"])[..., :3]
    left = np.asarray(raw["left_wrist_rgb"])[..., :3]
    right = np.asarray(raw["right_wrist_rgb"])[..., :3]
    target_h = head.shape[0] // 2
    each_w = head.shape[1] // 2
    left = cv2.resize(left, (each_w, target_h))
    right = cv2.resize(right, (each_w, target_h))
    wrists = np.concatenate((left, right), axis=1)
    wrists = cv2.resize(wrists, (head.shape[1], target_h))
    return np.concatenate((head, wrists), axis=0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--habitat-config", required=True)
    p.add_argument("--server-url", default="http://127.0.0.1:18116")
    p.add_argument("--output-dir", type=pathlib.Path, required=True)
    p.add_argument("--gpu-device-id", type=int, default=2)
    p.add_argument("--episodes", type=int, choices=(1, 2), default=2)
    p.add_argument("--prompt", default="pick up the target object and place it in the goal receptacle")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    health = post(args.server_url, {"op": "health"})
    if health.get("state_dim") != 14 or health.get("action_dim") != 16:
        raise RuntimeError(f"wrong policy contract: {health}")
    backend = TrossenHabitatBackend.make(
        TrossenHabitatConfig(
            habitat_config=args.habitat_config,
            prompt=args.prompt,
            gpu_device_id=args.gpu_device_id,
            max_episode_steps=100,
        )
    )
    records = []
    try:
        for episode in range(args.episodes):
            policy_obs = backend.reset()
            raw = backend.last_raw_observation
            initial_state = np.asarray(policy_obs["state"], dtype=np.float32)
            frames = [frame(raw)]
            post(args.server_url, {"op": "reset", "seed": episode})
            started = time.time()
            policy_action = np.asarray(
                post(
                    args.server_url,
                    {"op": "infer", "observation": policy_obs, "seed": episode},
                ),
                dtype=np.float32,
            )
            decoded = decode_openpi_action(policy_action)

            # Bounded actuator-only motion. Policy output is audited but is not
            # applied because Mobile ALOHA vx/vy is not the differential base's
            # forward/yaw interface and WXAI is not the ViperX training body.
            arms0 = np.r_[initial_state[:6], initial_state[7:13]]
            for step in range(16):
                targets = arms0.copy()
                phase = np.sin(np.pi * step / 15.0)
                targets[0] += (0.08 if episode == 0 else -0.08) * phase
                targets[6] -= (0.08 if episode == 0 else -0.08) * phase
                grip = step / 15.0 if episode == 0 else 1.0 - step / 15.0
                raw = backend.step_arm_grippers(targets, grip, grip)
                frames.append(frame(raw))
            for value in (0.12, 0.12, 0.0, -0.12, -0.12, 0.0):
                raw = backend.step_base_forward_yaw(value, 0.0)
                frames.append(frame(raw))

            import imageio.v2 as imageio

            video = args.output_dir / f"trossen_actuator_episode_{episode}.mp4"
            imageio.mimwrite(video, frames, fps=8, codec="libx264", quality=8)
            records.append(
                {
                    "episode_index": episode,
                    "habitat_episode_id": backend.episode_id,
                    "initial_state": initial_state.tolist(),
                    "policy_action": policy_action.tolist(),
                    "policy_base_velocity_xy": decoded.base_velocity_xy.tolist(),
                    "policy_action_actuated": False,
                    "policy_block_reason": "Mobile ALOHA vx/vy and ViperX embodiment do not exactly match differential-drive WXAI backend",
                    "actuator_steps": 22,
                    "video": str(video),
                    "elapsed_s": time.time() - started,
                }
            )
    finally:
        backend.close()
    report = {"health": health, "episodes": records}
    report_path = args.output_dir / "smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
