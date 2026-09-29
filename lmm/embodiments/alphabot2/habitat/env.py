"""Factory for OVMM tasks/scenes with the AlphaBot 2 dual-arm mobile embodiment.

Run inside the HomeRobot Habitat environment with the source trees first on
``PYTHONPATH`` (``scripts/alphabot2_habitat/run_habitat.sh``): the conda
env's site-packages carries an older, unpatched habitat.
"""
from __future__ import annotations

import os

import numpy as np

HAB_CONFIG = os.environ.get(
    "ALPHABOT2_OVMM_CONFIG",
    "/nix/yanjun/zts/lmm/home-robot/src/third_party/habitat-lab/habitat-lab/habitat/config/benchmark/ovmm/ovmm_alphabot2.yaml")
#: Stretch-specific pieces of the OVMM config, deleted (not edited) for AlphaBot 2
STRETCH_ONLY = ("habitat.task.actions.arm_action", "habitat.task.actions.base_velocity",
                "habitat.task.lab_sensors.joint_sensor", "habitat.task.lab_sensors.joint_velocity_sensor")


def make_env(gpu: int = 0, overrides: list | None = None, split: str | None = None):
    import habitat
    from habitat_baselines.config.default import get_config

    from . import alphabot2_actions as acts               # registers the action and the sensor
    acts.register_robot()
    ov = [f"~{k}" for k in STRETCH_ONLY]
    ov += [f"habitat.simulator.habitat_sim_v0.gpu_device_id={gpu}"]
    if split:
        ov.append(f"habitat.dataset.split={split}")
    ov += list(overrides or [])
    cfg = get_config(HAB_CONFIG, overrides=ov)
    env = habitat.Env(config=cfg)
    # OVMM episodes are stored serialized (object transforms and nav viewpoints are indices into
    # dataset-wide matrices); only OVMMEpisodeIterator.__next__ deserializes them, on a deep copy.
    env._alphabot2_raw = {e.episode_id: e for e in env.episodes}
    env._alphabot2_iter0 = env.episode_iterator
    return env


def reset_to(env, episode_id):
    """Reset into one specific OVMM episode, deserialized exactly as the stock iterator does."""
    from habitat.datasets.ovmm.ovmm_dataset import OVMMEpisodeIterator
    it0 = env._alphabot2_iter0
    env.episode_iterator = OVMMEpisodeIterator(it0.viewpoints, it0.transformations,
                                               [env._alphabot2_raw[str(episode_id)]], cycle=True, shuffle=False)
    return env.reset()


#: Simulated seconds per policy action and physics substeps (asserted by the smoke).
ACTION_DT_S, PHYSICS_DT_S = 0.1, 0.005


def robot(env):
    return env.sim.get_agent_data(0).articulated_agent


def action_obj(env):
    return env.task.actions["alphabot2_action"]


def step18(env, joint18):
    """ONE policy action (JointMobileAction18) = ONE env.step."""
    a = np.asarray(joint18, np.float32).reshape(18)
    return env.step({"action": "alphabot2_action", "action_args": {"alphabot2_action": a}})


def hold_action(env) -> np.ndarray:
    """JointMobileAction18 that holds the current arm joints and gripper fractions, base still."""
    p = robot(env).proprio16()
    return np.r_[p, 0.0, 0.0]
