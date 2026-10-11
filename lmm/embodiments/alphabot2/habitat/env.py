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
    "/workspace/runtime/home-robot/src/third_party/habitat-lab/habitat-lab/habitat/config/benchmark/ovmm/ovmm_alphabot2.yaml")
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


def episode_seed(episode_id, replica: int = 0) -> int:
    """Seed bank for OVMM's own start sampling: one deterministic seed per (episode_id, replica_id),
    stable across processes and hosts. Replica 0 keeps the original per-episode seed."""
    import zlib
    key = f"alphabot2-ovmm:{episode_id}" if int(replica) == 0 else f"alphabot2-ovmm:{episode_id}:{int(replica)}"
    return zlib.crc32(key.encode()) & 0x7FFFFFFF


def parse_unit(unit) -> tuple[str, int]:
    """Collection unit id "EPISODE#REPLICA" (or a bare episode id = replica 0)."""
    s = str(unit)
    if "#" in s:
        e, r = s.split("#", 1)
        return e, int(r)
    return s, 0


def unit_id(episode_id, replica: int = 0) -> str:
    return f"{episode_id}#{int(replica)}"


def reset_to(env, episode_id, replica: int = 0):
    """Reset into one specific OVMM episode, deserialized exactly as the stock iterator does.

    OVMM TRAIN episodes carry no start pose (all 37,477 are zero): OVMM samples one at every reset
    (random navigable point on the active island + random yaw + random goal). That sampler is kept,
    but seeded per episode (python/numpy/sim pathfinder/task via env.seed) so a fresh reset of the
    same episode is identical -- needed for fresh-reset replay. ``replica`` selects another seed from
    the (episode_id, replica_id) bank, i.e. another OVMM-sampled start. VAL episodes have fixed starts."""
    from habitat.datasets.ovmm.ovmm_dataset import OVMMEpisodeIterator
    it0 = env._alphabot2_iter0
    env.episode_iterator = OVMMEpisodeIterator(it0.viewpoints, it0.transformations,
                                               [env._alphabot2_raw[str(episode_id)]], cycle=True, shuffle=False)
    # Loading a new scene recomputes the AlphaBot navmesh, which resets the pathfinder's RNG after
    # env.seed(); measured: the first reset after a scene switch sampled a different start than every
    # later reset of the same episode. Load the scene with a throw-away reset first, then seed + reset.
    #
    # FROZEN RESET CONTRACT: hard world rebuild -> seeded task reset -> episode. Used by recording, replay,
    # pilot, collection and eval alike; there is no soft-reset fast path. rearrange_sim.reconfigure only
    # rebuilds the world when the scene or object set changes and otherwise re-poses objects in the running
    # Bullet world, which made object dynamics depend on earlier episodes in the process (pilot replay
    # rejections). A/B (8 units, one long-lived process): identical start poses; the history-dependent
    # divergence disappears; what remains is the placed target's own rotation after release (tipping).
    env.sim.prev_scene_id = None                  # force the hard rebuild ...
    env.sim._prev_obj_names = None
    env.seed(episode_seed(episode_id, replica))   # ... of a throw-away reset (also loads the navmesh, which
    try:                                          # resets the pathfinder RNG after env.seed)
        env.reset()
    except AssertionError:                        # OVMM's reachability assert: the real reset decides
        pass
    env._alphabot2_scene = scene
    env.seed(episode_seed(episode_id, replica))
    return env.reset()


def fix_start(env) -> dict:
    """Embodiment deviation, applied to every AlphaBot episode: OVMM start poses were sampled on the
    Stretch navmesh (radius 0.3); when one is not navigable for AlphaBot (radius per the audit), the
    robot is moved to the nearest navigable point, yaw unchanged. Returns the move (m) for the record."""
    import magnum as mn
    rob = robot(env)
    pf = env.sim.pathfinder
    p = rob.sim_obj.translation
    if pf.is_navigable(p):
        return {"moved_m": 0.0, "navigable": True}
    s = pf.snap_point(p)
    if not np.isfinite(np.array(s)).all():
        return {"moved_m": None, "navigable": False}
    tr = rob.sim_obj.transformation
    rob.sim_obj.transformation = mn.Matrix4.from_(tr.rotation(), s)
    return {"moved_m": float((s - p).length()), "navigable": True}


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
