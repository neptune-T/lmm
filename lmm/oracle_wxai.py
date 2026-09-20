#!/usr/bin/env python3
"""Minimal WXAI OVMM demonstration oracle.

Reuses OVMM episodes, HSSD scenes, object/receptacle metadata, the navmesh and
the official success measures. Replaces Stretch-specific manipulation with the
calibrated WXAI embodiment.

Privileged semantic state is used freely -- this is an OFFLINE demonstration
generator. Nothing it knows is exposed in the recorded observation: the
recorded stream is exactly head/wrist RGB + 14-D proprio + language, which is
all a learned policy will ever see. Phase labels are recorded only as episode
metadata for analysis, never as a policy input channel.

Temporal contract (see docs/WXAI_TEMPORAL_CONTRACT.md):
    simulator control frequency : 120 Hz   (measured)
    recorded dataset FPS        :  15 Hz   (chosen)
    sim ticks per recorded action:   8     (120 / 15)
One recorded 16-D action is applied unchanged for all 8 ticks, so arms,
grippers and base always belong to the same logical transition.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
import magnum as mn

sys.path.insert(0, "/nix/yanjun/zts/lmm/canonical_lmm")

CONTROL_HZ = 120.0
DATASET_FPS = 15.0
TICKS_PER_ACTION = int(round(CONTROL_HZ / DATASET_FPS))   # 8
LIN, ANG = 0.25, 0.5
GRIPPER_OPEN_M = 0.044
GRASP_THRESH_DIST = 0.15
SNAP_THRESHOLD_G = 0.5

ARM_SLOTS = {"left": slice(0, 6), "right": slice(6, 12)}
POLICY_ARM_DIMS = {"left": list(range(0, 6)), "right": list(range(7, 13))}
POLICY_GRIP_DIM = {"left": 6, "right": 13}


def wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


class WXAIOracle:
    def __init__(self, env, arm: str, rng: np.random.Generator):
        self.env, self.arm, self.rng = env, arm, rng
        self.sim = env.sim
        self.robot = env.sim.get_agent_data(0).articulated_agent
        self.safe_arm = np.asarray(self.robot.params.arm_init_params, dtype=np.float64).copy()
        self.frames: list[dict] = []
        self.events: list[dict] = []
        self.phase = "init"
        self.sim_tick = 0

    # ---------- kinematics ----------
    def heading(self) -> float:
        f = self.robot.base_transformation.transform_vector(mn.Vector3(1, 0, 0))
        return float(np.arctan2(f.z, f.x))

    def base_xz(self) -> np.ndarray:
        p = np.asarray(self.robot.base_pos, dtype=np.float64)
        return np.array([p[0], p[2]], dtype=np.float64)

    def snap_to_navmesh(self) -> float:
        """OVMM episodes can start slightly off-navmesh.

        Measured on episode 743: start_is_navigable=False with a 22.4 mm snap
        offset. BaseVelAction's step_filter then pins the robot after that same
        22.4 mm, which reads as a totally stuck base. Snapping once at reset
        removes the artefact; the pose moves by millimetres.
        """
        pf = self.sim.pathfinder
        cur = np.asarray(self.robot.base_pos, dtype=np.float64)
        if pf.is_navigable(cur):
            return 0.0
        snapped = np.asarray(pf.snap_point(cur), dtype=np.float64)
        if not np.all(np.isfinite(snapped)):
            return -1.0
        self.robot.base_pos = mn.Vector3(*snapped.astype(np.float32))
        return float(np.linalg.norm(snapped - cur))

    def ee_pos(self) -> np.ndarray:
        idx = 0 if self.arm == "left" else 1
        return np.asarray(self.robot.ee_transform(idx).translation, dtype=np.float64)

    def full_arm(self) -> np.ndarray:
        return np.asarray(self.robot.arm_joint_pos, dtype=np.float64).copy()

    def fk_ee(self, arm12: np.ndarray) -> np.ndarray:
        """Forward kinematics probe: set joints, read EE, restore."""
        saved = self.full_arm()
        self.robot.arm_joint_pos = arm12.astype(np.float32)
        out = self.ee_pos()
        self.robot.arm_joint_pos = saved.astype(np.float32)
        return out

    # ---------- 16-D action assembly ----------
    def make_action(self, arm12, grip_active, forward_mps, yaw_radps) -> np.ndarray:
        a = np.zeros(16, dtype=np.float64)
        a[0:6] = arm12[0:6]
        a[7:13] = arm12[6:12]
        a[6] = grip_active if self.arm == "left" else 0.0
        a[13] = grip_active if self.arm == "right" else 0.0
        a[14] = float(np.clip(forward_mps, -LIN, LIN))
        a[15] = float(np.clip(yaw_radps, -ANG, ANG))
        return a

    def step_control(self, action16: np.ndarray, phase: str) -> dict:
        """One recorded control transition = TICKS_PER_ACTION simulator ticks."""
        a = np.asarray(action16, dtype=np.float64)
        arm12 = np.concatenate([a[0:6], a[7:13]])
        gl, gr = float(a[6]), float(a[13])
        fwd = float(np.clip(a[14] / LIN, -1, 1))
        yaw = float(np.clip(a[15] / ANG, -1, 1))
        held_before = bool(self.sim.grasp_mgr.is_grasped) if hasattr(self.sim, "grasp_mgr") else None
        mgrs = self.env.sim.agents_mgr[0].grasp_mgrs
        grasped_before = [bool(m.is_grasped) for m in mgrs]

        obs = None
        for _ in range(TICKS_PER_ACTION):
            obs = self.env.step({"action": ("arm_action", "base_velocity"), "action_args": {
                "arm_action": arm12.astype(np.float32),
                "left_grip_action": np.asarray([2 * gl - 1], np.float32),
                "right_grip_action": np.asarray([2 * gr - 1], np.float32),
                "base_vel": np.asarray([fwd, yaw], np.float32)}})
            self.sim_tick += 1
        grasped_after = [bool(m.is_grasped) for m in mgrs]

        for i, (b, aft) in enumerate(zip(grasped_before, grasped_after)):
            if b != aft:
                self.events.append({
                    "dataset_step": len(self.frames), "sim_tick": self.sim_tick,
                    "ee_index": i, "event": "snap" if aft else "desnap",
                    "gripper_cmd": gl if i == 0 else gr,
                    "crossed_threshold": bool((gl if i == 0 else gr) >= SNAP_THRESHOLD_G),
                    "phase": phase})

        state = np.asarray(self.robot.pi05_state(), dtype=np.float64)
        frame = {
            "dataset_step": len(self.frames),
            "sim_tick": self.sim_tick,
            "sim_time_s": self.sim_tick / CONTROL_HZ,
            "dataset_time_s": len(self.frames) / DATASET_FPS,
            "ticks_this_action": TICKS_PER_ACTION,
            "state14": state.tolist(),
            "action16": a.tolist(),
            "grip_binary_left": bool(gl >= SNAP_THRESHOLD_G),
            "grip_binary_right": bool(gr >= SNAP_THRESHOLD_G),
            "holding": grasped_after,
            "phase": phase,
            "obs_shapes": {k: list(np.asarray(v).shape)
                           for k, v in obs.items() if k in
                           ("head_rgb", "left_wrist_rgb", "right_wrist_rgb")},
        }
        self.frames.append(frame)
        self._last_obs = obs
        return frame

    # ---------- behaviours ----------
    def drive_to(self, goal_xz: np.ndarray, stop_dist: float, phase: str,
                 max_steps: int = 1200, grip=0.0) -> bool:
        path = __import__("habitat_sim").ShortestPath()
        p = self.robot.base_transformation.translation
        path.requested_start = np.array([p.x, p.y, p.z], dtype=np.float32)
        path.requested_end = np.array([goal_xz[0], p.y, goal_xz[1]], dtype=np.float32)
        found = self.sim.pathfinder.find_path(path)
        waypoints = [np.array([w[0], w[2]]) for w in path.points] if found else [goal_xz]
        if len(waypoints) > 1:
            waypoints = waypoints[1:]
        arm = self.full_arm()
        steps = 0
        for wp in waypoints:
            while steps < max_steps:
                d = wp - self.base_xz()
                dist = float(np.linalg.norm(d))
                remaining = float(np.linalg.norm(goal_xz - self.base_xz()))
                if remaining <= stop_dist:
                    return True
                if dist < 0.12:
                    break
                desired = float(np.arctan2(d[1], d[0]))
                err = wrap(desired - self.heading())
                # positive base_vel[1] DECREASES heading -> negate
                yaw_cmd = -float(np.clip(3.0 * err, -1.0, 1.0)) * ANG
                fwd_cmd = LIN if abs(err) < 0.35 else 0.0
                before = self.base_xz()
                self.step_control(self.make_action(arm, grip, fwd_cmd, yaw_cmd), phase)
                steps += 1
                # stall guard: if commanded forward but pinned by the navmesh,
                # rotate away instead of grinding against it
                if fwd_cmd > 0 and float(np.linalg.norm(self.base_xz() - before)) < 1e-4:
                    stalled = getattr(self, "_stall", 0) + 1
                    self._stall = stalled
                    if stalled >= 3:
                        for _ in range(6):
                            self.step_control(self.make_action(
                                arm, grip, 0.0, ANG * (1 if (stalled // 3) % 2 else -1)), phase)
                            steps += 1
                        self._stall = 0
                else:
                    self._stall = 0
        return float(np.linalg.norm(goal_xz - self.base_xz())) <= stop_dist * 1.6

    def face(self, target_xz: np.ndarray, phase: str, tol=0.12, max_steps=120, grip=0.0):
        arm = self.full_arm()
        for _ in range(max_steps):
            d = target_xz - self.base_xz()
            err = wrap(float(np.arctan2(d[1], d[0])) - self.heading())
            if abs(err) < tol:
                return True
            self.step_control(self.make_action(
                arm, grip, 0.0, -float(np.clip(2.0 * err, -1, 1)) * ANG), phase)
        return abs(err) < tol * 2

    def reach(self, target_xyz: np.ndarray, phase: str, tol=0.12,
              max_steps=90, grip=0.0) -> float:
        """Damped-least-squares IK on the active arm's 6 joints."""
        dims = ARM_SLOTS[self.arm]
        arm = self.full_arm()
        best = np.linalg.norm(self.fk_ee(arm) - target_xyz)
        for _ in range(max_steps):
            cur = self.fk_ee(arm)
            err = target_xyz - cur
            dist = float(np.linalg.norm(err))
            if dist < tol:
                break
            J = np.zeros((3, 6))
            eps = 1e-3
            for j in range(6):
                probe = arm.copy(); probe[dims.start + j] += eps
                J[:, j] = (self.fk_ee(probe) - cur) / eps
            lam = 0.06
            dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(3), err)
            dq = np.clip(dq, -0.06, 0.06)
            nxt = arm.copy(); nxt[dims] = arm[dims] + dq
            nxt = self.clamp_arm(nxt)
            arm = nxt
            self.step_control(self.make_action(arm, grip, 0.0, 0.0), phase)
            best = min(best, float(np.linalg.norm(self.fk_ee(arm) - target_xyz)))
        return float(np.linalg.norm(self.ee_pos() - target_xyz))

    def clamp_arm(self, arm12: np.ndarray) -> np.ndarray:
        lo = np.array([-3.0543, 0.0, 0.0, -1.5708, -1.5708, -3.1416] * 2)
        hi = np.array([3.0543, 3.1416, 2.3562, 1.5708, 1.5708, 3.1416] * 2)
        return np.clip(arm12, lo + 1e-4, hi - 1e-4)

    def set_gripper(self, value: float, phase: str, steps: int = 4):
        arm = self.full_arm()
        start = 0.0 if value > 0.5 else 1.0
        for i in range(steps):
            g = start + (value - start) * (i + 1) / steps
            self.step_control(self.make_action(arm, g, 0.0, 0.0), phase)

    def hold(self, n: int, phase: str, grip: float):
        arm = self.full_arm()
        for _ in range(n):
            self.step_control(self.make_action(arm, grip, 0.0, 0.0), phase)

    # ---------- full episode ----------
    def run(self) -> dict:
        ep = self.env.current_episode
        obj_goal = ep.candidate_objects[0]
        obj_xyz = np.asarray(obj_goal.position, dtype=np.float64)
        _, tgt = self.sim.get_targets()
        goal_xyz = np.asarray(tgt[0], dtype=np.float64).reshape(-1)[:3] \
            if np.asarray(tgt).ndim <= 2 else np.asarray(tgt[0])[:3, 3]
        instruction = (f"pick up the {ep.object_category} from the "
                       f"{ep.start_recep_category} and place it on the "
                       f"{ep.goal_recep_category}")
        rep = {"episode_id": str(ep.episode_id), "scene_id": str(ep.scene_id),
               "arm": self.arm, "instruction": instruction,
               "object_category": str(ep.object_category),
               "start_recep": str(ep.start_recep_category),
               "goal_recep": str(ep.goal_recep_category),
               "object_xyz": obj_xyz.tolist(), "goal_xyz": np.asarray(goal_xyz).tolist()}

        rep["navmesh_snap_m"] = self.snap_to_navmesh()
        rep["start_navigable_after_snap"] = bool(
            self.sim.pathfinder.is_navigable(np.asarray(self.robot.base_pos)))

        # inactive arm stays at the verified safe pose for the whole episode
        arm = self.safe_arm.copy()
        self.robot.arm_joint_pos = arm.astype(np.float32)

        self.phase = "navigate_to_object"
        rep["nav_pick_ok"] = self.drive_to(obj_xyz[[0, 2]], 0.75, self.phase)
        self.face(obj_xyz[[0, 2]], self.phase)

        self.phase = "reach"
        rep["reach_dist_m"] = self.reach(obj_xyz, self.phase, tol=0.10)

        self.phase = "grasp"
        self.set_gripper(1.0, self.phase, steps=4)
        self.hold(2, self.phase, 1.0)
        mgrs = self.env.sim.agents_mgr[0].grasp_mgrs
        ee_i = 0 if self.arm == "left" else 1
        rep["grasped"] = bool(mgrs[ee_i].is_grasped)

        self.phase = "lift"
        lift = obj_xyz.copy(); lift[1] += 0.20
        self.reach(lift, self.phase, tol=0.12, max_steps=30, grip=1.0)

        self.phase = "transport"
        rep["nav_place_ok"] = self.drive_to(np.asarray(goal_xyz)[[0, 2]], 0.70,
                                            self.phase, grip=1.0)
        self.face(np.asarray(goal_xyz)[[0, 2]], self.phase, grip=1.0)

        self.phase = "place"
        above = np.asarray(goal_xyz, dtype=np.float64).copy(); above[1] += 0.12
        rep["place_reach_dist_m"] = self.reach(above, self.phase, tol=0.14,
                                               max_steps=60, grip=1.0)
        self.set_gripper(0.0, self.phase, steps=4)
        self.hold(8, "settle", 0.0)

        m = self.env.task.measurements.measures
        self.env.task.measurements.update_measures(
            episode=ep, action=None, task=self.env.task, observations=self._last_obs)
        def mv(k):
            try: return m[k].get_metric()
            except Exception: return None
        rep["measures"] = {k: mv(k) for k in (
            "ovmm_place_success", "obj_anywhere_on_goal", "did_pick_object",
            "pick_success", "object_at_rest", "ovmm_placement_stability",
            "ovmm_object_to_place_goal_distance", "robot_collisions",
            "ovmm_dist_to_pick_goal", "ovmm_dist_to_place_goal")}
        oag = rep["measures"].get("obj_anywhere_on_goal")
        rep["success"] = bool(oag.get("0")) if isinstance(oag, dict) else bool(oag)
        rep["n_dataset_steps"] = len(self.frames)
        rep["n_sim_ticks"] = self.sim_tick
        rep["events"] = self.events
        return rep
