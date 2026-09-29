"""AlphaBot 2 Habitat embodiment smoke in a real OVMM episode (no task policy).

Checks (numbered as in the milestone): 1 load; 2 reset to the safe posture;
3 16-D proprio; 4 three RGB cameras (+ orientation from depth); 5 each of the 14
arm joints moved alone (+ FK vs the Isaac-qualified reference); 6-8 left and
right gripper, independence; 9-11 base forward, backward, both rotations;
12 combined dual-arm + gripper + base action; 13 atomicity (one action = one
env.step = 0.1 s of simulation, everything applied in it); 14 reproducible
reset; 15 no self-collision at home. Writes smoke.json + camera PNGs.
"""
import json, math, os, sys, time

import numpy as np

REPO = os.environ.get("LMM_REPO", "/nix/yanjun/zts/alphabot2/lmm_repo")
sys.path.insert(0, REPO)
from lmm.embodiments.alphabot2 import kinematics as K  # noqa: E402
from lmm.embodiments.alphabot2.habitat import env as HE  # noqa: E402
from lmm.embodiments.alphabot2.habitat.alphabot2_robot import HOME, SIDES  # noqa: E402

OUT = os.environ.get("SMOKE_OUT", "/nix/yanjun/zts/alphabot2/habitat_runs/smoke")
os.makedirs(OUT, exist_ok=True)
R, OK = {}, {}
env = HE.make_env(gpu=int(os.environ.get("GPU", "0")))
rob = None


def check(name, ok, **detail):
    OK[name] = bool(ok)
    R[name] = {"pass": bool(ok), **detail}
    print(("PASS " if ok else "FAIL ") + name, json.dumps(detail, default=str)[:400], flush=True)


def heading():
    """Base forward/left axes (ROS base_link x/y) and yaw about world +y."""
    T = rob.base_frame()
    f, l = T[:3, 0], T[:3, 1]
    return f, l, T[:3, 3], math.atan2(-f[2], f[0])          # Habitat world: +y up; left of +x is -z


# 1 / 2 / 3
t0 = time.time()
ep_first = env.episodes[0].episode_id
obs = HE.reset_to(env, ep_first)
rob = HE.robot(env)
ep0 = env.current_episode.episode_id
check("01_load", rob is not None and rob.sim_obj.is_alive, episode=ep0, scene=env.current_episode.scene_id,
      load_reset_s=round(time.time() - t0, 1), movable_joints=len(rob.sim_obj.joint_positions))
p = rob.proprio16()
home = np.r_[HOME["left"], 1.0 - 1.0, HOME["right"], 0.0]
check("02_reset_safe_posture", np.allclose(p, home, atol=1e-5), proprio=np.round(p, 4).tolist())
op = np.asarray(obs["alphabot2_proprio"])
check("03_proprio16", op.shape == (16,) and np.isfinite(op).all() and np.allclose(op, p, atol=1e-5), shape=list(op.shape))
# 4 cameras
from PIL import Image  # noqa: E402
cams = {}
for k in ("head_rgb", "left_wrist_rgb", "right_wrist_rgb"):
    im = np.asarray(obs[k])[..., :3]
    cams[k] = {"shape": list(im.shape), "std": float(im.std()), "mean": float(im.mean())}
    Image.fromarray(im.astype(np.uint8)).save(f"{OUT}/{k}_home.png")
d = np.asarray(obs["head_depth"]).squeeze()
cams["head_depth_top_vs_bottom_m"] = [float(np.nanmedian(d[:40])), float(np.nanmedian(d[-40:]))]
check("04_three_rgb_cameras", all(c["shape"] == [240, 320, 3] and c["std"] > 2 for c in list(cams.values())[:3])
      and cams["head_depth_top_vs_bottom_m"][0] > cams["head_depth_top_vs_bottom_m"][1], **cams)
# 5 each arm joint alone + FK agreement
moved, fk_err = [], []
for side_i, side in enumerate(SIDES):
    for j in range(7):
        a = HE.hold_action(env)
        idx = (0 if side == "left" else 8) + j
        lo, hi = K.JOINT_LIMITS[j]
        delta = 0.25 if a[idx] + 0.25 < hi - 0.02 else -0.25
        a[idx] += delta
        before = rob.proprio16()
        HE.step18(env, a)
        after = rob.proprio16()
        ch = np.abs(after - before)
        others = np.delete(ch, idx)
        moved.append({"joint": f"{side}_joint{j + 1}", "delta_cmd": delta, "delta_meas": float(after[idx] - before[idx]),
                      "max_other_change": float(others.max())})
        # FK: Habitat gripper_base in the Habitat base_link frame vs kinematics.fk
        Tb = np.linalg.inv(rob.arm_frame()) @ rob.link_frame(f"{side}_gripper_base")      # arm frame A
        Tk = K.fk(side, after[(0 if side == "left" else 8):(7 if side == "left" else 15)])
        fk_err.append([float(np.linalg.norm(Tb[:3, 3] - Tk[:3, 3])), float(np.linalg.norm(K.rotvec(Tb[:3, :3].T @ Tk[:3, :3])))])
        a[idx] -= delta
        HE.step18(env, a)
fe = np.array(fk_err)
check("05_each_arm_joint_independent", all(abs(m["delta_meas"] - m["delta_cmd"]) < 1e-5 and m["max_other_change"] < 1e-6 for m in moved),
      joints=moved, fk_max_pos_err_m=float(fe[:, 0].max()), fk_max_rot_err_rad=float(fe[:, 1].max()))
check("05b_fk_matches_isaac_reference", fe[:, 0].max() < 1e-4 and fe[:, 1].max() < 1e-4,
      fk_max_pos_err_m=float(fe[:, 0].max()), fk_max_rot_err_rad=float(fe[:, 1].max()))
# 6-8 grippers
g = {}
for side, gi, oi in (("left", 7, 15), ("right", 15, 7)):
    a = HE.hold_action(env)
    rows = []
    for f in (0.25, 0.5, 1.0, 0.0):
        a[gi] = f
        HE.step18(env, a)
        p = rob.proprio16()
        sep = np.linalg.norm(rob.link_frame(f"{side}_left_pad")[:3, 3] - rob.link_frame(f"{side}_right_pad")[:3, 3])
        rows.append({"cmd": f, "meas": round(float(p[gi]), 5), "other_hand": round(float(p[oi]), 5), "pad_sep_m": round(float(sep), 4)})
    g[side] = rows
check("06_left_gripper", all(abs(r["meas"] - r["cmd"]) < 1e-4 for r in g["left"]), rows=g["left"])
check("07_right_gripper", all(abs(r["meas"] - r["cmd"]) < 1e-4 for r in g["right"]), rows=g["right"])
check("08_gripper_independence", all(r["other_hand"] == 0.0 for s in g for r in g[s]))
# 9-11 base: in open space (the OVMM start pose may sit on the navmesh boundary, where the
# navmesh snap would be mistaken for a base error); clearance >= 1 m from any obstacle
import magnum as mn  # noqa: E402
pf = env.sim.pathfinder
pf.seed(0)
start = rob.sim_obj.translation
isl = pf.get_island(start)                  # same navmesh island: OVMM goals must stay reachable
best = None
for _ in range(3000):
    pt = pf.get_random_navigable_point_near(start, 4.0, 100, isl)
    c = pf.distance_to_closest_obstacle(pt, 2.0)
    if best is None or c > best[1]:
        best = (pt, c)
    if c >= 1.0:
        break
pt = best[0]
tr = rob.sim_obj.transformation
rob.sim_obj.transformation = mn.Matrix4.from_(tr.rotation(), mn.Vector3(pt))
HE.step18(env, HE.hold_action(env))
R["base_test_point"] = {"xyz": [float(v) for v in pt], "clearance_m": float(pf.distance_to_closest_obstacle(pt, 2.0))}
base = {}
# backward first: the OVMM start pose may face an obstacle; forward then retraces free space
for name, v, w, n in (("backward", -0.3, 0.0, 5), ("forward", 0.3, 0.0, 5), ("ccw", 0.0, 0.5, 5), ("cw", 0.0, -0.5, 5)):
    f0, l0, p0, y0 = heading()
    a = HE.hold_action(env)
    a[16], a[17] = v, w
    filt = 0.0
    for _ in range(n):
        HE.step18(env, a)
        filt += HE.action_obj(env).last_info["base"]["navmesh_filtered_m"]
    f1, l1, p1, y1 = heading()
    dp = p1 - p0
    base[name] = {"along_forward_m": float(dp @ f0), "along_left_m": float(dp @ l0),
                  "dyaw_rad": float((y1 - y0 + math.pi) % (2 * math.pi) - math.pi),
                  "forward_turned_toward_left": float(f1 @ l0), "expected_m": v * 0.1 * n, "expected_rad": w * 0.1 * n,
                  "navmesh_filtered_m_total": filt}
check("09_base_forward", abs(base["forward"]["along_forward_m"] - 0.15) < 0.01, **base["forward"])
check("10_base_backward", abs(base["backward"]["along_forward_m"] + 0.15) < 0.01, **base["backward"])
check("11_base_rotate_both", base["ccw"]["forward_turned_toward_left"] > 0.2 and base["cw"]["forward_turned_toward_left"] < -0.2
      and abs(base["ccw"]["dyaw_rad"] - 0.25) < 0.02 and abs(base["cw"]["dyaw_rad"] + 0.25) < 0.02, ccw=base["ccw"], cw=base["cw"])
# 12 / 13 combined + atomicity
a = HE.hold_action(env)
a[2] += 0.2; a[10] -= 0.2; a[7] = 0.6; a[15] = 0.3; a[16], a[17] = 0.2, 0.3
p0, (_, _, b0, y0) = rob.proprio16(), heading()
wt0 = env.sim.get_world_time()
n_steps0 = env._elapsed_steps
HE.step18(env, a)
wt1 = env.sim.get_world_time()
p1, (_, _, b1, y1) = rob.proprio16(), heading()
changed = {"left_arm": abs(p1[2] - p0[2] - 0.2) < 1e-5, "right_arm": abs(p1[10] - p0[10] + 0.2) < 1e-5,
           "left_grip": abs(p1[7] - 0.6) < 1e-4, "right_grip": abs(p1[15] - 0.3) < 1e-4,
           "base_moved": float(np.linalg.norm(b1 - b0)) > 0.015, "base_turned": abs(y1 - y0) > 0.025}
check("12_combined_action", all(changed.values()), **changed)
check("13_atomic_one_env_step", env._elapsed_steps - n_steps0 == 1 and abs((wt1 - wt0) - 0.1) < 1e-6,
      env_steps=env._elapsed_steps - n_steps0, sim_time_per_action_s=wt1 - wt0,
      physics_substeps_per_action=round((wt1 - wt0) / 0.005), ctrl_freq=env.sim.ctrl_freq, ac_freq_ratio=env.sim.ac_freq_ratio)
# 14 reproducible reset (same episode)
o1 = HE.reset_to(env, ep0); s1 = (rob.proprio16().copy(), rob.base_frame().copy(), np.asarray(o1["head_rgb"]).copy())
HE.step18(env, np.r_[rob.proprio16(), 0.2, 0.1])
o2 = HE.reset_to(env, ep0); s2 = (rob.proprio16().copy(), rob.base_frame().copy(), np.asarray(o2["head_rgb"]).copy())
check("14_reset_reproducible", np.array_equal(s1[0], s2[0]) and np.allclose(s1[1], s2[1], atol=1e-6),
      proprio_equal=bool(np.array_equal(s1[0], s2[0])), base_max_diff=float(np.abs(s1[1] - s2[1]).max()),
      head_rgb_mean_abs_diff=float(np.abs(s1[2].astype(float) - s2[2]).mean()))
# 15 self-collision at home
sim = env.sim
sim.perform_discrete_collision_detection()
cps = sim.get_physics_contact_points()
rid = rob.sim_obj.object_id
kin = rob.kin
parent_of = {kin.joints[j].child: kin.joints[j].parent for j in kin.joints}
names = {i: rob.sim_obj.get_link_name(i) for i in rob.sim_obj.get_link_ids()}
self_pairs, env_pairs = set(), set()
for c in cps:
    if c.object_id_a == rid and c.object_id_b == rid:
        a_, b_ = names.get(c.link_id_a, "?"), names.get(c.link_id_b, "?")
        if parent_of.get(a_) != b_ and parent_of.get(b_) != a_:
            self_pairs.add(tuple(sorted((a_, b_))))
    elif rid in (c.object_id_a, c.object_id_b):
        env_pairs.add((names.get(c.link_id_a if c.object_id_a == rid else c.link_id_b, "?")))
check("15_no_self_collision_at_home", not self_pairs, self_collision_pairs=sorted(self_pairs), contacts_total=len(cps),
      robot_env_contact_links=sorted(env_pairs))
R["summary"] = {"passed": sum(OK.values()), "of": len(OK), "failed": [k for k, v in OK.items() if not v]}
print("###SMOKE###" + json.dumps(R["summary"]), flush=True)
json.dump(R, open(f"{OUT}/smoke.json", "w"), indent=1, default=str)
env.close()
