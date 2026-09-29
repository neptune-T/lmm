"""Observation timing probe + automated synchronization test (Stage 1).

Run once per app mode (``ALPHABOT2_ZERO_DELAY=0|1``):

1. **Frame-code test (object).** A magenta block is teleported before EVERY
   physics substep to a grid cell that encodes the physics frame index
   (mod 40). Stepping is exactly the production stepping (targets staged once,
   20 substeps, render on the last). Decoding the block in the head image gives
   the physics frame the image actually shows -> render lag in frames.
2. **Robot-motion test (the pipeline).** Large, visible robot motion (base yaw
   at 1.0 rad/s, continuous, both arms swinging +-12 cm) observed only through
   ``SimWorld.observe()``. Each observation is then re-rendered from the
   recorded full robot state at the frame the pipeline CLAIMS the images show,
   and at +-1, +-2 actions. Synchronized <=> for every camera the recorded image
   matches the claimed state best (offset 0), and proprio equals the state at
   that frame.
3. Wall-clock cost per action (act + observe with 4 cameras).
"""
import json, math, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import simworld as SW  # noqa: E402

app, ZD = SW.launch()
import numpy as np  # noqa: E402
from isaacsim.core.api.objects import FixedCuboid  # noqa: E402
from lmm.embodiments.alphabot2 import cartesian as C  # noqa: E402

OUT = os.environ.get("PROBE_OUT", "/nix/yanjun/zts/alphabot2/stage1/timing")
os.makedirs(OUT, exist_ok=True)
W = SW.SimWorld(app, ZD)
R = {"zero_delay": ZD, "configured_render_lag": W.render_lag}
code_block = FixedCuboid(prim_path="/World/code_block", name="code_block", position=np.array([0.0, 30.0, 0.06]),
                         scale=np.array([0.12, 0.12, 0.12]), color=np.array([1.0, 0.0, 1.0]))
NCODE = 40
#: a teleported kinematic marker occasionally shows up one 5 ms substep late
TOL_FRAMES = 1


def code_xy(c):
    i, j = c % 8, c // 8
    return np.array([1.35 + 0.30 * j, -0.70 + 0.20 * i, 0.06])


def set_code(c):
    code_block.set_world_pose(position=code_xy(c % NCODE))


def centroid(img):
    m = (img[..., 0] > 150) & (img[..., 2] > 150) & (img[..., 1] < 90)
    if m.sum() < 15:
        return None
    ys, xs = np.nonzero(m)
    return np.array([xs.mean(), ys.mean()])


# ---------------- 1. frame-code test ----------------
W.reset_robot((0.0, 0.0), 0.0)
calib = {}
for c in range(NCODE):
    set_code(c)
    for _ in range(4):
        W.apply_joint_action(W.A.home_action())
    calib[c] = centroid(W.B.render("cam_front"))
vis = {c: v for c, v in calib.items() if v is not None}
cc = np.array([vis[c] for c in sorted(vis)])
dmin = min(float(np.linalg.norm(cc[i] - cc[j])) for i in range(len(cc)) for j in range(i + 1, len(cc)))
R["frame_code_calibration"] = {"visible_codes": len(vis), "of": NCODE, "min_centroid_separation_px": round(dmin, 1)}


def decode(img):
    p = centroid(img)
    if p is None:
        return None
    return min(vis, key=lambda c: np.linalg.norm(vis[c] - p))


rows = []
j_home = W.A.home_action()
for t in range(60):
    # production stepping, with the code block moved before every substep
    cmd = W.A.action_to_commands(j_home)
    W.B.set_joint_targets(cmd["joint_targets"])
    W.B.set_base_velocity(0.0, 0.0)
    for i in range(W.IA.PHYSICS_STEPS_PER_ACTION):
        set_code(W.physics_frame + 1)          # the state AFTER this step carries code(frame)
        W.world.step(render=(i == W.IA.PHYSICS_STEPS_PER_ACTION - 1))
        W.B.steps_taken += 1
    W.A.sim_step_count += W.IA.PHYSICS_STEPS_PER_ACTION
    W._log_render()
    f_now = W.physics_frame
    img = W.B.render("cam_front")
    c = decode(img)
    lag = None if c is None else (f_now - c) % NCODE
    rows.append({"t": t, "frame_now": f_now, "decoded_code": c, "lag_frames": lag})
lags = [r["lag_frames"] for r in rows[2:] if r["lag_frames"] is not None]
mode_lag = max(set(lags), key=lags.count)
R["frame_code_test"] = {"rows": rows, "lag_frames_histogram": {str(v): lags.count(v) for v in sorted(set(lags))},
                        "modal_lag_frames": mode_lag, "modal_lag_actions": mode_lag / W.IA.PHYSICS_STEPS_PER_ACTION,
                        "within_tolerance": all(mode_lag <= v <= mode_lag + TOL_FRAMES for v in lags),
                        "tolerance_frames": TOL_FRAMES}
set_code(0)
code_block.set_world_pose(position=np.array([0.0, 30.0, 0.06]))

# ---------------- 2. robot-motion test through SimWorld.observe() ----------------


def scene():
    W.place_table("table_a", 1.45, 0.0, 0.0)
    W.place_table("table_b", 0.0, 1.65, math.pi / 2)
    W.place_box(1.35, -0.2, 0.3)
    W.place_object("red cube", 1.35, 0.15, 0.4)


full = {}                                   # physics frame -> full robot state at that render
_orig_log = W._log_render


def _log_full():
    _orig_log()
    pos, ori = W.art.get_world_pose()
    full[W.physics_frame] = {"q": np.asarray(W.art.get_joint_positions(), float).copy(),
                             "root": (np.asarray(pos, float).copy(), np.asarray(ori, float).copy())}


W._log_render = _log_full
cart = W.reset_robot((0.0, 0.0), 0.0, scene=scene)
recs = []
t0 = time.time()
N = 30
for t in range(N):
    ob = W.observe()
    recs.append(ob)
    a = cart.copy()
    a[C.CART_RIGHT_XYZ.start + 1] += 0.12 * math.sin(2 * math.pi * t / 6)
    a[C.CART_RIGHT_XYZ.start + 2] += 0.06 * math.cos(2 * math.pi * t / 6)
    a[C.CART_LEFT_XYZ.start + 1] -= 0.12 * math.sin(2 * math.pi * t / 5)
    a[C.CART_LEFT_XYZ.start + 2] += 0.06 * math.cos(2 * math.pi * t / 5)
    a[16], a[17] = 0.0, 1.0            # continuous rotation: no reversal, no stationary instant
    W.act(a)
R["wall_s_per_action_act_plus_observe_4cams"] = round((time.time() - t0) / N, 4)
W._log_render = _orig_log


def small(img):
    return img.astype(float).reshape(15, 16, 20, 16, 3).mean(axis=(1, 3))


def render_state(st):
    W.art.set_world_pose(*st["root"])
    W.art.set_joint_positions(st["q"])
    W.art.set_joint_velocities(np.zeros_like(st["q"]))
    W.B.set_base_velocity(0.0, 0.0)
    q = st["q"]
    W.B.set_joint_targets({d: q[W.B.index[d]] for d in W.IB.CONTROLLED_DOFS})
    for _ in range(3):                                     # >= lag + 1 renders
        W.world.step(render=True)
    return {k: small(W.B.render(k)) for k in W.cams}


checks, cam_ok = [], {k: [] for k in W.cams}
OBSERVABLE = 6.0      # neighbour separation must exceed the re-render noise floor (err at offset 0 is ~3-5)
frames = sorted(full)
for ob in recs[3:-3]:
    f = ob.image_frame
    offs = {}
    for o in (-2, -1, 0, 1, 2):
        g = f + o * W.IA.PHYSICS_STEPS_PER_ACTION
        if g in full:
            offs[o] = render_state(full[g])
    rec = {k: small(ob.images[k]) for k in W.cams}
    # observable = the claimed state differs visibly from its +-1-action neighbours
    obsv = {k: min(float(np.abs(offs[o][k] - offs[0][k]).mean()) for o in offs if o != 0) for k in W.cams}
    row = {"policy_step": ob.policy_step, "image_frame": f, "proprio_frame": ob.proprio_frame, "obs_frame": ob.obs_frame}
    for k in W.cams:
        d = {o: float(np.abs(rec[k] - offs[o][k]).mean()) for o in offs}
        best = min(d, key=d.get)
        row[k] = {"best_offset_actions": best, "err_at_0": round(d[0], 2),
                  "err_next_best": round(sorted(v for o, v in d.items() if o != 0)[0], 2),
                  "neighbour_separation": round(obsv[k], 2), "observable": obsv[k] >= OBSERVABLE}
        if obsv[k] >= OBSERVABLE:
            cam_ok[k].append(best == 0)
    checks.append(row)
R["robot_motion_test"] = {"rows": checks,
                          "per_camera_fraction_best_at_claimed_frame": {k: round(float(np.mean(v)), 3) for k, v in cam_ok.items()},
                          "per_camera_observable_steps": {k: len(v) for k, v in cam_ok.items()},
                          "proprio_frame_equals_image_frame": all(o.proprio_frame == o.image_frame for o in recs),
                          "obs_minus_image_frames": sorted({o.obs_frame - o.image_frame for o in recs})}
R["per_reset_sync_check"] = W.last_sync_check
R["synchronized"] = bool(all(len(v) >= 8 and all(v) for v in cam_ok.values())
                         and R["frame_code_test"]["within_tolerance"]
                         and R["frame_code_test"]["modal_lag_actions"] == W.render_lag
                         and W.last_sync_check["ok"])
print("###PROBE###" + json.dumps(R, default=str), flush=True)
with open(f"{OUT}/timing_probe_zd{int(ZD)}.json", "w") as fh:
    json.dump(R, fh, indent=1, default=str)
app.close()
