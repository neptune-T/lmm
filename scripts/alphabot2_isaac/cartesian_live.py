"""Live qualification of the Cartesian execution layer (Isaac Sim 4.5, 200 Hz, 10 Hz actions).

CartesianMobileAction18 -> CartesianExecutor (frame B, tool E, xacro IK)
-> JointMobileAction18 -> AlphaBot2IsaacAdapter -> Isaac. Measured EE poses are
read back from PhysX and expressed in the checkpoint's frame B.

1. reachability of the checkpoint's training workspace (IK only, both arms)
2. synthetic paths per arm, live: line, circle, orientation sweep
3. replay of recorded dataset target trajectories, live, compared with the
   joints the real robot recorded
"""
import json, math, os, sys, time, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("LMM_REPO", os.path.dirname(os.path.dirname(HERE)))
USDS = os.environ.get("ROBOT_USDS", "/nix/yanjun/zts/alphabot2/deps/robot_usds")
EPS = os.environ.get("REPLAY_NPZ", "/nix/yanjun/zts/alphabot2/hf/replay_episodes.npz")
OUT = os.environ.get("SMOKE_OUT", "/nix/yanjun/zts/alphabot2/smoke/cartesian")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, REPO)
from lmm.embodiments.alphabot2 import isaac_adapter as IA  # noqa: E402
from lmm.embodiments.alphabot2 import isaac_backend as IB  # noqa: E402
sys.argv.append(IB.RTX_DRIVER_CHECK_BYPASS)
O = {"sections": {}}


def sec(name):
    def deco(fn):
        t = time.time()
        try:
            O["sections"][name] = {"ok": True, **(fn() or {})}
        except Exception as e:
            O["sections"][name] = {"ok": False, "error": repr(e), "tb": traceback.format_exc()[-1500:]}
        O["sections"][name]["wall_s"] = round(time.time() - t, 2)
        print(f"[cart] {name}: ok={O['sections'][name]['ok']}", flush=True)
        return fn
    return deco


from isaacsim import SimulationApp  # noqa: E402
app = SimulationApp({"headless": True})
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleArticulation  # noqa: E402
from lmm.embodiments.alphabot2 import cartesian as C  # noqa: E402
from lmm.embodiments.alphabot2 import kinematics as K  # noqa: E402
from lmm.embodiments.alphabot2 import pi05_contract as P  # noqa: E402
from lmm.embodiments.alphabot2.contract import AlphaBot2Contract  # noqa: E402

world = World(stage_units_in_meters=1.0, physics_dt=1 / IA.PHYSICS_HZ, rendering_dt=1 / IA.PHYSICS_HZ)
stage = omni.usd.get_context().get_stage()
O["build"] = IB.build_stage(app, stage, USDS)
world.scene.add_default_ground_plane()
IB.enable_contact_reports(stage)
art = SingleArticulation(prim_path=IB.ROBOT_PRIM, name="bot2")
world.scene.add(art)
world.reset()
B = IB.IsaacArticulationBackend(world, art)
A = IA.AlphaBot2IsaacAdapter(AlphaBot2Contract(), backend=B)
SIDE_PREFIX = {"left": "Left", "right": "Right"}
EP = dict(np.load(EPS))


def r4(x):
    return [round(float(v), 4) for v in np.asarray(x).reshape(-1)]


def vendor_from_dataset(state16):
    s = np.asarray(state16, float)
    return P.DATASET_JOINT_SIGN["left"] * s[0:7], P.DATASET_JOINT_SIGN["right"] * s[7:14]


def teleport(ql, qr):
    m = {}
    for side, q in (("left", ql), ("right", qr)):
        for d, v in zip(IA.LEFT_ARM_DOFS if side == "left" else IA.RIGHT_ARM_DOFS, A.arm_vendor_to_sim(side, q)):
            m[d] = float(v)
    m[IA.LEFT_GRIPPER_DOF] = m[IA.RIGHT_GRIPPER_DOF] = A.gripper_fraction_to_joint(0.0)
    for d in (*IA.BODY_DOFS, *IA.HEAD_DOFS):
        m[d] = 0.0
    B.teleport(m)
    hold = np.zeros(18)
    hold[0:7], hold[8:15] = ql, qr
    for _ in range(5):
        A.apply_action(hold)


def measured(side):
    p, r = B.link_pose(f"Gripper_{SIDE_PREFIX[side]}/gripper_base")
    pc, rc = B.in_chassis(p, r)
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = rc, pc
    return P.action_from_gripper_base(side, m)


def ang_deg(q1, q2):
    return math.degrees(2 * math.acos(min(1.0, abs(float(np.dot(q1, q2))))))


def self_contacts():
    pre = IB.ROBOT_PRIM + "/"
    return sorted(k for k in IB.contact_pairs() if all(x.startswith(pre) for x in k.split(" <-> ")))


START = EP["start_median_task3"]


@sec("reachability_training_workspace")
def _():
    # dataset action ranges (meta/stats.json q01..q99 would be tighter; use min..max)
    out = {}
    qmean = {"left": np.array([-0.0323, 0.004, -0.0558, 0.993]), "right": np.array([-0.215, 0.0183, 0.0368, 0.9449])}
    ranges = {"left": ((0.50, 0.71), (0.22, 0.43), (0.71, 0.97)), "right": ((0.46, 0.96), (-0.68, -0.09), (0.61, 0.97))}
    ql, qr = vendor_from_dataset(START)
    seed = {"left": ql, "right": qr}
    for side in ("left", "right"):
        xs, ys, zs = (np.linspace(lo, hi, 7) for lo, hi in ranges[side])
        ok = tot = 0
        fails = []
        for x in xs:
            for y in ys:
                for z in zs:
                    tgt = P.gripper_base_from_action(side, (x, y, z), qmean[side])
                    res = K.ik(side, tgt, seed[side])
                    if not res.converged:
                        res = K.ik(side, tgt, np.array(IA.HOME_LEFT_ARM if side == "left" else IA.HOME_RIGHT_ARM))
                    tot += 1
                    ok += res.converged
                    if not res.converged and len(fails) < 6:
                        fails.append({"xyz_B": r4((x, y, z)), "pos_err_mm": round(res.pos_err_m * 1000, 1), "at_limit": res.at_limit})
        out[side] = {"grid_points": tot, "ik_reachable_frac": round(ok / tot, 3), "example_failures": fails,
                     "orientation": "dataset mean quaternion"}
    return out


def run_path(side, targets, quats, label):
    """Execute a Cartesian path for one arm; the other arm holds its current target."""
    ql, qr = vendor_from_dataset(START)
    teleport(ql, qr)
    ex = C.CartesianExecutor(q={"left": ql.copy(), "right": qr.copy()})
    other = "right" if side == "left" else "left"
    o_xyz, o_q = measured(other)
    cmd, ach, reps, q4 = [], [], [], []
    contacts = set()
    for xyz, q in zip(targets, quats):
        cart = np.zeros(18)
        sl_x = C.CART_LEFT_XYZ if side == "left" else C.CART_RIGHT_XYZ
        sl_q = C.CART_LEFT_QUAT if side == "left" else C.CART_RIGHT_QUAT
        cart[sl_x], cart[sl_q] = xyz, q
        cart[C.CART_RIGHT_XYZ if side == "left" else C.CART_LEFT_XYZ] = o_xyz
        cart[C.CART_RIGHT_QUAT if side == "left" else C.CART_LEFT_QUAT] = o_q
        j, rep = ex.lower(cart)
        A.apply_action(j)
        contacts.update(self_contacts())
        mx, mq = measured(side)
        cmd.append((np.asarray(xyz), np.asarray(q)))
        ach.append((mx, mq))
        reps.append(rep[side])
        q4.append(ex.q[side][3])
    for _ in range(5):                                  # settle on the last target
        A.apply_action(j)
    fx, fq = measured(side)
    pe = [np.linalg.norm(c[0] - a[0]) * 1000 for c, a in zip(cmd, ach)]
    re = [ang_deg(c[1], a[1]) for c, a in zip(cmd, ach)]
    return {"path": label, "n": len(targets),
            "ik_valid_frac": round(float(np.mean([r["ik_valid"] for r in reps])), 3),
            "max_joint_step_rad": round(float(max(r["max_joint_step_rad"] for r in reps)), 4),
            "discontinuities": int(sum(r["discontinuous"] for r in reps)),
            "elbow_sign_changes": int(np.sum(np.diff(np.sign(np.asarray(q4) + 1e-9)) != 0)),
            "track_pos_mm_median_p95": [round(float(np.median(pe)), 1), round(float(np.percentile(pe, 95)), 1)],
            "track_rot_deg_median_p95": [round(float(np.median(re)), 2), round(float(np.percentile(re, 95)), 2)],
            "final_settled_err": [round(float(np.linalg.norm(fx - cmd[-1][0]) * 1000), 2), round(ang_deg(fq, cmd[-1][1]), 3)],
            "commanded_disp_B": r4(cmd[-1][0] - cmd[0][0]), "achieved_disp_B": r4(fx - ach[0][0]),
            "self_contacts": sorted(contacts)}


@sec("synthetic_paths")
def _():
    out = {}
    for side in ("left", "right"):
        ql, qr = vendor_from_dataset(START)
        teleport(ql, qr)
        x0, q0 = measured(side)
        n = 30
        line = [x0 + np.array([0.12, 0.0, 0.0]) * (i + 1) / n for i in range(n)]
        up = [x0 + np.array([0.0, 0.0, 0.08]) * (i + 1) / n for i in range(n)]
        circ = [x0 + 0.05 * np.array([0.0, math.sin(2 * math.pi * i / 40), 1 - math.cos(2 * math.pi * i / 40)]) for i in range(1, 41)]
        rz = []
        r0 = P.quat_xyzw_to_matrix(q0)
        for i in range(n):
            a = math.radians(20.0) * (i + 1) / n
            rot = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
            rz.append(P.matrix_to_quat_xyzw(rot @ r0))
        out[side] = [run_path(side, line, [q0] * n, "+x_B 0.12 m"),
                     run_path(side, up, [q0] * n, "+z_B 0.08 m"),
                     run_path(side, circ, [q0] * 40, "circle r=0.05 in y-z"),
                     run_path(side, [x0] * n, rz, "rotate +20 deg about z_B")]
        print(side, json.dumps(out[side]), flush=True)
    return out


@sec("dataset_trajectory_replay")
def _():
    out = {}
    for name in sorted(k for k in EP if k.startswith("ep")):
        arr = EP[name]
        S, Aa = arr[:, :16], arr[:, 16:]
        ql, qr = vendor_from_dataset(S[0])
        teleport(ql, qr)
        ex = C.CartesianExecutor(q={"left": ql.copy(), "right": qr.copy()})
        simq, reps, ach = [], [], []
        for t in range(len(Aa)):
            cart = np.r_[Aa[t], 0.0, 0.0]
            j, rep = ex.lower(cart)
            A.apply_action(j)
            s = A.observe_proprio()
            simq.append(s[8:15].copy())
            reps.append(rep["right"])
            ach.append(measured("right"))
        simq = np.asarray(simq)
        realq = np.stack([vendor_from_dataset(s)[1] for s in S])
        best = None
        for k in range(0, 11):                          # real state lags the targets
            if len(simq) - k < 10:
                break
            d = np.abs(simq[:len(simq) - k] - realq[k:])
            med = float(np.median(np.max(d, axis=1)))
            if best is None or med < best[1]:
                best = (k, med, float(np.median(d)), [round(float(v), 3) for v in np.median(d, axis=0)])
        pe = [np.linalg.norm(a[0] - Aa[t, 7:10]) * 1000 for t, a in enumerate(ach)]
        out[name] = {"frames": len(Aa), "right_ik_valid_frac": round(float(np.mean([r["ik_valid"] for r in reps])), 3),
                     "right_ik_pos_err_mm_p95": round(float(np.percentile([r["pos_err_m"] for r in reps], 95)) * 1000, 1),
                     "discontinuities": int(sum(r["discontinuous"] for r in reps)),
                     "sim_vs_recorded_joints_best_lag_frames": best[0],
                     "sim_vs_recorded_joints_median_maxabs_rad": round(best[1], 3),
                     "sim_vs_recorded_joints_median_per_joint_rad": best[3],
                     "ee_track_mm_median": round(float(np.median(pe)), 1),
                     "right_gripper_cmds": sorted(set(np.round(Aa[:, 15], 1).tolist()))}
        print(name, json.dumps(out[name]), flush=True)
    return out


print("###JSON###" + json.dumps(O, default=str) + "\n", flush=True)
with open(f"{OUT}/cartesian_live.json", "w") as f:
    json.dump(O, f, indent=1, default=str)
app.close()
