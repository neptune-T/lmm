"""Identify q_usd = s*q_xacro + o per arm from screw axes, then verify in Isaac.

1. In Isaac, hold each arm at a collision-free reference pose and perturb each
   joint in turn; the gripper_base motion gives that joint's screw axis (direction
   + a point on it) in the chassis frame. That is the USD arm's product-of-
   exponentials FK, measured rather than parsed.
2. From the vendor xacro, compute the screw axes at q_xacro = 0.
3. Sequentially solve o_1..o_7 (and signs) so the USD axes at q_usd = o coincide
   with the xacro axes at 0, then the gripper_base frame fixes o_7. Each step has
   a (theta, theta+pi) ambiguity; all 2^7 branches are scored on random poses.
4. Verify the best mapping by driving Isaac to mapped random xacro poses.
"""
import itertools, json, math, re, sys, traceback
sys.argv += ["--/rtx/verifyDriverVersion/enabled=false"]
sys.path.insert(0, "/nix/yanjun/zts/alphabot2/smoke")
X = "/nix/yanjun/zts/alphabot2/deps/robot_descriptions/humanoid/Ai2Robotics/ai2_bot2_description/xacro/components"
O = {}
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
try:
    import numpy as np, omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.types import ArticulationAction
    import bot2_build

    def rpyR(r, p, y):
        cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
        return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr], [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]])
    def Tm(xyz, rpy): M = np.eye(4); M[:3, :3] = rpyR(*rpy); M[:3, 3] = xyz; return M
    def rot(ax, q):
        a = np.asarray(ax, float); a = a / np.linalg.norm(a)
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        return np.eye(3) + math.sin(q)*K + (1-math.cos(q))*K@K
    def Ra(ax, q): M = np.eye(4); M[:3, :3] = rot(ax, q); return M
    def screw(a, c, q):  # rotation by q about the line through c with direction a
        M = np.eye(4); R = rot(a, q); M[:3, :3] = R; M[:3, 3] = c - R @ c; return M
    nums = lambda s: [float(v) for v in s.split()]
    def resolve(b, d):
        b = re.sub(r'<xacro:if value="\$\{direction == (-?1)\}">(.*?)</xacro:if>',
                   lambda m: m.group(2) if float(m.group(1)) == d else "", b, flags=re.S)
        return b.replace("${direction}", str(d))
    def joints(xml, d=None):
        out = {}
        for m in re.finditer(r'<joint name="([^"]+)" type="([^"]+)">(.*?)</joint>', xml, re.S):
            b = resolve(m.group(3), d) if d is not None else m.group(3)
            o = re.search(r'<origin([^>]*)/>', b)
            xyz = nums(re.search(r'xyz="([^"]+)"', o.group(1)).group(1)) if o and 'xyz=' in o.group(1) else [0, 0, 0]
            rpy = nums(re.search(r'rpy="([^"]+)"', o.group(1)).group(1)) if o and 'rpy=' in o.group(1) else [0, 0, 0]
            ax = re.search(r'<axis xyz="([^"]+)"', b)
            lim = re.search(r'<limit lower="([^"]+)" upper="([^"]+)"', b)
            out[m.group(1)] = (xyz, rpy, nums(ax.group(1)) if ax else [0, 0, 1],
                               (float(lim.group(1)), float(lim.group(2))) if lim else None)
        return out
    ARM = open(f"{X}/arm.xacro").read(); BODY = joints(open(f"{X}/body.xacro").read())
    JS = {s: joints(ARM, 1 if s == "left" else -1) for s in ("left", "right")}
    XLIM = {s: [JS[s][f"${{prefix}}joint{k}"][3] for k in range(1, 8)] for s in JS}
    O["xacro_limits"] = XLIM
    O["xacro_axes"] = {s: [JS[s][f"${{prefix}}joint{k}"][2] for k in range(1, 8)] for s in JS}
    def xfk(side, q7, frames=False):
        J = JS[side]; M = np.eye(4); axes = []
        for n in ("body_joint1", "body_joint2", "body_joint3", "body_joint4"):
            M = M @ Tm(BODY[n][0], BODY[n][1])
        M = M @ Tm(*J["${prefix}joint0"][:2])
        for k in range(1, 8):
            xyz, rpy, ax, _ = J[f"${{prefix}}joint{k}"]; M = M @ Tm(xyz, rpy)
            a = M[:3, :3] @ (np.asarray(ax) / np.linalg.norm(ax)); axes.append((a, M[:3, 3].copy()))
            M = M @ Ra(ax, q7[k-1])
        M = M @ Tm(*J["${prefix}joint8"][:2]) @ Tm([0, 0, 0.0695], [0, 0, 0]) @ Tm([0, 0, 0], [0, 0, 1.5708])
        return (M, axes) if frames else M

    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()
    bot2_build.build(app, stage, add_reference_to_stage, O)
    world.scene.add_default_ground_plane()
    art = SingleArticulation(prim_path="/World/Bot2", name="bot2")
    world.scene.add(art); world.reset()
    pv = lambda: art._articulation_view._physics_view
    arr = lambda x: x.numpy() if hasattr(x, "numpy") else np.asarray(x)
    DOF = [p.replace("/World/Bot2/", "") for p in pv().dof_paths[0]]
    LNK = [p.replace("/World/Bot2/", "") for p in pv().link_paths[0]]
    LIM = arr(pv().get_dof_limits()).reshape(-1, 2)
    di = lambda s: [i for i, p in enumerate(DOF) if p.endswith(s)][0]
    IDX = {"left": [di(f"Arm_Left/joints/left_joint{k}") for k in range(1, 8)],
           "right": [di(f"Arm_Right/joints/right_joint{k}") for k in range(1, 8)]}
    O["usd_limits"] = {s: [[round(float(a), 4), round(float(b), 4)] for a, b in LIM[IDX[s]]] for s in IDX}
    GB = {"left": LNK.index("Gripper_Left/gripper_base"), "right": LNK.index("Gripper_Right/gripper_base")}
    CH = [i for i, p in enumerate(LNK) if p.split("/")[-1] == "base_link" and not p.startswith(("Arm_", "Gripper_"))][0]
    def quatR(q):
        x, y, z, w = q
        return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    H = math.pi / 2
    # Isaac-side reference poses (collision-free homes found in the earlier smoke)
    HOME_USD = {"left": np.array([0.8333, -H, -H, -1.78, H, -0.641, 0]),
                "right": np.array([-0.8333, -H, -H, -1.78, H, -0.641, 0])}
    q_all = arr(art.get_joint_positions()).astype(float)
    for s in IDX: q_all[IDX[s]] = HOME_USD[s]
    def hold(side, q7):
        q = q_all.copy(); q[IDX[side]] = q7
        art.set_joint_positions(q); art.set_joint_velocities(np.zeros_like(q))
        art.apply_action(ArticulationAction(joint_positions=q))
        world.step(render=False)
        qa = arr(art.get_joint_positions()).astype(float)[IDX[side]]
        T = arr(pv().get_link_transforms()).reshape(-1, len(LNK), 7)[0]
        Rc, pc = quatR(T[CH, 3:7]), T[CH, :3]
        M = np.eye(4); M[:3, :3] = Rc.T @ quatR(T[GB[side], 3:7]); M[:3, 3] = Rc.T @ (T[GB[side], :3] - pc)
        return M, qa
    def axis_from(M0, M1):
        dR = M1[:3, :3] @ M0[:3, :3].T
        ang = math.acos(np.clip((np.trace(dR) - 1) / 2, -1, 1))
        a = np.array([dR[2, 1]-dR[1, 2], dR[0, 2]-dR[2, 0], dR[1, 0]-dR[0, 1]]) / (2*math.sin(ang))
        # point c with M1.p = dR (M0.p - c) + c  ->  (I - dR) c = M1.p - dR M0.p
        c = np.linalg.lstsq(np.eye(3) - dR, M1[:3, 3] - dR @ M0[:3, 3], rcond=None)[0]
        c = c - a * (a @ c)  # canonical point: closest to origin
        return a, c, ang
    def rel_pose_err(Ma, Mb):
        dR = Ma[:3, :3].T @ Mb[:3, :3]
        return float(np.linalg.norm(Ma[:3, 3] - Mb[:3, 3])), math.degrees(math.acos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
    def line_dist(a, c, b, d):  # distance between point d and line (a,c)
        v = d - c; return float(np.linalg.norm(v - a * (a @ v)))

    rng = np.random.default_rng(11)
    RESULT = {}
    for side in ("left", "right"):
        R = {}
        q0 = HOME_USD[side].copy()
        M0, qa0 = hold(side, q0)
        # --- measured USD screw axes at q0 (chassis frame) ---
        scr = []
        for k in range(7):
            dq = 0.3 if q0[k] + 0.3 <= LIM[IDX[side][k], 1] else -0.3
            q1 = q0.copy(); q1[k] += dq
            M1, qa1 = hold(side, q1)
            a, c, ang = axis_from(M0, M1)
            d_meas = qa1[k] - qa0[k]
            if d_meas < 0: a = -a  # a is the axis for the measured (possibly negative) rotation
            scr.append((a, c))
            R.setdefault("fd_check", []).append({"joint": k+1, "cmd": dq, "measured": round(float(d_meas), 4),
                                                   "rot_angle": round(ang, 4)})
        hold(side, q0)
        def usd_fk(qu):
            M = np.eye(4)
            for k in range(7):
                M = M @ screw(scr[k][0], scr[k][1], qu[k] - qa0[k])
            return M @ M0
        # POE self-check against Isaac on random in-limit USD poses
        lims = LIM[IDX[side]]
        chk = []
        for _ in range(4):
            qu = np.array([rng.uniform(lo + 0.2, hi - 0.2) for lo, hi in lims])
            Mi, qa = hold(side, qu)
            chk.append(rel_pose_err(Mi, usd_fk(qa)))
        R["poe_vs_isaac_max_pos_m"] = round(max(e[0] for e in chk), 5)
        R["poe_vs_isaac_max_rot_deg"] = round(max(e[1] for e in chk), 3)
        # --- xacro axes at q_x = 0 ---
        Mx0, xax = xfk(side, [0]*7, frames=True)
        R["axis_line_dist_joint1_m"] = round(line_dist(xax[0][0], xax[0][1], None, scr[0][1]) if False else line_dist(xax[0][0], xax[0][1], None, scr[0][1]), 4) if False else None
        # sequential solve with branch on theta / theta+pi
        def solve(branch):
            o = np.zeros(7); sgn = np.zeros(7)
            sgn[0] = 1.0 if scr[0][0] @ xax[0][0] > 0 else -1.0
            for k in range(1, 7):
                # choose o[k-1] so that axis k (after rotating joints <k to o) aligns with xacro axis k
                def axis_k_at(theta):
                    oo = o.copy(); oo[k-1] = theta
                    M = np.eye(4)
                    for j in range(k):
                        M = M @ screw(scr[j][0], scr[j][1], oo[j] - qa0[j])
                    return M[:3, :3] @ scr[k][0], (M @ np.append(scr[k][1], 1))[:3]
                a_prev = sgn[k-1] * xax[k-1][0]
                # rotate about axis (k-1) in its current (already-solved) placement
                M = np.eye(4)
                for j in range(k-1):
                    M = M @ screw(scr[j][0], scr[j][1], o[j] - qa0[j])
                ap = M[:3, :3] @ scr[k-1][0]
                # angle about ap taking current axis k (at theta = qa0) to xacro axis k
                u, _ = axis_k_at(qa0[k-1])
                t = xax[k][0]
                pu = u - ap * (ap @ u); pt = t - ap * (ap @ t)
                phi = math.atan2(ap @ np.cross(pu, pt), pu @ pt)
                theta = qa0[k-1] + phi + (math.pi if branch[k-1] else 0.0)
                theta = (theta + math.pi) % (2 * math.pi) - math.pi
                o[k-1] = theta
                ak, _ = axis_k_at(theta)
                sgn[k] = 1.0 if ak @ t > 0 else -1.0
            # last offset: align gripper_base orientation about axis 7
            best = None
            for th in np.linspace(-math.pi, math.pi, 3601):
                oo = o.copy(); oo[6] = th
                e = rel_pose_err(usd_fk(oo), Mx0)[1]
                if best is None or e < best[0]: best = (e, th)
            o[6] = best[1]
            return o, sgn
        cands = []
        for branch in itertools.product((0, 1), repeat=6):
            o, sgn = solve(branch)
            errs = []
            for _ in range(6):
                qx = np.array([rng.uniform(lo + 0.1, hi - 0.1) for lo, hi in XLIM[side]])
                errs.append(rel_pose_err(usd_fk(sgn * qx + o), xfk(side, qx)))
            cands.append((max(e[1] for e in errs), max(e[0] for e in errs), o, sgn, branch))
        cands.sort(key=lambda c: (c[0], c[1]))
        e_rot, e_pos, o, sgn, br = cands[0]
        R["best_offline"] = {"sign": sgn.astype(int).tolist(), "offset_rad": [round(float(v), 4) for v in o],
                             "offset_over_pi2": [round(float(v) / H, 3) for v in o],
                             "max_rot_err_deg": round(e_rot, 3), "max_pos_err_m": round(e_pos, 4)}
        R["runner_up"] = {"sign": cands[1][3].astype(int).tolist(), "offset_over_pi2": [round(float(v) / H, 3) for v in cands[1][2]],
                          "max_rot_err_deg": round(cands[1][0], 2), "max_pos_err_m": round(cands[1][1], 4)}
        # snap offsets to nearest multiple of pi/2 when within 1 deg
        o_snap = np.array([round(v / H) * H if abs(v / H - round(v / H)) * 90 < 1.0 else v for v in o])
        R["offset_snapped_over_pi2"] = [round(float(v) / H, 3) for v in o_snap]
        # --- live verification in Isaac on mapped xacro poses (incl. xacro home_2) ---
        HOME2 = {"left": [0.8333, 0, 0, -1.78, 0, -0.641, 0], "right": [-0.8333, 0, 0, -1.78, 0, -0.641, 0]}
        live, oob = [], 0
        poses = [np.array(HOME2[side])] + [np.array([rng.uniform(lo + 0.2, hi - 0.2) for lo, hi in XLIM[side]]) for _ in range(12)]
        for qx in poses:
            qu = sgn * qx + o_snap
            qu = (qu + math.pi) % (2 * math.pi) - math.pi
            if np.any(qu < lims[:, 0] - 1e-3) or np.any(qu > lims[:, 1] + 1e-3):
                oob += 1; continue
            Mi, qa = hold(side, qu)
            p, r = rel_pose_err(Mi, xfk(side, qx))
            live.append({"q_x": [round(float(v), 3) for v in qx], "track_err": round(float(np.max(np.abs(qa - qu))), 4),
                         "pos_err_m": round(p, 4), "rot_err_deg": round(r, 2)})
        R["live_poses_checked"] = len(live); R["live_poses_out_of_usd_limits"] = oob
        R["live_max_pos_err_m"] = max(v["pos_err_m"] for v in live) if live else None
        R["live_max_rot_err_deg"] = max(v["rot_err_deg"] for v in live) if live else None
        R["live"] = live[:4]
        R["home2_gripper_base_chassis"] = [round(float(v), 4) for v in xfk(side, HOME2[side])[:3, 3]]
        RESULT[side] = R
    O["mapping"] = RESULT
except Exception as e:
    O["error"] = repr(e); O["tb"] = traceback.format_exc()[-2000:]
print("###JSON###" + json.dumps(O, default=str) + "\n", flush=True)
app.close()
