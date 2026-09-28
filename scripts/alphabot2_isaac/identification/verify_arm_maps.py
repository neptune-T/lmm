"""Disambiguate the right-arm mapping by link COMs, then qualify both arms.

Phase 1 (arm DOF limits temporarily widened so any mapped pose is reachable):
  compare each arm link's world COM (chassis frame) with the xacro inertial
  origin under each candidate mapping. Kinematically equivalent shoulder
  solutions differ in where link1/link2 sit; only the true one matches.
Phase 2 (limits = xacro limits pushed through the chosen mapping):
  FK agreement on random poses, the mapped home_2, self-contacts at home,
  and left/right mirror symmetry of the home.
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
    from pxr import Usd, UsdPhysics, PhysxSchema
    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.types import ArticulationAction
    from omni.physx import get_physx_simulation_interface
    import bot2_build

    def rpyR(r, p, y):
        cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
        return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr], [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]])
    def Tm(xyz, rpy): M = np.eye(4); M[:3, :3] = rpyR(*rpy); M[:3, 3] = xyz; return M
    def Ra(ax, q):
        a = np.asarray(ax, float); a = a / np.linalg.norm(a)
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        M = np.eye(4); M[:3, :3] = np.eye(3) + math.sin(q)*K + (1-math.cos(q))*K@K; return M
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
    def links(xml, d):
        out = {}
        for m in re.finditer(r'<link name="\$\{prefix\}link(\d)">(.*?)</link>', xml, re.S):
            b = resolve(m.group(2), d)
            inert = re.search(r'<inertial>(.*?)</inertial>', b, re.S).group(1)
            o = re.search(r'<origin[^>]*xyz="([^"]+)"', inert)
            out[int(m.group(1))] = (nums(o.group(1)) if o else [0, 0, 0], float(re.search(r'<mass value="([^"]+)"', inert).group(1)))
        return out
    ARM = open(f"{X}/arm.xacro").read(); BODY = joints(open(f"{X}/body.xacro").read())
    JS = {s: joints(ARM, 1 if s == "left" else -1) for s in ("left", "right")}
    LK = {s: links(ARM, 1 if s == "left" else -1) for s in ("left", "right")}
    XLIM = {s: [JS[s][f"${{prefix}}joint{k}"][3] for k in range(1, 8)] for s in JS}
    def xfk(side, q7):
        """returns gripper_base frame and COM positions of link1..7 (chassis frame)"""
        J = JS[side]; M = np.eye(4); coms = []
        for n in ("body_joint1", "body_joint2", "body_joint3", "body_joint4"):
            M = M @ Tm(BODY[n][0], BODY[n][1])
        M = M @ Tm(*J["${prefix}joint0"][:2])
        for k in range(1, 8):
            xyz, rpy, ax, _ = J[f"${{prefix}}joint{k}"]; M = M @ Tm(xyz, rpy) @ Ra(ax, q7[k-1])
            coms.append((M @ np.append(LK[side][k][0], 1))[:3])
        M = M @ Tm(*J["${prefix}joint8"][:2]) @ Tm([0, 0, 0.0695], [0, 0, 0]) @ Tm([0, 0, 0], [0, 0, 1.5708])
        return M, coms

    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()
    bot2_build.build(app, stage, add_reference_to_stage, O)
    world.scene.add_default_ground_plane()
    for p in Usd.PrimRange(stage.GetPrimAtPath("/World/Bot2"), Usd.TraverseInstanceProxies()):
        if p.HasAPI(UsdPhysics.RigidBodyAPI) and not p.IsInstanceProxy():
            PhysxSchema.PhysxContactReportAPI.Apply(p).CreateThresholdAttr().Set(0.0)
    art = SingleArticulation(prim_path="/World/Bot2", name="bot2")
    world.scene.add(art); world.reset()
    pv = lambda: art._articulation_view._physics_view
    arr = lambda x: x.numpy() if hasattr(x, "numpy") else np.asarray(x)
    DOF = [p.replace("/World/Bot2/", "") for p in pv().dof_paths[0]]
    LNK = [p.replace("/World/Bot2/", "") for p in pv().link_paths[0]]
    N = len(DOF)
    LIM0 = arr(pv().get_dof_limits()).reshape(-1, 2).copy()
    di = lambda s: [i for i, p in enumerate(DOF) if p.endswith(s)][0]
    SIDE = {"left": "Left", "right": "Right"}
    IDX = {s: [di(f"Arm_{S}/joints/{s}_joint{k}") for k in range(1, 8)] for s, S in SIDE.items()}
    LIDX = {s: [[i for i, p in enumerate(LNK) if p.startswith(f"Arm_{S}/") and p.split("/")[-1] == f"link{k}"][0]
                for k in range(1, 8)] for s, S in SIDE.items()}
    GB = {s: LNK.index(f"Gripper_{S}/gripper_base") for s, S in SIDE.items()}
    CH = [i for i, p in enumerate(LNK) if p.split("/")[-1] == "base_link" and not p.startswith(("Arm_", "Gripper_"))][0]
    COMS = arr(pv().get_coms()).reshape(-1, len(LNK), 7)[0]
    MASS = arr(pv().get_masses()).reshape(-1, len(LNK))[0]
    O["link_mass_usd_vs_xacro"] = {s: [[round(float(MASS[LIDX[s][k]]), 4), LK[s][k+1][1]] for k in range(7)] for s in SIDE}
    def quatR(q):
        x, y, z, w = q
        return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    def set_limits(L):
        pv().set_dof_limits(np.asarray(L, np.float32).reshape(1, N, 2), np.array([0], np.int32))
    H = math.pi / 2
    MAP = {"left": (np.array([1, 1, 1, 1, 1, 1, 1.]), np.array([0, -H, -H, 0, H, 0, 0])),
           "right_A": (np.array([1, -1, -1, 1, -1, -1, 1.]), np.array([-math.pi, H, -H, 0, -H, 0, 0])),
           "right_B": (np.array([1, 1, -1, 1, -1, -1, 1.]), np.array([0, -H, H, 0, -H, 0, 0]))}
    HOME2 = {"left": np.array([0.8333, 0, 0, -1.78, 0, -0.641, 0]), "right": np.array([-0.8333, 0, 0, -1.78, 0, -0.641, 0])}
    q_all = arr(art.get_joint_positions()).astype(float)
    q_all[IDX["left"]] = MAP["left"][0] * HOME2["left"] + MAP["left"][1]
    q_all[IDX["right"]] = HOME2["right"]*0 + np.array([-0.8333, -H, -H, -1.78, H, -0.641, 0])  # upright, collision-free
    def frames(side):
        T = arr(pv().get_link_transforms()).reshape(-1, len(LNK), 7)[0]
        Rc, pc = quatR(T[CH, 3:7]), T[CH, :3]
        M = np.eye(4); M[:3, :3] = Rc.T @ quatR(T[GB[side], 3:7]); M[:3, 3] = Rc.T @ (T[GB[side], :3] - pc)
        coms = []
        for li in LIDX[side]:
            w = T[li, :3] + quatR(T[li, 3:7]) @ COMS[li, :3]
            coms.append(Rc.T @ (w - pc))
        return M, coms
    def hold(side, q7, steps=1):
        q = q_all.copy(); q[IDX[side]] = q7
        art.set_joint_positions(q); art.set_joint_velocities(np.zeros_like(q))
        art.apply_action(ArticulationAction(joint_positions=q))
        for _ in range(steps): world.step(render=False)
        return arr(art.get_joint_positions()).astype(float)[IDX[side]]
    def pose_err(Ma, Mb):
        dR = Ma[:3, :3].T @ Mb[:3, :3]
        return float(np.linalg.norm(Ma[:3, 3] - Mb[:3, 3])), math.degrees(math.acos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
    wrap = lambda v: (v + math.pi) % (2 * math.pi) - math.pi
    rng = np.random.default_rng(3)

    # ---------------- Phase 1: COM disambiguation ----------------
    wide = LIM0.copy()
    for s in SIDE: wide[IDX[s]] = [-2 * math.pi, 2 * math.pi]
    set_limits(wide)
    P1 = {}
    for name, (sg, of) in MAP.items():
        side = "left" if name == "left" else "right"
        poses = [HOME2[side]] + [np.array([rng.uniform(lo + 0.2, hi - 0.2) for lo, hi in XLIM[side]]) for _ in range(6)]
        per_link = np.zeros(7); gb = []
        for qx in poses:
            qa = hold(side, wrap(sg * qx + of))
            Mi, ci = frames(side)
            qx_act = (wrap(qa - of)) * sg  # xacro angles actually realised
            Mx, cx = xfk(side, qx_act)
            gb.append(pose_err(Mi, Mx))
            per_link = np.maximum(per_link, [np.linalg.norm(a - b) for a, b in zip(ci, cx)])
        P1[name] = {"max_com_err_per_link_m": [round(float(v), 4) for v in per_link],
                    "gb_max_pos_m": round(max(e[0] for e in gb), 4), "gb_max_rot_deg": round(max(e[1] for e in gb), 2)}
    O["phase1_com"] = P1
    chosen = min(("right_A", "right_B"), key=lambda n: max(P1[n]["max_com_err_per_link_m"]))
    O["right_chosen"] = chosen

    # ---------------- Phase 2: limits through the mapping ----------------
    lim = LIM0.copy()
    mapped_lim = {}
    for side, name in (("left", "left"), ("right", chosen)):
        sg, of = MAP[name]
        ml = []
        for k in range(7):
            lo, hi = XLIM[side][k]; a, b = sorted((sg[k] * lo + of[k], sg[k] * hi + of[k]))
            ml.append([round(a, 4), round(b, 4)])
        mapped_lim[side] = ml
        lim[IDX[side]] = ml
    O["usd_limits_original"] = {s: [[round(float(a), 4), round(float(b), 4)] for a, b in LIM0[IDX[s]]] for s in SIDE}
    O["usd_limits_mapped_from_xacro"] = mapped_lim
    set_limits(lim)
    P2 = {}
    for side, name in (("left", "left"), ("right", chosen)):
        sg, of = MAP[name]
        errs, track = [], []
        for _ in range(12):
            qx = np.array([rng.uniform(lo + 0.05, hi - 0.05) for lo, hi in XLIM[side]])
            qu = sg * qx + of
            qa = hold(side, qu)
            Mi, _ = frames(side)
            errs.append(pose_err(Mi, xfk(side, qx)[0])); track.append(float(np.max(np.abs(qa - qu))))
        P2[side] = {"n_poses": 12, "max_pos_err_m": round(max(e[0] for e in errs), 4),
                    "max_rot_err_deg": round(max(e[1] for e in errs), 2), "max_track_err_rad": round(max(track), 4)}
    # mapped homes for both arms together, hold 90 steps, count self contacts
    q = q_all.copy()
    for side, name in (("left", "left"), ("right", chosen)):
        sg, of = MAP[name]; q[IDX[side]] = sg * HOME2[side] + of
    art.set_joint_positions(q); art.set_joint_velocities(np.zeros_like(q))
    art.apply_action(ArticulationAction(joint_positions=q))
    for _ in range(3): world.step(render=False)
    hdr, _ = get_physx_simulation_interface().get_contact_report()
    selfc = sorted({f"{str(h.actor0).replace('/World/Bot2/', '')} <-> {str(h.actor1).replace('/World/Bot2/', '')}"
                    for h in hdr if "GroundPlane" not in str(h.actor0) + str(h.actor1)})
    for _ in range(90): world.step(render=False)
    qn = arr(art.get_joint_positions()).astype(float)
    home = {"usd_home_left": [round(float(v), 4) for v in q[IDX["left"]]],
            "usd_home_right": [round(float(v), 4) for v in q[IDX["right"]]],
            "self_contact_pairs": selfc,
            "track_err_after_90_steps": round(float(np.max(np.abs(qn[IDX["left"] + IDX["right"]] - q[IDX["left"] + IDX["right"]]))), 4)}
    for side in SIDE:
        Mi, _ = frames(side)
        home[f"gb_{side}_chassis"] = [round(float(v), 4) for v in Mi[:3, 3]]
        home[f"approach_{side}"] = [round(float(v), 4) for v in Mi[:3, 2]]
    home["mirror_residual_m"] = round(float(np.linalg.norm(np.array(home["gb_left_chassis"]) * [1, -1, 1] - np.array(home["gb_right_chassis"]))), 4)
    # proprio round-trip in xacro convention from actual readback
    back = {}
    for side, name in (("left", "left"), ("right", chosen)):
        sg, of = MAP[name]
        back[side] = [round(float(v), 4) for v in wrap(qn[IDX[side]] - of) * sg]
    home["proprio_xacro_readback"] = back
    P2["home"] = home
    O["phase2"] = P2
except Exception as e:
    O["error"] = repr(e); O["tb"] = traceback.format_exc()[-2000:]
print("###JSON###" + json.dumps(O, default=str) + "\n", flush=True)
app.close()
