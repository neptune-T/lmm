"""Shared builder: load Ai2 Bot2 with the gripper re-attachment fix applied.

Imported by every live script so the fix is defined exactly once.
"""
from pxr import Usd, UsdGeom, UsdPhysics, Gf

ROOT = "/nix/yanjun/zts/alphabot2/deps/robot_usds"
BOT2 = f"{ROOT}/humanoid/Ai2/Bot2/Bot2.usda"
GRIP = f"{ROOT}/grippers/Inspire/EG2_4C2/Inspire_EG2_4C2.usda"
SIDES = {"Left": "left", "Right": "right"}


def build(app, stage, add_reference_to_stage, report):
    add_reference_to_stage(usd_path=BOT2, prim_path="/World/Bot2")
    stage.GetPrimAtPath("/World/Bot2").GetVariantSets().GetVariantSet("ROS").SetVariantSelection("none")
    for _ in range(3): app.update()
    xc = UsdGeom.XformCache()
    poses = {}
    for S in SIDES:
        g = f"/World/Bot2/Arm_{S}/link7/link8/tcp/Inspire_EG2_4C2"
        poses[S] = {"root": xc.GetLocalToWorldTransform(stage.GetPrimAtPath(g)),
                    "base": xc.GetLocalToWorldTransform(stage.GetPrimAtPath(g + "/gripper_base"))}
    for S in SIDES:
        stage.GetPrimAtPath(f"/World/Bot2/Arm_{S}").GetVariantSets().GetVariantSet("EE").SetVariantSelection("None")
    for _ in range(3): app.update()

    # verify the nested gripper is really gone, not just an empty placeholder
    nested = {}
    for S in SIDES:
        t = stage.GetPrimAtPath(f"/World/Bot2/Arm_{S}/link7/link8/tcp")
        rbs = [str(q.GetPath()) for q in Usd.PrimRange(t, Usd.TraverseInstanceProxies())
               if q.HasAPI(UsdPhysics.RigidBodyAPI) or q.IsA(UsdPhysics.Joint)] if t else []
        nested[S] = rbs
    report["nested_leftover_bodies_or_joints"] = nested

    for S in SIDES:
        path = f"/World/Bot2/Gripper_{S}"
        add_reference_to_stage(usd_path=GRIP, prim_path=path)
        gp = stage.GetPrimAtPath(path)
        gp.GetVariantSets().GetVariantSet("Physics").SetVariantSelection("physx")
        for q in Usd.PrimRange(gp):
            if q.HasAPI(UsdPhysics.ArticulationRootAPI):
                q.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        # The standalone gripper ships `root_joint`, a FixedJoint with an EMPTY
        # body0 -- i.e. gripper_base welded to the WORLD. Leaving it in place
        # bolts the whole robot to the world through its hands: PhysX picks
        # gripper_base as the root link (fixed_base=True), the base cannot
        # drive, and the two anchors close a kinematic loop through both arms.
        # The gripper must hang only from attach_to_link7.
        anchored = []
        for q in Usd.PrimRange(gp):
            if q.IsA(UsdPhysics.Joint):
                jj = UsdPhysics.Joint(q)
                if not jj.GetBody0Rel().GetTargets() or not jj.GetBody1Rel().GetTargets():
                    q.SetActive(False); anchored.append(str(q.GetPath()))
        report.setdefault("deactivated_world_anchors", []).extend(anchored)
        xf = UsdGeom.Xformable(gp); xf.ClearXformOpOrder()
        parent_w = xc.GetLocalToWorldTransform(stage.GetPrimAtPath("/World/Bot2"))
        xf.AddTransformOp().Set(poses[S]["root"] * parent_w.GetInverse())
    for _ in range(3): app.update()
    xc = UsdGeom.XformCache()
    for S in SIDES:
        link7 = stage.GetPrimAtPath(f"/World/Bot2/Arm_{S}/link7")
        gbase = stage.GetPrimAtPath(f"/World/Bot2/Gripper_{S}/gripper_base")
        rel = xc.GetLocalToWorldTransform(gbase) * xc.GetLocalToWorldTransform(link7).GetInverse()
        j = UsdPhysics.FixedJoint.Define(stage, f"/World/Bot2/Gripper_{S}/attach_to_link7")
        j.CreateBody0Rel().SetTargets([link7.GetPath()])
        j.CreateBody1Rel().SetTargets([gbase.GetPath()])
        j.CreateLocalPos0Attr().Set(Gf.Vec3f(rel.ExtractTranslation()))
        j.CreateLocalRot0Attr().Set(Gf.Quatf(rel.ExtractRotationQuat()))
        j.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
        j.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
        report.setdefault("grip_base_pose_err_m", {})[S] = round(float(
            (xc.GetLocalToWorldTransform(gbase).ExtractTranslation()
             - poses[S]["base"].ExtractTranslation()).GetLength()), 6)
