"""Live Habitat load probe for the AlphaBot 2 URDF (no OVMM task).

The WXAI embodiment failed because a ROS z-up URDF was loaded into Habitat's
y-up world without the body ever being rotated (it lay on its side). This probe
loads the AlphaBot URDF into bare habitat_sim and measures, for candidate root
rotations, where the head camera, both gripper bases and the wheels end up in
the world, and compares them with the URDF FK (z-up) mapped by the candidate
rotation. It also reports the Habitat link/joint index <-> name tables that the
robot class must resolve by name.
"""
import json, math, os, sys

import numpy as np

URDF = os.environ.get("ALPHABOT2_URDF", "/workspace/runtime/alphabot2_assets/alphabot2/alphabot2.urdf")
REPO = os.environ.get("LMM_REPO", "/workspace/lmm")
sys.path.insert(0, REPO)
import habitat_sim  # noqa: E402
import magnum as mn  # noqa: E402
from lmm.embodiments.alphabot2.habitat.urdf_kin import URDFKin  # noqa: E402

cfg = habitat_sim.SimulatorConfiguration()
cfg.scene_id = "NONE"
cfg.enable_physics = True
cfg.gpu_device_id = int(os.environ.get("GPU", "0"))
agent = habitat_sim.agent.AgentConfiguration()
sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent]))
aom = sim.get_articulated_object_manager()
ao = aom.add_articulated_object_from_urdf(URDF, fixed_base=True, maintain_link_order=True)
assert ao is not None, "URDF failed to load"
links = {ao.get_link_name(i): i for i in ao.get_link_ids()}
R = {"num_links": ao.num_links, "dofs": len(ao.joint_positions), "links": len(links)}
table = []
for name, i in sorted(links.items(), key=lambda kv: kv[1]):
    table.append({"link": name, "link_id": i, "joint": ao.get_link_joint_name(i),
                  "joint_type": str(ao.get_link_joint_type(i)), "pos_offset": ao.get_link_joint_pos_offset(i),
                  "num_pos": ao.get_link_num_joint_pos(i)})
R["link_table"] = table
kin = URDFKin(URDF)
ROI = ("head_camera", "left_gripper_base", "right_gripper_base", "base_link", "driving_wheel_left_link",
       "driving_wheel_right_link", "left_link8", "right_link8")
ROI = [r for r in ROI if r in links]
cands = {"identity": mn.Quaternion(),
         "rot_x_minus_90": mn.Quaternion.rotation(mn.Rad(-math.pi / 2), mn.Vector3(1, 0, 0))}
res = {}
for cname, q in cands.items():
    ao.transformation = mn.Matrix4.from_(q.to_matrix(), mn.Vector3(0, 0, 0))
    ao.joint_positions = [0.0] * len(ao.joint_positions)
    rot = np.array(q.to_matrix())                                   # URDF root frame -> world
    rows = {}
    for r in ROI:
        w = np.array(ao.get_link_scene_node(links[r]).absolute_translation)
        want = rot @ kin.pose(r)[:3, 3]
        wi = rot @ (kin.pose(r) @ kin.inertial(r))[:3, 3]      # link frame composed with its inertial origin
        rows[r] = {"world": w.round(4).tolist(), "fk_mapped": want.round(4).tolist(),
                   "err_m": round(float(np.linalg.norm(w - want)), 5),
                   "err_vs_inertial_frame_m": round(float(np.linalg.norm(w - wi)), 6)}
    up = rows["head_camera"]["world"][1] if "head_camera" in rows else None
    res[cname] = {"links": rows, "head_camera_world_y": up,
                  "upright": bool(up is not None and up > 1.2),
                  "urdf_up_in_world": (rot @ np.array([0, 0, 1.0])).round(3).tolist(),
                  "urdf_forward_in_world": (rot @ np.array([1.0, 0, 0])).round(3).tolist()}
R["candidates"] = res
print("###LOAD###" + json.dumps(R, default=str), flush=True)
sim.close()
