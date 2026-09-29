"""AlphaBot 2 Habitat constants (numpy-free, importable without Habitat)."""

#: gripper_base -> pad midpoint along the approach axis (+z of gripper_base), measured in Isaac.
REACH_M = 0.1196

#: Vendor home posture (the xacro's home_2), identical to the Isaac adapter's.
HOME = {"left": (0.8333, 0.0, 0.0, -1.7800, 0.0, -0.6410, 0.0),
        "right": (-0.8333, 0.0, 0.0, -1.7800, 0.0, -0.6410, 0.0)}

#: Canonical fixed v1 torso posture (body_joint1..4), baked into the URDF by build_urdf.py.
TORSO = (-1.1, 2.2, -1.1, 0.0)

#: Navigation tuck (tuck_search.py): the collision-free posture closest to HOME whose arms stay
#: inside the chassis turning circle (0.369 m vs 0.378 m) at the crouched canonical torso, >= 2 cm from the
#: torso/head surfaces; 0.10 rad from HOME. Right arm = vendor mirror (joint1 negated).
NAV_TUCK = {"left": (0.8731, -0.0265, 0.0657, -1.7393, 0.0325, -0.6495, -0.0281),
            "right": (-0.8731, -0.0265, 0.0657, -1.7393, 0.0325, -0.6495, -0.0281)}
