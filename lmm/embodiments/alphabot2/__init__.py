"""AlphaBot 2 (AI² Robotics / 智平方) embodiment.

PROVISIONAL. No official URDF, SDK documentation or simulator asset for this
robot is public (audited 2026-09-21, see docs/ALPHABOT2_PUBLIC_DATA.md). The
arm/gripper/torso layout here is inferred from two independent public LeRobot
datasets; the mobile-base convention is **not known at all** and must be
declared explicitly before an action vector can be built.

Nothing in this package fabricates capability. The simulator adapter refuses to
run because there is no body to load, and the real adapter is an unimplemented
interface, not a stub that returns plausible numbers.
"""
