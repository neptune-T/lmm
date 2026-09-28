"""AlphaBot 2 (AI² Robotics / 智平方) embodiment.

No official URDF, SDK documentation or simulator asset for this robot is
public. The simulator body is the **public third-party** ``fiveages-sim`` Ai2
Bot2 asset (``PUBLIC_THIRD_PARTY_ASSET``), which has not been verified as the
internal AI² Robotics digital twin (docs/ALPHABOT2_SIM_ASSET_PROVENANCE.md).

* ``contract``      frozen v1 policy contract: 16-D proprio, 18-D action
* ``isaac_adapter`` contract <-> Ai2 Bot2 USD (joint maps, gripper polarity,
                    diff-drive, atomic stepping); numpy only
* ``isaac_backend`` live Isaac Sim 4.5 backend; imports Isaac lazily
* ``real_adapter``  real-robot interface, unimplemented (no public SDK)
* ``sim_adapter``   superseded historical record of the 2026-09-21 search

The generic LMM stack never imports Isaac.
"""
