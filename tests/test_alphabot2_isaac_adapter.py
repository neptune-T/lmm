"""AlphaBot 2 Isaac adapter tests — naming, joint maps, gates, atomicity.

No Isaac Sim required: the backend is a Protocol and these use a fake. The
constants under test (joint maps, gripper polarity, rates) were measured live
on the public Ai2 Bot2 asset; these tests pin them so a later edit cannot
silently undo a measured fact.
"""

from __future__ import annotations

import math
import sys
import unittest

import numpy as np

from _helpers import REPO_ROOT  # noqa: F401

from lmm.embodiments.alphabot2 import camera_spec as CS
from lmm.embodiments.alphabot2 import isaac_adapter as IA
from lmm.embodiments.alphabot2.contract import (
    ACTION_DIM, BASE_FORWARD_INDEX, BASE_YAW_INDEX, LEFT_ARM_INDICES,
    LEFT_GRIPPER_INDEX, PROPRIO_DIM, RIGHT_ARM_INDICES, RIGHT_GRIPPER_INDEX,
    AlphaBot2Contract, ContractError,
)
from lmm.embodiments.alphabot2.isaac_adapter import (
    GRIPPER_JOINT_LIMITS, HOME_LEFT_ARM, HOME_RIGHT_ARM, LEFT_ARM_DOFS,
    LEFT_ARM_MAP, LEFT_GRIPPER_DOF, NAV2_MAX_V_MPS, RIGHT_ARM_DOFS,
    RIGHT_ARM_MAP, RIGHT_GRIPPER_DOF, AlphaBot2IsaacAdapter, ArmJointMap,
    IsaacUnavailable,
)

H = math.pi / 2


class FakeIsaac:
    """Records commands and counts physics steps."""

    def __init__(self):
        self.joints: dict[str, float] = {
            **dict(zip(LEFT_ARM_DOFS, LEFT_ARM_MAP.to_sim(HOME_LEFT_ARM))),
            **dict(zip(RIGHT_ARM_DOFS, RIGHT_ARM_MAP.to_sim(HOME_RIGHT_ARM))),
            LEFT_GRIPPER_DOF: 0.82, RIGHT_GRIPPER_DOF: 0.82,
        }
        self.base = (0.0, 0.0)
        self.steps = 0
        self.order: list[str] = []

    def set_joint_targets(self, mapping): self.joints.update(mapping); self.order.append("joints")
    def get_joint_positions(self): return dict(self.joints)
    def set_base_velocity(self, f, y): self.base = (f, y); self.order.append("base")
    def step(self, n): self.steps += n; self.order.append(f"step:{n}")
    def render(self, camera): return np.zeros((8, 8, 3), np.uint8)


def _adapter(backend=None):
    return AlphaBot2IsaacAdapter(AlphaBot2Contract(), backend=backend)


class TestAssetNaming(unittest.TestCase):
    def test_seven_dofs_per_arm_addressed_by_path(self):
        self.assertEqual(len(LEFT_ARM_DOFS), 7)
        self.assertEqual(len(RIGHT_ARM_DOFS), 7)
        self.assertEqual(LEFT_ARM_DOFS[0], "Arm_Left/joints/left_joint1")
        self.assertEqual(RIGHT_ARM_DOFS[6], "Arm_Right/joints/right_joint7")

    def test_gripper_dofs_live_under_the_reattached_gripper(self):
        """Both grippers name their joint ``gripper_joint``; only the path differs."""
        self.assertEqual(LEFT_GRIPPER_DOF, "Gripper_Left/joints/gripper_joint")
        self.assertEqual(RIGHT_GRIPPER_DOF, "Gripper_Right/joints/gripper_joint")
        self.assertEqual(LEFT_GRIPPER_DOF.split("/")[-1], RIGHT_GRIPPER_DOF.split("/")[-1])

    def test_measured_dof_census(self):
        self.assertEqual(IA.EXPECTED_DOF_COUNT, 14 + 2 + 2 * IA.GRIPPER_MIMIC_DOFS_PER_SIDE + 4 + 2 + 2 + 8)

    def test_vendor_home_mirrors_only_joint1(self):
        """The xacro mirrors axes 3-6 itself, so in vendor convention only q1 flips."""
        self.assertAlmostEqual(HOME_LEFT_ARM[0], -HOME_RIGHT_ARM[0])
        for k in range(1, 7):
            self.assertAlmostEqual(HOME_LEFT_ARM[k], HOME_RIGHT_ARM[k])


class TestArmJointMaps(unittest.TestCase):
    def test_left_map_is_offsets_only(self):
        self.assertEqual(LEFT_ARM_MAP.sign, (1.0,) * 7)
        np.testing.assert_allclose(LEFT_ARM_MAP.offset, (0, -H, -H, 0, H, 0, 0))

    def test_right_map_differs_from_left(self):
        """The USD right arm reuses the left arm's joints; the xacro mirrors."""
        self.assertEqual(RIGHT_ARM_MAP.sign, (1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0))
        np.testing.assert_allclose(RIGHT_ARM_MAP.offset, (0, -H, H, 0, -H, 0, 0))
        self.assertNotEqual(LEFT_ARM_MAP, RIGHT_ARM_MAP)

    def test_home_maps_to_the_live_verified_usd_pose(self):
        np.testing.assert_allclose(LEFT_ARM_MAP.to_sim(HOME_LEFT_ARM),
                                   (0.8333, -H, -H, -1.78, H, -0.641, 0.0))
        np.testing.assert_allclose(RIGHT_ARM_MAP.to_sim(HOME_RIGHT_ARM),
                                   (-0.8333, -H, H, -1.78, -H, 0.641, 0.0))

    def test_round_trip(self):
        rng = np.random.default_rng(0)
        for m in (LEFT_ARM_MAP, RIGHT_ARM_MAP):
            for _ in range(20):
                q = rng.uniform(-2, 2, 7)
                np.testing.assert_allclose(m.to_vendor(m.to_sim(q)), q, atol=1e-12)

    def test_bad_sign_rejected(self):
        with self.assertRaises(ContractError):
            ArmJointMap(sign=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5), offset=(0.0,) * 7)
        with self.assertRaises(ContractError):
            ArmJointMap(sign=(1.0,) * 6, offset=(0.0,) * 6)

    def test_left_needs_no_limit_patch(self):
        np.testing.assert_allclose(IA.sim_arm_limits("left"), IA.ASSET_SHIPPED_ARM_LIMITS, atol=1e-3)

    def test_right_patch_is_exactly_joints_3_and_5(self):
        patch = IA.right_arm_limit_patch()
        self.assertEqual(sorted(patch), ["Arm_Right/joints/right_joint3", "Arm_Right/joints/right_joint5"])
        np.testing.assert_allclose(patch["Arm_Right/joints/right_joint3"], (-3.1 + H, 3.1 + H))
        np.testing.assert_allclose(patch["Arm_Right/joints/right_joint5"], (-3.1 - H, 3.1 - H))

    def test_shipped_right_limits_exclude_the_vendor_home(self):
        """Why the patch exists: without it the right arm cannot reach home."""
        q = RIGHT_ARM_MAP.to_sim(HOME_RIGHT_ARM)
        outside = [k + 1 for k, (v, (lo, hi)) in enumerate(zip(q, IA.ASSET_SHIPPED_ARM_LIMITS))
                   if not lo - 1e-9 <= v <= hi + 1e-9]
        self.assertEqual(outside, [3, 5])
        for side, home in (("left", HOME_LEFT_ARM), ("right", HOME_RIGHT_ARM)):
            m = LEFT_ARM_MAP if side == "left" else RIGHT_ARM_MAP
            for v, (lo, hi) in zip(m.to_sim(home), IA.sim_arm_limits(side)):
                self.assertTrue(lo <= v <= hi)

    def test_unverified_maps_refuse(self):
        a = _adapter()
        a.arm_maps_verified = False
        with self.assertRaises(ContractError):
            a.arm_vendor_to_sim("left", HOME_LEFT_ARM)


class TestProprioAssembly(unittest.TestCase):
    def _joints(self, ql, qr, gl=0.82, gr=0.82):
        j = dict(zip(LEFT_ARM_DOFS, LEFT_ARM_MAP.to_sim(ql)))
        j.update(zip(RIGHT_ARM_DOFS, RIGHT_ARM_MAP.to_sim(qr)))
        j[LEFT_GRIPPER_DOF], j[RIGHT_GRIPPER_DOF] = gl, gr
        return j

    def test_proprio_is_16d_in_vendor_convention(self):
        ql = 0.1 * np.arange(1, 8)
        qr = -0.1 * np.arange(1, 8)
        s = _adapter().proprio_from_joint_map(self._joints(ql, qr, gl=0.82, gr=0.0))
        self.assertEqual(s.shape, (PROPRIO_DIM,))
        np.testing.assert_allclose(s[list(LEFT_ARM_INDICES)], ql, atol=1e-12)
        np.testing.assert_allclose(s[list(RIGHT_ARM_INDICES)], qr, atol=1e-12)
        self.assertAlmostEqual(s[LEFT_GRIPPER_INDEX], 0.0)    # joint 0.82 = open
        self.assertAlmostEqual(s[RIGHT_GRIPPER_INDEX], 1.0)   # joint 0.0 = closed

    def test_missing_dof_raises_not_zero_fills(self):
        with self.assertRaises(ContractError) as ctx:
            _adapter().proprio_from_joint_map(dict.fromkeys(LEFT_ARM_DOFS, 0.0))
        self.assertIn("missing", str(ctx.exception))

    def test_left_right_independent(self):
        ql = np.array(HOME_LEFT_ARM)
        ql[2] = 1.234
        s = _adapter().proprio_from_joint_map(self._joints(ql, HOME_RIGHT_ARM))
        self.assertAlmostEqual(s[LEFT_ARM_INDICES[2]], 1.234)
        self.assertAlmostEqual(s[RIGHT_ARM_INDICES[2]], HOME_RIGHT_ARM[2])

    def test_no_base_pose_in_state(self):
        s = _adapter(FakeIsaac()).observe_proprio()
        self.assertEqual(s.shape, (16,))


class TestGripperMapping(unittest.TestCase):
    def test_measured_polarity_zero_is_closed(self):
        self.assertFalse(IA.GRIPPER_ZERO_IS_OPEN)
        seps = [IA.MEASURED_PAD_SEPARATION_M[k] for k in sorted(IA.MEASURED_PAD_SEPARATION_M)]
        self.assertEqual(seps, sorted(seps))          # wider as the joint grows

    def test_policy_open_maps_to_joint_upper_limit(self):
        a = _adapter()
        lo, hi = GRIPPER_JOINT_LIMITS
        self.assertAlmostEqual(a.gripper_fraction_to_joint(0.0), hi)
        self.assertAlmostEqual(a.gripper_fraction_to_joint(1.0), lo)
        self.assertAlmostEqual(a.gripper_fraction_to_joint(0.5), (lo + hi) / 2)

    def test_unverified_aperture_refuses(self):
        a = _adapter()
        a.gripper_aperture_verified = False
        with self.assertRaises(ContractError) as ctx:
            a.gripper_fraction_to_joint(0.5)
        self.assertIn("unverified", str(ctx.exception))

    def test_fraction_joint_round_trip(self):
        for zero_open in (True, False):
            a = _adapter()
            a.gripper_zero_is_open = zero_open
            for f in (0.0, 0.25, 0.5, 0.75, 1.0):
                self.assertAlmostEqual(a.gripper_joint_to_fraction(a.gripper_fraction_to_joint(f)), f, places=9)


class TestBaseMapping(unittest.TestCase):
    def test_unverified_signs_refuse(self):
        a = _adapter()
        a.base_signs_verified = False
        with self.assertRaises(ContractError) as ctx:
            a.base_to_twist(0.1, 0.0)
        self.assertIn("sign convention", str(ctx.exception))

    def test_physical_units_pass_through_unscaled(self):
        """Unlike WXAI/Habitat, the diff-drive consumes m/s directly."""
        self.assertEqual(_adapter().base_to_twist(0.42, -0.31), (0.42, -0.31))

    def test_nav2_bounds_enforced_only_on_request(self):
        a = _adapter()
        a.base_to_twist(5.0, 0.0)
        with self.assertRaises(ContractError):
            a.base_to_twist(NAV2_MAX_V_MPS + 0.5, 0.0, enforce_nav2_bounds=True)

    def test_nav2_forbids_reverse_but_contract_allows_it(self):
        a = _adapter()
        a.base_to_twist(-0.2, 0.0)
        with self.assertRaises(ContractError):
            a.base_to_twist(-0.2, 0.0, enforce_nav2_bounds=True)

    def test_differential_drive_kinematics(self):
        wl, wr = IA.twist_to_wheel_radps(0.17, 0.0)
        self.assertAlmostEqual(wl, 2.0)
        self.assertAlmostEqual(wr, 2.0)
        wl, wr = IA.twist_to_wheel_radps(0.0, 1.0)          # +omega = counter-clockwise
        self.assertAlmostEqual(wr, IA.WHEEL_TRACK_M / 2 / IA.WHEEL_RADIUS_M)
        self.assertAlmostEqual(wl, -wr)


class TestActionMappingAndAtomicity(unittest.TestCase):
    def test_action_splits_into_16_joint_targets_plus_base(self):
        act = _adapter().home_action()
        act[RIGHT_ARM_INDICES[2]] += 0.3
        act[RIGHT_GRIPPER_INDEX] = 1.0
        act[BASE_FORWARD_INDEX], act[BASE_YAW_INDEX] = 0.2, -0.4
        c = _adapter().action_to_commands(act)
        t = c["joint_targets"]
        self.assertEqual(len(t), 16)
        self.assertAlmostEqual(t["Arm_Right/joints/right_joint3"], H - 0.3)   # sign -1 on right q3
        self.assertAlmostEqual(t[LEFT_GRIPPER_DOF], GRIPPER_JOINT_LIMITS[1])
        self.assertAlmostEqual(t[RIGHT_GRIPPER_DOF], GRIPPER_JOINT_LIMITS[0])
        self.assertEqual((c["base_forward_mps"], c["base_yaw_radps"]), (0.2, -0.4))

    def test_only_actuated_dofs_are_commanded(self):
        t = _adapter().action_to_commands(np.zeros(ACTION_DIM))["joint_targets"]
        self.assertEqual(set(t), {*LEFT_ARM_DOFS, *RIGHT_ARM_DOFS, LEFT_GRIPPER_DOF, RIGHT_GRIPPER_DOF})
        for n in t:
            for banned in ("body_joint", "head_joint", "driving_", "pad", "inner", "outer"):
                self.assertNotIn(banned, n)

    def test_bad_actions_rejected(self):
        with self.assertRaises(ContractError):
            _adapter().action_to_commands(np.zeros(16))
        bad = np.zeros(ACTION_DIM)
        bad[3] = np.nan
        with self.assertRaises(ContractError):
            _adapter().action_to_commands(bad)

    def test_rates(self):
        self.assertEqual(IA.PHYSICS_HZ, 200.0)
        self.assertEqual(IA.PHYSICS_STEPS_PER_ACTION, 20)
        self.assertEqual(IA.POLICY_HZ, 10.0)
        self.assertEqual(_adapter().physics_steps_per_action, 20)

    def test_one_action_is_one_logical_transition(self):
        """Targets staged BEFORE physics advances, exactly once."""
        fake = FakeIsaac()
        r = _adapter(fake).apply_action(np.zeros(ACTION_DIM))
        self.assertEqual(fake.order, ["joints", "base", "step:20"])
        self.assertEqual((r["sim_steps_before"], r["sim_steps_after"], r["physics_steps_consumed"]), (0, 20, 20))
        self.assertEqual(fake.steps, 20)

    def test_arm_and_base_do_not_consume_two_transitions(self):
        """The WXAI defect: one policy action must not cost two env steps."""
        fake = FakeIsaac()
        act = np.zeros(ACTION_DIM)
        act[LEFT_ARM_INDICES[0]] = 0.1
        act[BASE_FORWARD_INDEX] = 0.3
        _adapter(fake).apply_action(act)
        self.assertEqual(sum(1 for o in fake.order if o.startswith("step")), 1)

    def test_unmeasured_timing_refuses_to_apply(self):
        a = _adapter(FakeIsaac())
        a.physics_steps_per_action = None
        with self.assertRaises(ContractError) as ctx:
            a.apply_action(np.zeros(ACTION_DIM))
        self.assertIn("MEASURED", str(ctx.exception))

    def test_no_backend_raises(self):
        with self.assertRaises(IsaacUnavailable):
            _adapter().apply_action(np.zeros(ACTION_DIM))

    def test_home_action_round_trips_through_fake(self):
        fake = FakeIsaac()
        a = _adapter(fake)
        a.apply_action(a.home_action())
        np.testing.assert_allclose(a.observe_proprio(), np.r_[HOME_LEFT_ARM, 0.0, HOME_RIGHT_ARM, 0.0], atol=1e-12)


class TestBackendModuleWithoutIsaac(unittest.TestCase):
    def test_imports_without_isaac(self):
        from lmm.embodiments.alphabot2 import isaac_backend as IB
        self.assertNotIn("isaacsim", sys.modules)
        self.assertNotIn("pxr", sys.modules)
        self.assertEqual(len(IB.CONTROLLED_DOFS), 22)
        for d in IA.WHEEL_DOFS:
            self.assertNotIn(d, IB.CONTROLLED_DOFS)

    def test_camera_rows_look_along_forward(self):
        from lmm.embodiments.alphabot2.isaac_backend import camera_rows
        r = camera_rows((1, 0, 0), (0, 0, 1))
        self.assertAlmostEqual(np.linalg.det(r), 1.0)
        np.testing.assert_allclose(-r[2], (1, 0, 0), atol=1e-12)   # USD cameras look along -Z
        np.testing.assert_allclose(r[1], (0, 0, 1), atol=1e-12)    # image up
        np.testing.assert_allclose(r[0], (0, -1, 0), atol=1e-12)   # image right = robot right
        with self.assertRaises(ValueError):
            camera_rows((1, 0, 0), (2, 0, 0))

    def test_quaternion_convention_is_xyzw(self):
        from lmm.embodiments.alphabot2.isaac_backend import quat_xyzw_to_matrix
        np.testing.assert_allclose(quat_xyzw_to_matrix((0, 0, 0, 1)), np.eye(3))
        s = math.sqrt(0.5)
        np.testing.assert_allclose(quat_xyzw_to_matrix((0, 0, s, s)) @ (1, 0, 0), (0, 1, 0), atol=1e-12)


class TestAvailabilityAndCameras(unittest.TestCase):
    def test_availability(self):
        av = AlphaBot2IsaacAdapter.availability()
        self.assertEqual(av["asset_classification"], "PUBLIC_THIRD_PARTY_ASSET")
        self.assertEqual(av["physics_hz"], 200.0)
        self.assertEqual(av["policy_hz"], 10.0)

    def test_three_cameras_mapped_to_asset_frames(self):
        self.assertEqual(len(CS.CANONICAL_CAMERAS), 3)
        self.assertEqual(CS.ASSET_FRAMES["cam_front"], "head_camera")
        self.assertEqual(CS.MODEL_KEYS["cam_front"], "base_0_rgb")
        self.assertEqual(set(CS.USD_CAMERA_MOUNTS), set(CS.CANONICAL_CAMERAS))

    def test_only_the_live_qualified_contract_claims_verification(self):
        d = CS.CameraContract().describe()
        self.assertFalse(d["frames_verified"] or d["orientation_verified"] or d["channel_order_verified"])
        q = CS.LIVE_QUALIFIED.describe()
        self.assertTrue(q["frames_verified"] and q["orientation_verified"] and q["channel_order_verified"])
        self.assertEqual(q["resolution"], (240, 320))
        self.assertFalse(q["depth"])


if __name__ == "__main__":
    unittest.main()
