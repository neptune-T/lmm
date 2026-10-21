import unittest
import numpy as np

from lmm.embodiments.alphabot2.gripper_geometry import audit_gripper, q_for_aperture
from lmm.embodiments.alphabot2.ovmm_chain_check import lowering_chain
from test_alphabot2_ovmm_chain_check import episode


class GripperEvidenceTests(unittest.TestCase):
    def test_wide_held_object_and_empty_hand(self):
        width = .07103033934369066
        fraction = 1 - q_for_aperture(width) / .82
        self.assertLess(fraction, .15)
        result = audit_gripper([1., 1., 0.], [fraction, 1., 0.], [0, -1, -1], [width, 0., 0.])
        self.assertTrue(result["pass"])
        self.assertAlmostEqual(fraction, .139068204482331, places=6)

    def test_jam_wrong_width_and_open_while_holding_fail(self):
        for cmd, nxt, held, width in [(1., .139, -1, 0.), (1., .139, 0, .03),
                                       (0., .139, 0, .07103), (1., 0., 0, .09)]:
            self.assertFalse(audit_gripper([cmd], [nxt], [held], [width])["pass"])

    def test_malformed_and_nonfinite_rejected(self):
        for held, widths in [([0, 1], [.03]), ([0], [np.nan]), ([.5], [.03])]:
            with self.assertRaises(ValueError):
                audit_gripper([1.], [.5], held, widths)

    def test_chain_uses_evidence_but_legacy_requires_old_gate(self):
        low, md = episode()
        width = .07103033934369066
        cl = low["joint18"][:, 7] >= .95
        low["proprio"][1:, 7] = np.where(cl[:-1], 1 - q_for_aperture(width) / .82, 0.)
        self.assertFalse(lowering_chain(low, md)["gripper"]["pass"])
        md["held_after"] = np.full((len(cl), 2), -1)
        md["held_after"][cl, 0] = 0
        md["held_width_after"] = np.zeros((len(cl), 2))
        md["held_width_after"][cl, 0] = width
        self.assertTrue(lowering_chain(low, md)["gripper"]["pass"])
        md["held_after"][:, 0] = -1
        self.assertFalse(lowering_chain(low, md)["gripper"]["pass"])
