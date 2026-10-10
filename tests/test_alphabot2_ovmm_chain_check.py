"""Lowering-chain check: passes on a consistent expert-style episode, fails on injected interface faults."""
import unittest

import numpy as np

from lmm.embodiments.alphabot2 import cartesian as C
from lmm.embodiments.alphabot2.habitat.constants import HOME
from lmm.embodiments.alphabot2.ovmm_chain_check import lowering_chain


def episode(fault=None, T=60):
    q0 = {"left": np.array(HOME["left"], float), "right": np.array(HOME["right"], float)}
    ex = C.CartesianExecutor.from_proprio(np.r_[q0["left"], 0.0, q0["right"], 0.0])
    pr, cart, j18, base = [], [], [], []
    p = np.r_[q0["left"], 0.0, q0["right"], 0.0]
    x = z = yaw = 0.0
    for t in range(T):
        pr.append(p.copy())
        base.append([x, z, yaw])
        u = t / T
        j = np.r_[q0["left"] + 0.2 * u, 1.0 if t > 30 else 0.0, q0["right"] - 0.1 * u, 0.0,
                  0.3 if t < 40 else 0.0, 0.4 if 20 < t < 50 else 0.0]
        c = C.cartesian_from_joint_action(j, ex.prev_quat)          # expert: label = FK of commanded joints
        jl, _ = ex.lower(c)                                        # executed = lowered label
        lab = c.copy()
        if fault == "label_arms_swapped":
            lab = np.r_[c[7:14], c[0:7], c[15], c[14], c[16:]]
        if fault == "omega_label_sign":
            lab[17] = -lab[17]
        cart.append(lab)
        j18.append(jl)
        p = np.r_[jl[0:7], min(jl[7], 0.6), jl[8:15], jl[15]]     # left gripper stops on an object
        x += np.cos(yaw) * jl[16] * 0.1
        z += -np.sin(yaw) * jl[16] * 0.1
        yaw += (-jl[17] if fault == "base_turns_wrong_way" else jl[17]) * 0.1
    return {"action": np.array(cart), "joint18": np.array(j18), "proprio": np.array(pr)}, {"base_xz_yaw": np.array(base)}


KEYS = ("lowering", "joint_tracking", "fk_next_state_matches_label", "gripper", "base")


class TestChainCheck(unittest.TestCase):
    def verdict(self, fault):
        r = lowering_chain(*episode(fault))
        return {k: r[k]["pass"] for k in KEYS}

    def test_consistent_episode_passes(self):
        self.assertTrue(all(self.verdict(None).values()))

    def test_arm_swap_detected(self):
        v = self.verdict("label_arms_swapped")
        self.assertFalse(v["fk_next_state_matches_label"])
        self.assertFalse(v["lowering"])

    def test_omega_label_sign_detected(self):
        self.assertFalse(self.verdict("omega_label_sign")["base"])

    def test_base_direction_detected(self):
        self.assertFalse(self.verdict("base_turns_wrong_way")["base"])


if __name__ == "__main__":
    unittest.main()
