"""Canonical implementation of Long-Horizon Mobile Manipulation (LMM).

Method: online RL with rollout scaling for long-horizon mobile manipulation.

    full-task rollout -> candidate snapshots -> continuation probes
    -> empirical frontier -> cheap scaling -> BF16 verification
    -> online RL update -> full-task evaluation

Design rules enforced across this package:

* The policy is **end-to-end**. No handcrafted semantic skill sequencer decides
  when navigation ends or manipulation begins. See ``lmm.policy``.
* Semantic events are **observers**, never controllers. See
  ``lmm.instrumentation``.
* A frontier state is defined **empirically** by continuation statistics
  (``V_hat``), not by a predefined skill boundary. See ``lmm.frontier``.
* Cheap and high-fidelity outcomes are **never mixed**. See
  ``lmm.verification``.
* Root/group identity is **explicit and stored**, never inferred from
  filenames. See ``lmm.rl``.

The scientific core of this package (schemas, frontier estimation, grouping,
advantages, evaluation) depends only on the standard library plus numpy, so it
is testable without a GPU, simulator or policy server.
"""

__version__ = "0.1.0"
