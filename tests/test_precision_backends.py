"""Precision-backend honesty rules (cleanup requirement 11)."""

from __future__ import annotations

import unittest

import _helpers  # noqa: F401,E402  (path setup)
from lmm.rollout.precision_backend import (  # noqa: E402
    ALIASES,
    BACKENDS,
    HIGH_FIDELITY_BACKEND,
    PrecisionError,
    require_high_fidelity,
    resolve,
)


class TestBackendHonesty(unittest.TestCase):
    def test_historical_fp8_is_storage_not_compute(self):
        b = resolve("fp8")
        self.assertEqual(b.name, "fp8_storage_bf16_compute")
        self.assertEqual(b.storage_dtype, "fp8")
        self.assertEqual(b.compute_dtype, "bf16")
        self.assertFalse(b.claims_native_fp8_compute)

    def test_only_native_fp8_claims_fp8_compute(self):
        claiming = [b.name for b in BACKENDS.values() if b.claims_native_fp8_compute]
        self.assertEqual(claiming, ["native_fp8"])

    def test_native_fp8_refused_on_sm80(self):
        """A800 is SM80; the deferral manifest forbids claiming native FP8."""
        with self.assertRaises(PrecisionError) as ctx:
            BACKENDS["native_fp8"].check_hardware(8.0)
        self.assertIn("Refusing to run", str(ctx.exception))

    def test_native_fp8_allowed_on_hopper(self):
        BACKENDS["native_fp8"].check_hardware(9.0)  # must not raise

    def test_q4_alias_maps_to_int4_weight_only(self):
        self.assertEqual(resolve("q4").name, "int4_weight_only")

    def test_int4_documents_the_round_index_collision(self):
        self.assertIn("training round", BACKENDS["int4_weight_only"].description)

    def test_unknown_backend_is_an_error_not_a_default(self):
        with self.assertRaises(PrecisionError):
            resolve("cheap_fast_v2")

    def test_only_bf16_is_high_fidelity(self):
        high = [b.name for b in BACKENDS.values() if b.is_high_fidelity]
        self.assertEqual(high, [HIGH_FIDELITY_BACKEND])

    def test_cheap_backends_refused_for_verification(self):
        for name in ("fp8_storage_bf16_compute", "int4_weight_only", "native_fp8"):
            with self.assertRaises(PrecisionError):
                require_high_fidelity(name)

    def test_every_alias_resolves(self):
        for alias in ALIASES:
            self.assertIn(resolve(alias).name, BACKENDS)

    def test_ambiguous_names_are_aliases_only_never_canonical(self):
        for ambiguous in ("cheap", "high", "champion", "fp8", "q4", "mixed", "normal"):
            self.assertNotIn(ambiguous, BACKENDS,
                             f"{ambiguous!r} must not be a canonical backend name")


if __name__ == "__main__":
    unittest.main()
