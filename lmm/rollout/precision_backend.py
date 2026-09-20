"""Precision backend registry.

Backend names in this project describe **what the runtime actually executes**,
not what a checkpoint file happens to be called.

The historical codebase used "FP8" for what was, on A800 / SM80 hardware, FP8
*storage* with dequantisation to BF16 compute. The measured hardware audit
(``experiments/fp8_rollout_scaling_v1/hardware_audit.json``) classified that
machine ``NATIVE_FP8_COMPUTE_UNSUPPORTED_ON_CURRENT_A800`` and explicitly
blocked the claims "native FP8 GEMM throughput" and "native FP8 latency
improvement". The accompanying deferral manifest names the forbidden shortcut:

    "Do not call storage-only or BF16-dequant execution native FP8 compute."

This module encodes that rule so it cannot be lost again: a backend advertises
its storage dtype and its compute dtype separately, and
``native_fp8_compute`` is a property of the *runtime*, never of a name.

See ``docs/QUANTIZATION_BACKENDS.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class PrecisionError(ValueError):
    """An unknown or misdeclared precision backend."""


@dataclass(frozen=True)
class PrecisionBackend:
    """Declarative description of a rollout execution precision."""

    name: str
    storage_dtype: str
    compute_dtype: str
    fidelity: str  # "high" | "cheap"
    requires_compute_capability: float | None
    description: str
    historical_aliases: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_high_fidelity(self) -> bool:
        return self.fidelity == "high"

    @property
    def claims_native_fp8_compute(self) -> bool:
        """True only if the backend genuinely executes FP8 arithmetic."""
        return self.compute_dtype == "fp8"

    def check_hardware(self, compute_capability: float | None) -> None:
        """Raise if this backend cannot honestly run on the given hardware."""
        need = self.requires_compute_capability
        if need is None:
            return
        if compute_capability is None:
            raise PrecisionError(
                f"backend {self.name!r} requires compute capability >= {need} "
                "but the hardware was not identified"
            )
        if compute_capability < need:
            raise PrecisionError(
                f"backend {self.name!r} requires compute capability >= {need}; "
                f"this host reports {compute_capability}. Refusing to run: "
                "storage-only or dequantised execution must not be reported as "
                f"{self.compute_dtype} compute."
            )


#: Canonical backends. ``bf16`` is the only high-fidelity backend.
BACKENDS: dict[str, PrecisionBackend] = {
    "bf16": PrecisionBackend(
        name="bf16",
        storage_dtype="bf16",
        compute_dtype="bf16",
        fidelity="high",
        requires_compute_capability=8.0,
        description=(
            "Canonical high-fidelity policy. The only backend whose outcomes "
            "count as verification."
        ),
        historical_aliases=("high", "bf16_oracle", "champion"),
    ),
    "fp8_storage_bf16_compute": PrecisionBackend(
        name="fp8_storage_bf16_compute",
        storage_dtype="fp8",
        compute_dtype="bf16",
        fidelity="cheap",
        requires_compute_capability=8.0,
        description=(
            "FP8 post-training-quantised weights, dequantised to BF16 for "
            "compute. This is what the historical A800 'FP8' experiments "
            "actually ran. Saves memory, NOT a native FP8 compute path."
        ),
        historical_aliases=("fp8", "fp8_preview", "fp8_ptq", "cheap"),
    ),
    "int4_weight_only": PrecisionBackend(
        name="int4_weight_only",
        storage_dtype="int4",
        compute_dtype="bf16",
        fidelity="cheap",
        requires_compute_capability=8.0,
        description=(
            "4-bit weight-only quantisation, BF16 activations and compute. "
            "NOTE: historically written 'Q4', which collides with the q0/q4/"
            "q20 *training round* indices. Never reuse the bare token 'Q4'."
        ),
        historical_aliases=("q4", "q4_preview", "servable_merged_q4"),
    ),
    "native_fp8": PrecisionBackend(
        name="native_fp8",
        storage_dtype="fp8",
        compute_dtype="fp8",
        fidelity="cheap",
        requires_compute_capability=8.9,  # Ada SM89 / Hopper SM90+
        description=(
            "True FP8 tensor-core execution. DEFERRED: never validated on any "
            "hardware this project has used. Requires Ada SM89 or Hopper "
            "SM90+. Declared here only so the name cannot be silently "
            "attached to a dequantising backend."
        ),
        historical_aliases=(),
    ),
}

#: Maps every historical alias to its canonical backend name.
ALIASES: dict[str, str] = {
    alias: b.name for b in BACKENDS.values() for alias in b.historical_aliases
}

HIGH_FIDELITY_BACKEND = "bf16"


def resolve(name: str) -> PrecisionBackend:
    """Resolve a canonical name or a historical alias to a backend.

    Raises :class:`PrecisionError` for unknown names, so a typo can never
    silently degrade into an untracked precision.
    """
    if name in BACKENDS:
        return BACKENDS[name]
    if name in ALIASES:
        return BACKENDS[ALIASES[name]]
    raise PrecisionError(
        f"unknown precision backend {name!r}. Known: "
        f"{sorted(BACKENDS)} (aliases: {sorted(ALIASES)})"
    )


def require_high_fidelity(name: str) -> PrecisionBackend:
    """Resolve ``name`` and assert it is the verification backend."""
    backend = resolve(name)
    if not backend.is_high_fidelity:
        raise PrecisionError(
            f"backend {backend.name!r} is cheap (fidelity={backend.fidelity}); "
            "verification outcomes must come from a high-fidelity backend"
        )
    return backend
