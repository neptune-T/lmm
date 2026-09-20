# Quantization Backends

> **Path note.** Paths such as `scripts/rollout/...`, `scripts/diagnostics/...`,
> `configs/branch_support/...` and historical `docs/<NAME>.md` refer to the
> companion repository **[neptune-T/llm_rollout](https://github.com/neptune-T/llm_rollout)**,
> not to this one. See [`LEGACY_CODE_LOCATION.md`](LEGACY_CODE_LOCATION.md).

Canonical registry: `lmm/rollout/precision_backend.py`. This document explains
what each name means and what the historical names meant.

## The rule

> **Backend names describe what the runtime actually executes, not what a
> checkpoint file is called.**

The prior FP8 work on A800 hardware was **FP8 storage with dequantisation to
BF16 compute**. It was never native FP8 tensor-core execution, and the project's
own hardware audit says so. From
`experiments/fp8_rollout_scaling_v1/hardware_audit.json`:

```json
{
  "classification": "NATIVE_FP8_COMPUTE_UNSUPPORTED_ON_CURRENT_A800",
  "gpu_name": "NVIDIA A800-SXM4-80GB",
  "compute_capability": "8.0",
  "architecture": "Ampere SM80",
  "native_fp8_compute_supported": false,
  "fp8_ptq_representation_supported": true,
  "blocked_claims": [
    "native FP8 GEMM throughput",
    "Transformer Engine FP8 acceleration",
    "native FP8 latency improvement",
    "native FP8 compute efficiency"
  ],
  "allowed_study": "FP8 PTQ storage with auditable dequantization to BF16 compute"
}
```

And from `experiments/fp8_rollout_scaling_v1/native_fp8_deferred_manifest.json`:

```json
{
  "status": "DEFERRED_PENDING_ADA_OR_HOPPER",
  "required_hardware": "Ada SM89 or Hopper SM90+ with a verified native FP8 Tensor Core path",
  "forbidden_shortcut": "Do not call storage-only or BF16-dequant execution native FP8 compute"
}
```

**This was recorded honestly at the time and must not be rewritten.** The prior
work did not claim native FP8 compute; it explicitly deferred it. The cleanup
encodes that discipline in code so it cannot be lost:
`PrecisionBackend.check_hardware()` raises on any attempt to run `native_fp8`
below compute capability 8.9.

## Canonical backends

| Name | Storage | Compute | Fidelity | Requires CC | Notes |
|---|---|---|---|---|---|
| `bf16` | bf16 | bf16 | **high** | 8.0 | Canonical policy. The only verification backend. |
| `fp8_storage_bf16_compute` | fp8 | bf16 | cheap | 8.0 | What the A800 "FP8" experiments actually ran. Saves memory, not a compute path. |
| `int4_weight_only` | int4 | bf16 | cheap | 8.0 | 4-bit weights, BF16 activations. Historically "Q4". |
| `native_fp8` | fp8 | **fp8** | cheap | **8.9** | True FP8 tensor cores. **Never validated on any hardware this project has used.** Declared so the name cannot be attached to a dequantising backend. |

`bf16` is the only backend with `fidelity == "high"`.
`require_high_fidelity()` raises for every other name, so a cheap outcome can
never enter the training set as a verified one.

## Historical aliases

Every alias resolves through `precision_backend.resolve()`.

| Historical name | Resolves to | Why it was ambiguous |
|---|---|---|
| `fp8`, `fp8_preview`, `fp8_ptq` | `fp8_storage_bf16_compute` | Read as native FP8 compute; it never was |
| `cheap` | `fp8_storage_bf16_compute` | Named a role, not a precision |
| `q4`, `q4_preview`, `servable_merged_q4` | `int4_weight_only` | **Collides with training round 4** |
| `high`, `bf16_oracle` | `bf16` | Named a role, not a precision |
| `champion` | `bf16` | The BEHAVIOR-challenge winning checkpoint |

### The worst hazard: `Q4`

`q0`, `q1`, `q4`, `q5`, `q10`, `q20` in checkpoint names are **training round
indices**, not precisions. `q4` in `paper_closure/` means *the checkpoint after
round 4*. `Q4` in the quantization scripts means *4-bit weight-only*.

These are unrelated concepts that shared a token. In canonical code:

- a precision is always spelled `int4_weight_only`;
- a round index is always a `policy_version` field, never a backend name.

`tests/test_precision_backends.py` asserts that no ambiguous token
(`cheap`, `high`, `champion`, `fp8`, `q4`, `mixed`, `normal`) is ever a
canonical backend name.

## Checkpoint generation

The historical quantisation path, preserved and un-moved:

| Step | Script |
|---|---|
| JAX FP8 PTQ quantisation | `../scripts/fp8_rollout_scaling/quantize_jax_checkpoint_fp8.py` |
| PTQ evaluation | `../scripts/fp8_rollout_scaling/evaluate_jax_fp8_ptq.py` |
| Quantised policy server | `../scripts/fp8_rollout_scaling/pi0_http_policy_server_fp8_q4_v3.py` |
| BF16 parity check | `../scripts/fp8_rollout_scaling/run_bf16_parity_explicit_lora.py` |
| LoRA merge to PyTorch | `../scripts/fp8_rollout_scaling/convert_task_matched_jax_to_pytorch_explicit_lora.py` |

Paths are relative to `lmm_rollout_project/`; these live at the workspace root
and are **not under version control** — see
[`CODEBASE_INVENTORY_20260919.md`](CODEBASE_INVENTORY_20260919.md) §5.

## Expected behaviour

| Backend | Memory | Speed | Caveat |
|---|---|---|---|
| `bf16` | baseline | baseline | ground truth for every scientific claim |
| `fp8_storage_bf16_compute` | reduced resident weights | **no GEMM speedup on SM80** | any wall-clock gain comes from memory pressure and batching, not arithmetic |
| `int4_weight_only` | lowest | dequant overhead per forward | largest quality risk; selection-only |
| `native_fp8` | unknown | unknown | unmeasured; do not cite |

Cost accounting from the A800 cascade benchmark is under
`paper_data/server_pull_20260913/.../cost_model/a800_q4_inference.json` and
`paper_closure/cascade_efficiency/`.

## If native FP8 hardware becomes available

The deferral manifest pins the checkpoints to reuse and forbids repeating the
full A800 physics campaign. Measure resident VRAM, peak VRAM, latency,
throughput, max inference batch and max rollout concurrency, plus a small
paired rollout sanity set against the characterised A800 FP8 PTQ behaviour.
Only then may `native_fp8` be used, and only on CC >= 8.9.
