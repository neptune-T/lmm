# LMM — Long-Horizon Mobile Manipulation

Online RL with rollout scaling for long-horizon mobile manipulation.

```
End-to-end VLA rollout -> candidate snapshots -> continuation probes
  -> empirical frontier -> cheap scaling -> BF16 verification
  -> online RL update -> full-task evaluation
```

This repository is the **canonical implementation**. The historical codebase is
not duplicated here — it stays in the companion repo
**[neptune-T/llm_rollout](https://github.com/neptune-T/llm_rollout)**. See
[`docs/LEGACY_CODE_LOCATION.md`](docs/LEGACY_CODE_LOCATION.md) for the path
convention used in these docs.

## Read these first

1. [`docs/CURRENT_METHOD.md`](docs/CURRENT_METHOD.md) — the method and the five open questions
2. [`docs/PIPELINE.md`](docs/PIPELINE.md) — exact execution path, stage by stage
3. [`docs/NEXT_EXPERIMENT.md`](docs/NEXT_EXPERIMENT.md) — the one experiment to run next
4. [`docs/CODEBASE_INVENTORY_20260919.md`](docs/CODEBASE_INVENTORY_20260919.md) — what exists and in what state
5. [`docs/LEGACY_EXPERIMENTS.md`](docs/LEGACY_EXPERIMENTS.md) — historical results, including the negative ones that drove this redesign

Also: [`docs/DATA_FORMATS.md`](docs/DATA_FORMATS.md),
[`docs/QUANTIZATION_BACKENDS.md`](docs/QUANTIZATION_BACKENDS.md),
[`legacy/README.md`](legacy/README.md).

## Layout

```
lmm/                       canonical package (37 modules)
  schemas.py               all record types (docs/DATA_FORMATS.md)
  manifest.py              experiment manifest, stamped with git state
  policy/                  end-to-end pi0.5 interface
  env/                     simulator adapter + snapshots (NO task planning)
  instrumentation/         event observers (NOT controllers)
  rollout/                 runner, client, server contract, precision backends
  frontier/                candidate proposal, probes, V_hat, selection
  verification/            BF16 verification with cheap->verified lineage
  rl/                      dataset, reward, advantages, GRPO, trainer
  eval/                    frontier / full-task / fidelity

scripts/                   8 canonical entry points
configs/{env,policy,rollout,frontier,rl,eval}/
tests/                     103 tests, no GPU or simulator required
docs/                      method, pipeline, data formats, legacy map
```

## Three rules the code enforces

1. **The policy is end-to-end.** No skill sequencer decides when navigation
   ends or manipulation begins. Full-task evaluation rejects rollouts produced
   with one active.
2. **The frontier is empirical.** A root is selected by measured continuation
   success `V_hat`, never by semantic position. Thresholds live in config, not
   in code.
3. **Cheap never masquerades as verified.** Only `bf16` is high fidelity;
   cheap outcomes select candidates but never become training targets, and a
   cheap run cannot be relabelled as its own verification.

Also enforced: groups never span two roots; degenerate groups are reported
rather than silently zeroed; experiment names cannot assert a conclusion;
native FP8 cannot be claimed below compute capability 8.9.

## Run the tests

No GPU, simulator or policy server needed — every heavy dependency sits behind
a Protocol, and the scientific core depends only on the standard library plus
numpy.

```bash
cd tests && python3 -m unittest discover -s . -p 'test_*.py'
```

Every entry point also accepts `--dry-run`, which resolves and writes the
experiment manifest, validates configs and prints the plan without touching a
simulator:

```bash
python scripts/discover_frontier.py \
  --experiment-name frontier_probe_smoke_v1 \
  --output-dir runs/smoke \
  --policy-checkpoint /path/to/ckpt --policy-version q0 \
  --rollouts runs/e2e/full_rollouts.jsonl \
  --frontier-config configs/frontier/frontier_smoke_v1.json \
  --dry-run
```

## Status

The end-to-end policy path is **not yet validated** — see
[`docs/NEXT_EXPERIMENT.md`](docs/NEXT_EXPERIMENT.md). Three stages
(`discover_frontier`, `scale_frontier` / `verify_candidates`, `train_online`)
stop with an explicit message naming the simulator or policy backend still to
be bound, rather than silently doing nothing.

The A800 cluster was lost on 2026-09-14; the q20 checkpoint was recovered from
Hugging Face, exact q0/q5/q10 were not.
