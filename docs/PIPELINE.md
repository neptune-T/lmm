# Canonical Pipeline

Exact execution path and the module implementing each stage. Every script lives
in `scripts/`; every module lives in `lmm/`.

## Stage map

| # | Stage | Script | Core modules |
|---|---|---|---|
| 1 | Full-task rollout | `rollout_full_task.py` | `lmm/rollout/runner.py`, `lmm/policy/pi05_policy.py`, `lmm/env/behavior_env.py` |
| 2 | Candidate snapshots | `discover_frontier.py` | `lmm/frontier/candidate_proposer.py`, `lmm/env/snapshot.py` |
| 3 | Continuation probes | `discover_frontier.py` | `lmm/frontier/continuation_probe.py` |
| 4 | Frontier estimation + selection | `discover_frontier.py` | `lmm/frontier/frontier_estimator.py`, `frontier_selector.py` |
| 5 | Cheap scaling | `scale_frontier.py` | `lmm/frontier/continuation_probe.py` (explore mode) |
| 6 | BF16 verification | `verify_candidates.py` | `lmm/verification/bf16_verifier.py` |
| 7 | Online RL update | `train_online.py` | `lmm/rl/{dataset_builder,reward,advantages,grpo,trainer}.py` |
| 8a | Frontier evaluation | `eval_frontier.py` | `lmm/eval/frontier_eval.py` |
| 8b | Full-task evaluation | `eval_full_task.py` | `lmm/eval/full_task_eval.py` |
| 8c | Cheap/BF16 fidelity | `eval_fidelity.py` | `lmm/eval/fidelity_eval.py` |

## Data flow

```
                     configs/policy/pi05_bf16_v1.json
                                  |
  [1] rollout_full_task.py  -->  full_rollouts.jsonl        (FullRolloutRecord)
                                  |
  [2] discover_frontier.py  -->  snapshots.jsonl            (SnapshotRecord)
                                  |
  [3]        "              -->  probes.jsonl               (ProbeRecord, mode=probe)
                                  |
  [4]        "              -->  frontier_summaries.jsonl   (FrontierSummary)
                                  |            selected_as_frontier == true
                                  v
  [5] scale_frontier.py     -->  cheap_exploration.jsonl    (CheapExplorationRecord)
                                  |            ranked by cheap_score
                                  v
  [6] verify_candidates.py  -->  verifications.jsonl        (VerificationRecord)
                                  |            verified outcomes only
                                  v
  [7] train_online.py       -->  training_samples.json      (TrainingSampleRecord)
                                  training_steps.jsonl      (TrainingStepRecord)
                                  |
  [8] eval_*.py             -->  frontier_eval.json / full_task_eval.json / fidelity.json
```

Every stage writes `resolved_manifest.json` into its output directory before
doing work. See [`DATA_FORMATS.md`](DATA_FORMATS.md) for record fields.

## Invariants enforced in code

| Invariant | Enforced by |
|---|---|
| Verification backend is high fidelity | `precision_backend.require_high_fidelity` |
| Cheap run cannot be its own verification | `VerificationRecord.validate` |
| Probes are not pooled across backends or policy versions | `frontier_estimator.estimate_root` |
| A frontier is never inferred from one continuation | `ProbeConfig.__post_init__`, `estimate_root(min_probes=)` |
| A group never spans two roots | `advantages.group_samples` |
| Every sample carries root and group identity | `TrainingSampleRecord` required fields |
| Degenerate groups are reported, not silently zeroed | `advantages.normalize_groups`, `degenerate_fraction` |
| Full-task eval rejects skill-sequencer rollouts | `full_task_eval._assert_end_to_end` |
| Experiment names cannot assert a conclusion | `ExperimentManifest.validate` |
| Native FP8 cannot be claimed on SM80 | `PrecisionBackend.check_hardware` |

## Determinism contract

`docs/FRESH_POLICY_PROCESS_FIRST_DIVERGENCE_AUDIT_V1.md` recorded
`MODEL_FORWARD_BACKEND_DIVERGENCE`: two cold policy processes fed byte-identical
requests produced 274/320 differing float32 values in the raw model output.

Consequences, which the canonical code assumes:

1. **One persistent policy server per experiment.** A server restart
   invalidates a complete root block. `Pi05Policy` holds a client to an
   already-running server and never spawns one per continuation.
2. **Fresh simulator process per execution**, with snapshot restore for the
   root state.
3. **Explicit per-continuation seeds**, so intended stochasticity is
   separable from backend noise.

## Running the pipeline

Stages 1-6 need a GPU host with the simulator stack. Every script accepts
`--dry-run`, which resolves and writes the manifest, validates configs and
prints the plan without touching a simulator — usable on any machine.

```bash
python scripts/rollout_full_task.py \
  --experiment-name e2e_bf16_sanity_v1 \
  --output-dir runs/e2e_sanity \
  --policy-checkpoint /path/to/bf16_ckpt --policy-version q0 \
  --env-config configs/env/behavior_ovmm_v1.json \
  --rollout-config configs/rollout/full_task_v1.json
```

```bash
python scripts/discover_frontier.py \
  --experiment-name frontier_probe_smoke_v1 \
  --output-dir runs/frontier_smoke \
  --policy-checkpoint /path/to/bf16_ckpt --policy-version q0 \
  --rollouts runs/e2e_sanity/full_rollouts.jsonl \
  --frontier-config configs/frontier/frontier_smoke_v1.json
```

Stages 7-8 are pure analysis over JSONL and run anywhere:

```bash
python scripts/eval_full_task.py \
  --rollouts runs/e2e_sanity/full_rollouts.jsonl \
  --baseline-rollouts runs/baseline/full_rollouts.jsonl \
  --output runs/e2e_sanity/full_task_eval.json
```

## Unimplemented boundaries

Three stages stop with an explicit `SystemExit` naming what must be bound,
rather than silently doing nothing:

- `discover_frontier.py` — snapshot capture needs the Habitat backend behind
  `SnapshotBackend` (`lmm/env/snapshot.py`).
- `scale_frontier.py` / `verify_candidates.py` — need a running policy server
  wired through `RolloutRunner.run_continuation`.
- `train_online.py` — needs a `PolicyBackend` implementation
  (`lmm/rl/trainer.py`) against the JAX/OpenPI checkpoint.

The historical code to port from is named in each message. See
[`NEXT_EXPERIMENT.md`](NEXT_EXPERIMENT.md).
