# Canonical Data Formats

Schemas: `lmm/schemas.py`. Serialisation: JSON Lines, one record per line.
Every line carries `schema_version` (`lmm_schema_v1`) and `record_type`, so
`read_jsonl` reconstructs the concrete type without inspecting the path.

> **No ad-hoc filename parsing for scientific analysis.** Identity lives inside
> the record. A record is valid wherever the file sits.

## Full rollout record — `FullRolloutRecord`

End-to-end trajectory from the episode initial state.

| Field | Type | Notes |
|---|---|---|
| `rollout_id` | str | required |
| `episode_id` | str | required |
| `scene_id` | str | required |
| `task` | str | required |
| `policy_version` | str | required |
| `precision_backend` | str | required; resolved name |
| `seed` | int | |
| `steps` | list[StepRecord] | rehydrated on read |
| `terminal_success` | bool | |
| `terminal_metrics` | dict[str, float] | |
| `trajectory_length` | int | derived on write |

`StepRecord`: `step`, `action`, `observation_meta`, `events`, `reward`.
`events` are **annotations** produced by observers; they never influenced the
action stored alongside them.

## Snapshot / root record — `SnapshotRecord`

| Field | Notes |
|---|---|
| `root_id` | required; `root::<episode>::s<step>::<policy_version>` |
| `episode_id`, `scene_id` | required |
| `snapshot_step` | trajectory step captured |
| `snapshot_path` | on-disk state |
| `policy_version_that_generated_it` | required |
| `candidate_reason` | required; `fixed_interval:30`, `semantic_event:object_held`, `manual_debug`, `refine:<step>` |
| `source_rollout_id` | rollout it came from |

`candidate_reason` is what keeps a coarse pass and a refinement pass separable
after the fact.

## Probe record — `ProbeRecord`

One stochastic continuation from a restored root.

| Field | Notes |
|---|---|
| `root_id`, `probe_id` | required; `probe_id` = `<root>::<mode>::<idx>::seed<seed>` |
| `precision_backend` | required |
| `seed` | explicit per continuation |
| `policy_version` | required |
| `mode` | `probe` (discovery) or `explore` (scaling); validated |
| `terminal_success`, `terminal_metrics` | |
| `trajectory_length`, `trajectory_path` | |

## Frontier summary — `FrontierSummary`

| Field | Notes |
|---|---|
| `root_id` | required |
| `n_probes`, `n_success` | `n_success > n_probes` is rejected |
| `success_rate` | `V_hat(root)` |
| `uncertainty` | half-width of the Wilson interval |
| `ci_low`, `ci_high` | Wilson score interval |
| `precision_backend`, `policy_version` | required; probes are never pooled across either |
| `selected_as_frontier` | set by the selector |
| `selection_reason` | `frontier`, `too_hard`, `solved`, `no_outcome_variance`, `too_uncertain`, `insufficient_probes`, `over_budget` |

## Cheap exploration record — `CheapExplorationRecord`

| Field | Notes |
|---|---|
| `root_id`, `group_id`, `rollout_id` | all required |
| `precision_backend` | a cheap backend |
| `seed`, `policy_version` | |
| `cheap_score` | continuous ranking signal |
| `cheap_outcome` | cheap binary prediction |

## Verification record — `VerificationRecord`

Links a cheap candidate to its independent high-fidelity re-run.

| Field | Notes |
|---|---|
| `source_root` | required |
| `cheap_rollout_id`, `cheap_score`, `cheap_outcome` | the cheap side |
| `selection_rank` | rank under the cheap ordering |
| `verification_rollout_id` | required; **must differ** from `cheap_rollout_id` |
| `verification_backend` | required; must be high fidelity |
| `verified_success`, `verified_metrics` | the BF16 side |
| `policy_version` | required |

The two sides are separate fields by construction. Rewards read
`verified_success` only.

## RL training records

`TrainingSampleRecord`: `root_id`, `group_id`, `rollout_id`, `policy_version`,
`precision_backend`, `reward`, `advantage`, `trajectory_path`. `root_id` and
`group_id` are required and rejected when blank.

`TrainingStepRecord`: `policy_before`, `policy_after`, `optimizer_step`,
`root_ids`, `group_ids`, `mean_reward`, `mean_advantage`, `loss`, `approx_kl`.

Checkpoint lineage is a separate ledger (`lmm/rl/checkpoint_manager.py`,
`lmm_checkpoint_ledger_v1`): `policy_version`, `path`, `parent_version`,
`optimizer_step`, `created_utc`, `sha256`. Registering a child before its
parent, or reusing a version, is rejected.

## Experiment manifest

`lmm/manifest.py`, written as `resolved_manifest.json` into every output
directory before work begins. Fields: `experiment_name`, `timestamp`,
`git_commit`, `git_dirty`, `policy_checkpoint`, `parent_checkpoint`,
`environment`, `task`, `scene_ids`, `episode_ids`, `precision_backend`,
`rollout_mode`, `candidate_strategy`, `k_probe`, `k_cheap`, `k_verify`,
`rl_algorithm`, `learning_rate`, `update_epochs`, `number_of_roots`,
`random_seeds`, `output_directory`, `wandb_run`, `notes`.

`git_commit` and `policy_checkpoint` are mandatory: a run whose code state or
weights cannot be identified is not a scientific record.

## Reading records

```python
from lmm.schemas import read_jsonl, write_jsonl

records = list(read_jsonl("runs/x/full_rollouts.jsonl"))
write_jsonl("runs/x/out.jsonl", records)
```

Group by record fields, never by path:

```python
from collections import defaultdict
by_root = defaultdict(list)
for p in read_jsonl("runs/x/probes.jsonl"):
    by_root[p.root_id].append(p)
```
