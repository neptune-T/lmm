# Current Method — Online RL with Rollout Scaling for Long-Horizon Mobile Manipulation

Canonical as of 2026-09-19. This document supersedes older design assumptions in
`docs/`. Where an older protocol conflicts with this file, this file wins and
the older one is history.

## 1. The claim

Long-horizon VLA post-training is a **rollout-allocation problem**. Full
trajectories spend most of their compute on states that are already solved or
that provide no usable learning signal. Instead:

1. execute the VLA end-to-end;
2. identify states near its empirical competence frontier using branched
   continuation probes;
3. scale exploration from those states using cheap low-precision rollouts;
4. verify promising continuations at high fidelity;
5. update the same end-to-end VLA through online RL.

## 2. The loop

```
End-to-end VLA rollout
        |
candidate state snapshots
        |
cheap continuation probes
        |
estimate empirical continuation success
        |
discover competence frontier
        |
large-scale cheap branching
        |
high-fidelity BF16 verification
        |
online RL update
        |
full-task evaluation
```

Compactly: **Roll out -> Probe -> Locate Frontier -> Scale -> Verify -> Update
-> Evaluate.**

## 3. The policy is end-to-end

```
RGB / proprioception / language / policy history
                    |
                  pi0.5
                    v
           base + arm + gripper actions
```

The VLA executes the long-horizon task itself. No handcrafted semantic state
machine decides when navigation ends, when reaching begins, when to switch to
pick, when to start navigating to the receptacle, or when to enter place.

The previous OVMM-style sequencer

```
NAV_TO_OBJ -> GAZE -> PICK -> NAV_TO_REC -> GAZE_AT_REC -> PLACE -> FALL_WAIT
```

**is no longer the canonical policy architecture.** It is preserved as
historical infrastructure; see [`LEGACY_EXPERIMENTS.md`](LEGACY_EXPERIMENTS.md).

### The one permitted exception: simulator plumbing

A thin adapter is required and allowed:

```
continuous gripper command -> Habitat / BEHAVIOR SNAP or DESNAP API
```

`lmm/env/simulator_adapter.py` may do action scaling, coordinate transforms,
robot control interfaces, simulator stepping, snapshot save/restore, gripper
event translation and collision bookkeeping. It **must not** do task planning.
Anything that would change *which behaviour the robot attempts* — as opposed to
*how a command is transmitted* — does not belong there.

## 4. Semantic events are observers, not controllers

```
event_detector == observation / instrumentation only
event_detector != policy controller
```

`lmm/instrumentation/event_detector.py` detects object approached, grasp
attempted, object held, object released, receptacle approached, object on goal,
object at rest, task success and task failure. These are used for logging,
debugging, proposing candidate snapshots and trajectory analysis.

They must not determine the VLA's actions or force state transitions. This is
enforced by `tests/test_event_detector_noninterference.py`, which
asserts the detector exposes no action-producing API, that attaching it leaves
the emitted action sequence bit-identical, and that its executable code
references no controller symbols.

## 5. Frontier is empirical, not a skill boundary

Old concepts — `E5_PRE_PLACE_MINUS_ONE`, `PRE_PLACE`,
`first_official_place_step` — remain accessible for historical experiments but
are **not** the canonical definition.

> A frontier state is a trajectory state whose continuation success under the
> current policy lies near the policy's empirical competence boundary.

For candidate state `s`:

```
V_hat(s) = successes among branched stochastic continuations
           / total continuation probes

V_hat(s) ~= P_pi(task success | s)
```

- `V_hat(s) ~ 0` — usually too difficult
- `V_hat(s) ~ 1` — already solved
- intermediate — useful learning signal

Starting band: `frontier_low = 0.2`, `frontier_high = 0.8`. **These live in
`configs/frontier/*.json`, not in code.** They are a starting hypothesis, not a
result.

A frontier is never inferred from a single failed trajectory. The estimator
refuses fewer than `min_probes` continuations, refuses to pool probes across
precision backends or policy versions, and reports a Wilson interval so a 0/4
estimate cannot masquerade as certainty.

### Why this change was necessary

The frozen TRAIN K=8 pilot branched 8 hand-picked semantic pre-place roots x 8
siblings. All 64 terminated `released_outside`: 0 successes, 0 mixed roots,
terminal verdict `BRANCH_SUPPORT_NOT_OBSERVED`. A root bank with zero outcome
variance carries no contrastive signal, so group-relative RL had nothing to
learn from. Choosing roots by semantic position cannot detect this in advance;
estimating `V_hat` can, and refuses to proceed. `lmm/rl/advantages.py` reports
that condition explicitly rather than silently emitting zero advantages.

## 6. Cheap rollout has two roles, one implementation

**A. Frontier discovery** — cheap continuations estimate `V_hat(s)` across
candidate states (`K_probe`, small: 4-8).

**B. Frontier exploration** — once a frontier state is selected, draw many
cheap continuations from it (`K_cheap`, large).

Both go through `ContinuationProbeRunner`, differing only in `K` and in the
`mode` tag (`probe` / `explore`) written to each record. There is deliberately
no second implementation.

## 7. High-fidelity verification

Cheap rollout is **not** ground truth. Candidates selected under cheap
precision are re-run with BF16 pi0.5. Records keep `cheap_prediction` and
`BF16_verification` in separate fields; `VerificationRecord` rejects a record
whose cheap and verification rollout ids are equal, so a cheap run can never be
relabelled as its own verification. Rewards are computed from the verified
outcome only.

Lineage preserved per verified candidate: `source_root`, `cheap_rollout_id`,
`cheap_score`, `selection_rank`, `BF16_rollout_id`, `BF16_success`,
`policy_checkpoint`.

## 8. Root-wise grouping stays explicit

Siblings from the same restored root share an explicit `group_id`. Every
training sample stores `root_id`, `group_id`, `rollout_id`, `policy_version`,
`reward` and `advantage`. Grouping is never inferred from filenames later.
Siblings from unrelated roots never share a group. Cross-root aggregation
happens at the optimizer batch level, after per-root normalisation, and the
per-group statistics are retained so that step stays auditable.

## 9. Evaluation is three separate things

| Evaluation | Module | Question |
|---|---|---|
| Frontier continuation success | `lmm/eval/frontier_eval.py` | Did continuations from training roots improve? |
| Held-out frontier success | same, `split="held_out"` | Does it transfer to unseen roots? |
| **Full-task from episode initial state** | `lmm/eval/full_task_eval.py` | Did the actual task get solved? |

Because the method is end-to-end, full-task evaluation is **mandatory** and
cannot be replaced by "success from pre-place root". `full_task_eval.evaluate`
rejects any rollout whose metadata shows a skill sequencer was active.

## 10. Current scientific questions

The project should currently answer only these.

**Q1 — Frontier localization.** Can branched continuation probes identify
states with useful stochastic outcome diversity in an end-to-end long-horizon
VLA trajectory?

**Q2 — Rollout scaling.** Under a fixed or comparable compute budget, do cheap
scaled continuations discover more successful or promising trajectories than
high-fidelity-only collection?

**Q3 — Fidelity.** Does cheap rollout preserve enough candidate ordering /
selection quality that BF16 verification recovers genuinely useful
trajectories?

**Q4 — Online learning.** Can verified successful continuations be converted
into a measurable improvement in the same VLA?

**Q5 — Generalization.** Does improvement transfer to held-out frontier roots
and/or full-task initial-state evaluation?

Do not add algorithmic components unless they are necessary to answer these.

## 11. Explicitly out of scope right now

These are research hypotheses, not frozen method components. Where old code
implements them, it stays under legacy.

- adaptive bidirectional curriculum
- moving frontier curriculum
- continual-learning replay system
- landing-zone regularization
- adversarial skill chaining
- automatic semantic task decomposition
- SFT consolidation
- complex root archive replacement
- full multi-stage reverse curriculum

## 12. Naming

Prefer: `full_task_rollout`, `candidate_state`, `root` / `snapshot`,
`continuation_probe`, `frontier`, `cheap_rollout`, `BF16_verification`,
`verified_group`, `online_update`, `full_task_eval`.

Avoid, unless explicitly mapped in metadata: `q0`, `q1`, `q2`, `q3`, `Q4`,
`champion`, `mixed`, `normal`, `high`, `cheap`. The registry in
`lmm/rollout/precision_backend.py` holds those mappings; historical checkpoints
keep their original names and canonical code exposes explicit aliases.

Scripts and output directories are named for **what they test**, never for the
hoped-for conclusion. `ExperimentManifest.validate()` rejects experiment names
containing words like `successful`, `optimal`, `stable` or `proven`.
