# Next Experiment

**One experiment only.** Do not launch anything larger until it has run and
been read.

## Recommended: End-to-End BF16 Sanity Check

**Question.** Does the fine-tuned BF16 pi0.5 do anything coherent when it
executes the long-horizon task end-to-end, with the semantic skill sequencer
disabled?

This is the smallest experiment that validates the new architecture. Every
downstream stage — frontier probes, cheap scaling, verification, online RL —
assumes an end-to-end policy path exists. **That assumption has never been
tested.** Every historical result was produced with the semantic sequencer
driving NAV/MANIP mode, or from a restored pre-place root. There is no evidence
on file about what the policy does from an episode initial state on its own.

### Setup

- Disable the semantic skill sequencer entirely.
- Keep only the simulator adapter (action scaling, SNAP/DESNAP, stepping).
- Attach `EventDetector` as a passive observer for logging.
- Run the current fine-tuned BF16 pi0.5 from normal task initialization.

### Scale

**Small.** 4-8 episodes, 1-2 seeds each. 8-16 rollouts total. Enough to see
behaviour, not enough to conclude anything statistical.

### Command

```bash
python scripts/rollout_full_task.py \
  --experiment-name e2e_bf16_sanity_v1 \
  --output-dir runs/e2e_bf16_sanity_v1 \
  --policy-checkpoint <BF16_CHECKPOINT> \
  --policy-version <VERSION> \
  --env-config configs/env/behavior_ovmm_v1.json \
  --rollout-config configs/rollout/full_task_v1.json \
  --server-url http://127.0.0.1:8080
```

Validate the plan first on any machine:

```bash
python scripts/rollout_full_task.py ... --dry-run
```

### What to measure

Report each **from the observed trajectories**, not from assumption:

1. Does the base navigate toward the object?
2. Does the policy initiate reaching?
3. Does it attempt / achieve grasp?
4. After grasp, does it move toward the receptacle?
5. Does it attempt placement / release?
6. Does it ever complete the full task?
7. **Where do failures empirically occur?** (step index and last event fired)

`EventDetector` annotates each step, so answering 1-6 is a query over
`full_rollouts.jsonl`, and 7 falls out of the last event before termination.

### Do not

- Infer conclusions before watching the actual trajectories.
- Re-enable the sequencer "just to get a number".
- Scale up if the first batch looks bad — a bad result here is the most
  valuable information available, and it determines whether the end-to-end
  setup is viable at all.

## Execution host status

The workstation still cannot run this experiment, but the restored A800 host
can supply the simulator and GPU resources. See [`A800_SERVER.md`](A800_SERVER.md).

| Requirement | This machine |
|---|---|
| Habitat / HomeRobot | not installed |
| JAX + OpenPI | not installed |
| GPU for pi0.5 BF16 | RTX 3070 Laptop, 8 GB |
| Free disk | **4.9 GB of 916 GB** |
| A800 cluster | **restored: 4× A800 80GB; GPU1-3 idle at 2026-09-20 check** |

Available on the restored host: Habitat OVMM, HomeRobot/HSSD assets, OpenPI,
the Python 3.9/3.11 environments and three currently idle A800s. The dual-arm
simulator contract has passed `reset -> render -> step`.

Still blocking the experiment: `BehaviorEnv.make()` has not yet been bound to
the real Trossen Habitat backend, and there is no validated dual-arm OVMM
pi0.5 checkpoint. Recovered q20 is a Stretch 10D-state/10D-action checkpoint
and cannot be used as a 14D-state/16D-action Trossen policy.

The q20 BF16 model is recoverable from Hugging Face. Exact q0/q5/q10 are not —
see `LEGACY_EXPERIMENTS.md` §J.

## After it passes: milestone 2 — frontier probe smoke test

Only once the end-to-end path works.

Take one valid full trajectory. Save 4-8 candidate snapshots. Run `K_probe` =
2-4 short continuations each, using `configs/frontier/frontier_smoke_v1.json`.

```bash
python scripts/discover_frontier.py \
  --experiment-name frontier_probe_smoke_v1 \
  --output-dir runs/frontier_probe_smoke_v1 \
  --policy-checkpoint <CKPT> --policy-version <VERSION> \
  --rollouts runs/e2e_bf16_sanity_v1/full_rollouts.jsonl \
  --frontier-config configs/frontier/frontier_smoke_v1.json
```

Verify: snapshot restore is deterministic enough; stochastic continuations
differ when intended; outcome records are correct; `V_hat` is computable; the
whole thing works without the semantic controller.

This is a pipeline smoke test, not a paper experiment.

## Then: milestone 3 — cheap vs BF16 probe consistency

Very small state bank. Compare cheap and BF16 continuation estimates on
ranking, success-rate correlation, top-candidate retention and gross failure
modes, via `scripts/eval_fidelity.py`.

Note the open question this must settle honestly: the historical V3 ranking
result (Spearman 0.9107) was computed over a **proxy** metric with zero
successes on either side. It is not evidence about success ordering. Q3 is
open.

## Only then: prepare online RL

Confirm all of the following before any new RL run:

- [ ] end-to-end policy path works
- [ ] snapshot restore works
- [ ] frontier probes work
- [ ] cheap rollout lineage is correct
- [ ] BF16 verification works
- [ ] training records preserve root/group identity
- [ ] held-out and full-task evaluation scripts work

Then write the exact proposed experiment into this file and **do not launch it
unless explicitly instructed**.

Note for whoever plans that run: 20 rounds of online GRPO from semantic roots
already returned `NO_RETAINED_IMPROVEMENT` over ~47 hours. Repeating it with a
different root-selection rule is the point; repeating it unchanged is not.
