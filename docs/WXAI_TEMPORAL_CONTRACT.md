# WXAI Temporal Contract

Three distinct rates. Conflating any two of them is how a dataset silently
ends up describing a different robot than the one that will execute it.

| Quantity | Value | Status |
|---|---:|---|
| Simulator physics / control frequency | **120 Hz** | measured |
| Recorded dataset FPS | **15 Hz** | chosen |
| Simulator ticks per recorded action | **8** | derived (120 / 15) |

## 1. Simulator control frequency — 120 Hz, measured

`ctrl_freq = 120.0`, `ac_freq_ratio = 4`. One `env.step` advances exactly
1/120 s of simulated time, confirmed independently:

* yaw: 0.0041667 rad/step = `ang_speed / 120`
* forward: 2.0833 mm/step = `lin_speed / 120`, at six headings in free space

This is a property of the runtime. It is not a choice and must not be changed
to make a dataset convenient.

## 2. Recorded dataset FPS — 15 Hz, chosen

**Recording at 120 FPS would be wrong**, and the reasoning is worth keeping:

* OVMM is long-horizon and mobile. At 0.25 m/s a 10 m traverse takes 40 s =
  4,800 ticks. At 120 FPS that is 4,800 frames of head + two wrist RGB streams
  for the navigation portion of a single episode.
* At `action_horizon = 50`, a 120 FPS policy sees 0.42 s of lookahead — far too
  short to express a long-horizon intent.
* The base advances 2.08 mm per tick, well below any OVMM success tolerance, so
  most consecutive frames would be near-duplicates carrying no new supervision.

15 Hz was chosen because:

* it gives 3.3 s of lookahead at `action_horizon = 50`, which spans a
  reach or a place rather than a fragment of one;
* 16.7 mm of base motion per recorded action is fine-grained relative to OVMM
  tolerances while keeping episode length tractable;
* it matches DROID, a mobile-manipulation dataset already in pi0's
  pre-training mixture, so the temporal statistics are not alien to the model;
* 120 / 15 = 8 exactly, so no tick is dropped or double-counted.

This is a decision, not a measurement. It should be revisited if manipulation
turns out to need finer resolution than 66.7 ms — in which case 30 Hz
(decimation 4) is the natural next step, and it divides exactly too.

## 3. Ticks per recorded action — 8

One recorded 16-D action is applied **unchanged** for all 8 ticks:

```
recorded action a[16]
    |
    +-- tick 1 .. tick 8   env.step({"action": ("arm_action","base_velocity"), ...})
    |                      identical arm targets, gripper, base velocity
    v
one recorded frame: state14 + action16 + 3 RGB + instruction + metadata
```

Symmetry: training and inference both apply one action for 8 ticks. There is no
oracle-side interpolation that a policy would be unable to reproduce.

Because the arm is kinematically driven (`ArmAbsPosKinematicAction`), repeating
an absolute joint target across the 8 ticks is idempotent — the arm reaches the
target on the first tick and holds. The base integrates all 8 ticks, so one
recorded action produces `8 x lin_speed / 120 = 16.7 mm` of travel at full
forward command.

## 4. Atomicity

Arms, grippers and base for one recorded action belong to the same transition.
Habitat's `embodied_task` accepts a tuple of action names, so both are applied
within a single `env.step`; the composite is then repeated 8 times. The earlier
implementation spent two separate environment transitions per policy action and
has been removed.

## 5. Recorded per frame

`dataset_step`, `sim_tick`, `sim_time_s` (= `sim_tick / 120`),
`dataset_time_s` (= `dataset_step / 15`), and `ticks_this_action` (= 8).

Both clocks are stored explicitly so that a later change to the decimation is
detectable in the data rather than inferred from a config file.
