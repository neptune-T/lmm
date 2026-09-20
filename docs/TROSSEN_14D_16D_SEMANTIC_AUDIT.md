# Trossen Mobile 14D / 16D semantic audit

Audit date: 2026-09-20. OpenPI source revision:
`15a9616a00943ada6c20a0f158e3adb39df2ccac`.

## Verdict

The OpenPI `trossen_mobile` policy contract is fully identified, but the
installed Habitat robot is not an exact actuator-level match for that training
embodiment. OpenPI's asset is Mobile ALOHA: two ViperX arms on a holonomic
Slate base. Habitat currently loads Trossen Mobile AI: two WXAI arms on a
differential-drive base. Consequently policy inference may be connected and
audited, and each Habitat actuator may be smoke-tested, but raw policy actions
must not yet be executed as a full 16D command.

The previous `forward/yaw` interpretation of action dimensions 14 and 15 was
wrong. OpenPI defines them as x/y base velocity. The canonical adapter now
fails closed instead of sending lateral y velocity to a yaw actuator.

## Dimension-level contract

State and action dimensions 0–5 are left arm joint angles in radians. State
and action dimension 6 is left gripper closed fraction (`0=open`, `1=closed`).
Dimensions 7–12 are right arm joint angles in radians. Dimension 13 is right
gripper closed fraction. State ends at dimension 13. Action dimensions 14 and
15 are base x and y velocity, respectively.

Left and right are defined looking from behind the robot toward the workspace.
Trossen control frequency is 50 Hz. The state contains no base pose or base
velocity.

## OpenPI transform audit

`AlohaInputs(adapt_to_pi=True)` converts standard ALOHA arm/gripper values to
the internal training convention. Its joint sign mask is
`[+, -, -, +, +, +, +, +, -, -, +, +, +, +]`; grippers are converted from
the public linear closed-fraction convention to the internal angular
convention.

`LeRobotAlohaDataConfig(use_delta_joint_actions=True)` uses mask
`make_bool_mask(6, -1, 6, -1)`. Therefore only the 12 arm joints are modeled
as deltas. Both grippers remain absolute. Dimensions 14–15 are outside the
14D delta mask and remain base velocities. At inference, `AbsoluteActions`
restores the arm outputs to absolute joint targets before the final public
space conversion.

The upstream `AlohaOutputs` slices `actions[:, :14]`. Using it for Mobile
ALOHA silently discards the base channels. `scripts/serve_trossen_mobile_pi05.py`
replaces only that final output transform, applies standard ALOHA conversion
to dimensions 0–13, and preserves dimensions 14–15 unchanged.

The server loads the checkpoint's `assets/trossen_mobile/norm_stats.json`,
requires 14D state and 16D action statistics, uses quantile normalization as
required by pi0.5, and rejects any non-finite or incorrectly shaped output.

## Checkpoint dataset evidence

The checkpoint normalization statistics are computed from OpenPI's pretraining
data and are part of the dataset-to-policy contract. They contain 14 values for
state and 16 for action. The observed action quantiles for gripper dimensions
span approximately 0 to 1, confirming closed-fraction semantics. The final two
action dimensions have signed distributions (q01/q99 approximately
`-0.219/0.398` and `-0.314/0.252`), consistent with bidirectional planar base
velocity rather than pose.

No public source inspected specifies a different unit or frame convention for
the two base channels beyond “x-y base velocity”. The adapter therefore does
not invent a yaw mapping, scale, or coordinate rotation.

## Habitat actuator audit

The installed `TrossenMobileAIRobot` resolves 12 arm links, four gripper
carriage links, two end-effectors, and two wheel links by URDF name. The 12 arm
values drive `ArmAbsPosKinematicAction`, so their Habitat unit is absolute
radians. `pi05_state()` returns
`[left6, left_closed_fraction, right6, right_closed_fraction]`.

`BimanualArmAction` accepts independent `[-1,1]` grip inputs. The canonical
bridge maps OpenPI closed fraction `g` to `2g-1`, preserving `0=open` and
`1=closed`. Habitat's current base action accepts normalized
`[forward, yaw]`, scaled by the overlay to 0.25 m/s and 0.5 rad/s. It is not an
x/y actuator.

The exact-match blockers are:

1. Mobile ALOHA/ViperX joint zero and mechanical convention has not been shown
   equivalent to the newer WXAI URDF.
2. A holonomic Slate x/y command cannot be represented by the differential
   drive interface in one 50 Hz command.
3. Habitat currently exposes arm and base as separate task actions, so a full
   simultaneous mobile-manipulation command is not atomic.

Until those three items are resolved, policy actions are inference-audited but
not actuated. This is an intentional safety/result-integrity gate, not a
planner or task heuristic.

## Allowed smoke scope

The smoke runner performs one policy inference per episode, verifies the 16D
result, then independently exercises small absolute arm motions, both gripper
conventions, and low-amplitude forward differential-base motion. It records
head and both wrist cameras. It runs one or two episodes only. It contains no
frontier probing, continuation search, policy optimization, or RL.

