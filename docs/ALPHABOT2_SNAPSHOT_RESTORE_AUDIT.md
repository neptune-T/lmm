# AlphaBot 2 snapshot / restore audit (Isaac Sim)

**Verdict: `USABLE_WITH_TOLERANCE`.** An **exact-branching protocol exists**
(canonical restore, below). It makes every branch from a snapshot bitwise
identical to every other branch. No branch reproduces an *uninterrupted* run
exactly.

Measured live 2026-09-28 on the public third-party Ai2 Bot2 asset (Isaac Sim
4.5, CPU PhysX, 200 Hz, 10 Hz actions), in the hardest case the smoke has:
the left gripper **holding a cube in contact** while the arm lifts and the
**base drives** backwards and turns. Evidence:
`artifacts/alphabot2_live_smoke.json` → `grasp_snapshot_restore`; code:
`IsaacArticulationBackend.snapshot()/restore()`.

## What a snapshot captures

| State | Captured | How |
|---|:--:|---|
| Base pose (articulation root) | ✅ | `get_world_pose` → `set_world_pose` |
| Base linear / angular velocity | ✅ | root velocities |
| All 42 DOF positions (arms, grippers + mimics, torso, head, wheels, casters) | ✅ | `get_joint_positions` |
| All 42 DOF velocities | ✅ | `get_joint_velocities` |
| Controller state: position targets of the 22 driven DOFs | ✅ | backend target buffer, re-applied |
| Controller state: wheel velocity targets | ✅ | backend buffer, re-applied |
| Object poses and linear / angular velocities | ✅ | registered rigid objects |
| Backend step counter | ✅ | restored |
| Sim clock | recorded only | Isaac's sim time cannot be rewound; nothing in the controller reads it |
| PhysX solver warm-start, contact manifolds, friction anchors | ❌ | not exposed by the API |

Torso and head are ordinary DOFs here (held by position drives), so they are
covered by the DOF rows.

## Results (grasp in contact + base motion, 25 actions = 2.5 s after the snapshot)

The snapshot was taken mid-lift. At that point the cube had risen 3.1 cm and 7
gripper bodies were in contact with it. Over the next 25 actions the arm
finished the lift (cube +8.2 cm in total) and the base travelled 0.146 m while
still holding the cube.

| Comparison | actuated arm+gripper | gripper mimic | casters | wheels | chassis | cube |
|---|---:|---:|---:|---:|---:|---:|
| Restore read-back vs snapshot | 0.0 | — | — | — | — | 0.0 |
| In-place restore: replay vs replay | **0.0** | 0.0 | 0.0 | 0.0 | 0.0 | **0.0** |
| Canonical restore: branch vs branch | **0.0** | 0.0 | 0.0 | 0.0 | 0.0 | **0.0** |
| In-place restore vs uninterrupted original | 0.094 rad | 0.034 rad | 0.092 rad | 0.0013 rad | 0.65 mm | **23.2 mm** |
| Canonical restore vs uninterrupted original | 0.094 rad | 0.034 rad | 0.031 rad | 0.0020 rad | 0.43 mm | **22.5 mm** |

(max over the 25-action trajectory; 0.0 means bitwise identical.) Every branch
still ends with the cube held and lifted 8.2 cm.

## Interpretation

* **The captured state is complete for the physics we control.** Read-back is
  exact, and two branches from the same snapshot are bitwise identical.
* **The divergence from the uninterrupted run is solver state, not missing
  physical state.** The original run had warm-start impulses and contact
  manifolds from the step before the snapshot. A restored branch starts from
  whatever caches the engine currently holds. Contact-rich grasping amplifies
  that difference: 2.3 cm on the held cube and ~0.09 rad on a gripper-coupled
  DOF within 2.5 s. Base pose stays within 0.65 mm.
* The earlier 60 Hz run without grasp contact diverged less (0.028 rad,
  0.8 mm), as expected when there are fewer contacts to amplify.

## Canonical-restore protocol (exact branching)

```python
world.reset()      # rebuilds the PhysX scene: discards warm-start + contact caches
backend.refresh()  # re-acquire the physics view (reset invalidates it)
backend.restore(snap)
```

Measured cost: **2.8 s** per restore. Every branch then starts from identical
solver state and is bitwise identical. To make the *original* rollout part of
the same equivalence class, apply the same canonical restore at the moment the
snapshot is taken. The original is then one of the identical branches, at the
price of one cache flush on the main rollout.

## Consequence for the method

For continuation probes, a restored branch is a valid physical continuation of
the captured state. Branches are reproducible, so any randomness a prober wants
must be injected explicitly. What cannot be claimed is "restoring a snapshot
re-plays the recorded future of the original rollout". Near contact that
future is only reproduced within the tolerances above.

## Not yet covered

* Deformable or articulated scene objects (none in the smoke).
* Snapshots across a scene rebuild or process restart. Everything here stayed
  in one Isaac process.
* GPU PhysX (the smoke uses CPU dynamics, as declared by the asset's scene).
