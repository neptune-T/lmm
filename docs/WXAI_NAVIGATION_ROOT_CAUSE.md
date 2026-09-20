# WXAI navigation: root cause of the 11,817 collisions

Measured on the A800, episode 743, 2026-09-21. Diagnosis only — no fix applied.

## Root cause

**The robot is lying on its side. The URDF is ROS z-up and the articulation is
loaded into Habitat's y-up world without the coordinate correction ever being
applied to the physical body.**

URDF evidence (`mobile_ai.urdf`):

```
base_footprint_joint  xyz="0 0 0.211"                 <- height in z
cam_high_joint        xyz="0.209805 0 1.031778"       <- camera 1.03 m up, in z
left_wheel            xyz="0  0.192 -0.1259" axis="0 1 0"
right_wheel           xyz="0 -0.192 -0.1259" axis="0 1 0"   <- wheels split in y
```

So URDF +z = up, +y = lateral, +x = forward. Standard ROS.

Measured world positions with the robot at a random *navigable* free-space
point (base at y = 0.087):

| Link | world y | expected if upright |
|---|---:|---|
| `cam_high_link` | **0.096** | ~1.4 |
| `left_wheel_link` | **0.279** | ~0.0 |
| `follower_left_link_6` | **+0.417** | same as right |
| `follower_right_link_6` | **−0.208** | same as left |
| EE left / right (above base) | **+0.357 / −0.288** | equal, positive |

The right end-effector is *underground*. The head camera is at floor level. The
two wheels are at different heights. Left and right arms are displaced
symmetrically up and down about the base — the exact signature of the lateral
(±y) axis having been mapped to world vertical.

`TrossenMobileAIRobot.base_transformation` does post-multiply
`rotation(-pi/2, x)` with the comment "Trossen's URDF is ROS z-up; Habitat is
y-up". But that property only changes the *reported* transform used for pose
and heading maths. It never rotates the loaded articulation. Consistently:

```
sim_obj_up_axis_world  = [0, 1, 0]        <- object frame believes it is upright
base_tf_up_axis_world  = [0.744, 0, 0.668] <- reported base frame is horizontal
```

## Why this produced exactly the observed symptoms

* **`base_link` penetrates everywhere, including open free space.** Contact
  points sit at world y = −0.30, −0.19, −0.03 with penetration depths to
  **−0.31 m**: the base collision geometry is buried in the floor at every
  pose, navigable or not. Hence a non-zero collision count independent of
  location, and hence 11,817 accumulated over a long episode.
* **Only the *left* arm collides with the scene.** The right arm is below the
  floor where there is nothing to hit; the left arm is swept up to head height
  where it sweeps through walls and furniture.
* **"Stuck after exactly 22.4 mm."** Buried geometry catches immediately;
  the residual 22.4 mm is the navmesh snap offset, not real travel.

## Colliding links

At the navmesh-snapped start: 6 penetrating contacts, **100 % `base_link`**.

Along the planned path (28 ghost poses, all navigable, **28/28 colliding**,
max 76 penetrations): `base_link` at every single pose, plus
`follower_left_link_1..6`, `follower_left_base_link`,
`follower_left_gripper_left/right`, `follower_left_carriage_left/right`,
`follower_left_camera_mount_d405`, `cam_high_link`. No right-arm link ever
appears.

## What is NOT the cause

* **Not the navmesh agent radius.** `base_link` penetrates in open free space
  at a navigable point. No radius fixes a buried body.
* **Not the arm safe pose.** Current safe pose and a tucked pose both give
  exactly **6** penetrating contacts at the snapped start, all `base_link`.
  The pose is not what makes navigation invalid.
* **Not the navmesh path.** A geodesic path exists (7.42 m, 28 samples, every
  sample navigable). The path is fine; the body occupying it is not.

## Footprint measurement — withdrawn

The AABB figures produced during this pass are not trustworthy and are not
reported as fact: `cumulative_bb` on a link scene node includes the whole
child subtree, so "base only" silently measured the entire robot, and the
arms-only result was internally inconsistent (0.52 x 0.41 m extents with a
1.51 m circumscribed radius). Measuring a footprint on a robot lying on its
side is meaningless regardless. Footprint must be re-measured *after* the
orientation is fixed.

## Not attempted, and why

The radius sweep (0.30–0.50 m) was deliberately not run. Every configuration
would report the same base penetration, because the defect is orientation, not
clearance — the sweep would produce five equally meaningless rows and could
easily be misread as evidence that some radius "works". Re-run it once the
robot stands up, when the numbers will mean something.

## Fix direction

Rotate the loaded articulation, not the reported transform: apply the z-up →
y-up correction to `sim_obj` at load in `reconfigure()`, then re-derive
`base_transformation` accordingly so pose, heading and the verified
forward/yaw signs stay consistent. The full calibration (joint indexing,
gripper, 120 Hz, atomic action) must be re-verified afterwards, since every
world-frame quantity changes.
