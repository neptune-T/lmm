# AlphaBot 2 observation timing contract (Stage 1)

2026-09-29, Isaac Sim 4.5 (A800), 200 Hz physics, 10 Hz policy (20 physics
steps per action, render on the last substep). This contract replaces the
bootstrap's implicit "images lag one action, read them after the action"
caveat ([`ALPHABOT2_PI05_BOOTSTRAP.md`](ALPHABOT2_PI05_BOOTSTRAP.md) §7).

## 1. The contract

For every policy step `t`:

| Quantity | Definition |
|---|---|
| `policy_step` | index `t` of the action this observation decides (0 = first action after reset) |
| `obs_frame` | physics frame (200 Hz steps since reset) at which `observe()` is called |
| `image_frame` | physics frame of the render the RGB images come from |
| `proprio_frame` | physics frame of the state the 16-D proprio was read at |
| `action_frame` | physics frame at which action `t` starts to be applied |
| `*_time` | the same instants in simulation seconds (`frame / 200`) |

**Invariant (canonical, zero-delay app):**
`image_frame == proprio_frame == obs_frame == action_frame`, and consecutive
policy steps are exactly 20 frames apart. `RGB_t` and `proprio_t` show the same
physical state, and that state is the one the action is applied to. There is
**no observation latency**.

The one implementation of this lives in
[`scripts/alphabot2_isaac/stage1/simworld.py`](../scripts/alphabot2_isaac/stage1/simworld.py)
(`SimWorld.observe()` / `SimWorld.act()`). Demonstration recording
(`collect.py`), replay validation, SFT dataset generation (`stage1_to_lerobot.py`
reads only the recorded pairs) and closed-loop inference/evaluation
(`evaluate.py`) all go through it; none of them reads a camera or the proprio
directly.

## 2. How the zero-delay pairing is achieved

Isaac's default python app delivers the **previous** render from the camera
annotators (frames in flight). The app is therefore launched with Isaac's own
`apps/isaacsim.exp.base.zero_delay.kit` (`app.hydraEngine.waitIdle = true`,
`app.updateOrder.checkForHydraRenderComplete = 1000`), selected by
`simworld.launch()` (`CANONICAL_ZERO_DELAY = True`).

`SimWorld` does not *assume* the lag; it carries a render ring buffer. Every
render is logged with its physics frame and the proprio read at that frame.
`observe()` returns the images tagged with the frame of the render they come
from (`RENDER_LAG` renders back), paired with the proprio buffered **at that
frame**, never with a newer proprio. With the default app (`RENDER_LAG = 1`)
the same code yields synchronized pairs 20 frames (0.1 s) old; with the
zero-delay app (`RENDER_LAG = 0`) the pairs are current.

## 3. Measurements (`scripts/alphabot2_isaac/stage1/timing_probe.py`)

**Frame-code test (object stimulus).** A magenta block is teleported before
*every* physics substep to a grid cell encoding the physics frame (mod 40). The
stepping is exactly the production stepping. Decoding the block in the head
image gives the frame the image actually shows. 40/40 codes are visible, at a
minimum centroid separation of 12.6–14.0 px.

| App | Image lag, frames (58 reads) | Modal lag, actions |
|---|---|---|
| default | 20 ×58 (earlier runs: 20 ×57 / 21 ×1; 20 ×21 / 21 ×1) | 1 |
| **zero-delay** | **0 ×54, 1 ×4** | **0** |

The occasional +1-frame reads (5 ms, one substep) occur in both apps. They come
from the teleported kinematic marker's transform reaching the renderer one
substep late, not from the renderer. Tolerance: ≤ 1 physics frame.

**Robot-motion test (the pipeline itself).** Continuous base yaw (1 rad/s) and
both arms swinging ±12 cm (8.6° and up to 24 cm of hand travel per action) are
observed only through `SimWorld.observe()`. Afterwards each observation is
re-rendered from the full robot state recorded at the frame the pipeline
*claims* the images show, and at ±1 and ±2 actions. The recorded image must
match offset 0 best. A step is scored only if its ±1 neighbours differ from
offset 0 by ≥ 6 (mean |Δ| on 16×16 blocks), which is above the re-render
noise floor (median error at offset 0 ≈ 4.0; median next-best ≈ 8.4–8.6).

| App | head | left wrist | right wrist | chest | `obs − image` frames | proprio frame = image frame |
|---|---|---|---|---|---|---|
| default | 24/24 | 12/12 | 10/10 | 23/23 | 20 | yes |
| **zero-delay** | **23/23** | **11/11** | **9/9** | **22/22** | **0** | **yes** |

**The default app's lag is not stable.** In one run (probe 3) a strongly
observable step returned the *current* render: error 9.97 at the claimed frame
vs 5.46 one action newer, on head and chest together. A fixed-lag buffer
mis-pairs such a step silently. The zero-delay app showed no such event. This
is the reason for choosing it, together with the zero latency.

An earlier version of the robot test reversed the yaw mid-sequence. At the
reversal the ±1 reference states are nearly identical and one step was
ambiguous (4.57 vs 4.35). The stimulus was redesigned rather than the
threshold loosened.

**Cost.** Act + observe with 4 cameras (320×240): 0.26 s per action in the
default app, 0.32 s in the zero-delay app (probe 4; +17–23 %).

## 4. Per-episode enforcement

* **Live sync check on every reset** (`SimWorld._verify_sync`, 3 unrecorded
  actions). A visual-only marker (no collider) is put 0.4 m in front of the
  head camera for one action and removed. The observation must show it exactly
  in the observation whose `image_frame` is the marker frame. Any failure raises
  and the episode is quarantined. The result is stored in each episode's
  `meta.json` (`sync_check`) and each evaluation record (`sync_check_ok`).
* **Recorded timing per step**: `policy_step`, `image_frame`, `proprio_frame`,
  `obs_frame`, `action_frame` and the matching times, saved in `lowdim.npz`.
* **Quality gate `temporal_sync`**
  ([`lmm/embodiments/alphabot2/stage1_gates.py`](../lmm/embodiments/alphabot2/stage1_gates.py)):
  `image_frame == proprio_frame`, `obs_frame − image_frame == 20·RENDER_LAG`
  (0 here), `action_frame == obs_frame`, 20-frame spacing, contiguous policy
  steps, and a passed reset check. Episodes that fail are not trained on.

## 5. Consequences

* Training pairs `(RGB_t, proprio_t, instruction) → a_t…a_{t+49}` and closed-loop
  inputs are produced by the same function, with the same (zero) latency.
* The bootstrap's pilot data (image read after action t in the default app)
  was synchronized but not latency-free, and it would be mis-paired on the
  intermittent lag event. It is not used in Stage 1.
* Isaac's own `rendering_time` metadata is recorded (`isaac_render_time`)
  but never trusted (it was off by one frame in the default app).
