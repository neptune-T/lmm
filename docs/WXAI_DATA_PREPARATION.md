# WXAI Data Preparation

What must exist before WXAI normalization statistics can be computed, and the
exact command that computes them. **No data was collected and no training was
started.**

## 1. Demonstration source — the blocking gap

**There is no dual-arm OVMM demonstration source.** This is the largest gap.

What exists:

| Asset | State |
|---|---|
| OVMM episode datasets | present (`benchmark/home-robot/data/datasets/ovmm`) |
| HSSD scenes, objects, navmeshes | present |
| Stretch demonstration pipeline | present but **single-arm** |
| Dual-arm demonstration pipeline | **does not exist** |

The historical pipeline
(`collect_stretch_sft_bootstrap.py`, `stretch_support_collection.py`,
`build_stretch_lerobot_dataset.py`) produced Stretch data by running the OVMM
**baseline skill agent** and keeping successful suffixes. Those skills
(`NAV_TO_OBJ`, `GAZE`, `PICK`, `PLACE`) are implemented against Stretch's
single 6-DoF arm plus telescoping lift. They do not produce 12-DoF dual-arm
joint trajectories and cannot be retargeted without rewriting the controllers.

So the three candidate sources, ranked:

**A. Scripted dual-arm oracle in Habitat (recommended).** Write an IK-based
scripted controller that performs OVMM pick-and-place using one arm at a time,
recording both arms' joint states. Reuses the existing OVMM episodes, scenes
and success metrics. Produces exactly the required 14-D/16-D stream. Lowest
risk, no new assets, no teleoperation rig. The other arm holds its safe pose
(`arm_init_params = [0, π/12, π/12, 0, 0, 0]` per arm), which is honest
single-arm behaviour recorded in a dual-arm format.

**B. Teleoperation.** Highest quality, requires hardware and an operator.
Out of reach on this machine.

**C. Retarget existing Stretch demonstrations.** Attractive but **not
recommended**: Stretch's kinematics (telescoping lift + 4-DoF arm) have no
joint-space correspondence to a 6-DoF WidowX, so retargeting would be a
Cartesian IK replay whose joint trajectories are invented rather than
demonstrated. The resulting norm statistics would describe the IK solver, not
the task.

Whichever is chosen, **the gripper action must be recorded as the binarized
value Habitat actually executed** (0 or 1), not the continuous command — see
the audit's §2.

## 2. Proposed dataset schema

LeRobot v2.1, matching the precedent set by `build_stretch_lerobot_dataset.py`
(lerobot 0.1.0 writer, pinned commit `0cf8648`, matching openpi's `uv.lock`).

| Feature key | Shape | dtype | Meaning |
|---|---|---|---|
| `observation.state` | `(14,)` | float32 | `[left 0-5 rad, left grip, right 0-5 rad, right grip]` |
| `action` | `(16,)` | float32 | same 14 + `[forward m/s, yaw rad/s]` |
| `observation.images.cam_high` | `(480, 640, 3)` | uint8 | head RGB, HWC |
| `observation.images.cam_left_wrist` | `(224, 224, 3)` | uint8 | left wrist RGB, HWC |
| `observation.images.cam_right_wrist` | `(224, 224, 3)` | uint8 | right wrist RGB, HWC |
| `task` | — | str | language instruction |

Resolutions come from the overlay yaml. **HWC, not CHW** — the WXAI transforms
take Habitat's native layout (see `test_images_reach_the_model_as_hwc_uint8`).

Invariants the builder must enforce:

1. Arm dims are **absolute joint targets in radians**, never deltas. OpenPI
   applies the delta transform itself via a 14-element mask.
2. Gripper dims are **closed fractions**, `0 = open`, `1 = closed`, recorded
   post-binarization.
3. Base dims are **forward m/s and yaw rad/s**, bounded by `±0.25` and `±0.5`.
   Never `vx/vy`.
4. `action[:14]` and `observation.state` index the same actuators.
5. Language instruction populated per episode from the OVMM object and
   receptacle categories.

A repack transform maps these keys into the contract's `state` / `actions` /
`images` / `prompt`, exactly as `LeRobotAlohaDataConfig` does.

## 3. Training config required for the norm-stat computation

`compute_norm_stats.py` takes a **config name**, not a dataset path. It calls
`config.data.create(...)` and applies `repack_transforms.inputs` **and**
`data_transforms.inputs` before accumulating statistics.

That detail is what makes the statistics correct: they are computed in the same
space the model trains in, *after* `WXAIInputs` and the delta transform. A
config that used the ALOHA transforms would produce statistics for a different
space than the one this repo serves.

Register in `openpi/src/openpi/training/config.py`:

```python
@dataclasses.dataclass(frozen=True)
class LeRobotWXAIDataConfig(DataConfigFactory):
    """WXAI dual-arm differential-drive: 14-D state, 16-D action."""

    use_delta_joint_actions: bool = True
    default_prompt: str | None = None

    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation.images.cam_high",
                        "cam_left_wrist": "observation.images.cam_left_wrist",
                        "cam_right_wrist": "observation.images.cam_right_wrist",
                    },
                    "state": "observation.state",
                    "actions": "action",
                })
            ]
        )
    )
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs, model_config):
        # The SAME function the policy server uses. Training and inference
        # cannot diverge because there is only one definition.
        from lmm.policy.wxai_transforms import wxai_data_transform_group

        data_transforms = wxai_data_transform_group(
            use_delta_joint_actions=self.use_delta_joint_actions
        )
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(
            model_config
        )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


TrainConfig(
    name="pi05_wxai_ovmm",
    model=pi0_config.Pi0Config(pi05=True),
    data=LeRobotWXAIDataConfig(
        repo_id="lmm/wxai_ovmm_v1",
        default_prompt=None,
        assets=AssetsConfig(asset_id="wxai_ovmm_v1"),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "<checkpoint>/params"
    ),
    num_train_steps=20_000,
    batch_size=32,
)
```

`asset_id` is **`wxai_ovmm_v1`**, never `trossen` or `trossen_mobile`.
`serve_wxai_pi05.py` refuses those ids unless explicitly overridden.

## 4. Exact norm-stat computation command

Once the dataset exists at `repo_id = lmm/wxai_ovmm_v1` and the config above is
registered:

```bash
cd basecode/openpi && uv run scripts/compute_norm_stats.py --config-name pi05_wxai_ovmm
```

To sanity-check on a subset first:

```bash
cd basecode/openpi && uv run scripts/compute_norm_stats.py --config-name pi05_wxai_ovmm --max-frames 20000
```

Output lands in `config.assets_dirs / "lmm/wxai_ovmm_v1"` as `norm_stats.json`,
containing `state` (14-D) and `actions` (16-D) mean/std/quantiles.

### Verification after computing, before any fine-tuning

```bash
# 1. dimensions
python -c "
import json; s=json.load(open('<assets>/lmm/wxai_ovmm_v1/norm_stats.json'))
assert len(s['state']['mean'])==14 and len(s['actions']['mean'])==16
print('dims OK')"

# 2. base statistics must describe forward/yaw, bounded by the actuator
#    limits -- if |mean| or std on dims 14-15 exceeds 0.25 / 0.5, the data
#    was recorded in the wrong units or the wrong convention.

# 3. the server must now start WITHOUT the override, and report calibrated
python scripts/serve_wxai_pi05.py --checkpoint <ckpt> --norm-stats-asset-id wxai_ovmm_v1
curl -s localhost:18116/healthz | python -m json.tool | grep norm_stats_calibrated
# expect: "norm_stats_calibrated": true
```

## 5. Ordered prerequisites

1. Install the URDF (`habitat_patch/trossen_mobile_ai/install.sh`), then run
   `scripts/audit_wxai_urdf.py` and resolve every reported problem.
2. Reconcile `allow_dyn_slide` between the overlay yaml and
   `TrossenHabitatBackend.make`.
3. Confirm `link_id == joint_position_index` in a live Habitat session by
   commanding one joint at a time and checking only that dimension moves.
4. Measure the achieved control rate; replace the nominal 50 Hz label.
5. Build the dual-arm demonstration source (option A).
6. Build the LeRobot dataset to the schema above.
7. Register the config and compute the statistics.
8. Only then consider fine-tuning.

Steps 1–4 need a GPU host with the simulator; this machine has an 8 GB RTX 3070
with no `habitat`, `jax`, `torch` or `openpi` installed, and 4.8 GB free disk.
