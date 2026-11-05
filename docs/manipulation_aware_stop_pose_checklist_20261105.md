# 停姿对操作可行性数据采集清单

使用前先读 [`manipulation_aware_stop_pose_protocol_20261105.md`](manipulation_aware_stop_pose_protocol_20261105.md)。每个方框都必须有可追溯记录；失败项进入 quarantine，不删除。

## 启动前

- [ ] 已冻结 dataset version、git 状态、scene/object roster、train/val/held-out split 和 random seeds。
- [ ] 已核对 Frozen SFT checkpoint、normalizer、policy server、FP32、batch=1、chunk/history/timeout。
- [ ] 已完成 CPU schema/import/replay smoke test；已获得单独的 compute 或实机安全授权。
- [ ] Habitat/OVMM、AlphaBot2 URDF/navmesh、三路 RGB camera 和 `proprio16` 可用。
- [ ] 未触碰 Online RL campaign、旧 rollout、Frozen SFT checkpoint 和 reward 定义。

## 每个 scene-object configuration

- [ ] 记录 `configuration_group_id`、scene/object ID、instruction、reset seed/replica、初始 object/base/arm state hash。
- [ ] 候选清单已冻结并包含 original planner、inner feasible、boundary、heading contrast、hard geometry、invalid/unreachable（安全方式）。
- [ ] 每个 candidate 有 `candidate_id`、sampling reason/seed、目标位置/heading、target relative pose。
- [ ] 已记录 navmesh/path/geodesic/footprint、候选 IK/limits/aperture/collision diagnostics。

## 每个 candidate / trial

- [ ] 从同一初始配置 fresh reset 或 exact replay 开始。
- [ ] 已实际执行到候选停姿；记录 planned pose、actual final pose、distance/heading error、arrival gate、collision/support。
- [ ] 已连续 hold 3 steps，确认 base pose 稳定。
- [ ] 已保存 pre-stop `camera_top/left/right` RGB、history、timestamp、shape/dtype/hash 和 validity。
- [ ] 已保存 canonical proprio16、checkpoint state16、arm q、gripper、selected arm、instruction。
- [ ] privileged object/grasp/IK/collision 数据放在 diagnostics，未进入 policy input。
- [ ] 使用同一 Frozen SFT π0.5；BASE_LOCK=1；未使用 SNAP/DESNAP/expert fixture/人工夹取。
- [ ] 记录 raw Cartesian chunk、lowered joint18、IK/safe-hold/rate-limit、query/response/RNG/hash-chain。
- [ ] 分别标记 pregrasp、contact、closure、stable grasp、lift、carry、detach、PICK、PLACE、full。
- [ ] 失败有 `failure_category`；工程失败与 task failure 分开；未评估字段为 `null`。

## 收尾与提交

- [ ] rich recorder 有 seal；数组/RGB/视频有 SHA256；JSONL 可解析且 ID 无重复。
- [ ] fresh-reset replay 通过，或有明确的 replay failure receipt；不把 replay failure 当 policy failure。
- [ ] 检查 timestamp、坐标系、candidate/trial lineage、IK consistency、缺失/损坏文件。
- [ ] 保存 attempted/completed/accepted/quarantined 分母和 failure breakdown。
- [ ] 同一 configuration 的所有 candidate/repeat 保持在同一 split。
- [ ] README、manifest、schema 版本和 QC summary 已随 shard 提交。
