# 停姿对操作可行性 Pilot 计划（2026-11-05）

这是采集前的可审核计划，不是 launch 命令，也不授权 GPU 或实机动作。详细 SOP、字段和复用接口见 [`manipulation_aware_stop_pose_protocol_20261105.md`](manipulation_aware_stop_pose_protocol_20261105.md)。

## 固定设计

| 维度 | 计划 |
|---|---|
| 配置 | 6 个不同 scene-object configuration；所有 candidate/repeat 继承同一 `configuration_group_id` |
| roster | TRAIN 优先从 `docs/v2/results/data_collection_protocol_20261101/episode_split.json` 的既有 units 选取，例如 `20387#0, 22439#1, 14331#1, 36992#0, 33669#0, 35294#0`；held-out 至少保留 `17694#0, 18583#0` 等既有 roster，最终清单在采集前冻结 |
| candidate | 每配置 6 个：original planner、inner feasible、R_MAX boundary、heading -30°、heading +30°、hard/invalid geometry |
| repeat | 每个安全可执行 candidate 2 个独立 Frozen SFT trials；重复 deterministic replay 不计为独立 trial |
| 目标数量 | 约 72 个 manipulation trials；geometry-only negative 不启动 policy |
| policy | Frozen SFT3000 π0.5；FP32、batch=1、固定 checkpoint/normalizer/server/hash、固定 history/EXEC/timeout；BASE_LOCK=1 |
| split | scene + object + base episode/configuration group 分组；candidate 级别不能跨 split |

## 候选范围

`OVMMExpert` 当前实现的工作区为 `R_MIN=0.25m, R_MAX=0.64m`。pilot 用 `{0.30, 0.40, 0.50, 0.60, 0.64}m` 作为采样目标，再加入 original planner pose；实际值以到达后的 `base_pose_actual` 为准。每个配置都要有 easy positive、hard positive、easy negative、hard negative；没有形成对照时停止扩展并报告 `contrast_missing`。

## 估算与先决闸门

仓库现有 comparable full-task source trial 的记录是 `0.232066 GPU-hour/trial`（`docs/data_collection_protocol_20261101.md`）。候选停姿吞吐尚未实测，所以只把它作为保守 proxy：

- `72 × 0.232066 = 16.71 GPU-hour`；加 20% 录制/replay/启动余量约 20.1 GPU-hour。
- 每 trial 600 manipulation steps 是 60 秒 logical time；如跑 120-step settle，再加 12 秒。
- query-level 原始 RGB（约 120 queries、3 路 240×320×3）约 83 MB/trial，72 trial 约 6 GB；若保留每个 10 Hz frame，约 36 GB 上界。建议至少预留 50 GB。
- 开始正式 pilot 前必须完成：3 个 CPU/import smoke、recorder exact-replay preflight、3 个实际 trial 的 wall/GPU/storage 计时。用这 3 个 trial 重算预算；本计划不会自动启动后续 trial。
- 前 3 个配置的 recorder/replay pass rate 不是 100% 时停止，修复工程链路后重新审核。

## 输出与判定

每个 shard 提交 `manifest.json`、`episodes.jsonl`、`candidates.jsonl`、`observations.jsonl`、`trials.jsonl`、`outcomes.jsonl`、`qc.jsonl`、数组/视频 hash、replay receipts 和 `qc_summary.json`。统计 attempted/completed/accepted/quarantined 以及按 failure category、candidate class、scene-object group 的结果。

Pilot 只回答数据可用性和 stop-pose contrast 是否存在，不训练 Cross-Attention、不改 planner、不改 reward、不选择模型超参。只有在 accepted data 通过 split、replay、coverage 和安全审查后，才另行设计 geometric-ranking 与 learned-ranking ablation。

## 主要风险

- 导航到达误差必须单列；不可达或碰撞 candidate 在仿真中几何标注，不在实机执行。
- `IK feasible` 与 `π0.5 grasp success` 必须保留为两个标签；IK 成功但抓取失败是目标 hard negative。
- SNAP/DESNAP、expert fixture、人工摆位和 recorder failure 均不进入 unassisted PICK success 分母。
- 所有未来 outcome 只写 label/diagnostics，不能写进 pre-stop policy input。
