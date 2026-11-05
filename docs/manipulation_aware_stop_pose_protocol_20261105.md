# 操作感知的长时程移动操作停姿选择：数据采集协议（2026-11-05）

**状态：仅完成仓库审计、协议、schema 和 pilot 设计；没有启动 GPU 训练、策略 rollout 或大规模采集。**

这份协议针对研究问题

`P(grasp success | scene, object, candidate base pose, robot state, pre-stop observation)`

数据单位是“同一 scene–object 配置下的多个候选停姿及其 Frozen SFT π0.5 操作结果”。候选停姿之间必须形成可比较的正负对照；单独增加随机的 full-task demonstration 不满足本协议。

## 1. 仓库审计结论

审计依据是当前 `/home/plote/longhorizon/lmm` 工作树的实际实现，坐标和接口以代码为准。当前工作树已有其他实验的未提交改动；本次只新增文档/schema，不覆盖它们。

| 项目 | 当前实现与已有字段 | 本项目仍需补齐 |
|---|---|---|
| Embodiment | `lmm/embodiments/alphabot2/contract.py`：双 7-DoF 手臂、双单自由度夹爪；canonical proprio 是 16-D；底盘动作是 18-D，最后两维为 `v(m/s), omega(rad/s)`。`habitat/alphabot2_robot.py` 校验 URDF 关节顺序（26 个可动 joint）。 | 采集记录必须把 `contract_version`、维度和标签写入每个 manifest；不得从历史假定维度。 |
| 观测/动作接口 | `ovmm_policy_io.py`：policy state 由 `state_from_proprio()` 从 canonical proprio 转换；`policy_images()` 使用 `head_rgb`、`right_wrist_rgb`、`left_wrist_rgb`，分别映射到 `camera_top`、`camera_left`、`camera_right`；`camera_front` 在 AlphaBot2 OVMM 中是 masked empty camera。 | 每个候选停姿必须保存实际 policy input、history offsets、camera key 映射、dtype/shape、时间戳和 hash；Depth 在 v1 不可假定，缺失需显式标记。 |
| 控制频率 | `habitat/env.py`：一个 `step18` 是一个逻辑动作，`ACTION_DT_S=0.1`（10 Hz）；`pi05_contract.py`：预测 chunk 50、flow sampling 10 步。`eval_policy.py`/`pa_eval.py` 的 `EXEC_STEPS` 是实际重查询间隔。 | manifest 固定 `EXEC_STEPS`、history、batch=1、precision 和 checkpoint；不能把 query frame 当成每个 simulator frame。 |
| 导航 planner | `habitat/nav.py`：`plan()` 调 Habitat shortest path，`follow()` 沿 navmesh 顶点行走，返回 `geodesic_m`、`travelled_m`、`arrived`、碰撞和 `arrival_dist_m`。`habitat/expert.py`：`OVMMExpert.standoff_candidates()` 生成停姿，`plan_pick()` 按 geodesic + grasp tilt 选 feasible standoff；当前 `R_MIN=0.25m, R_MAX=0.64m` 是代码中的 arm-frame/底盘中心窗口。`pa_eval.py:navigate()` 用 `ArrivalGate` 确认距离和 heading。 | 没有“给定候选停姿集合、逐个执行并回到同一初始状态”的公开 collector。需要新增薄 harness，复用这些函数，不得另写一套几何或导航定义。 |
| IK/动作 lowering | `kinematics.py:ik()` 是 seeded damped least squares；`cartesian.py:CartesianExecutor.lower()` 把 CartesianMobileAction18 降到 JointMobileAction18，记录 per-arm `ik_valid`、position/rotation error、limits、joint step。阈值为 5 mm / 2°；`manipulation_control.py:BoundedCartesianExecutor` 处理 invalid IK safe-hold 和 joint-step bound。 | 每个候选在操作前要保存左右臂 IK 结果、目标 grasp pose、margin、limits、safe-hold；当前 `pilot.py` 只保存 `joint18` 和粗粒度轨迹，不能直接作为这些字段。 |
| 碰撞 | `habitat/nav.py:scene_contacts()` 在实际 simulator step 后报告 robot-scene contacts（support contact 单列）；`overlap_geometry.py` 提供带 callback 的 IK/limit/swept collision feasibility 检查。 | 需区分“几何预检查失败”“到达停姿后的实际接触”“操作中碰撞”。物理机器人上的碰撞 negative 不执行，用模拟/几何检查标注。 |
| 相机 | `habitat/alphabot2_robot.py:_camera_params()` 绑定 head/wrist；`CAMERA_HFOV` 为 head 69°、wrist 87°；`camera_spec.py` 记录 240×320×3 RGB 和 frame 说明。 | `pilot.py` 不把 calibration 写入每集 metadata；新 manifest 要保存 calibration/静态 contract hash，观测要有 camera timestamp 和 validity flag。 |
| Frozen SFT π0.5 | `scripts/alphabot2_habitat/eval_policy.py` 是 normal-init end-to-end；`pa_eval.py` 是 planner-assisted NAV + π0.5 manipulation，并支持 `BASE_LOCK`、`continuous_v1`、H=5/10 等合同。当前实验文档绑定 Frozen SFT3000 SHA256 `d66831b7b4bef0d19050bdc1ba165c1ebc3b73008c872d8865354ecab696b413`；运行时仍需以 manifest 再核对。 | 候选停姿实验必须锁定同一个 checkpoint、normalizer、server precision、inference seed schedule、history、timeout 和 success definition；不要用 `eval_policy.py` 的 full-task 初始状态路径冒充候选停姿数据。 |
| 现有 recorder | `scripts/alphabot2_habitat/pilot.py` 已写 RGB、proprio、Cartesian action、lowered `joint18`、stage、sim time、base xz/yaw、held metadata，并做 replay；`scripts/alphabot2_habitat/rich_rollout_recorder.py` 提供 hash-chain `events.jsonl`、packed arrays、reset/render/action/query/response、controller snapshot、RNG 和 `label_sequence()`；`lmm/rl/rollout_recording.py` 有 `read_log()`、`qualify_recording()`、replay/action reconstruction。`overlap_recording.py` 保留原始 RGB frame journal。 | rich recorder 没有 candidate-set/pose identity，也没有现成“候选 pose 批量采集”入口；需把 candidate metadata 放进 header/run_init，并把 pre-stop observation 与 trial outcome 绑定。 |
| 成功定义 | `eval_policy.py:metrics()` / `pa_eval.py:phases()` 返回 OVMM funnel：find、grasp、carry、place、full；`HE.action_obj(env).events` 记录 grasp/release；`rich_rollout_recorder.label_sequence()` 可从 diagnostic 标注 grasp initiation、holding、lift、carry-ready、physical release、stable full success。 | 数据集必须同时保存事件级字段和 funnel 字段。不能用一个 `success` 覆盖 IK、稳定抓取、PICK、PLACE、full task；不可把 SNAP/DESNAP 或 expert fixture 算作 unassisted policy grasp。 |
| replay/eval | `scripts/alphabot2_habitat/nav_replay_check.py` 可做 fresh-reset action/image/proprio/base/funnel replay；`lmm/eval/full_task_eval.py` 只聚合 end-to-end `FullRolloutRecord`，且拒绝 skill sequencer。 | 新数据需增加 candidate-level exact replay receipt、pose error、query/response hash、missing/engineering 分类；同一 scene-object 的 candidate/repeat 必须作为 group 分割。 |
| 现有数据与失败记录 | `artifacts/alphabot2_pilot/`、`previews/holding_pose_failures_20261030/`、`docs/holding_pose_pilot_20261029*.md`、`docs/reliable_collection_20261101*.md`、`docs/data_collection_protocol_20261101.md`、`docs/data_coverage_interference_20261101.md` 及其 `docs/v2/results/...` 是已有 pilot、失败、replay 和 coverage 证据。它们是 provenance/reference，不自动成为本项目候选停姿标签。 | 新数据必须有新的 `dataset_version` 和 lineage；不要移动、覆盖或重标旧 rollout。 |

### 重要解释

1. `eval_policy.py` 是端到端 policy 直接控制 base 的路径；`pa_eval.py` 才有可复用的 planner/arrival/BASE_LOCK 分段。新 collector 要在候选停姿已经到达且确认稳定后，调用同一 manipulation policy phase，并锁定 base（`BASE_LOCK=1`），否则“停姿效果”会和 policy 自己继续移动底盘混在一起。
2. `OVMMExpert` 的 privileged object pose、navmesh、IK 和目标 receptacle 只可进入诊断/候选生成/标签；不能进入 policy observation。新数据的 `observation` 与 `diagnostics` 必须分文件或至少分 namespace。
3. 当前 `pilot.py` 的 `meta.json` 有丰富但非本研究专用的字段；没有 candidate ID、planned-vs-actual stop pose delta、完整 per-arm IK report、reachability margin、camera calibration、candidate-level failure taxonomy。因此不能直接把它重命名为本 schema。

## 2. 数据包布局与记录原则

遵循仓库的 JSONL 约定（每行带 `schema_version` 和 `record_type`），建议一个 pilot 目录如下：

```
<dataset_root>/
  manifest.json                 # dataset/policy/env/split/hash contract
  episodes.jsonl                # 一行一个 scene-object 初始配置
  candidates.jsonl              # 一行一个 candidate stop pose
  observations.jsonl            # 一行一个 candidate 的 pre-stop policy input 索引
  trials.jsonl                  # 一行一次 Frozen SFT manipulation trial
  outcomes.jsonl                # 一行一次 trial 的事件/阶段结果
  qc.jsonl                      # 一行一次 trial 的质量门
  arrays/<sha256>.npy           # RGB/state/action/diagnostic 数组；JSONL 只存引用
  replay/<trial_id>/receipt.json
  videos/<trial_id>.mp4         # 可选派生文件，原始 RGB 保留
```

`docs/manipulation_aware_stop_pose_schema_20261105.json` 是 machine-readable JSON Schema 和合成示例。实际数组用 `.npy/.npz` 或 rich recorder packed store；JSONL 不内嵌大图。

所有坐标、单位和时间约定必须写入 manifest：

- Habitat world：`[x,y,z]`，`+y` 向上；`NAV.pose2d()` 的二维位置是 `[x,z]`；yaw 绕 `+y`，0 指向 base `+x`，正方向按 `nav.py` 的 `yaw_to()/wrap()`。
- Base frame：`+x` forward；`base_pose_world` 使用 4×4 SE(3) 或 `[x,y,z,yaw]`，不能混用未声明的 ROS z-up。
- Arm frame A / `gripper_base`：由 `kinematics.py` 定义；`gripper_base +z` 是 approach；Cartesian π0.5 action 在 checkpoint frame B/tool E，必须注明转换链。
- 角度是 rad，距离是 m，速度是 m/s 或 rad/s；RGB 是 `uint8`，policy v1 为 240×320×3；时间统一用 simulator world seconds，并保留 wall-clock UTC 字符串。

## 3. 协作者 SOP（模拟器优先）

### 3.1 开始前

1. 只使用已审核的 Habitat/OVMM AlphaBot2 环境、指定 Frozen SFT checkpoint、对应 policy server 和现有配置。先完成 CPU/schema/import smoke test，再申请 GPU/硬件授权；本协议本身不授权启动。
2. 固定并写入 manifest：`dataset_version`、git commit/dirty 状态、checkpoint SHA256、normalizer/server hash、环境/config hash、`precision=fp32`、`max_batch=1`、`chunk_size=50`、`num_inference_steps=10`、`EXEC_STEPS`、history offsets、timeouts、random seeds、split roster。
3. 为每个 `scene-object` 建唯一 `episode_id` 和 `configuration_group_id`。候选 pose、repeat trial、replay 都继承它；不得把同一配置拆到 train/val/test。
4. 先做 1 个 dry-run：确认三路 RGB、proprio16、policy state 转换、server camera keys 和 policy response shape `(50,18)`；确认 `camera_front` 是 masked/invalid，而不是伪造胸前图。
5. 初始化机器人/模拟器并记录 reset 后的实际 base pose、arm q、gripper state、target object ID、scene/object hash。Real robot 另需安全审核、急停和人工监督；不得执行碰撞 negative。

### 3.2 为一个 scene-object 生成候选停姿

1. 从同一个 fresh reset 和同一个 object 初始状态开始。保存 reset seed、replica、scene/object transform hash。
2. 生成候选集合，至少包含：原 planner 选择；`OVMMExpert.standoff_candidates()` 的几何可行点；靠近 `R_MIN/R_MAX` 的边界点；heading 对照点；IK 可行但视觉/夹爪几何困难点；以及仅几何标注的无效/不可达点。
3. 对每个候选先做静态检查：navmesh 可达、候选 base footprint 不越界、目标相对 base 方向/距离、左右臂目标 grasp pose、IK/limit、gripper aperture、几何 collision callback。把每个 arm 的结果写进 `candidates.jsonl`。
4. 候选排序只用于收集覆盖，不用 outcome 选择 candidate。预先冻结 candidate list 和顺序，给每个 candidate 固定 `candidate_id`、`sampling_reason`、`sampling_seed`。
5. 在模拟器中从同一 reset 实际执行到候选点（不要只 teleport 后声称 navigation success）。使用 `NAV.follow()`/`pa_eval.navigate()` 的 arrival gate，记录 path length、collision、planned pose、actual final pose 和误差。到达后先做 3 个 hold steps，确认位置和 heading 稳定，再 capture pre-stop observation。
6. 如果候选在静态几何检查已 invalid/unreachable，模拟器可保留为 hard negative，但不要执行会造成明显碰撞的动作；其 `trial_status=geometry_only`，outcome 字段对未执行部分为 `null`。

### 3.3 记录 pre-stop observation

在任何操作动作之前、候选 base 已稳定后，记录一份不可变 observation：

- `camera_top`、`camera_left`、`camera_right` 的原始 RGB、shape/dtype、camera timestamp、validity 和 sha256；保留 policy 的 wrist cross-map。
- 若当前 Frozen SFT 合同使用 `camera_top_h5/h10/h15`，保存实际历史图像引用和 exact offsets；历史不足只允许协议规定的 episode-start clamp，不能用未来帧。
- canonical `proprio16`、checkpoint `state16`、base pose/heading、arm q、gripper state、selected arm、instruction；保存转换函数和 hash。
- 将 privileged object pose、target grasp pose、IK/collision/reachability margin 放入 `diagnostics` namespace，不放入 policy observation。
- 为该 observation 写 `observation_id`，并把它和 `candidate_id`、`trial_id` 绑定；不覆盖同一 candidate 的旧 observation。

### 3.4 执行 Frozen SFT manipulation trial

1. 固定 `policy_checkpoint_id`、server manifest、inference seed、batch=1、precision、history、`MANIP1_MAX/MANIP2_MAX` 和 episode timeout。每个 pose 的 repeat 只改变预先声明的 stochastic seed；不要做 deterministic replay 当作独立 trial。
2. 复用 `pa_eval.py:policy_phase()`、`CartesianExecutor`/`BoundedCartesianExecutor` 和 `HE.step18()`。候选 stop-pose 阶段把 base 锁为 0（`BASE_LOCK=1`），只让 π0.5 输出 arm/gripper；保存 policy raw `CartesianMobileAction18`、lowered `JointMobileAction18`、per-arm IK report 和 safe-hold/rate-limit report。
3. 使用 `rich_rollout_recorder.install()` 包住 collector，保存 query/response、RNG/noise receipt、controller snapshot、render、contact、action/action_result、step_complete、terminal hash-chain。若 collector 尚未通过 CPU exact-replay preflight，停止，不要开始正式采集。
4. 每个 trial 只执行一次 Frozen SFT manipulation；不要 SNAP/DESNAP、expert pick、脚本式夹取或人工把物体放入手中。若为了安全使用 assisted interaction，写 `assisted=true` 和具体事件，并将该 trial 从 unassisted PICK success 分母排除。
5. manipulation 结束后保留全部事件和终态，即使失败、timeout、policy server error 或 recorder error。工程错误不是 task failure；两者分别写 `trial_status` 和 `failure_category`。

### 3.5 标注 outcome（字段必须分开）

按 simulator event、诊断帧和 unchanged OVMM metrics 三者交叉确认：

- `reached_pregrasp`：到达 pre-grasp pose 且 IK/limit/安全门通过。
- `contacted_object`：有目标物体 contact/attachment evidence；单独记录 contact timestamp。
- `gripper_closed`：有闭合命令和实际 gripper 状态；闭合本身不等于抓住。
- `stable_grasp`：目标保持 attached/held 达到协议规定的连续 steps（建议 5）；不要从 gripper 值推断。
- `object_lift`：目标相对初始高度达到预先固定阈值（建议 0.05 m），单独记录数值。
- `stable_carry`：holding 且满足 carry-ready/姿态条件；保存 unchanged `ovmm_find_recep_phase_success` 和 carry diagnostic。
- `object_detached`：真实 release/desnap event 后至少 3 个 unheld steps；张开夹爪本身不够。
- `pick_success`：本项目主标签，定义为 verified stable grasp（如有 lift 要求则同时满足）；与 `ovmm_pick_object_phase_success` 并列保存。
- `place_success`、`full_task_success`：只有实际执行并完成相应 task segment 才可为 true；未评估写 `null`。
- `failure_category`：从 `invalid_ik`、`collision`、`unreachable`、`no_contact`、`unstable_grasp`、`drop_after_lift`、`timeout`、`policy_server`、`missing_observation`、`recorder_failure`、`assisted_interaction`、`unknown` 等枚举中选；允许 `primary` + `secondary`。

### 3.6 每个 trial 完成时

1. 写 `outcomes.jsonl`、`qc.jsonl`、rich recorder seal 和 replay receipt；保存所有 raw/derived 文件。
2. 做 fresh-reset replay：验证 reset/scene/object identity、pre-stop observation hash、executed actions、images/query response lineage、base pose、events、terminal funnel。候选 pose 的 replay tolerance 要写入 manifest；默认 numeric state `1e-6`，base pose 与物体 pose 使用单独声明的 m/deg tolerance。
3. 运行完整性检查：缺文件、损坏 RGB、重复 ID、timestamp 非单调、coordinate frame mismatch、candidate/trial 断链、IK 与 lowered action 不一致、invalid label、未封存 journal 都 quarantine；困难失败不删除。
4. 只有 `qc.accepted=true` 的 unassisted trial 才进入训练候选；geometry-only、engineering failure、assisted interaction、missing outcome 仍保留在审计分母但不伪造标签。

## 4. 停姿采样与配对设计

### 候选类别（每个 configuration 建议 6 个）

1. `original_planner`：`OVMMExpert.plan_pick()` 实际选中的 standoff。
2. `feasible_inner`：距离/heading 处于可靠工作区的可行点。
3. `feasible_boundary`：接近代码边界 `R_MAX=0.64m` 的点；若该点不可导航，保留几何 negative 并注明原因。
4. `heading_minus` / `heading_plus`：相同或相近距离，heading 相对 nominal 偏 `-30°/+30°`（可导航时才执行）。
5. `hard_geometry`：IK 通过但 approach/夹爪宽度/视觉遮挡困难，或导航成功但预期 grasp margin 小。
6. `invalid_or_unreachable`：IK、limit、navmesh 或 collision 失败；simulation 中只执行安全可达的部分。

建议半径先覆盖 `{0.30, 0.40, 0.50, 0.60, 0.64}m`，再加入 planner pose；这些是候选生成目标，不是强行 teleport 的精确值。每个候选实际距离/heading 以记录的 final base pose 为准。

每个 configuration 至少要有 easy positive、hard positive、easy negative、hard negative；如果六个候选中没有形成对照，标记 `contrast_missing=true`，不要靠事后筛选补齐。

保持不变：scene/object 初始 transform、instruction、policy checkpoint、arm initialization、inference settings、timeout、success definitions、candidate generation seed。只改变 base stopping position/heading 和 trial stochastic seed。候选 repeat 不是独立 scene；分析时按 `configuration_group_id` 聚类。

## 5. 标签与 split

同时生成三类目标：

- `y_geometry`：指定 reachability + limit + collision criteria 均满足；来自 candidate diagnostics，不看未来 policy outcome。
- `y_grasp`：Frozen SFT 从该停姿达到本协议定义的 verified stable grasp；仅 unassisted、完整 observation/trial 才可为 0/1，否则 `null`。
- `y_downstream`：后续 segment/full task 成功；未执行或未评估写 `null`。

保存每个 repeat 的原始 outcome，再单独保存 `n_trials`、`n_valid`、`grasp_successes`、Wilson/empirical mean 等 aggregate。不要用 aggregate 覆盖 individual trial。

split 按 `scene_id + object_id + base episode/configuration_group_id` 分组：同一配置的所有候选、repeat、observation、replay 只能在一个 split。推荐冻结 `train/val/heldout` roster 后再运行；不要 random frame split。候选评分模型未来只能看到 pre-stop observation/state/candidate fields，不能看到 action、future frames、contact、outcome 或 post-stop pose error。

## 6. Pilot 计划与成本闸门

### 推荐规模

- **开发 TRAIN**：从现有冻结 roster 选 6 个不同 scene-object configuration；优先复用 `docs/v2/results/data_collection_protocol_20261101/episode_split.json` 中的 TRAIN units（例如 `20387#0`、`22439#1`、`14331#1`、`36992#0`、`33669#0`、`35294#0`），不要重新发明 episode ID。
- **锁定 evaluation**：从同一 artifact 的 held-out roster 预留至少 2 个不同 scene-object configuration（例如 `17694#0`、`18583#0` 仅作为已存在 roster 的候选，不得在采集后调换）。evaluation outcome 不回流训练或 pose selection。
- 每个 configuration 6 个 candidate；每个可执行 candidate 2 个 independent Frozen SFT trials。目标是约 `6 × 6 × 2 = 72` 次 manipulation trials；geometry-only candidates 不消耗 policy trial。
- 若第一批 3 个 configuration 的 recorder/replay pass rate < 100%，立即停止扩展，修 recorder，不增加样本。

### 估算（不是已测吞吐，也不是授权）

现有 `docs/data_collection_protocol_20261101.md` 的 comparable full-task source trial 为 `0.232066 GPU-hour/trial`；候选停姿 trial 的真实吞吐尚未测量，故先把它作为保守上界 proxy，并在 3 个 CPU/smoke + 3 个实际 trial 后重新估算。

| 项目 | 估算 |
|---|---:|
| 每 candidate 的 stop-pose setup + 3 hold steps | 20–60 s wall（需实测） |
| manipulation 上限 | 600 steps = 60 s logical time；若执行 settle，再加 120 steps = 12 s |
| 72 次 trial 的 proxy GPU 时间 | `72 × 0.232066 = 16.71 GPU-h` |
| 加 20% recorder/replay/启动余量 | 约 20.1 GPU-h；必须另行批准，不能由本协议自动启动 |
| 每 trial query frame 原始 RGB（约 120 queries、3×240×320×3） | 约 83 MB，未含 history/trace/压缩差异 |
| 72 trial 的 query-level 原始 RGB | 约 6.0 GB；完整每 10 Hz frame journal 的上界约 36 GB |
| 建议保留空间 | 至少 50 GB，并按 shard 做 hash/备份 |

真实机器人 pilot 先缩为 2 个 scene-object、3 个可执行候选、每候选 2 repeats；只收安全可达 pose，碰撞/不可达 negative 用仿真和几何检查，不在实机执行。需要硬件、急停、人工监督和独立安全签字。

### 风险与处理

- **导航到达误差**：保存 planned/actual delta；超出 gate 的 trial 不默认为 pose negative，先分为 `navigation_engineering`。
- **IK 可行但 π0.5 失败**：这是本研究最有价值的 hard negative，保留为 `y_geometry=1,y_grasp=0`（前提是 trial 完整）。
- **物体/场景漂移**：每个 candidate 从 fresh reset 或经 exact replay 的同一 root 开始；发生漂移则 quarantine 整个 configuration。
- **history 泄漏**：历史图像按 simulator step 索引，不能用 future frame；记录 history hash。
- **SNAP/DESNAP 或 assisted interaction**：单独标记并从 unassisted grasp 分母排除。
- **录制失败**：不删文件；`trial_status=engineering_failure`，保留 journal 和错误栈。

## 7. 现有代码复用清单与明确缺口

| 用途 | 必须复用 | 当前缺口 |
|---|---|---|
| 环境/episode reset | `lmm/embodiments/alphabot2/habitat/env.py:make_env, reset_to, fix_start, step18, hold_action` | 候选批量 reset/pose orchestration 尚无入口 |
| Base pose/frame | `lmm/embodiments/alphabot2/habitat/nav.py:pose2d, wrap, yaw_to`；`AlphaBot2Robot.base_frame/arm_frame` | candidate schema 和 planned-vs-actual receipt 需新增 |
| Navigation | `nav.plan, nav.follow, nav.scene_contacts`；`OVMMExpert.standoff_candidates, plan_pick`；`pa_eval.navigate, ArrivalGate` | 候选 stop-pose collector 尚无 CLI |
| IK/Cartesian | `kinematics.fk/ik`；`cartesian.CartesianExecutor.lower/cartesian_from_joint_action`；`manipulation_control.BoundedCartesianExecutor` | per-candidate report、reachability margin、swept collision receipt需新增 |
| Policy manipulation | `scripts/alphabot2_habitat/pa_eval.py:policy_phase`；`scripts/alphabot2_pi05/policy_server.py`/`wire.py`；`ovmm_policy_io.py` | 需要一个“已到 candidate pose 后只跑 manipulation”的 harness；不要改 Frozen SFT |
| Recording | `scripts/alphabot2_habitat/rich_rollout_recorder.py:install`；`lmm/rl/rollout_recording.py:PackedEvidenceLog, read_log, qualify_recording, reconstruct_five` | header/run_init 增加 candidate/observation/trial lineage；需先做 CPU exact-replay preflight |
| Replay | `scripts/alphabot2_habitat/nav_replay_check.py`；`habitat/snapshot.py:capture, restore`（仅用于已审核的 simulator diagnostic） | candidate-level replay receipt 和 invalid/engineering quarantine 需新增 |
| Outcome | `pa_eval.phases, target, held_priv`；`HE.action_obj(env).events`；`rich_rollout_recorder.label_sequence`; `eval_policy.metrics` | 将 event-level labels 和 `y_grasp/y_downstream` 规范化到新 schema |
| 既有 full-task eval | `scripts/alphabot2_habitat/eval_policy.py`；`lmm/eval/full_task_eval.py` | 不能直接回答 candidate stop-pose 问题；只作为未来 baseline/ablation |

## 8. 提交前硬性门

协作者提交 shard 前必须同时提供：`manifest.json`、六类 JSONL、数组/视频 hash manifest、每个 trial 的 replay receipt、`qc_summary.json` 和 README。README 要写清 accepted/attempted/quarantined 数量、按 failure category 的计数、每个 configuration 的 candidate coverage、每个 pose 的 repeat 数、以及哪些标签 unavailable。

任何缺少 policy-visible pre-stop observation、candidate identity、Frozen SFT identity、actual final pose、完整 outcome 或 failure reason 的 trial 都不能进入 accepted set。不要覆盖既有 rollout、Frozen SFT checkpoint、reward definition 或 Online RL campaign；不要启动 Cross-Attention 训练。
