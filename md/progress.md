# V1 Target-Link Trajectory Representation — 进度报告

> 更新时间：2026-09-02
> 项目目录：`/nfs/dataset-ofs-494-1/project/user/junao/target_link/`
> 规格文档：`representationV1.md`（唯一主要 specification）

---

## 0. 总体状态

| 阶段 | 内容 | 状态 |
|---|---|---|
| Step 0 | 阅读 spec（representationV1.md） | ✅ 完成 |
| Step 1 | 调研 repository / 现有工程与数据 | ✅ 完成 |
| Step 2 | 定位真实数据 + 理解 schema | ✅ 基本完成（1 个待确认问题） |
| Step 3 | trajectory → spatial profile 预处理 | ✅ 完成（ingest + 去重 + sub-link 切分 + profile） |
| Step 4 | Encoder 实现 | ✅ 完成（smoke 全过） |
| Step 5 | multi-trajectory link 聚合 | ⬜ 未开始 |
| Step 6 | 接入下游 RP / ETA | ⬜ 未开始（ETA 可直接做，RP 标签待定义） |
| Step 7 | debug 子集 overfit test | ⬜ 未开始 |
| — | 四组 Ablation + configs | ⬜ 未开始 |

---

## 1. 数据定位结论

### 1.1 用户提供的路径（待确认）

用户给出：`/user/bigdata-dp/user/liruifeng/traffic_traj_encoder/smoke/link125121851_20260816_10/`

**该路径在当前默认 HDFS 集群上不存在**（`hdfs dfs -ls` 报 No such file or directory，默认集群可正常访问、其它目录可见）。
可能原因：其它 HDFS 集群 / 其它机器本地路径 / 已被清理。
→ **待用户确认**；在此之前以 `beijing_week1/samples` 作为主数据源继续（同项目、同 schema）。

### 1.2 实际可用的真实数据（已确认可读）

```
hdfs:///user/bigdata-dp/user/liruifeng/traffic_traj_encoder/
├── beijing_links/
│   └── map_version=2026081412                 ← 地图
└── beijing_week1/
    ├── samples/event_hour=20260817XX/         ← 样本，56 个小时分区
    │     part-*.snappy.parquet               每分区 14 个文件，单文件 78MB~950MB
    ├── markers/                               ← 分区完成标记
    └── reports/report_*.json                  ← 每小时构建报告
```

规模（以 `event_hour=2026081708` 为例，来自 report）：
- `n_trajs=241,490`，`n_samples=11,378,823`，输出 ~36.6 GB / 小时
- 全量 56 小时 ≈ **2 TB**（全量不可行，需按小时/文件抽样）

已下载用于开发的样本：
`data/raw_hdfs/part-00002.parquet`（160 MB，7,600,000 行）

---

## 2. 真实数据 Schema（已验证）

数据粒度：**每行 = 一条车辆轨迹在某个 10 m spatial bin 上的一次观测**（`bin_size_m=10.0`，与 V1 规定的 10 m bin 完全一致）。

| 字段 | 类型 | 含义 |
|---|---|---|
| `sample_id` | string | `traj_id#target_link_id#pass_idx`，一次通过 |
| `traj_id` / `pass_idx` | string / int32 | 轨迹 id / 第几次通过 |
| `target_link_id` | string | 本样本的 target link |
| `seg_mark` | int8 | **0 = 上游缓冲，1 = target link 本体，2 = 下游缓冲** |
| `bin_idx` | int32 | 走廊内全局 10m-bin 序号 |
| `sub_idx` | int32 | 段内子序号（segment 重叠/拼接） |
| `ratio` | float | 该 bin 属于当前 link 的比例（1.0=整格，0.5=半格） |
| `T_cum` | double | 到达该 bin 的累计时间（s） |
| `T_diff` | double | 通过该 bin 耗时（s）→ 局部速度 = 10 m / T_diff |
| `observed` | int8 | 1 = 真实 GPS 点，0 = 插值点 |
| `n_gap_bins` / `n_zero_ratio` | int32 | 数据质量统计 |
| `L_link_m` | double | target link 长度（m） |
| `n_bins` / `n_segs` | int32 | 走廊总 bin 数 / 段数 |
| `y_travel_s` | double | **ETA 标签**（进入后剩余行驶时间，s） |
| `corridor_time_s` | double | 走廊总耗时 |
| `link_length / link_fc / link_speed_class / link_kind` | … | link 静态属性 |

### 2.1 关键语义验证结果

1. **V1 的 target-link-only 范围可直接用 `seg_mark == 1` 过滤**（上游/下游缓冲被天然排除）✓
2. `seg_mark==1` 的行中 ~3% `link_id != target_link_id`（多段 target 的地图匹配问题）→ **以 `seg_mark` 为准**。
3. 走廊结构示例（`sample 12763714378698517095#5408641#000`）：
   `seg_mark=0`（上游）→ `seg_mark=1`（target, bin 10~14, link=5408641）→ `seg_mark=2`（下游），T_cum 单调递增、bin 连续 ✓。

### 2.2 初步统计（来自 part-00002，row group 0，136,977 个样本）

| 统计量 | 数值 | 对实验设计的影响 |
|---|---|---|
| target link 长度中位数 | ~71 m（均值 112，p95 高，最大 6005） | **多数 target link 短**，≤200 m 时 sub-link 切分不生效（V1 规则：短于 L_sub 保留原 link）；Ablation 3 仍有效但需留意 |
| target bins / 样本 中位数 | 7（均值 11.5） | token 序列很短，与 V1 假设的 ~20 token 有差距，需在统计报告里如实记录 |
| observed ratio / 样本 中位数 | 0.5 | 约一半 bin 是插值 → **padding/missing mask 逻辑是真实需要的** |
| 隐含平均速度 `L/y` | 中位 6.23 m/s（≈22.4 km/h） | 城市拥堵水平合理 |
| `mean_speed` 字段 | **数据中没有现成字段** | 需从轨迹推导：同一 (target_link, 时间窗) 内通过速度均值 → 这正是 baseline 特征，会显式构造并输出 |

---

## 3. 环境与工程现状

### 3.1 计算环境

- Python 3.13.13（`/home/luban/miniconda3/bin/python`）
- torch 2.12.0+cu130，CUDA 可用：**1 × NVIDIA RTX A6000 (46 GB)**
- 库可用：pyarrow / pandas / yaml / numpy；**缺**：sklearn、matplotlib（指标用 numpy 自算，可视化先跳过或补装）
- `hdfs` CLI 可用（默认集群），可拉取真实数据

### 3.2 现有工程调研结论

- 本环境**没有现成的 RP / ETA 训练 pipeline 可直接插入**：
  - `/nfs/.../junao/baseline/HTP`：与本项目无关（用户已确认 "跟 HTP 没关系"）
  - `rpgpt`（zhouyuping / junao/ruiqian）：geohash-token 化路线生成模型，其轨迹数据**无 metric link 几何**，不可用于本任务
  - `rp_pylib`：只是在线服务客户端，不是训练代码
- 因此：**需要在本项目内搭建一个受控的下游 evaluation harness**（spec §9 允许：保持相同下游结构，只换动态表征输入），这是 V1 阶段的标准做法。

### 3.3 已创建的文件（兜底/调试用，非主线）

在拿到真实数据之前写过一个合成数据生成器，现在**保留作 debug/单测用途**：

```
target_link/
├── target_link_v1/
│   ├── __init__.py
│   ├── utils.py                  # config 加载 / seed / 指标（mae, rmse, mape, accuracy）
│   └── data/
│       ├── __init__.py
│       └── synthetic_city.py     # 合成城市：link 几何 + 隐藏空间瓶颈结构（仅供调试）
├── tools/
│   └── build_dataset.py          # 合成数据集构建（已停用主线，待确认）
├── data/
│   └── raw_hdfs/part-00002.parquet   # 真实数据样本（160MB）
└── configs/  scripts/  runtime/      # 空目录占位
```

---

## 4. 关键设计决策（已定）

1. **下游任务先做 ETA**：`y_travel_s` 是现成回归标签，可直接训练；**RP 任务本数据无显式标签**，待与用户讨论定义（候选：下一时间窗拥堵分类，需另外构造）。
2. **mean_speed baseline 特征**：由 (target_link, 时间窗) 内所有轨迹的通过速度（`L_link_m / target 段耗时`）求均值得到，作为所有实验共享的输入特征（显式构造、落盘）。
3. **target-link-only**：只取 `seg_mark==1` 行；不做 up/downstream、zone、approach token（V1 §2）。
4. **sub-link 切分**：默认 200 m（可配置），短于阈值的 link 保留原样；需要保存 original link id ↔ sub-link id 映射。
5. **数据抽样策略**：先 1~2 个 part 文件（~2 万样本）做开发与 overfit，跑通后再按小时扩展。

---

## 5. 下一步（按优先级）

1. ~~等用户确认数据源~~ ✅ 2026-09-02 已确认：用 `beijing_week1/samples`、先只做 ETA、抽样开发再扩展。
2. ~~写 ingest 脚本~~ ✅ 完成（`tools/ingest.py` + `configs/ingest.yaml`），产出见 §7。
3. ~~实现 spatial profile 构建~~ ✅ 完成（`tools/build_profiles.py` + `configs/profiles.yaml`），产出见 §8。
4. ~~实现 Encoder~~ ✅ 完成（`target_link_v1/models/encoder.py` + `tools/smoke_encoder.py`），产出见 §9。
5. **multi-trajectory → link 聚合**（下一步起点）：(sub-link, window) 内 K 条 r_traj 取 mean（spec §8）。
5. multi-trajectory → link 聚合（mean）。
6. 接入 ETA 下游 + debug 子集 overfit 验证（shape / loss / mask / NaN / 无泄漏）。
7. 之后才进入四组 Ablation（Stage 1 优先：MeanSpeed / MeanSpeed-MLP / Ours）。

---

## 6. 待用户确认的问题（已全部确认，2026-09-02）

1. ~~`smoke/` 目录位置~~ → **用 `beijing_week1/samples` 继续**（smoke 不存在，放弃）。
2. ~~RP 标签定义~~ → **V1 先只做 ETA**（`y_travel_s`），RP 后续讨论。
3. ~~数据规模~~ → **抽样开发再扩展**：先用本地 `part-00002`，跑通后按小时扩展。

---

## 7. Step 3a ingest 产出（2026-09-02 完成）

### 7.1 产出文件

```
configs/ingest.yaml                # 预处理配置（窗长 3600s、过滤阈值等）
tools/ingest.py                    # 流式逐 row-group 处理，可直接扩展多小时数据
data/processed/
├── samples.parquet      11.5 MB   # 222,063 样本 × 1 行（window_id, v_sample, y_travel_s, link 静态属性）
├── bins.parquet         13.0 MB   # 2,551,368 bin 行（rel_bin_idx, ratio, T_diff, observed）
├── link_window.parquet   3.3 MB   # 108,875 个 (link, window)：mean_speed 等 baseline 特征
└── ingest_stats.json             # spec §11 基础统计
```

### 7.2 本步新发现（schema 补充，progress 之前未记录）

| 字段 | 语义（已验证） |
|---|---|
| `t_enter` | unix 秒时间戳，样本级进入时刻；北京时间 2026-08-17 07~08 时，与 `event_hour=20260817xx` 对齐 → **时间窗 = floor(t_enter, 1h)** |
| `t_ref` | 参考时间，略早于 t_enter（暂未使用） |
| `bin_pt_ts` | bin 级 GPS 时间戳字符串，`-1`=插值 bin，与 `observed` 对应 |
| `X_in_m` / `X_out_m` | 进入/离开走廊位置（暂未使用） |

### 7.3 数据质量结论

1. **9,321 个 bin 的 `T_diff` 为 NaN**（0.37%，影响 2,396 样本；其中 5,490 个竟是 `observed=1`）→ 保留 NaN 落盘，Step 3 构建 profile 时标为 invalid（正好对应 spec §4 的 `m_i` missing flag，真实需求）。样本级 `td_target` 用 sum 跳过 NaN，不受影响。
2. `sum(ratio)×10` 与 `L_link_m` 偏差 p10~p90 约 ±9% → **样本速度按 progress §4.2 决策用 `L_link_m / ΣT_diff`**（生产定义）。
3. 39 个样本 `ΣT_diff ≤ 0.1s` 已过滤。
4. 复核统计（part-00002 全量 2 个 row group）：222,063 样本、108,768 link、2 个小时窗；bins/样本 p50=7、p90=23；observed_ratio p50=0.5；trajs/(link·window) p50=1、p90=4 —— **长尾 link-window 轨迹数很少，聚合时多数只有 1~2 条轨迹**（对 §8 聚合与下游样本量评估有影响）。

### 7.4 已验证的不变量

- `rel_bin_idx` 每样本从 0 开始、行数 == `n_bins_target` ✓
- bins 与 samples 样本集合一致 ✓
- `link_window.mean_speed` 抽查重算一致 ✓

---

## 8. Step 3b spatial profile + sub-link 切分产出（2026-09-02 完成）

### 8.1 产出文件

```
configs/profiles.yaml               # L_sub/max_bins/速度上限可配（--l-sub 覆盖，供 Ablation 3）
tools/build_profiles.py             # 全向量化：速度 -> sub 切分 -> padded npz
data/processed/
├── profiles_l200.npz       13 MB   # 263,400 profiles × pad 40：speeds/valid/observed/lengths + meta
├── link_sub_map_l200.parquet       # 122,617 个唯一 sub-link（link↔sub 映射，n_bins/eff_len_m/n_profiles）
└── profile_stats_l200.json         # spec §11 统计
```

npz meta：`sample_id / link_id / window_id / sub_id / y_travel_s / v_sample / td_target / n_bins_target`。
`--l-sub {100,200,300,400}` 即可重生成 Ablation 3 全部变体（max_bins=40 已覆盖 400m）。

### 8.2 本步关键决策（数据驱动）

1. **bin 速度定义（修正 spec 字面）**：`v_i = bin_size_m · ratio_i / T_diff_i`，而非 `10/T_diff`。
   验证：半格 bin 用 `10/T_diff` 得 p90=55.6 m/s（伪异常）；用 ratio 修正后 p50=6.5 与样本速度一致，
   且 `Σ(10·ratio)/ΣT_diff` vs `v_sample` 相关 0.987、中位偏差 2.9% → **T_diff 是"通过该 bin 内 link 所属部分"的耗时**。
2. **重复 bin 去重（重要，34% 样本受影响）**：corridor segment 重叠导致同 `(sample_id, bin_idx)` 出现两行
   （76,353 行，即 schema `sub_idx` 的来源），`ΣT_diff` 重复计段会低估 `v_sample` ~10-20%。
   ingest 已修：保留 `T_diff` 有效且 `ratio` 最大的行（拥有该 bin 主体的 segment）。
   去重后 bins 2,551,368→2,475,015，`v_sample` p50 6.35→6.44，重复与 gap 均为 0。
3. **invalid bin 判定**：`T_diff` NaN/≤0（8,972）或 `v>33.3 m/s`（120km/h 城市上限，10,796）
   → speed 置 0、valid=False，正对应 spec §4 的 `m_i` flag。
4. **sub-link 切分**：`s_start = (cumsum(ratio)-ratio)·10`，`sub_id = floor(s_start/L_sub)` clip 到 `ceil(L/200)-1`。
   0.4% link 因 `L_link_m` 与 bin 实际覆盖有 ±9% 偏差而尾部 sub 无 bin → 自然无 profile，无害。

### 8.3 规模统计（L_sub=200）

- 222,063 样本 → **263,400 profiles**（1.186/样本）；108,768 links → **122,617 sub-links**（12.7% link >200m 被切）
- bins/profile：p50=8、p90=20、mean=9.4（比 spec 假设的 ~20 短，多数 link <200m）
- v_bin（m/s）：p10/50/90 = 3.45/9.63/19.61；valid bin 占 99.2%

### 8.4 已验证的不变量

- 单样本内 sub_id 单调不减、(sample,sub) 组内 rel_bin 连续（去重后 diff==1 全成立）✓
- 切分边界 `s mod 200 == 0` 全部精确 ✓；首 bin sub==0 ✓；600m link 正确切 3×200m ✓
- `Σlengths == bins 行数` ✓；valid⇔speed>0、非 valid 非 pad⇔speed==0、valid⊂pad ✓
- npz meta 的 y/v_sample 与 samples 表一致 ✓

---

## 9. Step 4 Encoder 产出（2026-09-02 完成）

### 9.1 产出文件

```
target_link_v1/models/encoder.py     # TrajectoryEncoder（spec §§5-7）
target_link_v1/models/__init__.py
tools/smoke_encoder.py               # 真实数据 smoke（shape/mask/不变量/反向）
```

### 9.2 实现要点（与 spec 的对应 + 两个明确决策）

1. 输入 `x_i = [v_i/33.3, m_i]`（线性缩放，absolute/residual 两模式对称）；MLP(2→d→d)+LayerNorm；learnable pos（max 40）。
2. Transformer：pre-LN、GELU、batch_first，d=128/L=4/H=4/FFN=4d/dropout=0.1；mean pooling over valid，无 CLS。
3. **mask 语义（明确决策）**：pad 位（≥length）完全屏蔽出 attention；**profile 内 invalid 位参与 attention**（v=0, m=0，模型可见"这里有洞"）但不进 pooling —— 这才是 spec §4 `m_i` 作为输入特征的意义。
4. `input_mode="residual"`（Ablation 2）：v 减去该 profile 的 valid 均值；`out_dim`（Ablation 4）：统一投影到 128 维公平控制。

### 9.3 smoke 结果（真实 profiles_l200.npz 前 4096 条，CUDA）

- 默认 d=128 参数量 815,360；forward/backward 正常、无 NaN
- **pad 不变性**：N 40→20 截短，r 完全不变 ✓
- **invalid 上下文效应**：挖掉首个 valid bin → 4095/4096 的 r 改变 ✓
- **residual 平移不变**：全体速度 +5 m/s → residual 模式 r 不变（Ablation 2 语义在实现层成立）；absolute 模式则有反应（对照）✓
- d-sweep（Ablation 4）：32/64/128/256 → out 128 维，参数量 57.5K / 215K / 832K / 3.27M
