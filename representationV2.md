# representationV2 — 自监督 CLS 预训练 + 下游 ETA 适配

> 状态：设计定稿（2026-09-07）。stage① 三件套全部落地并过 overfit 双判据：语料构建（`tools/build_pretrain_corpus.py`）、模型（`target_link_v1/models/pretrain.py`）、训练器（`tools/train_pretrain.py` + `configs/pretrain.yaml`，overfit：rec 0.0035 / cls0 0.047 ×13.4 / acc_y 0.69）。未动工：stage② 改造（`train_eta.py --init-from`）、正式预训练（等 7 天数据上集群）。V1 监督基线 spec 见 `representationV1.md`。

## 1. 目标与总体架构

下游任务（ETA，后续 RP）的输入单元是 **target-link 片段**：一条轨迹通过目标 link 的 bin 序列（`bins_shards` 里一个 sample 的行，p50=7 bin）。本项目要为这个输入单元学一个**携带全局语义的 [CLS] 表征**——压缩"这辆车如何通过这条 link"的完整模式（速度分布、走走停停形态、时段与拥堵上下文），供多个下游任务复用。

获得方式是两段式：

```
① 自监督预训练   span 掩码 + MAE 重建，重建信息被结构性强制流过 CLS → CLS 学全局
② 下游适配       冻结 CLS linear probe（验证表征质量）→ 微调做 ETA
```

选自监督而非直接监督的理由：标签免费（遮住的位置即答案，7 天全量数据可用，免疫 `y_travel_s` 的标签噪声）；更关键的是结构性保证——监督信号能被"平均车速"标量捷径绕过（V1 双稳态：encoder 分支死亡退化成标量路径），而 CLS 瓶颈重建在结构上不存在绕行路径。

设计第一性原则：**loss 落在哪，表征学什么**。一切 loss/结构决策由此推导。

## 2. 数据架构（HDFS 主场，本地只留训练侧产物）

### 2.1 管线全景

```
raw: hdfs://DClusterNmg3/.../beijing_week_biz/samples   (7 整天 20260817~23, ~4.2亿行/时)
 │
 ▼  scripts/submit_ingest_yarn.sh  MODE=yarn  每天一个 yarn 任务
processed @ hdfs: processed_spark/dayYYYYMMDD/
 │    samples.parquet / bins_shards/ / link_window.parquet / ingest_stats.json.d/
 ▼  scripts/fetch_processed.sh  (新; hdfs dfs -get 封装; pyarrow 直读 hdfs 已实测不可靠)
processed @ 本地 NFS
 │
 ▼  build_profiles.py  (流式; 两条支线共用 —— 预训练单元 = 画像行, 见 §2.2)
 ├─▶ [监督支线] split_random.py → train_eta.py                       (V1 链路, 保持不变)
 └─▶ [预训练主线] build_pretrain_corpus.py (已落地) → train_pretrain.py (新) → pretrain ckpt
                                                    │
                                                    ▼  train_eta.py --init-from --freeze-encoder
                                              stage② 冻结/微调
```

### 2.2 预训练语料（build_pretrain_corpus，已落地）

- **单元 = 画像行**：一个样本在一条 200 米子路段上的 10 米粒度速度曲线（profiles npz 的一行）——这正是 TrajectoryEncoder 在下游吃的输入单元，encoder 权重可原样迁移。曲线数组 `speeds/valid/observed/lengths` 直接沿用画像契约（`valid` 内部可有观察空洞，与 encoder 原生"缺口参与注意力"语义一致）。
- **自由标签**：画像 npz 行内已自带 `y_travel_s / v_sample / n_bins_target`，**无需关联样本表**。构建时转成分位 bucket（各 16 桶，y/v 均衡；len 因离散取值轻微不均衡 1.9%~9.3%，可接受）+ 北京小时 `(window_id+8)%24`。（起终点 link 标签需要 corridor 上下文，留待 v2.5——见 §3.5。）
- **规模预算**：全量 ≈ 2.4 亿样本/天 × 7 ≈ 17 亿行，本机单卡（21G）跑不动一个 epoch。构建时**按天以样本为单位均匀抽样** `--per-day-cap`（起步 500 万样本/天），分天顺序加载，峰值内存 = 单天画像。
- **产出**：单 npz——曲线数组 + `attr_y/attr_v/attr_len/attr_hour`（int8）+ 分桶边界（评测期复用）+ 溯源字段。冒烟验证（smoke3，365 万行）：桶均衡 6.25%/6.25%（y/v）、小时标签正确（07 时数据只含 6/7）、无 NaN、valid 位置速度全正、边界单调。
- **遮挡在训练期做**，语料不预遮：只遮 valid 位置（无观察的位置没有重建目标）。

## 3. 模型架构（组件②，`target_link_v1/models/pretrain.py`）

### 3.1 Backbone：复用，不复制

encoder token 编码**复用 V1 `TrajectoryEncoder` 的同一个类**。预训练与下游共享同一实现是迁移成立的技术前提，也是代码不冗余的硬约束——任何情况下不允许出现第二份 encoder。

### 3.2 序列与掩码

- 输入序列：`[CLS] + bin_0 ... bin_n`，位置编码沿用 encoder 的 `pos_emb`（bin 下标）。
- **遮挡 = encoder 原生"缺口"语义**：被遮位置以 `速度=0、有效位=0` 参与注意力（TrajectoryEncoder 对 invalid bin 的既有处理），模型代码无需新增输入分支。只遮 valid 位置（无观察位置没有重建目标）；重建目标 = 被遮位置的真实速度（按 `v_norm` 缩放）。
- **span 掩码**：连续 span 长 2~4 bin，掩码比例 50~60%（画像行 p50 长度 ~8 bin，不照搬图像 MAE 的 75%）；曲线过短时降比例，保证至少 1 个可见 valid bin。

### 3.3 CLS 瓶颈 decoder（本设计的核心约束）

decoder **只吃 [CLS + mask token]，不给可见 token 的 skip connection**。被遮 bin 的重建信息必须全部经 CLS 流过——这是对"CLS 死亡"（被局部插值绕过）的结构性免疫，等价于把 V1 分支启用监控从事后检查升级为架构保证。

### 3.4 Loss

```
L = L_rec + λ · Σ_k L_attr_k                    λ 起步 0.2
L_rec   = masked-bin MSE（z-scored ratio/T_diff + observed 的 BCE）
L_attr  = CE(CLS→head_k, 自由标签 k)            k ∈ {y, v, len, hour}
```

### 3.5 v2.5 扩展路径（不在本期实现）

corridor 上/下游（`seg_mark` 0/2）作为**条件上下文**输入（不遮、不重建），让 CLS 在知道来龙去脉的前提下总结 target 片段；需要 ingest 加开关保留上下文行，并解锁起终点 link 自由标签。预训练单元与下游协议均不变。

## 4. 训练与评估协议

### 4.1 Stage ①（train_pretrain.py）

1. **overfit smoke**：1,000 样本重建到近零误差（验证可实现性）。
2. 正式预训练（抽样语料）。监控三项：
   - `L_rec` 曲线 + masked vs visible 重建误差比；
   - attr head 准确率（y/v/len bucket 应显著高于随机 1/16，hour 高于 1/24）;
   - **CLS 活性**：eval 时 CLS 置零，`ΔL_rec` 必须显著 > 0——不涨即 CLS 死亡，预训练无效，这是预训练版的分支启用监控。

### 4.2 Stage ②（train_eta.py 改造）

1. **冻结 probe**：CLS + encoder 冻结，只训回归头。判据：K=1 subset MAE ≥ mean 臂（1.508s）即表征有效（参照系：speed 1.937 / oracle 1.173）。
2. **微调**：全量微调 vs V1 端到端监督，判两段式是否成立。
3. 沿用 V1 协议：随机 8:1:1、分支/CLS 置零监控、3 seeds。

## 5. 代码改造清单

### 5.1 归档（`tools/archive/`，README 记录角色与继任者；不删除）

| 文件 | 原角色 | 归档理由 |
|---|---|---|
| `tools/build_dataset.py` | 合成城市 V1 数据生成器 | 真实数据接管后无场景 |
| `tools/ingest.py` | pandas 全量 ingest | OOM 元凶，被 ingest_streaming 取代 |
| `tools/ingest_streaming.py` | pandas 流式 ingest | Spark 等价性已判决（9.7），被 ingest_spark.py 取代；等价性证据存 md/9.7progress.md §6 |
| `tools/_build_profiles_old.py` | 旧版 profiles | 已被流式版取代 |
| `tools/check_ingest_equiv.py` | pandas↔Spark 等价对比 | 判决完成；且原版在 ref 重复 sample_id 时早退，真实对比由 ad-hoc 脚本完成 |

保留：`ingest_spark.py`（唯一数据入口）、`build_profiles.py`、`split_random.py`、`train_eta.py`、`check_k.py`（K audit 仍待跑）、`debug_overfit.py`、`smoke_*.py`（组件单测，models 仍在用）。configs 的历史项（`*_4f/_h3/_ts_day*` 等）在 M2 后随数据目录一并清理。

### 5.2 修改（最小改动点）

| 文件 | 改动 |
|---|---|
| `train_eta.py` | 加 `--init-from <ckpt>` 与 `--freeze-encoder`（stage② 入口，不新建训练脚本） |
| `configs/` | 新增 `pretrain.yaml`（语料路径/抽样/掩码/λ） |

### 5.3 新增（每文件单一职责）

| 文件 | 职责 |
|---|---|
| `scripts/fetch_processed.sh` | hdfs processed → 本地（get + `_SUCCESS` 校验 + 断点跳过） |
| `tools/build_pretrain_corpus.py` | **已落地**：profiles npz → 预训练语料 npz（画像行曲线 + 4 个免费标签 int8 + 分桶边界 + 按天以样本为单位抽样），冒烟验证通过 |
| `target_link_v1/models/pretrain.py` | **已落地**：`curve_representation()`（唯一 CLS 读出，预训练/stage② 共用）+ `span_mask()`（连续片段遮挡，只遮有效位置）+ `CurveMAE`（CLS 瓶颈 decoder + attr 四头 + `use_cls=False` 活性探针）；单元冒烟通过，encoder 子模块梯度已验证流入 |
| `tools/train_pretrain.py` | **已落地**：stage① 训练入口（AdamW+λ·attr，val 样本级切分，CLS 置零活性 eval，overfit 双 gate，ckpt 供 stage② 零重映射加载）；overfit 烟测通过（×13.4 CLS 信息比） |

### 5.4 不冗余三原则

1. **每阶段单入口**：ingest 唯一（ingest_spark）、profiles 唯一、切分唯一、监督训练唯一（train_eta）、预训练唯一（train_pretrain）。
2. **模型代码只在 `target_link_v1/models/`**，tools 只做 IO 与编排，不允许第二份 encoder。
3. **归档不删除**：`tools/archive/README.md` 记录每个文件的原始角色、继任者、归档日期；等价性/事故证据链引 md 进度文件。

## 6. 里程碑

```
M1  day20 全量 yarn ingest 首跑（命令就绪，用户执行）→ 通过后 7 天全提
M2  fetch_processed + profiles 全量跑通 → check_k 审计 → 归档 5.1 清单落地
M3  build_pretrain_corpus + 1,000 样本 overfit smoke（M3 起组件②动工）
M4  stage① 正式预训练（抽样语料）→ CLS 活性达标
M5  stage② 冻结 probe 判读 → 微调 → 对照 V1 基线出两段式判决
```
