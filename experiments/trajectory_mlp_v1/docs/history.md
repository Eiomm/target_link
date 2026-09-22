# 历史开发与运行说明

以下保留原 README 的详细记录，机器路径已替换为占位路径。历史测试数、数据可用状态、集群配置与运行指标属于当时记录，不代表本次复验或正式实验结论；当前安装与快速开始以[项目首页](../README.md)为准。所有命令仍从仓库根目录执行。本地图册、权重和真实样例不随仓库发布。

说明：reader 的 x 第三列保留为零，模型从独立 bin_valid 构造第三个输入特征；下文三通道说法指模型输入。原“实验.md”曾混入 Agent 工作规范，现已整理为实验说明，旧 A/B/C 引用不再适用。

---

# trajectory_mlp_v1：独立的整轨迹 MLP 实验

本目录是文件夹形式的实验分支，未创建/切换 Git branch。原 `target_link_v1/`、`tools/`、`tests/` 保持原样。

- [实验设计](design.md#实验设计)：新方法 Ours、可见轨迹算术均值 baseline、完整实验步骤与指标，供审阅。
- [技术协议](design.md#技术协议)：实现时使用的参数、数据划分与验收细则；建议先读主文中的 A/B/C 数字例子。
- 本地真实数据图册（`data_atlas/index.html`，不随仓库发布）与本地中文统计说明（`data_atlas/统计说明.md`）：5 个训练日随机抽取 1,280 个完整 cell、14,449 条轨迹记录，52 张真实图、7 张拼图。40 张随机展示和 12 张定向案例分开标注；统计使用全部 1,280 个 cell。
- `tools/plot_link_bin_times.py`：已实现的独立绘图脚本，复制原版本后局部修改，无需新依赖。
- `tests/test_plot_link_bin_times.py`：空间裁剪、短道路、缺失和CLI回归。
- `examples/`：真实参考cell与明确标注SYNTHETIC的短link/错位起点样图。

模型、重新分组 reader、raw-mean baseline、raw-MAE 训练与配对评测已实现。不使用 observed；输入固定50×3=[T_clean,ratio,valid]，无占位通道；无效耗时置零，ratio保留。上游 valid 插值标签直接使用，不重复插值。当前执行了训练子集 smoke，未进行正式三seed训练或23日真实验证。

## 两个时间编码版本（已知 ratio，MAE decoder v4）

CLS 按 [官方图像 MAE](https://github.com/facebookresearch/mae/blob/main/models_mae.py)
的样本级方式处理：一个 group 对应一张图像，一条轨迹对应一个 patch token。
模型只有一份 `[1,1,d]` 的可学习 `cls_token`，每次前向广播到 B 个 group；
各组独立执行 attention，输出各自的 `h_CLS`，形状为 `[B,d]`，不会跨组聚合。
各组重建损失的梯度汇总到同一份 CLS 参数；`h_CLS` 不会写回该参数或传递到下一批。
Decoder 使用投影后的 CLS、可见轨迹状态和恢复原槽位的 mask tokens，而非仅依赖 CLS。
这是 CLS/token 流程的对应关系；时间编码、已知 ratio 条件及 raw-seconds 损失仍是轨迹任务适配。
显式广播不改变参数形状、数量或 v4 checkpoint 兼容性。

- `--time-encoding seconds`（默认）：保留秒级连续时间，使用原共享时间MLP，默认5,003,058参数。
- `--time-encoding bucket30`：每30秒一个位置，共20个可学习向量，默认4,941,874参数。区间为[0,30)、[30,60)…[570,600]；真实时间越界报错。
- 两版均为4层encoder、2层decoder、256维、8 heads；`--decoder-layers`可调，构造模型时强制参数不超过10M。
- decoder输入为投影后的CLS、可见traj输出和恢复原slot的mask tokens；隐藏traj加已知ratio的Linear(50,256)投影，所有traj加时间编码后做self-attention，每条traj输出50个bin。没有逐bin位置query。
- 可见traj使用[T_clean,ratio,valid]：仅无效T清零，ratio不受valid影响。隐藏traj只隐藏T和bin_valid，已知ratio通过轻量线性层成为decoder条件；不预测ratio，不改变raw-seconds损失。真实traj的ratio须有限且非负，NaN/Inf报错；padding中的NaN不读取。
- 同组同30秒桶且ratio向量相同的隐藏traj在eval时得到相同预测，这是分桶的已知限制。桶不是轨迹ID，也不会合并轨迹。
- 本对照同时改变了时间分辨率和编码器形式（MLP vs embedding表），不能把效果差异全部归因于30秒分桶。
- 两版使用不同out目录，其余data seed、模型seed、分组和训练参数保持一致。分别汇总三seed，禁止混合时间编码版本求seed均值。
- 集群入口支持`TIME_ENCODING=seconds`或`TIME_ENCODING=bucket30`，自动batch探测使用同一版本。
- 新checkpoint格式为`trajectory_mlp_mae_known_ratio_v4`，与旧CLS-only及v3模型不兼容；需要重新训练。下文旧smoke/真实数据说明不构成v4效果验证。

在下方相同训练命令中分别添加`--time-encoding seconds`、`--time-encoding bucket30`并指定不同`--out`即可。

## 训练与评测入口

当前原始数据可直接在已分配的GPU节点上启动，不需要等待固定索引预处理：

```bash
bash scripts/train_traj_mae_raw.sh seconds
# 另一组独立实验（建议上一组结束后再启动）
bash scripts/train_traj_mae_raw.sh bucket30
# 仅检查七天共896个分区的目录及Parquet存在性，不启动训练
DRY_RUN=1 bash scripts/train_traj_mae_raw.sh seconds
```

该脚本明确使用`runtime/cell_mlp_train`和`runtime/cell_mlp_validation_20260823`，
每轮仍会排序和构建固定分组。默认batch=512、workers=0、epochs=10、seed=42，
可通过同名大写环境变量覆盖。脚本本身不提交YARN任务，也不自动后台运行。
输出目录自动命名为`runtime/traj_mae_v4_raw_<版本>_seed<seed>_<时间>_<pid>`；
`console.log`记录日志，`artifacts/best.pt`与`artifacts/last.pt`保存模型。
启动后先计算验证集baseline，再开始训练；目录预检不代替全量内容校验或GPU内存验证。

- `model.py`：整轨迹MLP→时间→轨迹Transformer→可见tokens+CLS+mask tokens的MAE decoder，输出非负秒数。
- `data.py`：只读 observations_v2，按cell重建64条group，尾组<3丢弃；奇数mask向下取整。
- `evaluation.py`：raw MAE损失、可见同bin均值及回退、配对指标/分层/cell bootstrap/行级输出。
- `run.py`：train、baseline、evaluate、smoke；默认17–22训练、23验证。没有sealed-test流程。
- `tools/summarize_runs.py`：核对同一验证集合与训练协议，汇总至少三个模型seed。

本机可用解释器是 `/path/to/env/bin/python`，下列命令中的 python 指该环境。run.py 当前预检本地文件系统；HDFS语料需先准备本地路径。输出目录必须不存在，避免覆盖已有结果。

```bash
# 17–22日已合并至 cell_mlp_train；training-only smoke不需要23
python experiments/trajectory_mlp_v1/run.py smoke --data \
  runtime/cell_mlp_train \
  --out experiments/trajectory_mlp_v1/runs/my_smoke \
  --device cpu --batch-size 8 --smoke-groups 128 --smoke-steps 20

# 正式训练：23日独立作为验证集；使用前确认该目录 validation_report.json 为 passed
python experiments/trajectory_mlp_v1/run.py train --data \
  runtime/cell_mlp_train \
  --val-data runtime/cell_mlp_validation_20260823 \
  --out experiments/trajectory_mlp_v1/runs/m64_seed42 \
  --device cuda --m-max 64 --seed 42
```

依次将seed换为43、44并使用不同out目录。默认完整遍历所选日期，不再隐含4096样本上限。`--max-groups`/`--groups-per-partition`是显式诊断采样，须在报告中注明；max-groups要求workers0。分组规模消融使用`--m-max 16/32/64`，每种配置单独固定分组/验证mask并重跑baseline，不跨不同group配置混用配对CI。

`baseline`模式使用相同data/val-data与分组参数，直接输出验证baseline。`evaluate`还需要`--checkpoint <best.pt>`；分组上限、data seed、验证日期必须与checkpoint一致。

```bash
python experiments/trajectory_mlp_v1/tools/summarize_runs.py \
  experiments/trajectory_mlp_v1/runs/m64_seed42 \
  experiments/trajectory_mlp_v1/runs/m64_seed43 \
  experiments/trajectory_mlp_v1/runs/m64_seed44 \
  --out experiments/trajectory_mlp_v1/runs/m64_summary.json
```

训练产物：config/data_manifest、baseline、每epoch指标、last/best checkpoint、best_val_metrics与逐bin预测/压缩mask身份JSONL。数据manifest是路径/大小/修改时间指纹，不冒充全量parquet内容hash。分组QC按day/bucket去重，是有产出分区的完整候选组统计；实际yield的group数另报，抽样时两者不同。

**23日 HDFS v2 已下载至 `runtime/cell_mlp_validation_20260823`，仅用于验证。** 内容验收结果见该目录的 `validation_report.json`（应为 `status: passed`）；下载清单、本地 SHA256 和逐桶检查日志保存在同目录。 17–22已用作训练，旧的22日验证成绩不适用于本版。GPU在当前环境不可用，本轮smoke使用CPU；没有正式效果或泛化结论。

## 批量随机抽样与统计图

新增 `tools/sample_cell_atlas.py`：每日随机 4/128 个分桶，每桶随机 64 个 K≥3 的完整 cell，保存全部轨迹、来源 manifest、逐 cell/逐轨迹 CSV、抽样权重和近似置信区间。只读取 17–21 日训练集。统计限定于这一抽样总体，不是对新模型性能的显著性检验。

```bash
python experiments/trajectory_mlp_v1/tools/sample_cell_atlas.py
# 从已保存的小型样本重新计算统计、绘图和生成中文说明
python experiments/trajectory_mlp_v1/tools/sample_cell_atlas.py --stage report
```

批量脚本使用现有 numpy、pyarrow、matplotlib 环境。PNG 热力图统一 0–5 秒色标，并增加直接 GPS/有效耗时的状态图；SVG 保留每图 P5–P95 自适应色标。详见图册说明。

本次抽样发现约 63.5% 的有效 bin 没有直接 GPS，因此 `observed=0` 不等于耗时缺失；不能据此重复插值。道路几何未知时图中显示记录覆盖上界，不能把短覆盖直接称为短道路。

## 绘图变化

1. 默认`--x-range coverage`：从segment原点0画到全部选中轨迹中最远的**有piece记录的bin**，包括耗时未知的记录；不是取最长轨迹长度，不按每条轨迹左移、压紧或拉伸。范围在300行显示上限之前计算。
2. 原始parquet有一致的`L_link_m`时，按既有segment规则计算`min(500, L_link_m - 500*seg_idx)`。可信长度可裁掉末端partial bin的道路外部分；坐标与长度冲突时明确报错，不静默丢数据。
3. `--x-range geometry`：显示该segment完整物理范围；若无可信几何，则报错并要求`--segment-length-m`。corpus当前未携带几何长度，可显式传入。
4. `--x-range grid`：保留0–500m检查视图；已知道路范围外用单独灰色标记。
5. 灰色表示“无piece记录，覆盖原因未知”；斜纹表示“有记录但耗时无效”；道路外与这两者分开。不能仅凭一行的首尾空白推断真实未经过。
6. 中位数折线和IQR带都在无有效数据的空间列处断开，不跨缺口连线。原始50-bin统计仍保存在summary中。

`coverage`未知几何时只保证10m网格上界，不声称精确piece终点。图上纵向等间距的是轨迹，按进入时间排序（早下晚上），不是连续时间轴。各图颜色默认各自P5–P95，不宜跨图仅凭颜色强弱比较绝对速度；实际范围保存在summary与图例。

## 使用

在仓库根目录运行，`python`需带numpy；读取本地parquet还需pyarrow（复用现有环境即可）。所有命令只向新目录或指定输出写图，不改输入。

```bash
# 真实本地参考数据；raw配色与实验的原始秒数标签一致
python experiments/trajectory_mlp_v1/tools/plot_link_bin_times.py \
  --raw-parquet data/ref_link_125121851/samples_flat.parquet \
  --metric raw \
  --out experiments/trajectory_mlp_v1/examples/real_cell.svg

# 选择segment，显示实际几何范围（原始数据必须有可信长度）
python experiments/trajectory_mlp_v1/tools/plot_link_bin_times.py \
  --raw-parquet data/ref_link_125121851/samples_flat.parquet \
  --seg-idx 1 --x-range geometry --metric raw \
  --out experiments/trajectory_mlp_v1/examples/real_segment_1.svg

# 生成可复现的合成示意：50m短link、较短轨迹却终点更远的170m覆盖
python experiments/trajectory_mlp_v1/tools/make_plot_examples.py

# 新旧绘图测试（两个测试文件同名，显式使用importlib收集）
python -m pytest -q --import-mode=importlib \
  experiments/trajectory_mlp_v1/tests/test_plot_link_bin_times.py \
  tests/test_plot_link_bin_times.py
```

服务器corpus入口保持原版参数，可用Spark运行新路径：

```bash
spark-submit experiments/trajectory_mlp_v1/tools/plot_link_bin_times.py \
  --corpus hdfs:///path/to/corpus_v1 \
  --link <实际link_id> --seg-idx 0 --metric raw --x-range coverage \
  --out <输出文件.svg>
```

如果已知当前segment只有50m，可增加`--segment-length-m 50`；该参数指当前segment长度，不能给第二个segment传入整条link长度。`--metric equivalent-10m`保持原版显示能力，表示`sum(T_diff)/sum(ratio)`，不得与本实验raw标签的指标混为一谈。

## 本次验证

- 当前独立模型、reader、指标、CLI集成、种子汇总与绘图相关测试共 **52 passed**。
- 合成parquet集成验证覆盖 train→best checkpoint→独立evaluate→baseline，验证mask与评分分母一致；合成23日仅是接口测试，不冒充真实23日数据。
- 真实训练数据 smoke：默认d256、4层、M_max64，固定128个group、20steps，CPU batch8；检查梯度、checkpoint重载、隐藏内容隔离。结果保存在 `runs/smoke_raw_mae_m64_seed42_verified/`。
- 最初实现验收未执行三seed全量训练，也未读取真实23日数据。后续23日下载和读取验收见上述 validation_report.json；尚未验证GPU/DDP；当前入口单进程训练（数据加载可用workers）。
- 复现全部本次测试：`python -m pytest -q --import-mode=importlib experiments/trajectory_mlp_v1/tests tests/test_plot_link_bin_times.py`。

以下是此前绘图阶段的验证记录：

- 新版空间/CLI测试与原版绘图测试：22 passed。
- 加上随机抽样、统计权重、缺口定义与零事件区间测试：共 28 passed。
- 真实参考数据本地CLI读取、出SVG与summary：通过；该样例seg0覆盖500m，不能用它单独证明短link适配。
- 合成50m短link、偏移起点170m示意：单独生成，标题明确标注SYNTHETIC，不冒充实验结果。
- HDFS/Spark路径未连接集群执行；读取逻辑沿用原版，新的presence与范围处理在共享绘图路径已测试。

原版绘图源文件SHA256（创建分支时，便于确认未改动）：

```text
tools/plot_link_bin_times.py
d026c527763ff33346424e867b02f31bf0508f07a5fc24bb847206bef5b6951e
tests/test_plot_link_bin_times.py
d4541cbd7ffd8a51dfc883ca622fa06540869b7a5c967ddb8eeb9482ba4c56f4
```

## L20 45G 单卡集群入口

在集群平台选择 **1 张 L20**，挂载同一 NFS，平台启动命令：

```bash
bash scripts/submit_cell_mlp_job.sh
```

该脚本在平台分配的 GPU 节点运行，不自行申请集群资源。默认读取 `runtime/cell_mlp_train`（17～22日）及 `runtime/cell_mlp_validation_20260823`（23日），完整数据、10 epochs、seed42、M=64、4个数据加载worker、FP32。启动时检查7天共896个分桶均存在非空文件列表；这是路径预检，不替代内容验收。可先 `DRY_RUN=1 bash scripts/submit_cell_mlp_job.sh` 检查路径，不要求GPU。

默认 `BATCH_SIZE=auto`：在目标GPU独立子进程中依次尝试64/128/256/384/512/768/1024组。使用实际默认模型、每组64条轨迹、50% mask及50个有效bin，做3次前向、反向、梯度裁剪和AdamW更新。以启动时可用显存的85%为预算，选择候选中通过的最大batch；发生CUDA OOM即退回上一个通过档位，其他错误直接终止。结果保存 `batch_probe.json`。这不是L20上的预先实测结论，也不是吞吐最优搜索；如果1024仍通过，当前搜索以1024为上限。保留余量应对运行时波动，不承诺绝不OOM。使用独占GPU；其他作业运行中占用显存会影响结果。

batch单位是组，不是轨迹；例如256组最多包含16384条轨迹。decoder会计算全部64×50位置，大batch增加激活显存；显存接近满载并不保证GPU计算利用率最高，数据读取/重分组也可能成为瓶颈。先运行auto获得实际容量和吞吐，再固定batch用于各seed对比；batch变化会改变每epoch优化步数，不能视为完全相同训练设置。脚本保持原学习率2e-4，不自动按batch放大。

手动覆盖示例（256是待实机验证的起始尝试，不是已测上限）：

```bash
BATCH_SIZE=256 WORKERS=4 SEED=42 bash scripts/submit_cell_mlp_job.sh
# 后续seed固定相同batch，输出目录默认自动区分
BATCH_SIZE=256 SEED=43 bash scripts/submit_cell_mlp_job.sh
```

支持覆盖 `ENV_ROOT`、`PYTHON`、`DATA`、`VAL_DATA`、`OUT`、`BATCH_SIZE`、`WORKERS`、`EPOCHS`、`SEED`、`M_MAX`。固定默认模型维度256、4层、8头；修改架构或精度时必须同步修改显存探测配置。现阶段单卡，不使用torchrun/DDP。

输出默认在 `runtime/cell_mlp_l20_<日期时间>_<PID>/`，已存在目录会拒绝覆盖：

- `console.log`：终端stdout/stderr完整副本，含报错；每60秒GPU心跳。
- `batch_probe.json`：auto模式的实测容量、候选耗时和最终batch。
- `command.sh`：本次完整训练命令；`exit_code.txt`：退出状态。
- `artifacts/`：config、manifest、baseline、逐epoch指标、best/last checkpoint及最终验证结果。

日志顺序：PREFLIGHT → BATCH PROBE → BASELINE → TRAIN/VALIDATION → EPOCH COMPLETE → DONE。baseline先完整遍历23日，随后才开始训练；首次加载分区可能较慢，心跳仅代表进程仍在运行。TRAIN打印当前与累计MAE（秒）、组数、吞吐和峰值显存；BASELINE/VALIDATION打印已处理组数、分桶数和耗时。epoch指标JSON保留原有0基索引，终端epoch使用1基显示。完整数据训练和最终逐bin输出可能耗时、占空间，脚本不暗中采样。

### 收敛曲线

每个epoch验证结束后自动更新 `artifacts/loss_curve.png`、`loss_curve.svg` 和 `loss_curve.csv`，无需等待所有epoch结束。图中包含训练MAE、验证MAE、可见轨迹均值baseline虚线和最佳验证epoch标记，单位均为秒。图片采用无界面Agg后端，适合集群；通过临时文件替换，读取时不会看到写到一半的图片。

训练线是epoch内随参数更新累计的有效bin加权MAE，验证线是epoch结束时固定mask上的MAE，两者不是同一数据集或同一计算时刻。训练/验证一起下降并逐渐趋稳说明趋于收敛；训练继续下降而验证持续上升提示过拟合。只有少数epoch或单次波动不足以下结论。当前保存epoch级曲线，不是每个batch一张图。

也可从已保存指标重新绘图：

```bash
python experiments/trajectory_mlp_v1/tools/plot_loss.py --out /实际运行目录/artifacts
```

### 阶段剩余时间 ETA

BASELINE、TRAIN、VALIDATION 的周期日志包含 `elapsed=HH:MM:SS`、`eta=HH:MM:SS` 和 `partitions_completed=已完成/总数`。ETA仅估计当前数据遍历阶段，不是整个10轮任务的倒计时，不包含之后的指标汇总、bootstrap、checkpoint保存与绘图。

首遍至少看到4个分桶（总数不足4时使用实际总数），并且至少完成一个分桶后，根据已见分桶的真实selected_groups均值估计总组数，再用主进程实际已处理组数/累计耗时估算剩余时间。此前显示 `eta=estimating`。`partitions_seen`仍表示接触过的分桶；`partitions_completed`按主进程实际消费的组数判断，不把worker预取算成完成。没有可用组的空分桶不会产生批次，因此完成计数可能小于总分桶数，以阶段complete日志为准。

baseline完成后，后续验证复用其实际总组数（`eta_basis=previous_full_pass`）；训练从第二轮起复用上一轮总组数。计时仍使用当前阶段速度，避免把均值baseline的速度用于模型验证。分桶大小、数据加载、GPU速度波动会使ETA变化，首遍显示 `eta_basis=sampled_partition_sizes` 提醒它依赖抽样估计。指标汇总阶段显示 `eta=unknown_for_aggregation`，不伪装为零秒完成。

代码更新对已启动的Python进程不热生效，下一次提交生效；不需要为显示ETA中断已有训练。

### 评测加速与中文阶段日志（2026-09-21）

评测主统计及分层统计改为NumPy批量float64归约，移除逐组/逐轨迹重复小张量运算。baseline直接使用CPU批次；模型验证仅将推理输入送到GPU，并将预测结果按批取回CPU。保留全部验证数据、同一mask与全部分层指标；valid_bin_length层仍以每条隐藏轨迹作为原有统计单元，不偷换group-balanced分母。归约顺序改变可能带来浮点末位差异。

终端有中文分隔标记：启动整轨迹MLP实验 → 探测GPU安全batch size → 数据与配置检查 → 计算均值基线 → 模型训练 → 验证模型 → 保存模型与指标 → 本轮训练完成 → 全部完成。英文阶段代码继续保留，方便搜索。ETA、log、loss曲线继续保留。

新旧指标一致性测试包含M=5/50/64、无可见轨迹、无有效标签、padding、无效位置NaN、同组重复长度分层、baseline与模型预测；同时核对bootstrap、mask标识和CSV内容。旧实现仅保留在tests/_evaluation_reference.py作为对照，不参与训练。

该优化不等于消除所有开销：数据加载、group构造、最终逐bin CSV导出和bootstrap仍有成本。CPU单批速度不能直接当成L20端到端提升；实际以集群新任务吞吐为准。运行中的旧Python进程不会热更新，不自动中断旧任务。

## 最终排序语料（runtime/final）

构建入口（默认单进程，不能按宿主机 `free -h` 的容量增加并发）：

```bash
python experiments/trajectory_mlp_v1/tools/prepare_final.py --out runtime/final --workers 1
```

目录为 `train/observations_v2`、`val/observations_v2`，对应17～22日及23日；各自的 `group_indices` 保存固定成员索引，`receipts` 保存逐分区源文件与输出文件校验值。默认索引协议为M_MAX=64、data_seed=20260921。其他M_MAX或data_seed必须重新生成对应版本，读取器拒绝误用。

每个分区按cell_id/sample_id排序，保留源表全部列和记录，重新读取核对写出值。索引记录候选行映射、group成员行号、K/K_raw和组序号；验证成员不重复、不跨cell、丢弃量记账一致，并用实际读取器对照分区首/中/末group的输入与身份。训练加载最终语料时直接恢复索引，不执行旧版排序和分组逻辑；每轮mask与group遍历顺序仍随机。文件内容SHA256是完整性依据，NFS时间戳仅作为记录；每次载入分区会核对数据与索引checksum，因此仍有读取和校验开销。

只有 `runtime/final/_SUCCESS.json` 及train/val内 `_FINAL_SUCCESS.json` 发布、且没有 `_BUILDING`，才是完成的数据集。构建期间不要交给训练。完成后可用：

```bash
DATA="$PWD/runtime/final/train" VAL_DATA="$PWD/runtime/final/val" \
  bash scripts/submit_cell_mlp_job.sh
```

构建会读取cgroup实际限额及占用，默认1进程，保守估计每进程8GiB并保留余量；每0.5秒检查容器内存，达到80%时终止本次构建的worker。当前容器限额30GiB，不是宿主机显示的1TiB。保护不能消除其他进程突然分配内存带来的风险。失败会保留已验收分区，使用同一配置重跑时核对源/目标hash后复用，未验收分区重新生成；不发布半成品。

## 已知覆盖比例的解码契约

设 R_k 是第k条轨迹的50维ratio向量，G(R_k)=Linear(50,256)(R_k)。

- 可见：d_k=P(h_k)+p(delta_t_k)，ratio已在encoder内容向量中。
- 隐藏：d_k=mask_token+p(delta_t_k)+G(R_k)。
- CLS：d_CLS=P(h_CLS)，不加轨迹时间或ratio。
- 将以上tokens按序拼接，进入decoder self-attention；不将CLS与traj逐元素相加。
- 隐藏T和bin_valid只参与目标/监督位置选择；修改它们不改变预测。修改隐藏ratio允许改变预测，但不能改变encoder CLS。
- 时间无效与几何未知不同：valid=0只清零T，已知ratio保留；无记录bin沿用数据读取层的ratio=0。
- ratio是条件，不强制耗时按ratio线性缩放，也不保证单调性。原raw同bin均值baseline不使用目标ratio，比较时应披露模型可使用额外的已知几何条件。
- 旧结构图中的第四个零通道、invalid ratio清零、隐藏ratio不进入decoder等标注已过时，以本节及model.py为准。

## 训练进度与趋势（2026-09-22）

启动脚本默认batch=512、epochs=10。已分配GPU节点上运行即可；512未在本机完成真实GPU显存验收。
训练环境qwen12已安装tqdm 4.67.1。均值基线、训练、验证均显示tqdm进度；首轮原始数据
总group数未知时显示已处理数量及速度，不伪造百分比，第二轮可按上轮实际group数显示总进度。
独立中文日志保留分区完成数与估计剩余时间，便于查看console.log。重定向日志可能包含进度条回车字符。

`artifacts/`内的文件：

| 文件 | 内容 / 更新频率 |
|---|---|
| step_metrics.csv | 首批、每20批及绘图时记录batch MAE、累计MAE、lr、裁剪前梯度范数、吞吐量和显存峰值 |
| step_trends.png | 首批、每1000批、每轮结束更新批次趋势；是采样曲线，累计MAE每轮重新统计 |
| loss_curve.png / svg / csv | 每轮训练MAE、固定mask验证MAE、基线与最优轮次 |
| epoch_diagnostics.csv | 每轮学习率、梯度范数均值/最大值、参数L2范数、验证RMSE、吞吐量和训练显存峰值 |
| training_diagnostics.png | 上述轮次指标的六宫格趋势图 |

直接运行run.py时可用`--log-every`和`--plot-every`调整刷新频率。
学习率仍固定2e-4，梯度仍按范数1.0裁剪；图中的梯度范数为裁剪前值，未增加学习率调度。
参数范数只说明权重整体尺度，不能单独用于判断效果；以固定验证集MAE及RMSE为主要效果指标。
显存为PyTorch训练阶段峰值allocated值，不等于nvidia-smi总占用；CPU运行显示0。

## 训练张量语料 v1（2026-09-22）

`trajectory_mlp_tensors_v1` 在数据准备阶段完成 piece→bin 聚合、有效性检查、
固定成员分组、尾组处理和 padding。保留基础 `observations_v2`，输出到独立目录。
默认 `m_max=64`、`data_seed=20260921`；它与模型初始化用的 `--seed` 不同。

每个 group 已保存 `x[64,50,3]`（float32 秒数、覆盖比例、零通道）、
`bin_valid[64,50]`、`traj_valid[64]`、`delta_t[64]`，以及真实成员 sample_id、
cell_id、组序号、K/K_raw/group_size。耗时保持原始秒数，模型内部的变换保持不变。
训练只读取并解压选中的 group、打乱顺序、堆叠 batch、按 epoch 生成 MAE mask。
验证继续使用原有固定 epoch 和 mask 协议。旧语料和旧索引读取路径仍兼容。

存储协议 `zlib-group-blocks-v1`：每个分区一个 `groups.bin`，每个 group 独立压缩；
`offsets.npy` 可直接定位组，`meta.npy` 保存组元数据。数组布局记录在收据内，
不使用 pickle。补零仍存在于解压后的张量中，但压缩后几乎不占空间。
读取一个 group 不需要解压其他 group；索引使用内存映射。

完整构建（先验证集，再训练集）：

```bash
PYTHON_BIN=/path/to/python bash scripts/prepare_training_tensors_v1.sh
```

输出默认是 `runtime/training_tensors_v1_m64_seed20260921/{train,val}`。
可设置 `TENSOR_OUT`、`DATA_SEED` 和 `BUILD_WORKERS=1|2`。两进程模式检查 cgroup
内存余量，并在占用超过限额 85% 时停止本次 worker；不会按宿主机内存推算容量。
单进程模式一次展开一个源分区；内存仍随该分区的 observation 和 group 索引量增长。

单独构建某些日期或独立校验：

```bash
python experiments/trajectory_mlp_v1/tools/prepare_tensors.py \
  --source runtime/cell_mlp_validation_20260823 --out runtime/my_tensors_v1 \
  --days 20260823 --workers 2
python experiments/trajectory_mlp_v1/tools/prepare_tensors.py \
  --out runtime/my_tensors_v1 --verify-only
```

每分区完成后记录源文件和输出的 SHA256，重读首/中/末 group 核对数值、成员与 padding。
合成测试覆盖所有 group 的新旧 batch 等价及真实 CPU train/evaluate 调用。
恢复时校验已完成分区的源/输出，复用通过的分区；未完成分区重新写。
协议、来源、日期或准备代码版本变化时要求新目录。文件锁防止并发构建同一个输出。
只有 `_TENSORS_SUCCESS.json` 存在且 `_TENSORS_BUILDING` 不存在时才允许训练。
训练检查尺寸、数组 schema 与组索引；不重复执行全量内容哈希。迁移数据后可使用
`--verify-only` 做完整 SHA256 校验，检测同尺寸内容变化。

两个 split 均发布完成后，训练启动方式：

```bash
DATA="$PWD/runtime/training_tensors_v1_m64_seed20260921/train" \
VAL_DATA="$PWD/runtime/training_tensors_v1_m64_seed20260921/val" \
  bash scripts/submit_cell_mlp_job.sh
```

张量文件不是独立测试集，也不改变已有训练/验证日期划分。训练吞吐收益应以实际
模型任务测量，不能从离线转换耗时推算 GPU 加速比例。

### 在 YARN 转换（2026-09-22 启动流程修订）

提交节点只打包代码并提交任务。Driver 先创建 SparkSession，然后通过 Spark JVM
里的 Hadoop FileSystem 访问 HDFS；不在 Spark 初始化之前启动 `hdfs` 子进程。
执行节点默认使用 Spark 分发的 Java/Hadoop jars、配置及容器凭据进行文件传输，
不要求节点 PATH 中有 `hdfs`。`HDFS_BIN` 仅保留为明确指定的兼容覆盖选项。

分开配置两套 Python，避免把转换环境的 Python 版本误用于 Spark：

- Spark Driver 和 Python worker 共用调度解释器。默认仍为现有 `minipy3` Python 3.7，
  对应旧提交机的 Spark 3.2。更高版本 Spark 可通过 `SPARK_PYTHON_ARCHIVE` 与
  `SPARK_PYTHON_REL` 指定匹配的调度环境。
- 张量转换作为子进程运行，需要 Python >=3.9、NumPy、PyArrow、PyTorch，无需 GPU。
  提交时必须明确提供 `TENSOR_ENV_ARCHIVE`，或明确设置执行节点可访问的 `TENSOR_PYTHON`。
  不再默认认为提交机的 NFS Python 路径一定存在于 YARN 容器中。

推荐用可重定位的环境包，例如设置 `TENSOR_ENV_ARCHIVE=hdfs://.../tensor_env.tar.gz`。
默认解包后转换解释器为 `./tensor_env/bin/python`，可用 `TENSOR_ENV_PYTHON` 修改。
这份环境包仅用于转换，不自动替换 Spark 调度解释器。代码 zip 同时作为 py-files
和解压目录 `tensor_code` 分发，子进程从解压后的代码运行。

四种模式：

```bash
# 本机检查：仅检查 Spark 启动器，不代表 YARN 容器环境通过
MODE=check bash scripts/submit_training_tensors_yarn.sh

# 打印命令，不提交任务
MODE=dry bash scripts/submit_training_tensors_yarn.sh

# 真正提交到 YARN，只检查一个执行节点：Python版本、依赖、源分区读取
# 必须先设置真实的 TENSOR_ENV_ARCHIVE，或显式的 TENSOR_PYTHON。
MODE=preflight bash scripts/submit_training_tensors_yarn.sh

# 使用相同环境配置正式转换
MODE=yarn bash scripts/submit_training_tensors_yarn.sh
```

`MODE=preflight` 不转换 observation，不创建输出或锁文件；日志出现
`[TENSOR_PREFLIGHT_PASSED]` 后才说明实际执行节点的检查通过。它不是全量数据验收，
也不能证明所有节点环境相同；正式任务仍会在每个转换进程上检查依赖。

默认输入 `hdfs:///path/to/corpus_v1/observations_v2`，
输出同级 `training_tensors_v1_m64_seed20260921/{train,val}`。支持 `SOURCE`、`OUT_DIR`、
`QUEUE`、`PARALLELISM`、`TRAIN_DAYS`、`VAL_DAYS`、`BUCKETS`、`M_MAX`、`DATA_SEED` 覆盖。
输入须在转换期间保持不变，输出必须使用新目录。默认训练17–22日，验证23日。

每个 day/bucket 是一个 Spark task。正式转换默认20个executor，每个1核，JVM内存2g，
额外内存12GiB用于Python子进程。每个节点仅下载和转换自己的分区，使用节点临时磁盘，
不会将全量数据下载到提交机器。失败重试使用独立 attempt 路径，避免覆盖或重复成员。

Driver 对输出加独占锁；只有全部896个分区成功，才生成 split 的 `_TENSORS_SUCCESS.json`，
并通过 Hadoop FileContext 的不覆盖重命名发布整套数据，根目录带 `_SUCCESS.json`。
报错会输出 `[TENSOR_PREPARE_FAILED] phase=...`；若临时输出目录已经建立，还会保存
`failure.json`。清理或停止 Spark 的错误不会掩盖原始错误。Driver异常强杀仍可能遗留锁，
应确认相关application结束后再处理；程序不会自动删除其他任务的锁。
当前支持Spark task重试，不支持跨application断点续跑。

HDFS输出沿用同一个张量版本和训练读取协议，文件位于 `_parts/.../attempt=.../`，
manifest记录其相对路径。训练前下载整个发布目录到本地或共享NFS，无需再次转换：

```bash
hdfs dfs -get \
  hdfs:///path/to/training_tensors_v1_m64_seed20260921 \
  /path/to/fresh/local_tensor_copy

DATA=/path/to/fresh/local_tensor_copy/train \
VAL_DATA=/path/to/fresh/local_tensor_copy/val \
  bash scripts/submit_cell_mlp_job.sh
```

复制后可对train/val分别执行 `prepare_tensors.py --out ... --verify-only` 检查内容。
测试包含新旧batch等价、失败发布保护、重试隔离、模拟提交、Spark先初始化、预检不写数据，
并用本机实际Spark JVM验证文件系统读写、独占锁、不覆盖发布和Java FsShell传输。
本机JVM验证不等于目标YARN集群验证；实际环境须先运行MODE=preflight。


---

# GitHub 发布准备

本实验保留在当前仓库的 `experiments/trajectory_mlp_v1/`，所有命令从仓库根目录执行。

## 提交范围

- 包含：模型、reader、训练/评估入口、辅助工具、测试、依赖清单、实验说明及合成示意图。
- 排除：真实语料、真实轨迹样例、图册、训练权重、运行结果、统计数据库、spill 文件、缓存和编辑器交换文件。文件仍保留在本地，规则位于本实验的 `.gitignore`。
- 本实验当前约 39 GB 的目录中，待提交文件约 0.5 MB。不要使用强制添加来绕过忽略规则，也不要直接打包整个实验目录上传。
- 历史说明保存在 `docs/history.md`；其中测试数量、运行数字及集群环境属于历史记录，不能替代本次检查或正式效果评估。

仅检查和暂存此实验，避免将仓库中其他既有改动一并带入：

```bash
git status --short --untracked-files=all -- experiments/trajectory_mlp_v1
git add -- experiments/trajectory_mlp_v1
git diff --cached --stat -- experiments/trajectory_mlp_v1
git diff --cached --name-only
```

最后一条命令检查整个暂存区；其中如果有其他项目文件，应先确认其是否属于本次提交。以上仅为操作说明，本次整理未执行暂存、提交或推送。

## 本次检查（2026-09-22）

- 使用 Linux / Python 3.12.11；依赖版本见两个 requirements 文件，实际 PyTorch 构建为 `2.6.0+cu126`，检查在 CPU 上运行。
- 原工作区：`100 passed`。
- 将 Git 待上传文件复制至独立临时目录，保留 `experiments/trajectory_mlp_v1` 层级、排除本地数据与产物后：`100 passed`。
- README 合成数据生成及两步 CPU smoke 通过；checkpoint 重载、隐藏输入预测和表示差异均为 `0.0`。
- 两幅合成示意图成功重新生成，JSON 使用相对 SVG 文件名，不再记录机器绝对路径。
- 当前环境的 `pip check` 通过。上述测试使用已有环境，未在全新环境联网安装依赖。
- PyTorch 报告 `norm_first=True` 时不启用 nested tensor 的提示；不影响本次测试通过。
- 最终核对发现 `tools/census_seven_days.py` 在测试副本建立后另有并发更新，本次整理保留了该更新。该脚本仅完成语法检查；上面的 100 项通过记录不能作为其新增聚合逻辑的运行验证。

本次没有执行真实数据的正式训练、GPU 性能测试或 YARN/Spark 集群验证。没有新增模型效果结论。


## 早期实现验证快照（过时，仅供追溯）

以下原样保留原 `implementation_validation.json` 内容。其中测试数、四通道描述和验证日状态均不是当前版本结论。

```json
{
  "tests": 52,
  "source_hashes_match": {
    "run.py": true,
    "data.py": true,
    "model.py": true,
    "evaluation.py": true
  },
  "real_smoke": {
    "groups": 128,
    "steps": 20,
    "model_dimension": 256,
    "layers": 4,
    "m_max": 64,
    "device": "CPU",
    "batch_size": 8
  },
  "checkpoint_reload": "exact match",
  "hidden_input_isolation": "exact match",
  "validation_day23": "not locally available; not run",
  "formal_training": "not run",
  "observed_used": false,
  "features": [
    "T_clean",
    "ratio_clean",
    "valid",
    "constant_zero"
  ],
  "loss": "raw_seconds_micro_MAE"
}
```
