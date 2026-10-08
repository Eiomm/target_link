# Target Link

唯一维护主线是 `trajectory_mae/` 的 **v6**。`V4/` 是独立的实验复现归档，主线不导入它，也不接受 v4/v5 权重。

## 安装

Python ≥3.11，推荐使用独立虚拟环境。先安装适合本机 GPU 驱动的 PyTorch 2.6.0，再安装其余依赖；本仓库验证环境为 Python 3.12、PyTorch 2.6.0+cu126。

```bash
git clone https://github.com/Eiomm/target_link.git
cd target_link
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# 运行测试时额外安装
python -m pip install -r requirements-dev.txt
```

仓库包含代码、配置及 v4 实验指标，不包含训练数据、模型权重或本机运行目录。完整训练需自行准备符合 [数据协议](docs/data_contract.md) 的数据；没有真实数据时先执行下面的合成数据测试。

## 最小 GPU 测试

以下命令使用完整 v6 模型，生成 20 条合成轨迹，训练 1 轮、更新 2 次参数，并完成验证、权重保存与最终重载验证。仅检查流程，不用于评估模型精度。输出目录必须不存在；重复执行时更换目录名。

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
python -m trajectory_mae.tools.make_synthetic_corpus --out runtime/synthetic_smoke
python -m trajectory_mae.run train \
  --data runtime/synthetic_smoke \
  --train-days 20260817 --val-days 20260823 \
  --out runtime/v6_gpu_smoke --device cuda \
  --batch-size 1 --workers 0 --threads 1 --epochs 1 \
  --bootstrap 0 --log-every 1 --plot-every 0
```

2026-10-08 已在 RTX A6000 上验证：训练、验证、保存及独立重载评估均通过，重载指标完全一致；测试进程峰值内存 1.57 GiB，实例总内存采样峰值 8.60 GiB（上限 30 GiB），训练及最终验证的 PyTorch 张量显存分配峰值约 170 MiB。测试期间设有 20 GiB 实例内存停止阈值，上述手动命令本身不附带内存监控。详见 [GPU 测试记录](docs/gpu_smoke.json)。

**30 GiB 内存注意事项：** 平台默认 `batch_size=1024`、`workers=4` 尚未完成全量内存验证。首次运行真实数据应从 `batch_size=1`、`workers=0` 开始；读取器仍可能一次加载整个数据分区，`--max-groups` 和 `--smoke-groups` 只限制组数，不能限制分区加载内存。合成小数据测试通过不代表全量训练不会 OOM。

## 运行

平台分配 GPU 后，训练入口只有根目录 `train.sh`；训练参数集中在 `train.toml`。

```bash
# 指定安装好依赖的解释器及真实数据路径
PYTHON="$PWD/.venv/bin/python" \
DATA=/path/to/train VAL_DATA=/path/to/validation \
DRY_RUN=1 bash train.sh

# 检查通过后启动；数据分区本身也必须能装入内存
PYTHON="$PWD/.venv/bin/python" \
DATA=/path/to/train VAL_DATA=/path/to/validation \
bash train.sh --batch-size 1 --workers 0 --threads 1
# 可通过 --time-encoding bucket30 切换时间编码
```

秒级和 30 秒分桶是同一个模型的时间编码参数，不是两份训练实现。默认保持秒级配置：batch=1024、epochs=5、workers=4、seed=42、M=64。平台解释器默认使用已有 qwen12 环境；可通过 `PYTHON` 明确指定另一个解释器。`DATA`、`VAL_DATA`、`OUT` 可显式指定路径，不按磁盘上是否存在其他数据目录自动切换。

默认训练数据为 `data/cell_mlp_train`（20260817–20260822），验证数据为 `data/cell_mlp_validation_20260823`。平台预设要求每天 128 个分区。输出为 `runtime/YYYY-MM-DD/traj_mae_v6_*/`，包含日志、退出码和 `artifacts/` 下的权重、指标与曲线。

直接使用 Python 时，在仓库根目录运行；未传 `--config` 时使用 CLI 的小规模 CPU 默认值。命令行参数覆盖显式加载的 TOML 预设：

```bash
python -m trajectory_mae.run --help
python -m trajectory_mae.run evaluate --data data/cell_mlp_validation_20260823 \
  --checkpoint /path/to/v6/best.pt --out /path/to/new/evaluation --device cpu
python -m trajectory_mae.tools.evaluate_visible_mean \
  --model-run /path/to/v6/artifacts --out /path/to/new/comparison
python -m trajectory_mae.run smoke --data runtime/synthetic_smoke \
  --train-days 20260817 --val-days 20260823 \
  --out runtime/v6_cpu_smoke --smoke-groups 2 --smoke-steps 2 \
  --batch-size 1 --workers 0 --threads 1 --device cpu
```

`evaluate` 输出逐 bin 预测、监督身份及指标；完整验证的明细可能很大。独立均值基线对比工具只导出汇总，核对数据清单和监督身份一致后才计算改善率。训练、每轮验证均只计算模型指标。

## 目录

```text
target_link/
├── train.sh / train.toml       # 唯一平台训练入口与配置
├── trajectory_mae/             # v6 模型、数据、训练和评估
│   ├── tools/                 # 数据准备、统计与绘图
│   ├── data_browser/          # 本地数据浏览器
│   └── tests/
├── tools/                     # 上游数据转换与审计
├── scripts/                   # 数据构建与 YARN 提交
├── tests/                     # 主线契约和集成测试
├── V4/                        # 冻结实验代码与结果，非主线依赖
├── docs/                      # 数据协议、重构和验证记录
└── requirements*.txt          # 训练、开发及可选工具依赖
```

## 调用链与职责

```text
train.sh → train.toml + trajectory_mae.run
                      ├─ cli / manifests / progress / checkpoints
                      ├─ data → grouping → columns
                      │          └─ 原始 piece 折叠、稳定分组、padding、遮挡
                      ├─ prepared / observation_v3 / tensor_corpus（存储适配）
                      ├─ model（唯一 v6）
                      └─ evaluation → 指标与绘图
```

`model.py` 的 encoder 每条轨迹只读取 50 个 `T_clean`；隐藏轨迹的 ratio 只作为 decoder 几何条件。`bin_valid` 仅用于预处理和监督，`traj_valid` 控制真实轨迹与 padding。时间不除以 ratio；训练目标仍为原始秒数的有效隐藏 bin 微平均 MAE。

数据协议、监督口径和已知标签问题见 [docs/data_contract.md](docs/data_contract.md)。重构清单及验证见 [docs/refactor_report.md](docs/refactor_report.md)。

## 数据准备与检查

不同存储格式服务于同一套分组与模型逻辑，数据版本号与模型 v4/v6 无关。保留 observation-v2、已发布的预计算索引、observation-v3 和压缩 tensor 读取；读取失败直接报错，不退回另一份数据。

- 本地 observation → tensor：`python -m trajectory_mae.tools.prepare_tensors --help`
- 上游已分 bin 的 raw Parquet → tensor：`scripts/submit_raw_training_tensors_yarn.sh`，核心实现 `tools/build_raw_training_tensors.py`。
- HDFS observation → tensor：`scripts/submit_training_tensors_yarn.sh`。
- HDFS observation → 紧凑 observation-v3：`scripts/submit_observation_v3_yarn.sh`。
- 边界审计、样本提取和统计：根目录 `tools/`；绘图和语料统计：`trajectory_mae/tools/`。
- 本地数据浏览器：`bash trajectory_mae/data_browser/start.sh`，仅监听本机；依赖已有统计产物 `outputs/trajectory_reports/`。

YARN 脚本默认仅预览命令；完整构建需要可用 Spark/HDFS 环境及明确指定的输出目录。旧 CellMAE 的 raw→旧 groups→旧训练语料发布链已移除；重新构建主线训练数据使用 raw→tensor 路径。上游 GPS→bin 的生产作业不在本仓库。

## 依赖与验证

Python ≥3.11。训练依赖统一在根目录 `requirements.txt`，测试依赖在 `requirements-dev.txt`。可选本地 Spark 检查使用 `requirements-spark.txt`；额外 SQL 普查工具使用 `requirements-analysis.txt`，DuckDB 尚未在当前训练解释器中安装。平台 YARN 使用平台提供的 Spark 环境。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
```

重构时全套回归为 226 项通过、3 项跳过，见 [测试记录](docs/tests_summary.json)。当时隔离环境无法访问 CUDA；随后在实际 GPU 上通过了上述最小训练测试。真实 Spark 集成测试需单独配置环境；新克隆未提供 v4 权重时，对应历史权重测试也会跳过。

测试覆盖两种时间编码、隐藏标签隔离、数据格式一致性、训练/验证/权重重载、归档隔离和发布完整性。`tests` 内的数值参考实现只用于回归对照，不属于运行入口。

## V4 实验归档

已归档的 **0.555 秒 / 6.134 秒 / 15.938 秒** 来自 v4，不能作为 v6 的实验结果。对应代码、原始指标、权重在服务器上的位置及来源校验清单见 [V4/README.md](V4/README.md)。原始 `data/`、`runtime/`、`outputs/` 继续保存实验数据与证据；它们不参与主线代码版本选择。
