# 整轨迹 MLP / MAE 实验

根据同一路段、同一时间窗口内的可见轨迹，预测被遮挡轨迹的耗时。支持秒级时间编码和 30 秒分桶两种版本。

## 当前环境

本项目已有 `qwen12` Python 环境，启动脚本已配置解释器，**不用新建环境、重新安装依赖或手动激活环境**。直接在平台选择本目录中的启动脚本即可；脚本会自动定位仓库根目录，不需要切换工作目录。

| 项目 | 当前配置 |
| --- | --- |
| 训练数据 | `runtime/cell_mlp_train`，20260817–20260822 |
| 验证数据 | `runtime/cell_mlp_validation_20260823`，20260823 |
| 运行位置 | 平台已分配的 GPU 节点，挂载当前仓库和数据所在 NFS |
| 默认训练参数 | batch 512、10 轮、seed 42、每组最多 64 条轨迹、workers 0 |

## 怎么启动

在平台分配好 GPU 后，选择下面其中一个脚本，**只填脚本路径，不加任何参数**：

| 版本 | 启动脚本 |
| --- | --- |
| 秒级时间编码 | [train_seconds.sh](train_seconds.sh) |
| 30 秒分桶编码 | [train_bucket30.sh](train_bucket30.sh) |

脚本位于 `experiments/trajectory_mlp_v1/`。平台若要求绝对路径，填写对应文件的完整路径。

两种版本分别运行、分别保存结果。脚本直接在当前节点执行，不会申请 GPU、提交 YARN 任务或自动转入后台。启动后先计算验证集均值基线，再逐轮训练和验证；第一次读取原始分区可能较慢。

## 结果在哪里

启动日志会打印本次输出目录，默认位于**仓库根目录的 `runtime/`** 下：

```text
runtime/traj_mae_v4_raw_<seconds或bucket30>_seed42_<时间>_<pid>/
├── console.log                    # 完整运行日志
├── exit_code.txt                  # 进程退出后生成；0 表示成功
└── artifacts/
    ├── loss_curve.png             # 每轮更新的训练、验证误差曲线
    ├── best.pt                    # 验证误差最低的模型
    ├── last.pt                    # 最近一轮的模型
    └── best_val_metrics.json       # 最优模型的验证指标
```

日志出现 `[全部完成]` 且 `exit_code.txt` 为 `0`，表示训练流程完成。数据检查模式也会生成日志和退出码，但不会生成训练结果。

## 怎么改参数

直接打开所选脚本，修改顶部的 `BATCH_SIZE`、`EPOCHS`、`SEED` 等配置，再向平台提交同一个脚本路径。无需在平台输入额外变量或参数。

如果只想检查数据，把脚本中 `DRY_RUN` 的默认值从 `0` 改为 `1`。看到 `[DRY RUN] Paths checked` 表示 7 天、每天 128 个分区通过目录检查；这一步不会训练。检查完成后改回 `0` 再正式运行。

其他机器复用时，保留完整仓库，通过 `PYTHON`、`DATA`、`VAL_DATA` 指定已有解释器和语料路径即可。数据必须满足当前入口的日期及分区要求；依赖版本见 [requirements.txt](requirements.txt)，数据格式见[技术协议](docs/design.md#技术协议)。真实数据和训练权重不随 GitHub 仓库上传。

## 读代码看哪里

日常运行看本页；理解实现按下面的顺序看，不必先读所有工具和测试。

| 文件 | 内容 |
| --- | --- |
| `train_seconds.sh` / `train_bucket30.sh` | 平台启动入口，配置在脚本顶部 |
| `train.sh` | 两种版本共用的启动逻辑 |
| `run.py` | 训练、验证、保存模型与进度显示 |
| `data.py` | 读取数据、轨迹分组与遮挡 |
| `model.py` | MLP 与 MAE 模型 |
| `evaluation.py` | 损失函数、均值基线与评估指标 |

`prepared.py` / `tensor_corpus.py` 只在了解预处理和张量存储时阅读。`tools/` 放按需使用的准备、统计和绘图工具，`tests/` 放测试；`docs/` 仅保留设计协议和历史记录两份说明。已有图册、报告和训练结果仍在原处，由 Git 忽略。

[设计与数据协议](docs/design.md) · [历史与发布记录](docs/history.md)
