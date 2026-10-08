# V4 实验复现归档

这是冻结的三通道模型实验；唯一维护主线仍是仓库根目录的 `trajectory_mae/`（v6）。V4 encoder 输入 T_clean、ratio、valid；隐藏轨迹 ratio 作为 decoder 条件。`code/` 独立导入自己的模块，主线不依赖此目录。

| 指标 | 可见均值基线 | V4 模型 | 变化 |
|---|---:|---:|---:|
| 单 bin MAE | 0.782252 秒 | 0.555092 秒 | 降低 29.04% |
| 轨迹有效部分总耗时 MAE | 7.159232 秒 | 6.133908 秒 | 降低 14.32% |
| 轨迹有效部分总耗时 RMSE | 14.637136 秒 | 15.937816 秒 | 升高 8.89% |

源训练运行：`runtime/2026-09-22/traj_mae_v4_raw_seconds_seed42_20260922_232119_982`，对比采用已完整保存的第 1 轮验证结果。

- `results/training/`：对应 best.pt、训练配置、数据清单、逐轮指标和曲线。
- `results/comparison/`：原始基线指标、对比 JSON 和报告。
- `results/provenance.json`：原始文件路径、归档路径、大小及 SHA-256。

代码提取自重构前仓库仍存在的显式 v4 路径，已移除 v5/v6 分支；参数名和形状保持不变。没有当时训练的完整源码快照，不能声称归档源码与历史 source_sha256 一致。原始产物中的哈希、配置和历史路径原样保存；已验证固定输入的输出、损失和梯度与重构前 v4 路径完全一致。

在仓库根目录使用与主线相同的 Python 依赖：

```bash
python -m V4.code.run evaluate \
  --checkpoint V4/results/training/best.pt \
  --data data/cell_mlp_validation_20260823 \
  --out runtime/v4_reproduction_new --device cpu --workers 0
```

输出目录必须不存在。完整验证耗时较长，以上命令未在重构中全量执行。原始 metadata 中的数据路径曾迁移，不能直接用旧绝对路径重新读取；复现时显式传当前数据根目录。验证分组、遮挡和有效监督口径保持不变。权重只保存在服务器，Git 忽略 `*.pt`。
