# 主线收敛与重构记录（2026-10-08）

## 最终选择

根据用户最终确认，v6 为唯一维护主线；v4 是明确授权保留的复现归档。用户给出的三项指标与 2026-09-23 保存的 v4 对比记录完全一致，不能用来评价 v6。秒级与 bucket30 时间编码均由同一模型的配置选择。

## 最终目录（核心文件）

```text
target_link/
├── README.md
├── train.sh
├── train.toml
├── requirements.txt
├── requirements-dev.txt
├── requirements-spark.txt
├── requirements-analysis.txt
├── pytest.ini
├── trajectory_mae/
│   ├── run.py
│   ├── cli.py
│   ├── model.py
│   ├── data.py
│   ├── grouping.py
│   ├── columns.py
│   ├── evaluation.py
│   ├── checkpoints.py
│   ├── manifests.py
│   ├── progress.py
│   ├── prepared.py
│   ├── observation_v3.py
│   ├── tensor_corpus.py
│   ├── tools/
│   ├── data_browser/
│   └── tests/
├── tools/
├── scripts/
├── tests/
├── V4/
│   ├── README.md
│   ├── code/
│   ├── tests/
│   └── results/
│       ├── training/
│       ├── comparison/
│       └── provenance.json
└── docs/
```

数据、训练输出和审计产物继续保存在 data/、runtime/、outputs/，不作为模型源码树展示。

## 删除、合并、拆分

- 原 `experiments/trajectory_mlp_v1` 迁为 `trajectory_mae`，同步所有运行代码、测试和集群分发包的导入。
- 删除 legacy、旧 target_link_v1 的 CellMAE/ETA/window/pretrain 模型、对应配置/启动器/专属测试、Qwen 环境检查及重复训练转发脚本。
- 移除旧 groups 语料发布链；主线重建训练数据使用现有 raw→tensor 构建器。已有 observation 可直接读取或转换为 tensor/observation-v3。
- 两个按时间编码复制的启动器和多层包装收敛为根目录 train.sh + train.toml，取消按目录存在情况自动切换数据源。
- run.py 按职责拆为 cli、checkpoints、manifests、progress；分组与折叠拆为 grouping，Arrow 列操作拆为 columns，移除读取器/存储模块循环依赖。
- 数据构建器直接使用 ObservationGroups，不再绕过 Dataset 初始化，也不依赖 getattr/hasattr 默认属性。
- 删除旧绘图副本，保留覆盖自适应空间显示的绘图工具；SVG 和普查图表分别拆为 bin_svg.py、census_plot.py，所有 Python 文件不超过 600 行。
- 训练依赖统一移到根目录；Spark 和 DuckDB 分开声明为数据工具依赖。源码指纹包含拆分后的全部读取模块。
- 旧进度文档已从运行仓库移除，关键数据约定与未解决问题整合进 data_contract.md。报告与样例产物移至 outputs/，不删除已有训练数据或结果。

## 行为不变及保留边界

v6 输入通道、Transformer 参数、时间编码、ratio 条件、softplus 输出、分组种子、尾组丢弃、验证遮挡、loss 和指标口径不变。主线只接受显式 v6 权重；v4 权重通过 V4.code.run 复现。

独立基线的组内均值补值属于原有数学定义，不是异常吞掉后的程序回退，因此保留。无有效监督的零损失、不可评分指标 null 及 padding mask 同样保留其明确语义。

原始源码和所有已有工作区差异已备份到仓库外：
`/nfs/dataset-ofs-494-1/project/user/junao/target_link_refactor_backup/20261008_171501/`
其中有效源码包为 `code_before.tar.gz`，另有 git_status.txt、git_diff.patch。AGENTS.md 与任务开始时完全一致。未开启多 agent。

## 验证

- 重构前核心回归：74 项通过。
- 数值对照：v4/v6 × seconds/bucket30，固定输入的输出、损失、所有参数梯度最大绝对差均为 0，见 numerical_equivalence.json。
- V4 原始 best.pt 已严格加载，并验证隐藏标签不影响预测；16 个归档产物逐一核对原文件及副本 SHA-256，见 archive_checks.json。
- 统一平台入口在真实数据上完成 dry-run：7 天、每天 128 个分区，未启动训练或分配 GPU。日志位于 `/tmp/target_link_v6_dryrun_20261008/console.log`。
- Python 语法、600 行限制、shell 语法及 git diff --check 通过，计数见 static_checks.json。
- 最终全套测试：226 项通过、3 项跳过（2 项真实 Spark 集成、1 项 CUDA 性能检查）；用时 163.35 秒。结果见 tests_summary.json，完整日志为 tests.log。

## 未解决或未验证

- 初次回归所在的隔离环境无法访问 CUDA；随后已在实际 RTX A6000 上通过 v6 合成数据最小 GPU 训练、验证、保存及独立重载测试，见 gpu_smoke.json。未执行全量真实数据训练、GPU 基线性能测试或全量验证，因此没有新的 v6 精度结论。
- 未提交真实 YARN/HDFS 作业。真实 Spark/JVM 集成需显式启用环境开关；分发包、预检失败行为、发布完整性和本地转换契约已自动测试。
- DuckDB 未安装在当前训练解释器，SQL 普查工具未进行全量运行；训练流程不需要它。
- v4 归档是从现存显式 v4 实现提取的可运行版本，缺少当时训练的完整源码快照，不能宣称源码哈希与历史记录相同。
- 上游边界和长尾标签口径问题仍存在，详见 data_contract.md。本次未改变标签、核心算法或重建全量数据。
