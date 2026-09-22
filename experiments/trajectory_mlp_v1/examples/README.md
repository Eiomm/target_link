# 合成示意图

本目录提交到 Git 的样例均为人工合成，不含真实轨迹。

- `short_link_50m.svg`：50 米短路段、缺失 bin 和无效耗时。
- `offset_coverage_170m.svg`：起点错位时，较短轨迹仍可能覆盖更远终点。
- 对应的 `*.summary.json` 包含 `synthetic: true` 标记与绘图统计。

在仓库根目录重新生成：

```bash
python experiments/trajectory_mlp_v1/tools/make_plot_examples.py
```

用于训练入口自检的小型合成语料由另一个脚本生成：

```bash
python -m experiments.trajectory_mlp_v1.tools.make_synthetic_corpus --out runtime/trajectory_mlp_demo
```

本地可能另有真实数据抽样、导出记录与图册，已排除 Git 跟踪；它们不属于公开示例，也不是运行测试所需的文件。
