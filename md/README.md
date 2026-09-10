# 文档导航

## 当前主线

- [`最新讨论想法.md`](最新讨论想法.md)：现役 V1 Cell MAE 设计与数据契约。
- [`本地开发工作流.md`](本地开发工作流.md)：本地改码、服务器训练、真实语料事实和当前待办。
- 根目录 [`数据说明.md`](../数据说明.md)：上游数据字段与来源说明。

当前主线采用 `(map_version, target_link_id, seg_idx, 10min window)` 的 500m cell，
语料为 `corpus_v1/observations_v2 + training_groups_k3`。若其他文档出现 200m sub-link、
`window_curves` 或 curves/stream pretraining，它们描述的是历史路线。

## 进度日志（全部保留）

- `9.2progress.md`
- `9.4progress.md`
- `9.7progress.md`
- `9.8progress.md`
- `9.9progress.md`
- `9.10progress.md`

进度日志保留当时的判断、实验数据和变更过程，不作为当前接口契约。

## 历史与审计材料

- `9_4.md`：早期 200m trajectory representation v2 设计。
- `数据处理过程.md`：早期 ingest + 200m spatial profile 流程。
- `窗口管线使用.md`、`窗口语料数据格式.txt`：已归档 causal-window 管线。
- `目标组件两文件审计.md`：9.9 两个本地数据文件的 ratio/边界审计。

新合入的旧方案与原始样本已放到 `legacy/md/`，避免与当前规范混淆，同时保留追溯信息。
