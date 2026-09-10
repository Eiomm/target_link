# legacy/ — 已归档的旧版本代码与产物(2026-09-07 架构清理)

归档原则:只移不死,需要时可随时搬回。

## 归档清单

| 文件 | 身份 | 被什么取代 |
|---|---|---|
| `tools/build_dataset.py` + `tools/synthetic_city.py` | V0 合成城原型(造假数据调管线) | 真实轨迹数据(raw_hdfs) |
| `tools/_build_profiles_old.py` | 旧版 profiles 构建(已是 underscore 标死) | `tools/build_profiles.py` |
| `configs/{ingest,profiles}_h3.yaml` + `runtime/run_h3.sh` | h3 实验线 | target-link 表示定型后废弃 |
| `configs/{ingest,profiles}_4f.yaml` + `data/processed_4f/` | 4 文件小样本实验线 | smoke3/spark3 链路 |
| `data/raw/`(空目录,直接删除) | 早期本地数据位 | `data/raw_hdfs/` |

## 有意未归档(容易误判,勿动)

- `tools/ingest_streaming.py` + `runtime/run_ts.sh` + `configs/*_ts_day*.yaml` + `data/processed_ts/`:
  ts 管线现役引擎(day21 待补跑),`ingest_spark.py` 是它的分布式重写但 ts 线尚未迁移。
- `tools/ingest.py` + `tools/check_ingest_equiv.py` + `data/processed_smoke_pandas3/`:
  等价性参照实现,yarn 全量等价验证完成前保留。
- `data/processed/`:当前 6000ep eta 训练(`configs/eta*.yaml`)正在读的数据,**不是**陈旧产物。

---

# 第二轮清理(2026-09-10,建模单元切到 cell 级之后)

背景:9.9 晚拍板 cell 级建模(`md/最新讨论想法.md`)。V1 主线变成
`tools/build_corpus.py`(yarn)→ HDFS `corpus_v1/` → `target_link_v1/data/cell_corpus.py`
+ `tools/train_cells.py`(k8s A100);windows 线、curves 段、pretrain 段随之退出主线。

归档原则不变:**只移不死**。这轮只移入口脚本和 config,`tools/`、`target_link_v1/`、`tests/`
下的代码全部留在原位。

## 归档清单

### `legacy/scripts/` — sh 入口

| 文件 | 身份 | 被什么取代 |
|---|---|---|
| `submit_windows_yarn.sh` + `submit_windows_24h.sh` | 因果窗口语料出料(yarn) | `submit_build_corpus_yarn.sh` |
| `submit_train_windows_job.sh` + `submit_train_smoke_job.sh` | 窗口 MAE 训练 / 平台连通性 smoke(k8s) | `submit_cell_train_job.sh` / `submit_cell_smoke_job.sh` |
| `submit_adapt_yarn.sh` | raw samples → 因果 events(windows 上游) | cell 线直接从 samples 出料 |
| `submit_curves_yarn.sh` | curves 线 spark 作业 | 9.8 起已降级 |
| `submit_pretrain_job.sh` | profile 行 CurveMAE 预训练(k8s) | 窗口/cell MAE |
| `submit_week_stats_yarn.sh` | 7 天 link/pass 普查(一次性) | 已跑完,产物在 `data/_stats/` |

移动后改了两处路径:6 个脚本的 `REPO` 从 `${BASH_SOURCE}/..` 改成 `/../..`(legacy/scripts →
仓库根),并给 `submit_windows_24h.sh`、`submit_train_smoke_job.sh` 加了 `SCRIPTS=` 指向
legacy/scripts,让 `24h → windows_yarn`、`train_smoke → train_windows_job` 这类兄弟调用仍然成立。
两个原本就写死绝对路径的 k8s 脚本(`submit_pretrain_job.sh`、`submit_train_windows_job.sh`)没动。
8 个脚本都过了 `bash -n`,并实测 `MODE=dry` 仍能算出完整 spark-submit 命令。

### `legacy/configs/` — config

| 文件 | 身份 |
|---|---|
| `pretrain.yaml` / `pretrain_stream.yaml` | CurveMAE(npz)/ 流式(curves shards)两版预训练配置 |
| `pretrain_windows.yaml` | 窗口 MAE 配置 |
| `ingest_smoke3.yaml` / `profiles_smoke_spark3.yaml` | smoke3 一次性实验 |
| `eta_smoke_cls.yaml` / `eta_smoke_mean.yaml` | ETA CLS vs mean 短 overfit A/B |

被引用的默认值同步改了,不留断链:`tools/train_pretrain.py --config` 的默认值
→ `legacy/configs/pretrain.yaml`;`md/窗口管线使用.md` 与 `configs/profiles_job.yaml` 注释里
指向已移走 config 的路径也一并更正。

## 有意未归档(本轮判断,容易误判,勿动)

- `scripts/check_windows_server.sh`:名字带 windows,内容是**通用**的 pod 环境体检(选 python、
  探可 bind 的 `SPARK_LOCAL_IP`、import 依赖、跑 pytest),windows 停了它还有用。
- `scripts/submit_ingest_yarn.sh` + `submit_profiles_job.sh` + `configs/{ingest,profiles}*.yaml`
  + `configs/eta*.yaml` + `tools/train_eta.py`:`md/窗口管线使用.md` 写明"旧的完整通行 curves/
  NPZ/ETA 入口继续保留",且 ETA 仍是 V1 下游,profiles/ingest 是它的数据来源 —— 等 cell 线出
  第一批结果再定去留。
- `scripts/watch_yarn.sh`、`scripts/submit_cells_stats_yarn.sh`、`tools/stats_cells.py`:
  cell 线的普查/监控,现役。
- `data/windows_day20260821/`(150GB,NFS)、`data/curves_spark/`、`data/pretrain_corpus/`、
  `data/processed_smoke_*`:对应产物。窗口线虽然停了,但删数据不可逆,留到 cell 线跑通再决定。
