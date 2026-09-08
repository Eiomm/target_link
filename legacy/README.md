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
