"""Export one cell's observations, legacy groups and current MAE groups (CPU only)."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pyarrow.dataset as pads
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.trajectory_mlp_v1.data import CellDataset, collate_cells


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2,
                               allow_nan=False) + '\n')


def render_member(group, index, observation):
    """Show stored pieces and the actual model features without inventing IDs."""
    hidden = bool(group['mae_mask'][index])
    lines = [f"=== 轨迹 slot[{index}] ==="]
    for key, value in observation.items():
        lines.append(f"{key}: {json.dumps(clean(value), ensure_ascii=False)}")
    lines += [
        f"group_id: {group['group_id']}",
        f"delta_t_seconds: {float(group['delta_t'][index]):.6f}",
        "traj_valid: True",
        f"mae_mask: {hidden}",
        "encoder_input: 整条轨迹不进入 Encoder" if hidden else
        "encoder_input: 下方 50×3 特征展平，经 trajectory_mlp 和时间编码后成为一个 token",
        "decoder_input: MASK + 已知 ratio 条件 + 时间编码" if hidden else
        "decoder_input: 投影后的可见轨迹状态 + 时间编码",
        "注意：下面的 T 是遮挡前数据；隐藏轨迹的 T/valid 仅用于监督。",
        "observed 仅展示来源，当前模型不使用。原记录未提供的 uid/traj_id 不从 sample_id 猜测。",
        "=== 按训练 bin tensor（固定 50 个位置）===",
    ]
    for b in range(50):
        pieces = [j for j, pos in enumerate(observation['bin_pos']) if pos == b]
        t, ratio = (float(v) for v in group['x'][index, b, :2])
        valid = bool(group['bin_valid'][index, b])
        lines.append(
            f"bin[{b:2d}] segment_range_m=[{b*10},{(b+1)*10}) "
            f"piece_indices={pieces} T_clean={t:.6f} ratio={ratio:.6f} "
            f"valid={valid} feature=[{t:.6f}, {ratio:.6f}, {int(valid)}] "
            f"encoder_uses={not hidden} loss_uses={hidden and valid}"
        )
    return '\n'.join(lines)


def write_text_examples(out, groups, by_id, info):
    header = [
        "当前 trajectory_mlp_v1 真实训练数据展示",
        f"corpus: {info['corpus']}",
        f"source_files: {json.dumps(info['source_files'], ensure_ascii=False)}",
        f"day: {info['day']}  bucket: {info['bucket']}  cell_id: {info['cell_id']}",
        f"map_version: {info['map_version']}  target_link_id: {info['target_link_id']}  seg_idx: {info['seg_idx']}",
        f"window: {info['window']}  window_local: {info['window_local']}",
        f"seed: {info['seed']}  epoch: {info['epoch']}  m_max: {info['m_max']}",
        f"cell_observations: {info['observations']}  group_sizes: {info['current_group_sizes']}",
        f"dropped_no_valid: {info['dropped_no_valid']}  dropped_tail: {info['dropped_tail']}",
        "这是 target 内单个 segment 的数据，不是 in 100m + 完整 target + out 100m 的原始片段。",
        "数组是 observation 的原始 piece 数据；下方 bin tensor 使用训练读取器的实际聚合结果。",
        "ratio_pct / 10 = ratio；同 bin 的 piece 求和，全部 piece 有效时该 bin 才有效。",
        "segment_range_m 是名义网格范围；实际覆盖由 ratio 表示。",
        "CLS 是共享模型参数，不在 observation 中；每个 group 前放一个 CLS。",
        "数据容器 x 第三通道为零，模型实际用 bin_valid 构造 [T_clean, ratio, valid]。",
        "以下按 group 展示，可将每组视为 batch 中的一个样本；未运行训练。",
    ]
    sections = ['\n'.join(header)]
    for group in groups:
        n = group['group_size']
        visible = [i for i in range(n) if not group['mae_mask'][i]]
        hidden = [i for i in range(n) if group['mae_mask'][i]]
        sections.append('\n'.join([
            f"=== GROUP {group['group_id']} ===",
            f"group_size: {n}  visible_slots: {visible}  hidden_slots: {hidden}",
            f"单组组批后 x_shape: [1, {info['m_max']}, 50, 3]",
            f"padding_slots: {list(range(n, info['m_max']))}（traj_valid=False，不进入注意力有效键或损失）",
            "encoder_sequence: [CLS, " + ', '.join(f'traj[{i}]' for i in visible) + ']'
        ]))
        sections.extend(render_member(group, i, by_id[sid])
                        for i, sid in enumerate(group['sample_ids']))
    (out / 'training_groups.txt').write_text('\n\n'.join(sections) + '\n', encoding='utf-8')
    first = groups[0]
    (out / 'training_trajectory_one.txt').write_text(
        '\n'.join(header) + '\n\n' + render_member(first, 0, by_id[first['sample_ids'][0]]) + '\n',
        encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus', type=Path, default=REPO / 'runtime/cell_mlp_train')
    p.add_argument('--day', default='20260820')
    p.add_argument('--bucket', default='123')
    p.add_argument('--cell-id', type=int, help='Default: deterministically choose a cell with 65–128 rows, else >=3.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epoch', type=int, default=0)
    p.add_argument('--m-max', type=int, default=64)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        p.error('Output already exists; choose a new directory.')
    rel = f'day={a.day}/bucket={a.bucket}'
    files = sorted((a.corpus / 'observations_v2' / rel).glob('*.parquet'))
    if not files:
        p.error(f'No observations in {rel}')
    source = pads.dataset([str(f) for f in files], format='parquet')
    cid = a.cell_id
    if cid is None:
        counts = Counter(source.to_table(columns=['cell_id'])['cell_id'].to_pylist())
        preferred = sorted(k for k, n in counts.items() if a.m_max < n <= 2*a.m_max)
        fallback = sorted(k for k, n in counts.items() if n >= 3)
        if not fallback:
            p.error('No cell with >=3 rows in this partition.')
        cid = (preferred or fallback)[0]
    table = source.to_table(filter=pads.field('cell_id') == cid)
    rows = sorted(table.to_pylist(), key=lambda r: r['sample_id'])
    if not rows:
        p.error('Cell not found.')

    # Feed only this cell to the unchanged production loader. Cell membership
    # depends on day/bucket/cell/seed, never on unrelated cells in the partition.
    class SelectedCellStore:
        base = str((a.corpus / 'observations_v2').resolve())

        def files(self, rel):
            return ['selected-cell-in-memory']

        def read(self, path):
            return table

    reader = CellDataset([str(a.corpus)], [a.day], m_max=a.m_max, seed=a.seed, epoch=a.epoch)
    reader._prepared = {}  # Explicitly reconstruct from exported observations.
    reader._tensors = {}
    loaded = reader._load_partition(SelectedCellStore(), a.day, a.bucket)
    if loaded is None:
        p.error('Selected cell has no valid observations.')
    specs = list(reader._group_specs(loaded, a.day, a.bucket))
    if not specs:
        p.error('Selected cell yields no usable training groups.')
    groups = [reader._pack(loaded, spec) for spec in specs]
    legacy_files = sorted((a.corpus / 'training_groups_k3' / rel).glob('*.parquet'))
    legacy = (pads.dataset([str(f) for f in legacy_files], format='parquet')
              .to_table(filter=pads.field('cell_id') == cid).to_pylist()) if legacy_files else []
    by_id = {r['sample_id']: r for r in rows}
    if len(by_id) != len(rows):
        raise AssertionError('Duplicate sample IDs')
    member_ids = []
    members = []
    for g in groups:
        batch = collate_cells([g], m_max=a.m_max, epoch=a.epoch)
        g['mae_mask'] = batch['mae_mask'][0, :g['group_size']].numpy()
        for i, sid in enumerate(g['sample_ids']):
            r = by_id[sid]
            member_ids.append(sid)
            # Independently fold ragged pieces to verify export/member alignment.
            for b in range(50):
                ix = [j for j, bp in enumerate(r['bin_pos']) if bp == b]
                valid = bool(ix) and all(r['valid'][j] for j in ix)
                expected_t = sum(r['T_diff'][j] for j in ix) if valid else 0
                expected_r = sum(r['ratio_pct'][j] for j in ix)/10
                np.testing.assert_allclose(g['x'][i,b], [expected_t, expected_r, 0], rtol=1e-6, atol=1e-6)
                assert bool(g['bin_valid'][i,b]) == valid
            np.testing.assert_allclose(g['delta_t'][i], r['dt'])
            members.append(dict(group_id=g['group_id'], member_index=i, sample_id=sid,
                                dt=r['dt'], n_pieces=r['n_pieces'],
                                valid_bins=int(g['bin_valid'][i].sum()),
                                mae_hidden=bool(g['mae_mask'][i])))
        assert int(g['mae_mask'].sum()) == g['group_size']//2
    assert len(member_ids) == len(set(member_ids))

    a.out.mkdir(parents=True)
    pq.write_table(table, a.out / 'observations.parquet')
    write_json(a.out / 'observations.json', rows)
    write_json(a.out / 'observation_one.json', by_id[groups[0]['sample_ids'][0]])
    write_json(a.out / 'current_groups.json', groups)
    write_json(a.out / 'legacy_training_groups_k3.json', legacy)
    with (a.out / 'group_members.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(members[0])); w.writeheader(); w.writerows(members)
    one = by_id[groups[0]['sample_ids'][0]]
    with (a.out / 'observation_one_pieces.csv').open('w', newline='') as f:
        keys = ['bin_pos', 'T_diff', 'ratio_pct', 'observed', 'valid']
        w = csv.DictWriter(f, fieldnames=['piece_index']+keys); w.writeheader()
        for i in range(one['n_pieces']):
            w.writerow(dict(piece_index=i, **{k:one[k][i] for k in keys}))
    info = dict(corpus=str(a.corpus.resolve()), day=a.day, bucket=a.bucket, cell_id=cid,
                target_link_id=rows[0]['target_link_id'], map_version=rows[0]['map_version'],
                seg_idx=rows[0]['seg_idx'], window=rows[0]['window'],
                window_local=datetime.fromtimestamp(rows[0]['window'], ZoneInfo('Asia/Shanghai')).isoformat(),
                selection='explicit cell' if a.cell_id else 'deterministic illustrative cell; not statistical sampling',
                seed=a.seed, epoch=a.epoch, m_max=a.m_max, observations=len(rows),
                current_group_sizes=[g['group_size'] for g in groups],
                legacy_group_sizes=[g['group_size'] for g in legacy],
                legacy_directory_exists=(a.corpus/'training_groups_k3').is_dir(),
                source_files=[str(f.resolve()) for f in files],
                validation='PASS: unique membership, observation-to-tensor folding, dt, mask counts',
                partition_stats_scope='selected cell only',
                dropped_no_valid=groups[0]['dropped_no_valid'], dropped_tail=groups[0]['dropped_tail'])
    write_json(a.out / 'manifest.json', info)
    write_text_examples(a.out, groups, by_id, info)
    first = {k:v for k,v in groups[0].items() if k not in ['x','bin_valid','traj_valid','sample_ids','delta_t','mae_mask']}
    first['sample_ids_preview'] = groups[0]['sample_ids'][:3]
    first['x_shape'] = list(groups[0]['x'].shape)
    first['first_member_first_3_bins'] = groups[0]['x'][0,:3].tolist()
    preview = dict(one); preview.update({k:one[k][:3] for k in ['bin_pos','T_diff','ratio_pct','observed','valid']})
    report = f'''# Observation 与训练 Group 实例

来源：`{a.corpus.resolve()}`，{info['window_local']}，link {info['target_link_id']}，segment {info['seg_idx']}。

完整 cell 有 **{len(rows)} 条 observation**。当前分组大小：**{info['current_group_sizes']}**；旧文件分组大小：**{info['legacy_group_sizes']}**。

仿原始轨迹打印格式的文本：先看 [单条训练轨迹](training_trajectory_one.txt)，再看 [全部训练分组](training_groups.txt)。包含原始字段数组、每条轨迹全部 50 个 bin、Encoder 可见性和重建监督位置。

这是结构展示样本，不代表全量数据的统计分布。seed={a.seed}，epoch={a.epoch}，M={a.m_max}。

## 1. Observation：一条轨迹在这个 segment、窗口内的记录

完整单条见 [observation_one.json](observation_one.json)，展开组件见 [observation_one_pieces.csv](observation_one_pieces.csv)。下方数组仅展示前三项：

```json
{json.dumps(clean(preview), ensure_ascii=False, indent=2)}
```

- `cell_id`：地图版本、link、segment、时间窗对应的 cell 标识；多个 observation 共享它。
- `sample_id`：轨迹经过的标识；`dt`：相对窗口起点的秒数。
- `n_pieces`：组件数；五个列表逐项对齐。同一 bin 可以有多个组件。
- `bin_pos`：segment 内 0–49 的格子位置，每格名义上 10m。
- `T_diff`：该组件耗时（秒）；`ratio_pct / 10` 为格子占比，10 表示完整一格，4 表示 0.4。
- `observed`：是否有 GPS 观测；`valid`：耗时是否有效，二者不同。当前模型不读取 observed。
- `window`：Unix 秒；`map_version`、`target_link_id`、`seg_idx`：所属地图、道路和分段。

## 2. 当前训练 Group：从 observation 现场构建

```json
{json.dumps(clean(first), ensure_ascii=False, indent=2)}
```

每个 cell 先排除没有有效 bin 的 observation，再按 sample_id 排序、固定 seed 打乱，以最多 {a.m_max} 条分组；不足 3 条的尾组丢弃。成员不会跨 cell。K 是整个 cell 可用轨迹数，group_size 才是本组成员数。

完整成员、数组见 [current_groups.json](current_groups.json)；简表见 [group_members.csv](group_members.csv)。

`x` 形状为 `[组内轨迹数, 50, 3]`，每个 bin 的三个数是 `[组件耗时之和, ratio之和, 0]`。`bin_valid` 判断该 bin 是否可作有效目标；`delta_t` 对应 observation 的 dt。

导出的是遮盖前的数据和监督目标；`mae_mask=true` 表示本轮要隐藏并重建的轨迹，不代表把完整耗时直接送给 encoder。每组隐藏 floor(group_size/2) 条。拼 batch 时再补齐到 {a.m_max} 条，padding 由 traj_valid 区分。

## 3. 磁盘上的旧 training_groups_k3

训练集中的旧文件是成员索引（含 sample_ids），并不复制 observation 的全部耗时数组。最新 MAE v4 不读取它；验证集可仅包含 observations_v2。

旧文件原样导出见 [legacy_training_groups_k3.json](legacy_training_groups_k3.json)。空列表表示选定位置没有旧记录；目录是否存在见 manifest.json。

## 验证与复现

{info['validation']}。导出包含该 cell 的全部 observation，独立核对了每个当前 group 成员的 50 个 bin、dt 和遮盖数量。JSON 中非有限数统一写为 null；Parquet 保留原始数据类型。无训练、无源数据修改。

```bash
{sys.executable} {Path(__file__).resolve()} --corpus {a.corpus.resolve()} --day {a.day} --bucket {a.bucket} --cell-id {cid} --seed {a.seed} --epoch {a.epoch} --m-max {a.m_max} --out /tmp/group_example_new
```
'''
    (a.out / 'README.md').write_text(report)
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
