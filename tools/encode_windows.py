"""Export one learned representation per nonempty road snapshot to Parquet."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

from target_link_v1.data.window_stream import WindowDataset, collate_windows
from target_link_v1.models.window_mae import WindowMAE


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--start", type=int)
    p.add_argument("--end", type=int)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    if a.batch_size <= 0:
        p.error("batch-size must be positive")
    out = Path(a.out)
    if out.exists():
        p.error("output exists; choose a new filename")
    saved = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    if saved.get("format") != "target_link_window_mae_v1":
        raise ValueError("Require a window MAE checkpoint, not the old CurveMAE")
    model = WindowMAE(**saved["model_kwargs"]).to(a.device)
    model.load_state_dict(saved["model"])
    model.eval()
    cfg = saved["config"]
    ds = WindowDataset(a.data, start_ts=a.start, end_ts=a.end,
                       age_bucket_seconds=cfg["age_bucket_seconds"],
                       max_curves_per_snapshot=cfg["max_curves_per_snapshot"])
    for key in ["lookback_seconds", "sub_length_m", "max_bins", "boundary"]:
        if ds.meta.get(key) != saved["data_meta"].get(key):
            raise ValueError("Checkpoint/data preprocessing mismatch: " + key)
    loader = DataLoader(ds, batch_size=a.batch_size, collate_fn=collate_windows)
    schema = pa.schema([("snapshot_id", pa.string()), ("map_version", pa.string()),
                        ("target_link_id", pa.string()), ("anchor_ts", pa.int64()),
                        ("representation", pa.list_(pa.float32(), saved["model_kwargs"]["d_model"]))])
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with pq.ParquetWriter(out, schema) as writer, torch.no_grad():
        for b in loader:
            moved = {k: v.to(a.device) if torch.is_tensor(v) else v for k, v in b.items()}
            rep = model(moved)["representation"].cpu().tolist()
            rows = [dict(snapshot_id=s, map_version=m, target_link_id=l, anchor_ts=t, representation=r)
                    for s, m, l, t, r in zip(b["snapshot_ids"], b["map_versions"], b["link_ids"], b["anchor_ts"], rep)]
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))
            count += len(rows)
    print("[encode_windows] exported %d nonempty snapshots to %s" % (count, out))


if __name__ == "__main__":
    main()
