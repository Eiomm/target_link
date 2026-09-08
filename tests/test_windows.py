import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pyspark.sql import SparkSession

from tools.build_windows_spark import (prepare_events, window_members, build_curves,
                                       snapshots, validate_args, FORMAT)
from target_link_v1.data.window_stream import WindowDataset, collate_windows
from target_link_v1.models.window_mae import WindowMAE, reconstruction_mask, reconstruction_loss


@pytest.fixture(scope="module")
def spark():
    s = (SparkSession.builder.master("local[2]").appName("window-tests")
         .config("spark.ui.enabled", "false").config("spark.sql.shuffle.partitions", 2)
         .config("spark.sql.session.timeZone", "UTC").getOrCreate())
    s.sparkContext.setLogLevel("ERROR")
    yield s
    s.stop()


def args(**kw):
    d = dict(anchor_start=600, anchor_end=720, lookback_seconds=600, stride_seconds=60,
             max_passes=0, seed=42, sub_length_m=200.0, max_bins=40, max_speed=33.3,
             availability_column="available_ts", position_column="spatial_start_m",
             time_source="explicit")
    d.update(kw)
    return SimpleNamespace(**d)


def event(sample="S1", idx=0, start=540.0, end=550.0, available=None, pos=None, **kw):
    row = dict(map_version="m", target_link_id="A", sample_id=sample, bin_idx=idx,
               ratio=1.0, bin_size_m=10.0, T_diff=float(end-start), L_link_m=1000.0,
               observed=1, seg_mark=1, spatial_start_m=float(idx * 10 if pos is None else pos),
               available_ts=float(end if available is None else available),
               bin_start_ts=float(start), bin_end_ts=float(end))
    row.update(kw)
    return row


def members_for(spark, rows, a=None):
    a = a or args()
    e, _ = prepare_events(spark.createDataFrame(rows), a)
    return window_members(e, a)


def test_partial_boundary_availability_and_future_invariance(spark):
    rows = [event(idx=0), event(idx=1, start=550, end=560),
            event(idx=2, start=560, end=610),  # still inside bin at t=600
            event("late", start=500, end=510, available=620),
            event("at_end", start=590, end=600),
            event("at_start", start=0, end=10),
            event("cross_start", start=-5, end=5)]
    members, _ = members_for(spark, rows)
    got = {(r.sample_id, r.bin_idx) for r in members.where("anchor_ts=600").collect()}
    assert got == {("S1", 0), ("S1", 1), ("at_start", 0)}
    got_later = {(r.sample_id, r.bin_idx) for r in members.where("anchor_ts=660").collect()}
    assert ("late", 0) in got_later and ("S1", 2) in got_later
    # Appending an entirely future suffix must not change the t=600 input.
    more, _ = members_for(spark, rows + [event(idx=3, start=610, end=650)])
    assert {(r.sample_id, r.bin_idx) for r in more.where("anchor_ts=600").collect()} == got


def test_cap_is_per_anchor_and_preserves_all_subs(spark):
    rows = [event(s, i, start=540+i, end=550+i, pos=position)
            for s in ["S1", "S2", "S3"] for i, position in [(0, 0), (1, 210)]]
    a = args(max_passes=1)
    one, counts = members_for(spark, rows, a)
    two, _ = members_for(spark, list(reversed(rows)), a)
    key = lambda df: {(r.anchor_ts, r.sample_id, r.bin_idx) for r in df.collect()}
    assert key(one) == key(two)
    assert all(r.n_passes_before_cap == 3 and r.n_passes_kept == 1 for r in counts.collect())
    assert one.select("anchor_ts", "sub_id").distinct().count() == 4


def test_contract_half_bin_and_conflicts(spark):
    a = args()
    half = event(ratio=0.5, start=549, end=550)
    e, audit = prepare_events(spark.createDataFrame([half, half]), a)
    r = e.first()
    assert e.count() == 1 and r.distance_m / r.duration == 5
    assert audit["target_rows"] == 2
    with pytest.raises(ValueError, match="Missing causal input"):
        prepare_events(spark.createDataFrame([half]).drop("available_ts"), a)
    with pytest.raises(ValueError, match="Conflicting"):
        prepare_events(spark.createDataFrame([half, dict(half, ratio=0.6)]), a)
    with pytest.raises(ValueError, match="align"):
        validate_args(args(anchor_start=601))


def test_cumulative_times_still_require_availability(spark):
    a = args(time_source="cumulative")
    row = dict(event(), t_ref=500., T_cum=50.)
    raw = spark.createDataFrame([row]).drop("bin_start_ts", "bin_end_ts")
    events, _ = prepare_events(raw, a)
    result = events.first()
    assert result.bin_start_ts == 540. and result.bin_end_ts == 550.
    with pytest.raises(ValueError, match="Missing causal input"):
        prepare_events(raw.drop("available_ts"), a)


def test_geometry_holes_empty_windows_and_reader(spark, tmp_path):
    a = args()
    members, counts = members_for(spark, [event(idx=0), event(idx=3, start=570, end=580)], a)
    curves = build_curves(members, a)
    assert [b.position_m for b in curves.first().bins] == [0.0, 30.0]
    roads = spark.createDataFrame([("m", "A"), ("m", "B")], ["map_version", "target_link_id"])
    table = snapshots(members, counts, roads, spark, a)
    assert table.where("target_link_id='B' AND empty_flag").count() == 2
    (curves.repartition(1).sortWithinPartitions("snapshot_id", "sub_id", "sample_id")
     .write.parquet(str(tmp_path / "window_curves")))
    meta = tmp_path / "window_meta.json.d"
    meta.mkdir()
    (meta / "part-000.txt").write_text(json.dumps(dict(vars(a), format=FORMAT)))
    items = list(WindowDataset(tmp_path))
    assert len(items) == 2
    assert items[0]["position"][0, :2].tolist() == [0.0, 30.0]
    assert items[0]["valid"].sum() == 2
    batch = collate_windows(items)
    assert batch["n_groups"] == 2
    assert batch["duration"].shape == (2, 40)


def model_batch():
    def item(sid):
        return dict(duration=np.array([[2, 3, 4, 0], [5, 6, 0, 0], [4, 7, 0, 0]], np.float32),
                    distance=np.array([[10, 10, 5, 0], [10, 10, 0, 0], [10, 5, 0, 0]], np.float32),
                    position=np.array([[0, 30, 40, 0], [0, 10, 0, 0], [200, 210, 0, 0]], np.float32),
                    age=np.zeros((3, 4), np.float32), observed=np.ones((3, 4), np.float32),
                    valid=np.array([[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 0, 0]], bool),
                    lengths=np.array([3, 2, 2]), link_lengths=np.array([400, 400, 400], np.float32),
                    curve_pass=np.array([0, 1, 1]), snapshot_id=sid, anchor_ts=600)
    return collate_windows([item("a"), item("b")])


def test_model_mask_no_target_leakage_and_gradients():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    b = model_batch()
    mask = reconstruction_mask(b, whole_pass_probability=1)
    # Whole passage masking must hide both sub-curves if passage 1 is selected.
    for g in range(2):
        for pid in b["curve_pass"][b["curve_group"] == g].unique():
            rows = b["curve_pass"] == pid
            fully = (mask[rows] == b["valid"][rows]).all(1)
            assert fully.all() or not fully.any()
    model = WindowMAE(d_model=16, heads=2, layers=1, group_layers=1, dropout=0).eval()
    output = model(b, mask)
    changed = dict(b, duration=b["duration"].clone(), age=b["age"].clone())
    changed["duration"][mask] += 100
    changed["age"][mask] = 999
    altered = model(changed, mask)
    torch.testing.assert_close(output["prediction"], altered["prediction"])
    torch.testing.assert_close(output["representation"], altered["representation"])
    loss = reconstruction_loss(output, mask, b["curve_group"], b["n_groups"])
    loss.backward()
    assert output["representation"].shape == (2, 16)
    assert torch.isfinite(loss) and model.aggregate.cls_token.grad.abs().sum() > 0


def test_reader_keeps_group_across_record_batches(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    row = dict(snapshot_id="one", anchor_ts=600, sample_id="s", sub_id=0,
               link_length_m=1000., bins=[dict(position_m=0., bin_idx=0, duration=10.,
               distance_m=10., observed=1, event_id="e", bin_start_ts=540., bin_end_ts=550., available_ts=550.)])
    root = tmp_path / "window_curves"
    root.mkdir()
    pq.write_table(pa.Table.from_pylist([dict(row, sample_id=str(i)) for i in range(2050)]),
                   root / "part-0.parquet", row_group_size=400)
    meta = tmp_path / "window_meta.json.d"
    meta.mkdir()
    (meta / "part-0.txt").write_text(json.dumps(dict(vars(args()), format=FORMAT)))
    items = list(WindowDataset(tmp_path, max_curves_per_snapshot=3000))
    assert len(items) == 1 and len(items[0]["lengths"]) == 2050
    with pytest.raises(ValueError, match="reader limit"):
        list(WindowDataset(tmp_path))
