import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pyspark.sql import SparkSession

from legacy.tools.build_windows_spark import (prepare_events, window_members, build_curves,
                                              snapshots, validate_args, parser, FORMAT)
from legacy.tools.adapt_samples_windows import adapt
from legacy.target_link_v1.data.window_stream import WindowDataset, collate_windows
from legacy.target_link_v1.models.window_mae import (WindowMAE, reconstruction_mask,
                                                      reconstruction_loss)


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
             max_passes=0, seed=42, sub_length_m=200.0, max_bins=21, max_speed=33.3,
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
    members, counts = window_members(e, a)
    # Mirror main(): members is persisted before curves/snapshots, so the multi-way
    # reuse below does not re-expand the validation lineage into one huge plan.
    return members.cache(), counts


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


def adapt_row(sample, idx, ratio, t_ref, t_cum, t_diff, observed):
    return dict(map_version="m", target_link_id="A", sample_id=sample, bin_idx=idx,
                ratio=ratio, bin_size_m=10.0, T_diff=float(t_diff), L_link_m=15.0,
                observed=observed, seg_mark=1, t_ref=float(t_ref), T_cum=float(t_cum),
                link_id="L1")


def test_adapt_audit_identity_matches_written_rows(spark):
    """adapt writes its meta row count from the audit, not from a second count()
    over the cached frame, so rows - dropped_quality - dropped_tail must equal the
    number of written events (and the derived geometry must survive the cache)."""
    rows = [
        adapt_row("s1", 0, 0.6, 100.0, 2.0, 2.0, 1),   # bin 0 split into two components
        adapt_row("s1", 0, 0.4, 102.0, 1.0, 1.0, 0),
        adapt_row("s1", 1, 0.5, 103.0, 3.0, 3.0, 1),
        adapt_row("s2", 0, 0.5, 100.0, 2.0, 2.0, 0),   # observed=0, closed by bin 1
        adapt_row("s2", 1, 0.0, 103.0, 3.0, 3.0, 1),   # zero distance -> bad quality
        adapt_row("s3", 0, 0.5, 100.0, 2.0, 2.0, 0),   # no downstream fix -> tail
    ]
    events, audit = adapt(spark.createDataFrame(rows), args())
    kept = audit["seg1_rows"] - audit["dropped_quality"] - audit["dropped_tail_no_fix"]
    assert audit["seg1_rows"] == 5
    assert audit["dropped_quality"] == 1 and audit["dropped_tail_no_fix"] == 1
    assert events.count() == kept == 3
    got = {(r["sample_id"], r["bin_idx"]): r for r in events.collect()}
    assert abs(got[("s1", 0)]["ratio"] - 1.0) < 1e-9      # 6m + 4m merged into one bin
    assert got[("s1", 1)]["spatial_start_m"] == 10.0      # prefix sum over merged bins
    assert got[("s1", 0)]["available_ts"] == 106.0        # closed by bin 1's fix
    assert not any(key[0] == "s3" for key in got)


def test_cumulative_times_still_require_availability(spark):
    a = args(time_source="cumulative")
    row = dict(event(), t_ref=500., T_cum=50.)
    raw = spark.createDataFrame([row]).drop("bin_start_ts", "bin_end_ts")
    events, _ = prepare_events(raw, a)
    result = events.first()
    assert result.bin_start_ts == 540. and result.bin_end_ts == 550.
    with pytest.raises(ValueError, match="Missing causal input"):
        prepare_events(raw.drop("available_ts"), a)


def test_default_window_and_stride_are_disjoint():
    """V1 main setting: window == stride, so no event is referenced twice."""
    a = parser().parse_args(["--inputs", "x", "--out", "y",
                             "--anchor-start", "600", "--anchor-end", "1200"])
    assert a.lookback_seconds == 600 and a.stride_seconds == 600


def test_snapshot_identity_is_modeling_unit(spark):
    """One sample per (modeling unit, anchor): sub_id splits the snapshot, and the
    unit extent is the nominal split, not the physical link length."""
    a = args()
    rows = [event(idx=0), event(idx=21, pos=210., start=550, end=560),
            event("tail", 0, pos=310., start=550, end=560, L_link_m=350.)]
    members, _ = members_for(spark, rows, a)
    got = {(r.sample_id, r.sub_id, r.unit_start_m, r.unit_length_m)
           for r in members.where("anchor_ts=600").collect()}
    assert got == {("S1", 0, 0.0, 200.0), ("S1", 1, 200.0, 200.0), ("tail", 1, 200.0, 150.0)}
    assert len({r.snapshot_id for r in members.where("anchor_ts=600").collect()}) == 2


def test_geometry_holes_empty_windows_and_reader(spark, tmp_path):
    a = args()
    members, counts = members_for(spark, [event(idx=0), event(idx=3, start=570, end=580)], a)
    curves = build_curves(members, a)
    assert [b.position_m for b in curves.first().bins] == [0.0, 30.0]
    roads = spark.createDataFrame([("m", "A", 1000.), ("m", "B", 300.)],
                                  ["map_version", "target_link_id", "link_length_m"])
    table = snapshots(members, counts, roads, spark, a)
    # B is empty for 2 units x 2 anchors; A's unit 0 is the only nonempty one, so
    # A contributes 4 empty units x 2 anchors and B all 2 units x 2 anchors.
    assert table.where("target_link_id='B' AND empty_flag").count() == 4
    assert table.where("empty_flag").count() == 12
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
    assert batch["duration"].shape == (2, 21)


def model_item(sid, n_curves=3):
    """Three curves over two passes; n_curves=1 gives a single-trajectory snapshot."""
    return dict(duration=np.array([[2, 3, 4, 0], [5, 6, 0, 0], [4, 7, 0, 0]], np.float32)[:n_curves],
                distance=np.array([[10, 10, 5, 0], [10, 10, 0, 0], [10, 5, 0, 0]], np.float32)[:n_curves],
                position=np.array([[0, 30, 40, 0], [0, 10, 0, 0], [200, 210, 0, 0]], np.float32)[:n_curves],
                age=np.zeros((n_curves, 4), np.float32), observed=np.ones((n_curves, 4), np.float32),
                valid=np.array([[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 0, 0]], bool)[:n_curves],
                lengths=np.array([3, 2, 2])[:n_curves],
                unit_starts=np.array([0, 0, 200], np.float32)[:n_curves],
                unit_lengths=np.array([200, 200, 200], np.float32)[:n_curves],
                curve_pass=np.array([0, 1, 1])[:n_curves], snapshot_id=sid, anchor_ts=600)


def model_batch():
    return collate_windows([model_item("a"), model_item("b")])


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
    model = WindowMAE(d_model=16, heads=2, layers=1, group_layers=1, dropout=0,
                      time_features="curve").eval()
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


def test_bin_features_and_target_follow_design_doc():
    """f_i = [dt, ratio, observed] + p_i = s_i/L_unit; the target is dt on the
    fixed bin-length basis, so a partial bin is scaled instead of dropped."""
    torch.set_num_threads(2)
    torch.manual_seed(0)
    b = model_batch()
    model = WindowMAE(d_model=16, heads=2, layers=1, group_layers=1, dropout=0,
                      time_features="none")
    assert model.feature[0].in_features == 4, "coarse age must not be a default feature"
    assert WindowMAE(d_model=8).feature[0].in_features == 4
    per_bin = b["duration"] * 10.0 / b["distance"].clamp_min(1e-6)
    torch.testing.assert_close(model(b)["target"], torch.log1p(per_bin))
    raw = WindowMAE(d_model=16, heads=2, layers=1, group_layers=1, dropout=0,
                    time_features="none", target_transform="raw")
    torch.testing.assert_close(raw(b)["target"], per_bin)
    with pytest.raises(ValueError, match="target_transform"):
        WindowMAE(target_transform="sqrt")
    # Geometry of a hidden bin still reaches the model (ratio/position); its
    # motion and quality do not, so only the geometry edit moves a prediction.
    mask = torch.zeros_like(b["valid"])
    mask[0, :2] = True
    base = model(b, mask)["prediction"]
    geometry = dict(b, distance=b["distance"].clone())
    geometry["distance"][mask] *= 0.5
    assert not torch.allclose(base, model(geometry, mask)["prediction"])
    motion = dict(b, duration=b["duration"].clone(), observed=b["observed"].clone())
    motion["duration"][mask] += 100
    motion["observed"][mask] = 0
    assert torch.allclose(base, model(motion, mask)["prediction"])


def test_whole_trajectory_mask_is_the_main_path():
    """§14: the mask unit is a pass. A multi-trajectory snapshot never gets a
    partial span, and a snapshot is never fully hidden."""
    torch.set_num_threads(2)
    torch.manual_seed(0)
    b = model_batch()
    seen = set()
    for _ in range(20):
        mask = reconstruction_mask(b, ratio=0.5, whole_pass_probability=0.5)
        assert (mask & ~b["valid"]).sum() == 0
        for g in range(b["n_groups"]):
            passes = b["curve_pass"][b["curve_group"] == g].unique().tolist()
            assert len(passes) == 2
            for pid in passes:
                p = b["curve_pass"] == pid
                assert not mask[p].any() or (mask[p] == b["valid"][p]).all()
                seen.add(bool(mask[p].any()))
            assert any(not mask[b["curve_pass"] == pid].any() for pid in passes)
            assert any(mask[b["curve_pass"] == pid].any() for pid in passes), \
                "every multi-trajectory snapshot must carry reconstruction signal"
    assert seen == {True, False}, "both masked and visible passes must occur"


def test_mask_never_empties_a_snapshot_and_falls_back_for_one_trajectory():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    b = model_batch()
    for _ in range(20):
        mask = reconstruction_mask(b, whole_pass_probability=1.0)
        for g in range(b["n_groups"]):
            passes = b["curve_pass"][b["curve_group"] == g].unique().tolist()
            visible = [p for p in passes if not mask[b["curve_pass"] == p].any()]
            assert len(visible) == 1
    # whole_pass_probability=0 is the span-only ablation, also for K>1 snapshots.
    span_all = reconstruction_mask(b, ratio=0.5, whole_pass_probability=0.0)
    for g in range(b["n_groups"]):
        rows = b["curve_group"] == g
        assert span_all[rows].any()
        for pid in b["curve_pass"][rows].unique().tolist():
            p = b["curve_pass"] == pid
            assert not (span_all[p] == b["valid"][p]).all(), "p=0 must not hide whole passes"
    solo = collate_windows([model_item("solo", n_curves=1)])
    span = reconstruction_mask(solo, ratio=0.5, whole_pass_probability=1.0)
    assert span.sum() == 2 and (span & ~solo["valid"]).sum() == 0
    assert not (span == solo["valid"]).all(), "a lone trajectory keeps visible bins"


def test_level2_sees_only_visible_tokens_plus_mask_token():
    """§14: level 2 reads [CLS] + visible trajectory tokens + one MASK token per
    hidden trajectory; the decoder has no level-1 skip connection."""
    torch.set_num_threads(2)
    torch.manual_seed(0)
    b = model_batch()
    model = WindowMAE(d_model=16, heads=2, layers=1, group_layers=1, dropout=0,
                      time_features="none").eval()
    assert model.decoder[0].in_features == 32, "decoder input is query + level-2 state"
    hidden = b["curve_pass"] == 1
    mask = torch.zeros_like(b["valid"])
    mask[hidden] = True
    base = model(b, mask)
    # A fully hidden trajectory is replaced by the shared MASK token: its own
    # level-1 state must not reach level 2.
    moved = dict(b, position=b["position"].clone())
    moved["position"][hidden] += 50.0
    torch.testing.assert_close(base["representation"], model(moved, mask)["representation"])
    # A visible trajectory still moves the representation.
    visible_edit = dict(b, position=b["position"].clone())
    visible_edit["position"][b["curve_pass"] == 0] += 50.0
    assert not torch.allclose(base["representation"], model(visible_edit, mask)["representation"])
    # Zeroing the level-2 state must hurt: otherwise the decoder ignores it.
    ablated = model(b, mask, ablate_aggregate=True)
    assert not torch.allclose(base["prediction"], ablated["prediction"])
    with pytest.raises(ValueError, match="decoder_input"):
        WindowMAE(decoder_input="both")


def test_reader_keeps_group_across_record_batches(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    row = dict(snapshot_id="one", anchor_ts=600, sample_id="s", sub_id=0,
               link_length_m=1000., unit_start_m=0., unit_length_m=200.,
               bins=[dict(position_m=0., bin_idx=0, duration=10.,
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
