"""End-to-end protocol checks using synthetic parquet, never real validation."""
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.trajectory_mlp_v1 import run


def corpus(root):
    for day, cells in [("20260817", [10, 11]), ("20260823", [20, 21])]:
        rows = []
        for cell in cells:
            for i in range(5):
                rows.append(dict(cell_id=cell, sample_id=f"{cell}-{i}", dt=float(i * 20),
                                 T_diff=[1.0 + i / 10, 2.0 + i / 10], ratio_pct=[10, 7],
                                 valid=[True, True], bin_pos=[0, 12]))
        p = root / "observations_v2" / f"day={day}" / "bucket=0"
        p.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(rows), p / "part.parquet")


@pytest.mark.parametrize("encoding", ["seconds", "bucket30"])
def test_train_checkpoint_evaluate_and_baseline_share_protocol(tmp_path, encoding):
    corpus(tmp_path)
    common = ["--data", str(tmp_path), "--train-days", "20260817", "--val-days", "20260823",
              "--device", "cpu", "--threads", "1", "--batch-size", "2", "--bootstrap", "20",
              "--time-encoding", encoding]
    run.main(["train", *common, "--out", str(tmp_path / "fit"), "--epochs", "2",
              "--d-model", "16", "--heads", "2", "--layers", "1", "--dropout", "0", "--plot-every", "1"])
    import csv
    curve_rows = list(csv.DictReader((tmp_path / "fit/loss_curve.csv").open()))
    assert [r["epoch"] for r in curve_rows] == ["1", "2"]
    history = [json.loads(line) for line in (tmp_path / "fit/metrics.jsonl").read_text().splitlines()]
    assert float(curve_rows[-1]["val_mae_seconds"]) == history[-1]["val"]["ours"]["bin_mae_seconds"]
    assert (tmp_path / "fit/loss_curve.png").read_bytes().startswith(b"\x89PNG")
    assert "<svg" in (tmp_path / "fit/loss_curve.svg").read_text()
    for name in ['step_trends.png', 'training_diagnostics.png']:
        assert (tmp_path / 'fit' / name).read_bytes().startswith(b'\x89PNG')
    step_rows = list(csv.DictReader((tmp_path / 'fit/step_metrics.csv').open()))
    diagnostic_rows = list(csv.DictReader((tmp_path / 'fit/epoch_diagnostics.csv').open()))
    assert len(step_rows) == 2
    assert [int(r['global_step']) for r in step_rows] == [1, 2]
    assert len(diagnostic_rows) == 2
    for step, diag, record in zip(step_rows, diagnostic_rows, history):
        assert float(step['batch_mae_seconds']) == pytest.approx(record['train']['loss_mae_seconds'])
        assert float(diag['grad_norm_mean']) == float(step['grad_norm'])
        assert float(diag['lr']) == .0002
        assert float(diag['parameter_norm']) > 0
        assert float(diag['val_rmse_seconds']) == record['val']['ours']['bin_rmse_seconds']
    fit = json.loads((tmp_path / "fit/best_val_metrics.json").read_text())
    baseline = json.loads((tmp_path / "fit/baseline.json").read_text())
    assert fit["eval_mask_id"] == baseline["eval_mask_id"]
    assert fit["bins"] == 8
    assert fit["ours"]["bin_mae_seconds"] >= 0
    run.main(["evaluate", *common, "--out", str(tmp_path / "evaluation"),
              "--checkpoint", str(tmp_path / "fit/best.pt")])
    again = json.loads((tmp_path / "evaluation/metrics.json").read_text())
    assert again["ours"] == fit["ours"]
    assert again["eval_mask_id"] == fit["eval_mask_id"]
    run.main(["baseline", *common, "--out", str(tmp_path / "baseline"), "--seed", "44"])
    alone = json.loads((tmp_path / "baseline/metrics.json").read_text())
    assert alone["baseline"] == baseline["baseline"]
    assert alone["eval_mask_id"] == baseline["eval_mask_id"]


def test_missing_validation_is_not_substituted(tmp_path):
    corpus(tmp_path)
    with pytest.raises(ValueError, match="missing"):
        run.main(["train", "--data", str(tmp_path), "--train-days", "20260817",
                  "--val-days", "20260824", "--out", str(tmp_path / "fit")])
    assert not (tmp_path / "fit").exists()


def test_training_only_smoke_records_isolation(tmp_path):
    corpus(tmp_path)
    run.main(["smoke", "--data", str(tmp_path), "--train-days", "20260817", "--val-days", "20260824",
              "--out", str(tmp_path / "smoke"), "--smoke-groups", "2", "--smoke-steps", "2",
              "--batch-size", "2", "--d-model", "16", "--heads", "2", "--layers", "1",
              "--threads", "1", "--device", "cpu"])
    s = json.loads((tmp_path / "smoke/smoke.json").read_text())
    assert s["steps"] == 2
    assert s["checkpoint_max_abs_diff"] == 0
    assert s["hidden_input_prediction_max_abs_diff"] == 0
    assert s["hidden_input_representation_max_abs_diff"] == 0


def test_duplicate_partition_and_overlapping_dates_rejected(tmp_path):
    corpus(tmp_path)
    with pytest.raises(ValueError, match="Duplicate"):
        run.file_manifest([str(tmp_path), str(tmp_path)], ["20260817"])
    with pytest.raises(SystemExit):
        run.parse_args(["train", "--data", str(tmp_path), "--out", str(tmp_path / "fit"),
                        "--train-days", "20260823"])


def test_seed_summary_rejects_unpaired_validation_sets(tmp_path):
    from experiments.trajectory_mlp_v1.tools.summarize_runs import summarize
    keys = ["bin_mae_seconds", "bin_rmse_seconds", "trajectory_mae_seconds",
            "trajectory_rmse_seconds", "group_balanced_bin_mae_seconds"]
    paths = []
    for i, seed in enumerate([42, 43, 44]):
        p = tmp_path / str(seed); p.mkdir(); paths.append(p)
        config = dict(m_max=64, input_channels=3, d_model=16, heads=2, layers=1, dropout=0,
                      epochs=2, lr=.0002, weight_decay=.01, batch_size=2, data_seed=20260921,
                      max_groups=0, groups_per_partition=0, train_days=["20260817"],
                      val_days=["20260823"], source_sha256={"fixture": "fixture"}, seed=seed)
        (p / "config.json").write_text(json.dumps(config))
        (p / "best_val_metrics.json").write_text(json.dumps(dict(eval_mask_id="same",
            baseline={k: 4 for k in keys}, ours={k: i + 1 for k in keys}, best_epoch=0, paired_ci={})))
    result = summarize(paths)
    assert result["ours"]["bin_mae_seconds"] == {"mean": 2, "std": 1}
    p = paths[-1] / "best_val_metrics.json"
    bad = json.loads(p.read_text()); bad["eval_mask_id"] = "different"; p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="Validation groups or masks differ"):
        summarize(paths)


# Progress estimates are part of the training runner.


def _eta_batch(bucket, count, size):
    return dict(day=['23']*count, bucket=[str(bucket)]*count,
                partition_stats=[dict(day='23', bucket=str(bucket),selected_groups=size)])


def test_prefetched_counts_are_not_completed():
    eta=run.StageETA(4)
    for i in range(4): eta.update(_eta_batch(i,2,10))
    assert eta.fields(8)['eta']=='estimating'
    eta.update(_eta_batch(0,8,10))
    fields=eta.fields(16)
    assert fields['partitions_completed']=='1/4'
    assert fields['estimated_total_groups']==40
    assert fields['eta']=='00:00:24'


def test_known_total_and_cap():
    eta=run.StageETA(128,max_groups=20,known_total=100)
    eta.update(_eta_batch(0,10,10))
    assert eta.fields(5)['eta']=='00:00:05'
    assert eta.fields(5)['eta_basis']=='previous_full_pass'


def test_estimate_does_not_claim_completion_when_exceeded():
    eta=run.StageETA(1,known_total=1)
    eta.update(_eta_batch(0,2,2))
    assert eta.fields(5)['eta']=='re-estimating'
