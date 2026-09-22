"""Render the validated seven-day P0 census outputs as a Chinese Markdown report.

This is deliberately a post-processing command: it reads compact CSV/JSON
artifacts emitted by ``census_seven_days.py`` and never opens observations or
Parquet source partitions.  A partial scan is rejected before ``REPORT.md`` is
written.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


DAYS = [str(day) for day in range(20260817, 20260824)]
BUCKETS = set(range(128))
EXPECTED_PARTITIONS = len(DAYS) * len(BUCKETS)
EXPECTED_WINDOWS = 7 * 24 * 6
STATIC_CITY_LINKS = 2_976_427


class ValidationError(ValueError):
    """The aggregate artifacts cannot support a formal seven-day report."""


def _json(path: Path) -> dict:
    if not path.exists():
        raise ValidationError(f"缺少必需文件：{path.name}")
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"无法读取 JSON：{path.name}") from exc


def _csv(path: Path, required: tuple[str, ...]) -> list[dict[str, str]]:
    if not path.exists():
        raise ValidationError(f"缺少必需文件：{path.name}")
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    fields = set(rows[0]) if rows else set()
    missing = set(required) - fields
    if missing:
        raise ValidationError(f"{path.name} 缺少列：{', '.join(sorted(missing))}")
    return rows


def _optional_csv(path: Path, required: tuple[str, ...]) -> list[dict[str, str]] | None:
    return _csv(path, required) if path.exists() else None


def _integer(row: dict[str, str], key: str, label: str) -> int:
    try:
        value = int(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValidationError(f"{label} 的 {key} 不是整数") from exc
    if value < 0:
        raise ValidationError(f"{label} 的 {key} 为负数")
    return value


def _number(row: dict[str, str], key: str, label: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValidationError(f"{label} 的 {key} 不是数值") from exc
    if value < 0:
        raise ValidationError(f"{label} 的 {key} 为负数")
    return value


def _sum(rows: list[dict[str, str]], key: str, label: str) -> int:
    return sum(_integer(row, key, label) for row in rows)


def _pct(numerator: float, denominator: float) -> str:
    # Gap and support rates often sit very close to 0% or 100%; two decimal
    # places would make a materially nonzero tail look exact.
    return "—" if denominator == 0 else f"{100 * numerator / denominator:.4f}%"


def _count(value: int | float) -> str:
    return f"{int(value):,}"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _check(condition: bool, message: str, checks: dict[str, bool]) -> None:
    checks[message] = bool(condition)
    if not condition:
        raise ValidationError(message)


def _expected_windows() -> list[int]:
    start = int(datetime(2026, 8, 17, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp())
    return list(range(start, start + EXPECTED_WINDOWS * 600, 600))


def _validate(out: Path) -> dict:
    """Load all report inputs and enforce cross-artifact accounting invariants."""
    checks: dict[str, bool] = {}
    complete = _json(out / "aggregation_complete.json")
    _check(complete.get("status") == "complete", "aggregation_complete.status 必须为 complete", checks)
    _check(complete.get("partitions") == EXPECTED_PARTITIONS, "完成标记的分区数必须为 896", checks)
    _check(complete.get("seed") == 20260921, "完成标记的 data_seed 必须为 20260921", checks)
    _check(complete.get("epoch") == 0 and complete.get("m_max") == 64,
           "完成标记的 epoch/m_max 必须为 0/64", checks)
    _check(complete.get("geometry_status") in {"dynamic_same_version", "static_same_version_subset"},
           "完成标记的 geometry_status 必须说明动态或静态同版本来源", checks)

    manifest = _json(out / "input_manifest.json")
    _check(manifest.get("partitions") == EXPECTED_PARTITIONS, "输入 manifest 的分区数必须为 896", checks)
    _check(manifest.get("days") == DAYS, "输入 manifest 的日期必须为 20260817..20260823", checks)

    # ``parts/`` also holds identity-audit sidecars.  Only the fixed
    # day_bucket receipt name participates in the source row accounting.
    receipts = [_json(path) for path in sorted((out / "parts").glob("????????_???.json"))]
    receipt_keys = {(str(row.get("day")), int(row.get("bucket", -1))) for row in receipts}
    expected_keys = {(day, bucket) for day in DAYS for bucket in BUCKETS}
    _check(len(receipts) == EXPECTED_PARTITIONS, "receipt 数量必须为 896", checks)
    _check(receipt_keys == expected_keys, "receipt 的 day/bucket 集合必须完整且唯一", checks)
    for receipt in receipts:
        _check(receipt.get("data_seed") == 20260921, "所有 receipt 的 data_seed 必须为 20260921", checks)
        for name in ("rows", "retained_observations", "dropped_no_valid", "dropped_tail",
                     "groups", "hidden_trajectories", "supervised_bins", "supported_supervised_bins"):
            _integer(receipt, name, "receipt")
        for name in ("n_present_hist", "n_valid_hist"):
            hist = receipt.get(name)
            _check(isinstance(hist, list) and len(hist) == 51,
                   f"receipt 的 {name} 必须为 51 项直方图", checks)
            _check(all(isinstance(value, int) and value >= 0 for value in hist),
                   f"receipt 的 {name} 必须为非负整数", checks)
            _check(sum(hist) == _integer(receipt, "rows", "receipt"),
                   f"receipt 的 {name} 总和必须等于 rows", checks)

    daily = _csv(out / "daily_totals.csv", ("day", "observations", "pieces", "recorded_bins",
                  "valid_bins", "internal_gap_observations", "usable_observations", "covered_m",
                  "cells", "links", "versioned_links", "distinct_traj_ids"))
    total_rows = _csv(out / "seven_day_totals.csv", ("observations", "pieces", "recorded_bins",
                       "valid_bins", "internal_gap_observations", "usable_observations", "covered_m",
                       "cells", "links", "versioned_links", "distinct_traj_ids"))
    _check(len(daily) == len(DAYS) and [row["day"] for row in daily] == DAYS,
           "daily_totals 必须恰含连续七天", checks)
    _check(len(total_rows) == 1, "seven_day_totals 必须恰一行", checks)
    total = total_rows[0]
    for key in ("observations", "pieces", "recorded_bins", "valid_bins", "internal_gap_observations",
                "usable_observations", "cells", "links", "versioned_links", "distinct_traj_ids"):
        _integer(total, key, "seven_day_totals")
    _number(total, "covered_m", "seven_day_totals")
    _check(_sum(receipts, "rows", "receipt") == _integer(total, "observations", "seven_day_totals"),
           "receipt rows 必须等于 seven_day_totals observations", checks)
    _check(_sum(daily, "observations", "daily_totals") == _integer(total, "observations", "seven_day_totals"),
           "daily observations 之和必须等于 seven_day_totals", checks)

    identity = _json(out / "identity_audit.json")
    _check(identity.get("status") == "passed", "identity_audit.status 必须为 passed", checks)
    _check(identity.get("partitions") == EXPECTED_PARTITIONS,
           "identity_audit 的分区数必须为 896", checks)
    _check(identity.get("rows") == _integer(total, "observations", "seven_day_totals"),
           "identity_audit rows 必须等于 seven_day_totals observations", checks)

    windows = _csv(out / "window_10min.csv", ("window_start", "local_time", "observations",
                    "distinct_traj_ids", "links", "cells"))
    observed_windows = [_integer(row, "window_start", "window_10min") for row in windows]
    _check(len(windows) == EXPECTED_WINDOWS, "window_10min 必须恰有 1008 个窗口", checks)
    _check(observed_windows == _expected_windows(), "window_10min 必须完整、连续且按 600 秒对齐", checks)
    _check(_sum(windows, "observations", "window_10min") == _integer(total, "observations", "seven_day_totals"),
           "window observations 之和必须等于 seven_day_totals", checks)

    integrity = _csv(out / "integrity.csv", ("observations", "partition_day_mismatch",
                     "malformed_sample_ids", "outside_days", "misaligned_windows"))
    _check(len(integrity) == 1, "integrity 必须恰一行", checks)
    _check(all(_integer(integrity[0], key, "integrity") == 0 for key in
               ("partition_day_mismatch", "malformed_sample_ids", "outside_days", "misaligned_windows")),
           "integrity 的异常计数必须全部为零", checks)

    coverage = _csv(out / "coverage_bins_distribution.csv", ("day", "n_present", "n_valid", "observations"))
    _check(_sum(coverage, "observations", "coverage_bins_distribution") ==
           _integer(total, "observations", "seven_day_totals"),
           "coverage 分布 observations 之和必须等于 seven_day_totals", checks)
    for row in coverage:
        present = _integer(row, "n_present", "coverage_bins_distribution")
        valid = _integer(row, "n_valid", "coverage_bins_distribution")
        _check(0 <= present <= 50 and 0 <= valid <= present,
               "coverage 的 n_valid/n_present 必须落在 0..50 且 valid<=present", checks)
    _check(sum(_integer(row, "n_present", "coverage") * _integer(row, "observations", "coverage")
               for row in coverage) == _integer(total, "recorded_bins", "seven_day_totals"),
           "coverage n_present 加权和必须等于 recorded_bins", checks)
    _check(sum(_integer(row, "n_valid", "coverage") * _integer(row, "observations", "coverage")
               for row in coverage) == _integer(total, "valid_bins", "seven_day_totals"),
           "coverage n_valid 加权和必须等于 valid_bins", checks)
    present_hist = [sum(receipt["n_present_hist"][index] for receipt in receipts) for index in range(51)]
    valid_hist = [sum(receipt["n_valid_hist"][index] for receipt in receipts) for index in range(51)]
    for index in range(51):
        _check(sum(_integer(row, "observations", "coverage") for row in coverage
                   if _integer(row, "n_present", "coverage") == index) == present_hist[index],
               "receipt n_present 直方图必须与 coverage 分布一致", checks)
        _check(sum(_integer(row, "observations", "coverage") for row in coverage
                   if _integer(row, "n_valid", "coverage") == index) == valid_hist[index],
               "receipt n_valid 直方图必须与 coverage 分布一致", checks)

    retention = _csv(out / "daily_training_retention.csv", ("day", "cells", "trainable_cells",
                     "observations", "dropped_no_valid", "dropped_small_tail", "retained_observations",
                     "training_groups"))
    mask = _csv(out / "daily_mask_reference.csv", ("day", "rows", "retained_observations",
                "dropped_no_valid", "dropped_tail", "groups", "hidden_trajectories", "supervised_bins",
                "supported_supervised_bins"))
    _check([row["day"] for row in retention] == DAYS and [row["day"] for row in mask] == DAYS,
           "retention 与 mask 参考必须恰含连续七天", checks)
    daily_by_day = {row["day"]: row for row in daily}
    for train, reference in zip(retention, mask):
        label = f"{train['day']} retention"
        raw = _integer(train, "observations", label)
        dropped_invalid = _integer(train, "dropped_no_valid", label)
        dropped_tail = _integer(train, "dropped_small_tail", label)
        retained = _integer(train, "retained_observations", label)
        _check(raw == dropped_invalid + dropped_tail + retained,
               "每日报告的 raw=无效剔除+尾组剔除+保留", checks)
        _check(raw == _integer(daily_by_day[train["day"]], "observations", "daily_totals"),
               "daily_training_retention observations 必须等于 daily_totals", checks)
        for left, right in (("observations", "rows"), ("dropped_no_valid", "dropped_no_valid"),
                            ("dropped_small_tail", "dropped_tail"),
                            ("retained_observations", "retained_observations"),
                            ("training_groups", "groups")):
            _check(_integer(train, left, label) == _integer(reference, right, "mask reference"),
                   "mask reference 必须与 SQL retention 一致", checks)
        _check(_integer(reference, "supported_supervised_bins", "mask reference") <=
               _integer(reference, "supervised_bins", "mask reference") <=
               _integer(reference, "hidden_trajectories", "mask reference") * 50,
               "same-bin 支持数必须落在监督目标范围内", checks)

    observations = _integer(total, "observations", "seven_day_totals")
    recorded = _integer(total, "recorded_bins", "seven_day_totals")
    valid = _integer(total, "valid_bins", "seven_day_totals")
    _check(valid <= recorded, "valid_bins 不得大于 recorded_bins", checks)
    _check(_integer(total, "internal_gap_observations", "seven_day_totals") <= observations and
           _integer(total, "usable_observations", "seven_day_totals") <= observations,
           "观测比例分子不得大于 observations", checks)

    return dict(complete=complete, manifest=manifest, receipts=receipts, daily=daily, total=total,
                windows=windows, coverage=coverage, retention=retention, mask=mask,
                present_hist=present_hist, valid_hist=valid_hist, identity=identity, checks=checks)


def _band_rows(coverage: list[dict[str, str]], observations: int) -> list[tuple[str, int, int]]:
    bands = (("0", 0, 0), ("1–5", 1, 5), ("6–10", 6, 10), ("11–20", 11, 20),
             ("21–40", 21, 40), ("41–50", 41, 50))
    result = []
    for label, lo, hi in bands:
        rows = [row for row in coverage if lo <= int(row["n_present"]) <= hi]
        n_obs = sum(int(row["observations"]) for row in rows)
        valid = sum(int(row["n_valid"]) * int(row["observations"]) for row in rows)
        result.append((label, n_obs, valid))
    assert sum(row[1] for row in result) == observations
    return result


def _k_bands(rows: list[dict[str, str]], kind: str) -> list[tuple[str, int]]:
    bands = (("0", 0, 0), ("1", 1, 1), ("2", 2, 2), ("3–9", 3, 9),
             ("10–63", 10, 63), ("64", 64, 64), ("65+", 65, 2**63 - 1))
    exact = [(int(row["k"]), int(row["cells"])) for row in rows if row["kind"] == kind]
    return [(label, sum(cells for k, cells in exact if lo <= k <= hi)) for label, lo, hi in bands]


def _table(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> str:
    line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join([line, separator, *body])


def _geometry_section(out: Path, observations: int, geometry_status: str) -> str:
    manifest_path = out / "geometry" / "manifest.json"
    match = _optional_csv(out / "geometry_match.csv", ("day", "match_status", "observations", "versioned_links"))
    coverage = _optional_csv(out / "exact_geometry_coverage.csv", ("day", "coverage_band", "observations"))
    differences = _optional_csv(out / "geometry_absolute_difference.csv", (
        "day", "length_band", "difference_band", "observations", "min_difference_m", "max_difference_m"))
    segment_types = _optional_csv(out / "geometry_segment_types.csv", (
        "day", "segment_type", "observations", "versioned_segments"))
    segment_lengths = _optional_csv(out / "exact_segment_lengths.csv", ("length_m", "segments"))
    sources = _optional_csv(out / "geometry_sources.csv", ("day", "geometry_source", "observations"))
    if not manifest_path.exists() or match is None or coverage is None:
        return """## 几何范围与缺项

当前正式产物缺少严格同版本的几何汇总 CSV，因此不报告道路长度或覆盖率。静态全城 `2,976,427` 条道路是独立道路底图基准，不能与本普查的 observed links 相加或替代。

`covered_m` 是 observation 中 `ratio_pct` 的求和统计，不是静态道路长度，不能反推路长。"""
    manifest = _json(manifest_path)
    city_lengths = _optional_csv(out / 'city_link_lengths.csv', ('length_band', 'links'))
    city_length_text = ""
    if city_lengths:
        city_total = sum(int(row['links']) for row in city_lengths)
        if city_total != int(manifest['unique_links']):
            raise ValidationError('静态路长分布总数与几何 manifest 不一致')
        city_length_text = "\n\n" + _table(
            ('静态全城道路长度', 'link 数', '占静态道路'),
            [(row['length_band'], _count(int(row['links'])), _pct(int(row['links']), city_total))
             for row in city_lengths],
        ) + "\n\n这张表按静态道路条数加权；observation 分布按观测条数加权，两者分母不同。"
    exact = sum(int(row["observations"]) for row in match if row["match_status"] == "exact_version")
    band_counts: dict[str, int] = {}
    for row in coverage:
        band_counts[row["coverage_band"]] = band_counts.get(row["coverage_band"], 0) + int(row["observations"])
    bands = sorted(band_counts.items())
    invalid_segment = sum(count for band, count in bands if band == "00_invalid_segment_geometry")
    over_105 = sum(count for band, count in bands if band == "06_over105pct")
    source_text = ""
    title = "仅严格同版本的静态路长参考覆盖比"
    if sources is None:
        reference_source = exact
        unmatched = observations - exact
        source_text = "几何来源分解 CSV 缺项；以下仅能按严格同版本键匹配汇报，不能判断动态与静态回退各自的实际覆盖。"
    else:
        by_source = {name: sum(int(row["observations"]) for row in sources if row["geometry_source"] == name)
                     for name in ("raw_dynamic_same_version", "static_same_version_fallback", "unmatched")}
        source_total = sum(by_source.values())
        if source_total != observations:
            raise ValidationError("geometry_sources observations 之和必须等于七天 observations")
        reference_source = by_source["raw_dynamic_same_version"] + by_source["static_same_version_fallback"]
        unmatched = by_source["unmatched"]
        if reference_source != exact:
            raise ValidationError("geometry_sources 的同版本来源数必须等于 geometry_match exact_version")
        if geometry_status == "dynamic_same_version":
            title = "同版本原始路长参考覆盖比"
            source_text = (f"来源：同版本 raw 动态 `L_link_m` {_count(by_source['raw_dynamic_same_version'])}（{_pct(by_source['raw_dynamic_same_version'], observations)}）；"
                           f"静态同版本回退 {_count(by_source['static_same_version_fallback'])}（{_pct(by_source['static_same_version_fallback'], observations)}）；"
                           f"未匹配 {_count(unmatched)}（{_pct(unmatched, observations)}）。")
        else:
            source_text = (f"来源：静态同版本回退 {_count(by_source['static_same_version_fallback'])}（{_pct(by_source['static_same_version_fallback'], observations)}）；"
                           f"未匹配 {_count(unmatched)}（{_pct(unmatched, observations)}）。动态同版本原始路长尚未在本次产物中提供。")
    reference_eligible = reference_source - invalid_segment
    geometry_unavailable = unmatched + invalid_segment
    difference_text = ""
    if differences is None:
        difference_text = "\n\n未生成按绝对差分档的几何参考表；该项为缺项，未以其他版本或 coverage 比替代。"
    else:
        difference_groups: dict[tuple[str, str], list[float]] = {}
        for row in differences:
            key = (row["length_band"], row["difference_band"])
            values = difference_groups.setdefault(key, [0, float("inf"), float("-inf")])
            values[0] += int(row["observations"])
            values[1] = min(values[1], float(row["min_difference_m"]))
            values[2] = max(values[2], float(row["max_difference_m"]))
        difference_text = "\n\n" + _table(
            ("参考段长档", "差值档（covered_m − 参考段长，m）", "观测数", "最小差(m)", "最大差(m)"),
            [(key[0], key[1], _count(values[0]), f"{values[1]:.4f}", f"{values[2]:.4f}")
             for key, values in sorted(difference_groups.items())],
        ) + "\n\n绝对差仅在严格同版本且参考子段长度为正的样本上计算。±5 m 仅表示与已知上游量化尺度一致，不能单独证明完整覆盖。"
    segment_text = ""
    if segment_types is None:
        segment_text += "\n\n未生成严格同版本的 segment 类型汇总；该项为缺项。"
    else:
        names = ("short_link_up_to_500m", "last_partial_segment", "full_500m_segment",
                 "invalid_segment_geometry")
        type_counts = {name: sum(int(row["observations"]) for row in segment_types
                                  if row["segment_type"] == name) for name in names}
        if sum(type_counts.values()) != exact:
            raise ValidationError("geometry_segment_types observations 之和必须等于 geometry_match exact_version")
        segment_text += "\n\n" + _table(
            ("严格同版本 segment 类型", "观测数", "占匹配观测"),
            [(name, _count(count), _pct(count, exact)) for name, count in type_counts.items()],
        ) + ("\n\n`short_link_up_to_500m` 是道路自身不超过 500 m；`last_partial_segment` 是较长道路的末段，"
           "二者不能混为同一种短覆盖现象；`full_500m_segment` 是完整 500 m 段。该分类不等同于 P0 有效 bin 覆盖。"
           "`versioned_segments` 是每日指标，不能跨日相加后称为七天去重段数。")
    if segment_lengths is None:
        segment_text += "\n\n未生成跨七天去重的严格同版本 segment 长度分布；该项为缺项。"
    else:
        lengths = [(float(row["length_m"]), int(row["segments"])) for row in segment_lengths]
        length_bands = (("0–50m", 0, 50), ("50–100m", 50, 100), ("100–200m", 100, 200),
                        ("200–500m", 200, 500))
        length_rows = [(name, sum(count for length, count in lengths if lo < length <= hi))
                       for name, lo, hi in length_bands]
        outside = sum(count for length, count in lengths if length <= 0 or length > 500)
        if outside:
            length_rows.append(("非正或 >500m（应审计）", outside))
        segment_text += "\n\n" + _table(
            ("跨七天去重的版本化 segment 长度", "segment 数"),
            [(name, _count(count)) for name, count in length_rows],
        ) + "\n\n此表来自已跨七天 `DISTINCT (map_version, target_link_id, seg_idx)` 的 `exact_segment_lengths.csv`，不是每日 segment 计数的求和。"
    return "\n".join([
        "## 几何范围与缺项",
        "",
        f"静态底图为 map_version={manifest.get('map_version', '未知')} 的 {_count(manifest.get('unique_links', STATIC_CITY_LINKS))} 条道路；全城基准仍单列为 {_count(STATIC_CITY_LINKS)}，不与 observed links 混算。",
        source_text,
        f"严格同版本 `(map_version, target_link_id)` 命中的观测为 {_count(exact)}（{_pct(exact, observations)}）；其中子段长度为正、可计算数值参考比的观测为 {_count(reference_eligible)}（{_pct(reference_eligible, observations)}）。未有可用同版本长度参考的观测为 {_count(geometry_unavailable)}（{_pct(geometry_unavailable, observations)}），包括来源未匹配 {_count(unmatched)} 与同版本但子段长度无效 {_count(invalid_segment)}。其他版本绝不以同数字 link id 跨版本 join。",
        "",
        _table((title, "观测数"), [(band, _count(count)) for band, count in bands]),
        "",
        f"`>105%` 参考比的观测为 {_count(over_105)}（占可计算同版本参考观测 {_pct(over_105, reference_eligible)}）。不截断该值，也不把它直接判为数据错误或“长轨迹不完整”：上游 `seg_mark` 为 bin 级，标记跨度相对 `L_link_m` 存在约 −5 至 +4 m 量化误差，且边界可包含非 target piece；短道路更容易出现放大。说明见 [上游切段记录](../../../../md/9.9progress.md)。",
        "",
        "上述档位比较量化后的 `covered_m` 与同版本参考子段长度；实际几何覆盖范围以上表为准，不能将 `covered_m` 本身当作路长。",
    ]) + difference_text + segment_text + city_length_text


def _plot(out: Path, bands: list[tuple[str, int, int]], daily: list[dict[str, str]], mask: list[dict[str, str]]) -> str | None:
    """Best-effort static overview; reporting remains dependency-free."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    total = sum(count for _, count, _ in bands)
    fig = plt.figure(figsize=(12, 7.5))
    grid = fig.add_gridspec(2, 2)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    temporal = fig.add_subplot(grid[1, :])
    windows = _csv(out / "window_10min.csv", ("local_time", "distinct_traj_ids"))
    timestamps = [datetime.strptime(row["local_time"], "%Y-%m-%d %H:%M:%S") for row in windows]
    temporal.plot(timestamps, [int(row["distinct_traj_ids"]) for row in windows], color="#2d756a", linewidth=1.1)
    temporal.set_title("Distinct traj IDs per 10-minute segment-entry window (Beijing time)")
    temporal.set_ylabel("distinct traj IDs")
    temporal.grid(alpha=0.2)
    import matplotlib.dates as mdates
    temporal.xaxis.set_major_locator(mdates.DayLocator())
    temporal.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    axes[0].bar([name for name, _, _ in bands], [100 * count / total for _, count, _ in bands], color="#3478bf")
    axes[0].set_ylabel("observations (%)")
    axes[0].set_title("P0 present-bin coverage")
    axes[1].plot([row["day"][-2:] for row in daily],
                 [100 * int(row["valid_bins"]) / int(row["recorded_bins"]) for row in daily],
                 marker="o", label="valid / recorded")
    axes[1].plot([row["day"][-2:] for row in mask],
                 [100 * int(row["supported_supervised_bins"]) / int(row["supervised_bins"])
                  if int(row["supervised_bins"]) else 0 for row in mask], marker="o", label="same-bin support")
    axes[1].set_ylabel("rate (%)")
    axes[1].set_title("Daily label availability")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    path = out / "census_overview.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def _render(out: Path, data: dict) -> str:
    total, daily, coverage = data["total"], data["daily"], data["coverage"]
    observations = int(total["observations"])
    recorded, valid = int(total["recorded_bins"]), int(total["valid_bins"])
    bands = _band_rows(coverage, observations)
    cell_rows = _csv(out / "cell_k_distribution.csv", ("day", "kind", "k", "cells"))
    raw_k, usable_k = _k_bands(cell_rows, "raw"), _k_bands(cell_rows, "usable")
    raw_cells, usable_cells = sum(count for _, count in raw_k), sum(count for _, count in usable_k)
    retention = data["retention"]
    mask = data["mask"]
    trainable = sum(int(row["trainable_cells"]) for row in retention)
    retained = sum(int(row["retained_observations"]) for row in retention)
    dropped_invalid = sum(int(row["dropped_no_valid"]) for row in retention)
    dropped_tail = sum(int(row["dropped_small_tail"]) for row in retention)
    training_groups = sum(int(row["training_groups"]) for row in retention)
    supervised = sum(int(row["supervised_bins"]) for row in mask)
    supported = sum(int(row["supported_supervised_bins"]) for row in mask)
    windows = _csv(out / "window_10min.csv", ("local_time", "distinct_traj_ids", "observations"))
    peak = max(windows, key=lambda row: int(row["distinct_traj_ids"]))
    trough = min(windows, key=lambda row: int(row["distinct_traj_ids"]))
    image = _plot(out, bands, daily, mask)

    daily_rows = []
    train_by_day = {row["day"]: row for row in retention}
    mask_by_day = {row["day"]: row for row in mask}
    for row in daily:
        train, reference = train_by_day[row["day"]], mask_by_day[row["day"]]
        daily_rows.append((row["day"], _count(int(row["observations"])),
                           _pct(int(row["valid_bins"]), int(row["recorded_bins"])),
                           _pct(int(row["internal_gap_observations"]), int(row["observations"])),
                           _pct(int(train["trainable_cells"]), int(train["cells"])),
                           _pct(int(reference["supported_supervised_bins"]), int(reference["supervised_bins"]))))

    sections = [
        "# 七天 observations_v2 P0 普查报告",
        "",
        "**范围：** 2026-08-17 至 2026-08-23，北京时间，每个 10 分钟窗口。全量扫描七天 observations_v2，覆盖 896 个分区；结果已通过 1008 个连续窗口、身份字段及跨表总数核对。",
        "",
        "## 口径",
        "",
        "P0 中 `present` 表示该绝对 bin 至少有一个 piece；`valid` 表示该 bin 的所有 piece 都有效。`n_valid>0` 才是可进入训练分组的轨迹。`covered_m` 为原始 `ratio_pct` 求和，名称沿用普查产物，不能当作静态道路长度。",
        "",
        "`distinct_traj_ids` 是 `sample_id` 第一段的去重轨迹标识数，**不是订单数**。掩码统计严格使用 `data_seed=20260921`、`epoch=0`、`m_max=64` 与训练读取器相同的成员和掩码规则。",
        "当前实验将 08-17 至 08-22 用于训练、08-23 用于验证。本报告对七天统一使用 epoch=0 做 P0 掩码参考；实际验证读取器固定使用 `VAL_EPOCH=1000000`，所以这里不是实际验证掩码的统计。下文“进入训练组”表示满足分组规则，并不把 08-23 改作训练集。",
        "",
        "## 七天总量",
        "",
        _table(("指标", "数值"), [
            ("观测", _count(observations)), ("pieces", _count(int(total["pieces"]))),
            ("出现 bin", _count(recorded)), ("有效 bin", _count(valid)),
            ("有效 / 出现", _pct(valid, recorded)),
            ("内部缺口观测", f"{_count(int(total['internal_gap_observations']))}（{_pct(int(total['internal_gap_observations']), observations)}）"),
            ("可用观测", f"{_count(int(total['usable_observations']))}（{_pct(int(total['usable_observations']), observations)}）"),
            ("观测道路 link", _count(int(total["links"]))),
            ("观测版本化 link", _count(int(total["versioned_links"]))),
            ("去重轨迹标识", _count(int(total["distinct_traj_ids"]))),
        ]),
        "",
        f"静态全城道路基准为 {_count(STATIC_CITY_LINKS)} 条（独立于观测道路计数，不能相加或互相替代）。",
        "",
        "## P0 覆盖与有效性",
        "",
        _table(("n_present 档", "观测", "观测占比", "有效 bin", "档内有效 / 出现"), [
            (name, _count(count), _pct(count, observations), _count(valid_bins), _pct(valid_bins, sum(
                int(row["n_present"]) * int(row["observations"]) for row in coverage
                if (name == "0" and int(row["n_present"]) == 0) or
                (name == "1–5" and 1 <= int(row["n_present"]) <= 5) or
                (name == "6–10" and 6 <= int(row["n_present"]) <= 10) or
                (name == "11–20" and 11 <= int(row["n_present"]) <= 20) or
                (name == "21–40" and 21 <= int(row["n_present"]) <= 40) or
                (name == "41–50" and 41 <= int(row["n_present"]) <= 50))))
            for name, count, valid_bins in bands]),
        "",
        "短覆盖档为 1–5、6–10、11–20、21–40、41–50；`0` 单列，避免把无出现 bin 的观测混入短覆盖。内部缺口比例见总量表。",
        "内部缺口指最小与最大已出现 bin 之间仍有未记录位置；首尾未出现的位置不计入这个指标，因此不能用它判断道路首尾是否完整。",
        "",
        f"每条 observation 平均有记录 bin {recorded/observations:.4f} 个、有效 bin {valid/observations:.4f} 个。固定 50-bin 网格的记录利用率为 {_pct(recorded, observations*50)}；这不是道路完整率，剩余位置可能只是短道路之外的 padding。",
        "",
        "## Cell K 与训练保留",
        "",
        _table(("K 档", "原始 cell", "可用 cell"), [(name, _count(raw), _count(usable))
               for (name, raw), (_, usable) in zip(raw_k, usable_k)]),
        "",
        f"可训练 cell（可用 K≥3）：{_count(trainable)} / {_count(raw_cells)}（{_pct(trainable, raw_cells)}）。",
        "",
        _table(("保留去向", "观测", "占原始观测"), [
            ("无有效 bin 剔除", _count(dropped_invalid), _pct(dropped_invalid, observations)),
            ("不足 3 条的尾组剔除", _count(dropped_tail), _pct(dropped_tail, observations)),
            ("进入训练组", _count(retained), _pct(retained, observations)),
            ("训练组", _count(training_groups), "—"),
        ]),
        "",
        "## 精确 epoch-0 掩码参考",
        "",
        f"隐藏轨迹的监督有效 bin：{_count(supervised)}；其中同 bin 至少存在一条可见轨迹支持的目标：{_count(supported)}（same-bin 支持率 {_pct(supported, supervised)}）。该比例是固定成员与 epoch=0 掩码的精确结果，不是随机重复训练的平均值。",
        "同一 cell 对应同一个地图版本、target link 子段和时间窗口，因此轨迹记录位置高度重合是可以预期的。这个比例衡量同位置参考是否存在，不衡量参考轨迹与目标轨迹的数值是否接近，也不代表模型预测精度。",
        "",
        "## 每日明细",
        "",
        _table(("日期", "observation", "去重 traj_id", "观测 link"), [
            (r["day"], _count(int(r["observations"])), _count(int(r["distinct_traj_ids"])), _count(int(r["links"])))
            for r in daily]),
        "",
        "同一 traj_id/link 可以跨天出现，七天去重总数不能用每日去重数相加。窗口中的 traj_id 表示该窗口有 segment observation 归属的实体，不是订单起点数或持续在途车辆数。",
        "",
        _table(("日期", "观测", "有效/出现", "内部缺口", "可训练 cell", "same-bin 支持"), daily_rows),
        "",
        "每 10 分钟完整 1008 个窗口（含零观测窗口）的明细见 [window_10min.csv](window_10min.csv)。",
        f"最多轨迹的窗口起点为 {peak['local_time']}，去重 traj_id {_count(int(peak['distinct_traj_ids']))}；"
        f"最少的窗口起点为 {trough['local_time']}，去重 traj_id {_count(int(trough['distinct_traj_ids']))}。"
        "窗口区间为 [起点, 起点+10分钟)，时间均为北京时间。",
        "",
        _geometry_section(out, observations, data["complete"]["geometry_status"]),
        "",
        "## 校验与限制",
        "",
        "`validation_summary.json` 记录了 receipt、identity audit、总计、窗口、coverage 直方图、训练保留与掩码参考的交叉校验。只有这些校验全部通过，才会生成本报告。",
        "",
        ("几何只接受 `(map_version, target_link_id)` 的严格同版本匹配。动态同版本原始路长已参与本次产物；实际未匹配比例以 `geometry_sources.csv` 为准，不能笼统说其他版本均未覆盖。"
         if data["complete"]["geometry_status"] == "dynamic_same_version" else
         "几何只接受 `(map_version, target_link_id)` 的严格同版本匹配。本次只有静态 2026081412 子集；其他版本的精确道路长度仍需要从七天 raw 业务表按同版本键提取并完成唯一性审计。现有 Spark 客户端未能继承集群定制认证，动态路长任务未成功启动；详见 [提取状态](geometry/dynamic_geometry_status.json)。这项缺失不影响七天 observation、traj_id、link、窗口和 bin 统计。"),
    ]
    if image:
        sections.extend(["", f"静态概览：![P0 census overview]({image})"])
    return "\n".join(sections) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="seven_day_p0 聚合输出目录")
    args = parser.parse_args()
    out = args.out
    try:
        data = _validate(out)
        body = _render(out, data)
    except ValidationError as exc:
        _write_json(out / "validation_summary.json", {"status": "rejected", "error": str(exc)})
        raise SystemExit(f"拒绝生成正式报告：{exc}")
    _write_json(out / "validation_summary.json", {
        "status": "passed", "checks": data["checks"], "partitions": EXPECTED_PARTITIONS,
        "windows": EXPECTED_WINDOWS, "data_seed": 20260921, "epoch": 0, "m_max": 64,
        "observations": int(data["total"]["observations"]),
    })
    (out / "REPORT.md").write_text(body)
    print(out / "REPORT.md")


if __name__ == "__main__":
    main()
