import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "extract_representative_cells", ROOT / "tools" / "extract_representative_cells.py")
extract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(extract)


def test_parse_days_and_slots():
    assert extract.parse_days("20260817,20260818") == ["20260817", "20260818"]
    assert extract.parse_slots("4,1,4") == [1, 4]
    with pytest.raises(ValueError):
        extract.parse_days("2026-08-17")
    with pytest.raises(ValueError):
        extract.parse_slots("6")


def test_tier_quotas_are_balanced_and_exact():
    assert extract.tier_quotas(48) == [12, 12, 12, 12]
    assert extract.tier_quotas(50) == [13, 13, 12, 12]
    assert sum(extract.tier_quotas(63)) == 63
    assert max(extract.tier_quotas(63)) - min(extract.tier_quotas(63)) == 1


def test_parser_coverage_defaults_are_week_wide():
    args = extract.parser().parse_args(["--corpus", "hdfs://c/corpus", "--out", "hdfs://c/out"])
    assert args.min_active_days == 7
    assert args.min_hour_slots == 24
    assert args.min_active_hours == 84
