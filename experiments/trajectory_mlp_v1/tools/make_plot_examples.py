"""Reproduce two explicitly synthetic spatial-axis examples, without training."""
import json
from pathlib import Path

import numpy as np

from plot_link_bin_times import _json_number, _quantiles_by_bin, render_svg


def main():
    root = Path(__file__).resolve().parents[1] / "examples"
    root.mkdir(exist_ok=True)
    cases = []
    matrix = np.full((3, 50), np.nan)
    matrix[0, :5] = [1.0, 1.1, 1.3, 1.2, 1.0]
    matrix[1, 1:4] = [1.2, 1.4, 1.1]
    matrix[2, [0, 1, 3, 4]] = [0.9, 1.0, 1.2, 1.1]
    present = np.isfinite(matrix)
    present[2, 2] = True  # a recorded bin with unknown time, not an absent bin
    cases.append(("short_link_50m", matrix, present, 50.0))

    matrix = np.full((2, 50), np.nan)
    matrix[0, :8] = [1.0, 1.1, 1.2, 1.1, 1.0, 1.2, 1.3, 1.0]
    matrix[1, 12:17] = [1.4, 1.5, 1.4, 1.3, 1.2]
    cases.append(("offset_coverage_170m", matrix, np.isfinite(matrix), None))
    for name, matrix, present, length in cases:
        meta = dict(target_link_id="SYNTHETIC-" + name, seg_idx=0,
                    window_local="Illustration only", n_trajectories=len(matrix),
                    metric="raw", lower_percentile=5, upper_percentile=95,
                    segment_length_m=length, x_range_mode="coverage", synthetic=True)
        stats = _quantiles_by_bin(matrix)
        summary = render_svg(matrix, [f"example-{i}" for i in range(len(matrix))],
                             np.arange(len(matrix)) * 60 + 1787000400, meta, stats,
                             str(root / (name + ".svg")), present=present)
        summary.update(meta)
        # Keep committed examples portable; do not embed the author's machine path.
        summary["svg"] = name + ".svg"
        summary["per_bin"] = [{k: _json_number(v) if k.endswith("_s") else v
                               for k, v in row.items()} for row in stats]
        (root / (name + ".summary.json")).write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        print(name, "axis:", summary["x_min_m"], summary["x_max_m"])


if __name__ == "__main__":
    main()
