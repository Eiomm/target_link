"""Census charts."""
from pathlib import Path
from datetime import datetime
import csv

def _plot(out: Path, bands: list[tuple[str, int, int]], daily: list[dict[str, str]], mask: list[dict[str, str]]) -> str | None:
    """Render the validated census tables."""
    import matplotlib.pyplot as plt
    total = sum(count for _, count, _ in bands)
    fig = plt.figure(figsize=(12, 7.5))
    grid = fig.add_gridspec(2, 2)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    temporal = fig.add_subplot(grid[1, :])
    with (out / "window_10min.csv").open() as stream:
        windows = list(csv.DictReader(stream))
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
