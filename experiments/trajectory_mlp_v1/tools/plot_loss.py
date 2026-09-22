"""Render epoch MAE curves from the recorded training metrics (headless)."""
import argparse
import csv
import json
from pathlib import Path


def plot_loss(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(out)
    records = [json.loads(line) for line in (out/'metrics.jsonl').read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError('No completed epochs in metrics.jsonl')
    rows = [dict(epoch=r['epoch']+1, train_mae_seconds=r['train']['loss_mae_seconds'],
                 val_mae_seconds=r['val']['ours']['bin_mae_seconds']) for r in records]
    baseline = json.loads((out/'baseline.json').read_text())['baseline']['bin_mae_seconds']
    with (out/'loss_curve.csv.tmp').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (out/'loss_curve.csv.tmp').replace(out/'loss_curve.csv')
    fig, ax = plt.subplots(figsize=(9,5))
    try:
        epochs = [r['epoch'] for r in rows]
        ax.plot(epochs, [r['train_mae_seconds'] for r in rows], 'o-', label='Train MAE (online epoch average)')
        ax.plot(epochs, [r['val_mae_seconds'] for r in rows], 'o-', label='Validation MAE (fixed masks)')
        ax.axhline(baseline, color='gray', linestyle='--', label=f'Visible-mean baseline ({baseline:.4f} s)')
        best = min(rows, key=lambda r:r['val_mae_seconds'])
        ax.scatter([best['epoch']], [best['val_mae_seconds']], marker='*', s=180,
                   color='red', zorder=5, label=f"Best validation: epoch {best['epoch']}")
        ax.set(xlabel='Epoch', ylabel='MAE / loss (seconds)', title='Trajectory MLP: training and validation loss')
        ax.set_xticks(epochs)
        ax.grid(alpha=.25); ax.legend(fontsize=9)
        fig.tight_layout()
        for suffix in ('png', 'svg'):
            temp = out/f'loss_curve.{suffix}.tmp'
            fig.savefig(temp, format=suffix, dpi=160)
            temp.replace(out/f'loss_curve.{suffix}')
    finally:
        plt.close(fig)
    plot_diagnostics(out, records)
    return out/'loss_curve.png'


def _save(fig, path, plt):
    try:
        fig.tight_layout()
        temp = path.with_suffix('.png.tmp')
        fig.savefig(temp, format='png', dpi=140)
        temp.replace(path)
    finally:
        plt.close(fig)


def plot_steps(out):
    """Refresh sampled step curves without requiring a completed epoch."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(out)
    path = out/'step_metrics.csv'
    if not path.exists():
        return None
    with path.open() as stream:
        rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
    if not rows:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    x = [r['global_step'] for r in rows]
    axes[0, 0].plot(x, [r['batch_mae_seconds'] for r in rows], alpha=.6, label='Sampled batch MAE')
    axes[0, 0].plot(x, [r['running_mae_seconds'] for r in rows], label='Running epoch MAE')
    axes[0, 0].set_ylabel('Loss (seconds)'); axes[0, 0].legend()
    for ax, key, title in [(axes[0, 1], 'grad_norm', 'Gradient norm (before clipping)'),
                            (axes[1, 0], 'lr', 'Learning rate'),
                            (axes[1, 1], 'groups_per_second', 'Throughput (groups/s)')]:
        ax.plot(x, [r[key] for r in rows]); ax.set_title(title)
    for ax in axes.flat:
        ax.set_xlabel('Global training step'); ax.grid(alpha=.25)
    fig.suptitle('Sampled training trends (not every batch; epoch average resets each epoch)')
    _save(fig, out/'step_trends.png', plt)
    return out/'step_trends.png'


def plot_diagnostics(out, records):
    import matplotlib.pyplot as plt
    if 'grad_norm_mean' not in records[0]['train']:
        return  # Old logs remain plottable.
    rows = [dict(epoch=r['epoch']+1,
                 **{k:r['train'][k] for k in ('lr', 'grad_norm_mean', 'grad_norm_max',
                                             'parameter_norm', 'groups_per_second', 'gpu_peak_GiB')},
                 val_rmse_seconds=r['val']['ours']['bin_rmse_seconds']) for r in records]
    with (out/'epoch_diagnostics.csv.tmp').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (out/'epoch_diagnostics.csv.tmp').replace(out/'epoch_diagnostics.csv')
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    x = [r['epoch'] for r in rows]
    for ax, key, title in zip(axes.flat,
            ('lr', 'grad_norm_mean', 'parameter_norm', 'val_rmse_seconds', 'groups_per_second', 'gpu_peak_GiB'),
            ('Learning rate', 'Gradient norm (before clipping)', 'Parameter L2 norm',
             'Validation RMSE (seconds)', 'Throughput (groups/s)', 'Training peak allocated GPU memory (GiB)')):
        ax.plot(x, [r[key] for r in rows], 'o-', label=key)
        if key == 'grad_norm_mean':
            ax.plot(x, [r['grad_norm_max'] for r in rows], 'o--', label='max'); ax.legend()
        ax.set(title=title, xlabel='Epoch'); ax.set_xticks(x); ax.grid(alpha=.25)
    _save(fig, out/'training_diagnostics.png', plt)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True, help='artifacts directory containing metrics.jsonl and baseline.json')
    print(plot_loss(p.parse_args().out))
