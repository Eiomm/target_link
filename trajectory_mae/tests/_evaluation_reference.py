"""Auditable raw-seconds evaluation for trajectory_mlp_v1.

The module deliberately owns the common denominator: hidden, valid bins in a
group with at least one valid visible value.  Both the learned prediction and
the same-bin-mean baseline are scored on exactly that set.
"""
from __future__ import annotations

import csv
import base64
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def _masks(batch):
    traj_valid = batch["traj_valid"].bool()
    hidden = batch["mae_mask"].bool() & traj_valid
    valid = batch["bin_valid"].bool() & traj_valid.unsqueeze(-1)
    visible_valid = (~batch["mae_mask"].bool() & traj_valid).unsqueeze(-1) & valid
    group_has_visible = visible_valid.any(dim=(1, 2))
    supervised = hidden.unsqueeze(-1) & valid & group_has_visible[:, None, None]
    return visible_valid, supervised, group_has_visible


def _selected_finite_nonnegative(value, mask, name):
    selected = value[mask]
    if selected.numel() and (not torch.isfinite(selected).all() or (selected < 0).any()):
        raise ValueError("%s must be finite and nonnegative on supervised positions" % name)


def visible_mean(batch):
    """Raw same-bin visible mean and its fallback, without reading hidden labels."""
    raw = batch["x"][..., 0]
    visible_valid, _, group_has_visible = _masks(batch)
    _selected_finite_nonnegative(raw, visible_valid, "visible target")
    safe_raw = torch.where(visible_valid, raw, torch.zeros_like(raw))
    same_count = visible_valid.sum(1)                    # [B, 50]
    same_sum = safe_raw.sum(1)
    all_count = same_count.sum(1)
    all_sum = same_sum.sum(1)
    fallback = all_sum / all_count.clamp_min(1).to(raw.dtype)
    same_mean = same_sum / same_count.clamp_min(1).to(raw.dtype)
    values = torch.where(same_count > 0, same_mean, fallback[:, None])
    return {"prediction_seconds": values[:, None, :].expand_as(raw),
            "same_bin_support": same_count > 0,
            "group_has_visible": group_has_visible}


def reconstruction_loss(prediction_seconds, batch):
    """Micro MAE in raw seconds over the shared hidden-valid denominator."""
    _, supervised, _ = _masks(batch)
    target = batch["x"][..., 0]
    if prediction_seconds.shape != target.shape:
        raise ValueError("prediction_seconds shape %s != target shape %s" %
                         (tuple(prediction_seconds.shape), tuple(target.shape)))
    _selected_finite_nonnegative(target, supervised, "target")
    _selected_finite_nonnegative(prediction_seconds, supervised, "prediction")
    if not supervised.any():
        # Preserve a differentiable zero without ever reducing an unselected
        # NaN (``NaN * 0`` is still NaN).
        safe = torch.where(torch.isfinite(prediction_seconds), prediction_seconds,
                           torch.zeros_like(prediction_seconds))
        return safe.sum() * 0.0
    # Selecting first prevents unselected NaNs from poisoning the reduction.
    return (prediction_seconds[supervised] - target[supervised]).abs().mean()


def _empty_stats():
    return dict(bins=0, trajectories=0, groups=0, abs_sum=0.0, sq_sum=0.0,
                total_abs_sum=0.0, total_sq_sum=0.0, group_mae_sum=0.0,
                same_supported=0, fallback=0, unscorable=0)


def _metrics(s):
    n, t, g = s["bins"], s["trajectories"], s["groups"]
    return dict(bin_mae_seconds=(s["abs_sum"] / n if n else None),
                bin_rmse_seconds=((s["sq_sum"] / n) ** .5 if n else None),
                trajectory_mae_seconds=(s["total_abs_sum"] / t if t else None),
                trajectory_rmse_seconds=((s["total_sq_sum"] / t) ** .5 if t else None),
                group_balanced_bin_mae_seconds=(s["group_mae_sum"] / g if g else None),
                bins=n, trajectories=t, groups=g,
                same_bin_coverage=(s["same_supported"] / n if n else None),
                fallback_count=s["fallback"], fallback_rate=(s["fallback"] / n if n else None),
                unscorable_groups=s["unscorable"])


class Evaluator:
    """Accumulate paired baseline/Ours metrics, optionally writing audit rows."""

    def __init__(self, output_dir=None):
        self.base = _empty_stats()
        self.ours = _empty_stats()
        self.groups_total = 0
        self.no_supervised_groups = 0
        self._cells = []
        self._identity_hashes = []
        self._strata = defaultdict(lambda: defaultdict(lambda: {"baseline": _empty_stats(), "ours": _empty_stats()}))
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self._csv = None
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._csv = open(self.output_dir / "predictions.csv", "w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._csv, fieldnames=[
                "cell_id", "group_id", "sample_id", "slot", "bin", "target_seconds",
                "baseline_seconds", "prediction_seconds", "same_bin_support", "used_fallback",
                "K", "valid_bin_count", "target_ratio", "ratio_sum", "full50", "fullratio"])
            self._writer.writeheader()
            self._identity_file = open(self.output_dir / "eval_mask_identity.jsonl", "w", encoding="utf-8")
        else:
            self._identity_file = None

    @staticmethod
    def _add(stats, target, prediction, mask, support, fallback, group_has_visible):
        """Update stats from a [B,M,N] tensor.  This only reads selected values."""
        # One transfer per batch keeps host aggregation accurate and avoids a
        # GPU synchronization for every bin.
        target = target.detach().to(device="cpu", dtype=torch.float64)
        prediction = prediction.detach().to(device="cpu", dtype=torch.float64)
        mask, support, group_has_visible = (x.detach().cpu() for x in (mask, support, group_has_visible))
        B = mask.shape[0]
        stats["unscorable"] += int((~group_has_visible).sum().item())
        for b in range(B):
            m = mask[b]
            if not bool(m.any()):
                continue
            e = prediction[b][m] - target[b][m]
            stats["bins"] += int(e.numel())
            stats["abs_sum"] += float(e.abs().sum().item())
            stats["sq_sum"] += float(e.square().sum().item())
            stats["same_supported"] += int(support[b].unsqueeze(0).expand_as(m)[m].sum().item())
            stats["fallback"] += int((~support[b].unsqueeze(0).expand_as(m))[m].sum().item())
            traj = m.any(-1)
            totals = torch.where(m, prediction[b] - target[b], torch.zeros_like(target[b])).sum(-1)[traj]
            stats["trajectories"] += int(traj.sum().item())
            stats["total_abs_sum"] += float(totals.abs().sum().item())
            stats["total_sq_sum"] += float(totals.square().sum().item())
            per_traj = torch.where(m, (prediction[b] - target[b]).abs(), torch.zeros_like(target[b])).sum(-1)
            per_traj = per_traj[traj] / m.sum(-1)[traj].to(per_traj.dtype)
            stats["groups"] += 1
            stats["group_mae_sum"] += float(per_traj.mean().item())

    def _identity(self, batch, supervised):
        for b, gid in enumerate(batch.get("group_id", range(supervised.shape[0]))):
            sample_ids = batch.get("sample_ids", [[] for _ in range(supervised.shape[0])])[b]
            visible = (~batch["mae_mask"][b].bool() & batch["traj_valid"][b].bool()).detach().cpu().numpy()
            hidden = batch["mae_mask"][b].bool().detach().cpu().numpy()
            valid = supervised[b].bool().detach().cpu().numpy()
            record = dict(group_id=str(gid), sample_ids=list(map(str, sample_ids)),
                visible_bits=base64.b64encode(np.packbits(visible).tobytes()).decode(),
                hidden_bits=base64.b64encode(np.packbits(hidden).tobytes()).decode(),
                valid_supervision_bits=base64.b64encode(np.packbits(valid.reshape(-1)).tobytes()).decode(),
                visible_shape=list(visible.shape), valid_supervision_shape=list(valid.shape))
            canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
            self._identity_hashes.append(hashlib.sha256(canonical.encode()).hexdigest())
            if self._identity_file is not None:
                self._identity_file.write(canonical + "\n")

    def update(self, prediction_seconds_or_none, batch):
        baseline = visible_mean(batch)
        _, mask, group_has_visible = _masks(batch)
        target = batch["x"][..., 0]
        _selected_finite_nonnegative(target, mask, "target")
        self.groups_total += int(mask.shape[0])
        self.no_supervised_groups += int((~mask.any(dim=(1, 2))).sum().item())
        prediction = prediction_seconds_or_none
        if prediction is not None:
            if prediction.shape != target.shape:
                raise ValueError("prediction_seconds shape %s != target shape %s" % (tuple(prediction.shape), tuple(target.shape)))
            _selected_finite_nonnegative(prediction, mask, "prediction")
        self._identity(batch, mask)
        self._add(self.base, target, baseline["prediction_seconds"], mask, baseline["same_bin_support"],
                  ~baseline["same_bin_support"], group_has_visible)
        if prediction is not None:
            self._add(self.ours, target, prediction, mask, baseline["same_bin_support"],
                      ~baseline["same_bin_support"], group_has_visible)
        self._add_strata(target, baseline["prediction_seconds"], prediction, mask, baseline["same_bin_support"], group_has_visible, batch)
        # One paired record per group/cell is enough for a correct cluster bootstrap.
        for b in range(mask.shape[0]):
            m = mask[b]
            if not bool(m.any()):
                continue
            be = baseline["prediction_seconds"][b][m] - target[b][m]
            oe = prediction[b][m] - target[b][m] if prediction is not None else None
            cell = int(batch.get("cell_id", torch.arange(mask.shape[0]))[b])
            self._cells.append(dict(cell_id=cell, bins=int(m.sum()), base_abs=float(be.abs().sum()),
                                    ours_abs=(float(oe.abs().sum()) if oe is not None else None)))
        self._write_rows(batch, target, baseline, prediction, mask)
        return self

    def _add_strata(self, target, baseline, prediction, mask, support, group_has_visible, batch):
        """Add the required diagnostic strata; all retain the paired mask."""
        B, M, N = mask.shape
        ratio = batch["x"][..., 1]
        for b in range(B):
            if not bool(mask[b].any()):
                continue
            valid_counts = batch["bin_valid"][b].sum(-1)
            hidden_rows = mask[b].any(-1)
            def add(category, value, selected):
                if not bool(selected.any()):
                    return
                entry = self._strata[category][value]
                sl, selected = slice(b, b + 1), selected.unsqueeze(0)
                self._add(entry["baseline"], target[sl], baseline[sl], selected, support[sl], ~support[sl], group_has_visible[sl])
                if prediction is not None:
                    self._add(entry["ours"], target[sl], prediction[sl], selected, support[sl], ~support[sl], group_has_visible[sl])

            add("K", str(int(batch.get("K", torch.tensor([M]))[b])), mask[b])
            # A valid-bin-length stratum is per hidden trajectory, not a
            # rounded mean over a group.
            for row in torch.nonzero(hidden_rows, as_tuple=False).flatten().tolist():
                row_mask = torch.zeros_like(mask[b])
                row_mask[row] = True
                add("valid_bin_length", str(int(valid_counts[row])), mask[b] & row_mask)

            fullratio = (valid_counts == N) & (((ratio[b] - 1).abs() * batch["bin_valid"][b]).amax(-1) < 1e-6)
            # Explicit axis handling is intentional: M can equal N (=50).
            add("same_bin_support", "same_bin", mask[b] & support[b][None, :])
            add("same_bin_support", "fallback", mask[b] & ~support[b][None, :])
            add("target_ratio", "full", mask[b] & ((ratio[b] - 1).abs() < 1e-6))
            add("target_ratio", "partial", mask[b] & ((ratio[b] - 1).abs() >= 1e-6))
            add("full50", "full50", mask[b] & (valid_counts == N)[:, None])
            add("full50", "partial", mask[b] & (valid_counts != N)[:, None])
            add("fullratio", "full", mask[b] & fullratio[:, None])
            add("fullratio", "partial", mask[b] & ~fullratio[:, None])

    def _write_rows(self, batch, target, baseline, prediction, mask):
        if self._csv is None:
            return
        B, M, _ = mask.shape
        for b in range(B):
            ids = batch.get("sample_ids", [[] for _ in range(B)])[b]
            for slot, bin_ in torch.nonzero(mask[b], as_tuple=False).tolist():
                valid_n = int(batch["bin_valid"][b, slot].sum())
                valid_ratio = batch["x"][b, slot, :, 1][batch["bin_valid"][b, slot]]
                ratio = float(valid_ratio.sum())
                is_fullratio = valid_n == 50 and bool(((valid_ratio - 1).abs() < 1e-6).all())
                support = bool(baseline["same_bin_support"][b, bin_])
                self._writer.writerow(dict(cell_id=int(batch.get("cell_id", torch.arange(B))[b]), group_id=batch.get("group_id", [b])[b],
                    sample_id=(ids[slot] if slot < len(ids) else ""), slot=slot, bin=bin_, target_seconds=float(target[b,slot,bin_]),
                    baseline_seconds=float(baseline["prediction_seconds"][b,slot,bin_]), prediction_seconds=("" if prediction is None else float(prediction[b,slot,bin_])),
                    same_bin_support=int(support), used_fallback=int(not support), K=int(batch.get("K", torch.zeros(B))[b]),
                    valid_bin_count=valid_n, target_ratio=float(batch["x"][b, slot, bin_, 1]), ratio_sum=ratio,
                    full50=int(valid_n == 50), fullratio=int(is_fullratio)))

    def finalize(self, bootstrap=0, seed=20260921):
        if self._csv is not None:
            self._csv.flush(); self._csv.close(); self._csv = None
        if self._identity_file is not None:
            self._identity_file.flush(); self._identity_file.close(); self._identity_file = None
        result = {"baseline": _metrics(self.base), "ours": (_metrics(self.ours) if self.ours["bins"] else None),
                  "groups": self.base["groups"], "groups_total": self.groups_total,
                  "no_supervised_groups": self.no_supervised_groups,
                  "bins": self.base["bins"], "trajectories": self.base["trajectories"],
                  "same_bin_coverage": self.base["same_supported"] / self.base["bins"] if self.base["bins"] else None,
                  "fallback_count": self.base["fallback"], "fallback_rate": self.base["fallback"] / self.base["bins"] if self.base["bins"] else None,
                  "unscorable_groups": self.base["unscorable"]}
        digest = hashlib.sha256("\n".join(sorted(self._identity_hashes)).encode()).hexdigest()
        result["eval_mask_id"] = digest
        if self.ours["bins"]:
            result["paired_ci"] = {"ours_minus_baseline_bin_mae_seconds": result["ours"]["bin_mae_seconds"] - result["baseline"]["bin_mae_seconds"],
                                    "ci95": self._bootstrap(bootstrap, seed),
                                    "effective_clusters": len({x["cell_id"] for x in self._cells}),
                                    "ci_status": ("ok" if len({x["cell_id"] for x in self._cells}) >= 2 else "insufficient_clusters")}
        result["stratified"] = {category: {value: {"baseline": _metrics(s["baseline"]), "ours": (_metrics(s["ours"]) if s["ours"]["bins"] else None)} for value, s in values.items()} for category, values in self._strata.items()}
        if self.output_dir is not None:
            (self.output_dir / "eval_mask.json").write_text(json.dumps({"eval_mask_id": digest, "groups": len(self._identity_hashes), "identity_file": "eval_mask_identity.jsonl"}, indent=2), encoding="utf-8")
            (self.output_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        return result

    def _bootstrap(self, bootstrap, seed):
        if not bootstrap or not self._cells or any(x["ours_abs"] is None for x in self._cells):
            return None
        clustered = defaultdict(lambda: [0, 0.0, 0.0])
        for x in self._cells:
            z = clustered[x["cell_id"]]; z[0] += x["bins"]; z[1] += x["base_abs"]; z[2] += x["ours_abs"]
        values = list(clustered.values())
        if len(values) < 2 or sum(x[0] for x in values) == 0:
            return None
        rng = np.random.default_rng(seed); n = len(values); draws = np.empty(int(bootstrap))
        a = np.asarray(values, dtype=float)
        for i in range(len(draws)):
            sample = a[rng.integers(0, n, size=n)]
            draws[i] = (sample[:, 2].sum() - sample[:, 1].sum()) / sample[:, 0].sum()
        return [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]
