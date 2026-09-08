"""Two-level window representation with partial-passage reconstruction.

No full-passage labels or exact bin timestamps are model inputs. Spatial
coordinates survive missing bins. A whole-pass mask hides every sub-curve of
that pass together, before encoding. Only finite available targets incur loss;
interpolated targets remain pseudo-labels, not direct GPS measurements.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from target_link_v1.models.level2 import TrajectoryLevelTransformer


def reconstruction_mask(batch, ratio=0.5, whole_pass_probability=0.5):
    if not 0 < ratio < 1 or not 0 <= whole_pass_probability <= 1:
        raise ValueError("require 0<ratio<1 and 0<=whole_pass_probability<=1")
    valid = batch["valid"]
    mask = torch.zeros_like(valid)
    for j in range(len(valid)):
        idx = torch.where(valid[j])[0]
        if len(idx) > 1:
            n = min(len(idx) - 1, max(1, int(round(len(idx) * ratio))))
            start = int(torch.randint(len(idx) - n + 1, (), device=valid.device))
            mask[j, idx[start:start + n]] = True
    for g in range(batch["n_groups"]):
        rows = batch["curve_group"] == g
        passes = batch["curve_pass"][rows].unique()
        if len(passes) > 1 and torch.rand((), device=valid.device) < whole_pass_probability:
            pick = passes[torch.randint(len(passes), (), device=valid.device)]
            mask[batch["curve_pass"] == pick] = valid[batch["curve_pass"] == pick]
    return mask


class WindowMAE(nn.Module):
    def __init__(self, d_model=128, heads=4, layers=2, group_layers=2, dropout=0.1,
                 time_features="curve"):
        super().__init__()
        if time_features not in ("none", "curve", "bin"):
            raise ValueError("time_features must be none, curve, or bin")
        self.time_features = time_features
        # q, distance, absolute/relative position, coarse age, observed,
        # artificial mask. Padding has a separate attention mask.
        self.input = nn.Sequential(nn.Linear(7, d_model), nn.GELU(), nn.LayerNorm(d_model))
        layer = nn.TransformerEncoderLayer(d_model, heads, d_model * 4, dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.cls = nn.Parameter(torch.randn(d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)
        self.aggregate = TrajectoryLevelTransformer(d_model, group_layers, heads, dropout=dropout)
        self.query = nn.Linear(3, d_model)
        self.decoder = nn.Sequential(nn.Linear(d_model * 3, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, b, mask=None):
        valid = b["valid"]
        mask = torch.zeros_like(valid) if mask is None else mask & valid
        visible = valid & ~mask
        q = torch.log1p(b["duration"] / b["distance"].clamp_min(1e-6))
        pos_abs = b["position"] / 1000.0
        pos_rel = b["position"] / b["link_lengths"].unsqueeze(1).clamp_min(1)
        geo = torch.stack([b["distance"] / 10.0, pos_abs, pos_rel], dim=-1)
        age = b["age"].masked_fill(~visible, 0)
        if self.time_features == "none":
            age = torch.zeros_like(age)
        elif self.time_features == "curve":
            # Coarse age of the newest VISIBLE bin, shared across that curve.
            # Hidden bin timing cannot change the summary used by the encoder.
            newest = b["age"].masked_fill(~visible, float("inf")).amin(dim=1, keepdim=True)
            newest = torch.where(visible.any(1, keepdim=True), newest, torch.zeros_like(newest))
            age = newest.expand_as(age).masked_fill(~visible, 0)
        features = torch.cat([
            q.masked_fill(~visible, 0).unsqueeze(-1), geo,
            age.unsqueeze(-1),
            b["observed"].masked_fill(~visible, 0).unsqueeze(-1),
            mask.float().unsqueeze(-1)], dim=-1)
        x = self.input(features)
        c, m = valid.shape
        x = torch.cat([self.cls.view(1, 1, -1).expand(c, 1, -1), x], dim=1)
        pad = torch.cat([torch.zeros(c, 1, dtype=torch.bool, device=x.device), ~valid], dim=1)
        z = self.norm(self.encoder(x, src_key_padding_mask=pad)[:, 0])
        r = self.aggregate(z, b["curve_group"], b["n_groups"])
        query = self.query(geo)
        pred = F.softplus(self.decoder(torch.cat([query, z[:, None].expand(-1, m, -1),
                                      r[b["curve_group"]][:, None].expand(-1, m, -1)], dim=-1)).squeeze(-1))
        return {"representation": r, "curve_representation": z, "prediction": pred, "target": q}


def reconstruction_loss(output, mask, curve_group, n_groups):
    # Equal curve weight within each snapshot, then equal snapshot weight.
    err = F.huber_loss(output["prediction"], output["target"], reduction="none")
    count = mask.sum(1)
    per_curve = (err * mask).sum(1) / count.clamp_min(1)
    total = err.sum() * 0
    n = 0
    for g in range(n_groups):
        rows = (curve_group == g) & (count > 0)
        if rows.any():
            total = total + per_curve[rows].mean()
            n += 1
    return total / max(n, 1)
