"""Two-level window representation with partial-passage reconstruction.

Bin features follow the V1 design: f_i = [dt_i, ratio_i, observed_i] with the
spatial position p_i = s_i/L_unit added as a separate embedding. dt is used
directly, never converted to speed: on a fixed-length bin it already encodes
how fast the vehicle moved. The coarse age is an ablation, not a default,
because available_ts is causal gating rather than a motion feature.

No full-passage labels or exact bin timestamps are model inputs. Spatial
coordinates survive missing bins, including on masked trajectories. A
whole-pass mask hides every sub-curve of that pass together, before encoding.
Only finite available targets incur loss; interpolated targets remain
pseudo-labels, not direct GPS measurements.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from target_link_v1.models.level2 import TrajectoryLevelTransformer


def reconstruction_mask(batch, ratio=0.5, whole_pass_probability=0.5):
    """Hide whole trajectories: the mask unit is a pass, not a bin.

    In a multi-trajectory snapshot each trajectory is hidden with probability
    whole_pass_probability, clamped so that at least one stays visible (the
    encoder must see some traffic) and at least one is hidden (otherwise the
    snapshot carries no reconstruction signal at all). whole_pass_probability=0
    disables the whole-pass path and uses the span fallback everywhere. A
    single-trajectory snapshot has nothing to reconstruct it from, so it always
    falls back to a contiguous span inside that curve. Padding and invalid bins
    are never masked.
    """
    if not 0 < ratio < 1 or not 0 <= whole_pass_probability <= 1:
        raise ValueError("require 0<ratio<1 and 0<=whole_pass_probability<=1")
    valid = batch["valid"]
    mask = torch.zeros_like(valid)
    for g in range(batch["n_groups"]):
        rows = torch.where(batch["curve_group"] == g)[0]
        passes = batch["curve_pass"][rows].unique()
        if len(passes) > 1 and whole_pass_probability > 0:
            picked = passes[torch.rand(len(passes), device=valid.device) < whole_pass_probability]
            if len(picked) == len(passes):
                keep = torch.randperm(len(picked), device=valid.device)[:len(picked) - 1]
                picked = picked[keep]
            elif len(picked) == 0:
                picked = passes[torch.randint(len(passes), (), device=valid.device)].unsqueeze(0)
            for pid in picked.tolist():
                mask[batch["curve_pass"] == pid] = True
        else:
            j = int(rows[0])
            idx = torch.where(valid[j])[0]
            if len(idx) > 1:
                n = min(len(idx) - 1, max(1, int(round(len(idx) * ratio))))
                start = int(torch.randint(len(idx) - n + 1, (), device=valid.device))
                mask[j, idx[start:start + n]] = True
    return mask & valid


class WindowMAE(nn.Module):
    def __init__(self, d_model=128, heads=4, layers=2, group_layers=2, dropout=0.1,
                 time_features="none", target_transform="log1p", bin_size_m=10.0,
                 decoder_input="hidden"):
        super().__init__()
        if time_features not in ("none", "curve", "bin"):
            raise ValueError("time_features must be none, curve, or bin")
        if target_transform not in ("raw", "log1p"):
            raise ValueError("target_transform must be raw or log1p")
        if decoder_input not in ("hidden", "cls"):
            raise ValueError("decoder_input must be hidden or cls")
        if not bin_size_m > 0:
            raise ValueError("bin_size_m must be positive")
        self.time_features = time_features
        self.target_transform = target_transform
        self.decoder_input = decoder_input
        self.bin_size_m = float(bin_size_m)
        # [dt, ratio, observed] (+ coarse age only in an ablation) + artificial
        # mask. Padding has a separate attention mask.
        self.feature = nn.Sequential(nn.Linear(4 if time_features == "none" else 5, d_model),
                                     nn.GELU(), nn.LayerNorm(d_model))
        self.pos = nn.Linear(1, d_model)
        layer = nn.TransformerEncoderLayer(d_model, heads, d_model * 4, dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.cls = nn.Parameter(torch.randn(d_model) * 0.02)
        # Stands in for a fully hidden trajectory at level 2. It carries no
        # level-1 information, so a hidden curve cannot be read back directly.
        self.mask_token = nn.Parameter(torch.randn(d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)
        self.aggregate = TrajectoryLevelTransformer(d_model, group_layers, heads, dropout=dropout)
        # Geometry query for the decoder: where the bin sits, not how fast it moved.
        self.query = nn.Linear(2, d_model)
        # query + level-2 state only: the level-1 curve state is deliberately not
        # a decoder input, or the decoder would bypass the CLS readout.
        self.decoder = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, b, mask=None, ablate_aggregate=False):
        valid = b["valid"]
        mask = torch.zeros_like(valid) if mask is None else mask & valid
        visible = valid & ~mask
        # Spatial coverage of this component, recovered from the fixed V1 bin grid.
        # It is geometry, so it survives masking: it says nothing about a hidden dt.
        ratio = b["distance"] / self.bin_size_m
        # Normalized by the modeling unit, not the physical link: p_i = s_i / L_unit.
        pos = ((b["position"] - b["unit_starts"].unsqueeze(1))
               / b["unit_lengths"].unsqueeze(1).clamp_min(1.0))
        parts = [b["duration"].masked_fill(~visible, 0).unsqueeze(-1),
                 ratio.unsqueeze(-1),
                 b["observed"].masked_fill(~visible, 0).unsqueeze(-1)]
        if self.time_features == "bin":
            parts.append(b["age"].masked_fill(~visible, 0).unsqueeze(-1))
        elif self.time_features == "curve":
            # Coarse age of the newest VISIBLE bin, shared across that curve.
            # Hidden bin timing cannot change the summary used by the encoder.
            newest = b["age"].masked_fill(~visible, float("inf")).amin(dim=1, keepdim=True)
            newest = torch.where(visible.any(1, keepdim=True), newest, torch.zeros_like(newest))
            parts.append(newest.expand_as(b["age"]).masked_fill(~visible, 0).unsqueeze(-1))
        parts.append(mask.float().unsqueeze(-1))
        x = self.feature(torch.cat(parts, dim=-1)) + self.pos(pos.unsqueeze(-1))
        c, m = valid.shape
        x = torch.cat([self.cls.view(1, 1, -1).expand(c, 1, -1), x], dim=1)
        pad = torch.cat([torch.zeros(c, 1, dtype=torch.bool, device=x.device), ~valid], dim=1)
        z = self.norm(self.encoder(x, src_key_padding_mask=pad)[:, 0])
        # Level 2 sees the visible trajectories plus one MASK token per fully
        # hidden trajectory. A hidden curve's own level-1 state is replaced, not
        # appended, so it can only come back through the aggregate.
        hidden_curve = (mask == valid).all(dim=1) & valid.any(dim=1)
        tokens = torch.where(hidden_curve.unsqueeze(1),
                             self.mask_token.view(1, -1).expand_as(z), z)
        h_cls, h_rows = self.aggregate.forward_tokens(tokens, b["curve_group"], b["n_groups"])
        state = (h_cls if self.decoder_input == "cls" else h_rows)[b["curve_group"]]
        # One level-2 state per curve, shared by every bin of that curve.
        state = state.unsqueeze(1).expand(-1, m, -1)
        if ablate_aggregate:
            state = torch.zeros_like(state)
        query = self.query(torch.stack([ratio, pos], dim=-1))
        pred = F.softplus(self.decoder(torch.cat([query, state], dim=-1)).squeeze(-1))
        # dt per fixed bin length: a partial bin is scaled onto the same basis, so
        # every curve is reconstructed in seconds-per-10m.
        target = b["duration"] * self.bin_size_m / b["distance"].clamp_min(1e-6)
        if self.target_transform == "log1p":
            target = torch.log1p(target)
        return {"representation": h_cls, "curve_representation": z,
                "prediction": pred, "target": target}


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
