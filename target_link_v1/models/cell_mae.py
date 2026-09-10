"""Cell-level whole-trajectory MAE (spec md/最新讨论想法.md §5).

A cell is (map_version, target_link_id, seg_idx, 10min window): one 500m segment
of one link in one 10-minute window. Its trajectories are the rows of a training
group (m_max=16); the model token is the 10m bin of that segment, so one
trajectory is a [50, 3] profile of (T_diff, ratio, observed) plus `bin_valid`.

    [50,3] bins + bin_valid -> TrajectoryEncoder -> r_k
    r_k + TimeEmbedding(delta_t)                  -> z_k
    keep visible trajectory tokens only           -> z_visible
    [CLS] + z_visible -> TrajectoryLevelTransformer -> h_CLS
    h_CLS + visible per-bin group profile
          + (bin position, delta_t) -> decoder     -> T_diff per bin

Choices worth naming, because they are what the smoke is meant to exercise:

  * The mask unit is a whole trajectory. Level 2 receives only visible
    trajectory tokens, so a hidden profile cannot leak into h_CLS.
  * The decoder reads h_CLS and a per-bin mean of visible trajectories' level-1
    states, never the target trajectory's own state. This preserves spatial
    congestion information that was previously destroyed by trajectory pooling.
    Validation can ablate the global and per-bin channels independently;
    `ablate_aggregate=True` remains the backward-compatible all-off probe.
  * T_diff stays in seconds. It is never converted to speed -- on a fixed-length
    bin the crossing time already is the motion feature.
  * The feature axis remains exactly (T_diff, ratio, observed). `bin_valid` is
    a separate attention and pooling mask and is never concatenated to x.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F

from target_link_v1.models.level2 import TrajectoryLevelTransformer

N_BINS = 50        # 500m segment / 10m bin, the fixed spatial grid
N_FEATURES = 3     # T_diff, ratio, observed


class TimeEmbedding(nn.Module):
    """The pinned V1 continuous-time encoder: delta_t / 600 -> small MLP."""

    def __init__(self, d_model, period=600.0):
        super().__init__()
        self.period = float(period)
        self.mlp = nn.Sequential(nn.Linear(1, d_model), nn.GELU(),
                                 nn.Linear(d_model, d_model))

    def forward(self, delta_t):
        """delta_t [P] float -> [P, d_model]."""
        return self.mlp((delta_t / self.period).unsqueeze(-1))


class CellTrajectoryEncoder(nn.Module):
    """Bin MLP + absolute position + Transformer + valid-bin mean pooling."""

    def __init__(self, d_model=256, heads=8, layers=4, n_bins=N_BINS, dropout=0.1):
        super().__init__()
        self.n_bins = int(n_bins)
        self.bin_proj = nn.Sequential(
            nn.Linear(N_FEATURES, d_model), nn.GELU(),
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model),
            nn.Dropout(dropout))
        self.pos_emb = nn.Embedding(self.n_bins, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, heads, d_model * 4, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d_model)

    def forward_tokens(self, x, bin_valid):
        """x [P, 50, 3] raw (T_diff seconds, ratio 0..1, observed 0/1),
        bin_valid [P, 50] bool. Returns pooled r and bin states."""
        P, n, _ = x.shape
        if n != self.n_bins:
            raise ValueError("trajectory width %d != n_bins %d" % (n, self.n_bins))
        pos = self.pos_emb(torch.arange(n, device=x.device))
        h = self.bin_proj(x) + pos
        # PyTorch attention produces NaN when every key is masked. Padding
        # trajectories use a harmless temporary key and are zeroed by pooling.
        safe = bin_valid.clone()
        empty = ~safe.any(1)
        safe[empty, 0] = True
        h = self.encoder(h, src_key_padding_mask=~safe)
        m = bin_valid.to(x.dtype).unsqueeze(-1)
        r = self.norm((h * m).sum(1) / m.sum(1).clamp_min(1.0))
        return r, self.norm(h)

    def forward(self, x, bin_valid):
        return self.forward_tokens(x, bin_valid)[0]


class CellMAE(nn.Module):
    def __init__(self, d_model=256, heads=8, traj_layers=4, level2_layers=4,
                 n_bins=N_BINS, dropout=0.1, target_transform="log1p"):
        super().__init__()
        if target_transform not in ("raw", "log1p"):
            raise ValueError("target_transform must be raw or log1p")
        self.target_transform = target_transform
        self.n_bins = int(n_bins)
        self.traj_encoder = CellTrajectoryEncoder(
            d_model, heads, traj_layers, n_bins, dropout)
        self.time_emb = TimeEmbedding(d_model)
        self.level2 = TrajectoryLevelTransformer(d_model, level2_layers, heads,
                                                 dropout=dropout)
        # Decoder query is geometry and time only: where the bin sits in the
        # segment, and when the trajectory entered it.
        self.dec_pos_emb = nn.Embedding(self.n_bins, d_model)
        self.dec_query = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU())
        self.profile_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                          nn.LayerNorm(d_model))
        self.decoder = nn.Sequential(nn.Linear(3 * d_model, d_model), nn.GELU(),
                                     nn.Linear(d_model, d_model), nn.GELU(),
                                     nn.Linear(d_model, 1))

    def forward(self, batch, ablate_aggregate=False, ablate_cls=False,
                ablate_group_bins=False):
        """batch: the dict `collate_cells` returns, already on the model's device.

        Returns `representation` h_CLS [B, d], `group_bin_state` [B, n_bins, d],
        `trajectory_state` [B, m_max, d] (zero on padding), and `prediction` /
        `target` [B, m_max, n_bins].
        """
        x = batch["x"]                                    # [B,M,50,3]
        bin_valid = batch["bin_valid"]                    # [B,M,50]
        traj_valid = batch["traj_valid"]                  # [B,M]
        mae_mask = batch["mae_mask"]                      # [B,M]
        B, M, n, F = x.shape
        if n != self.n_bins or F != N_FEATURES:
            raise ValueError("expected [B,M,%d,%d], got %s"
                             % (self.n_bins, N_FEATURES, tuple(x.shape)))
        P = B * M
        t = self.time_emb(batch["delta_t"].reshape(P))    # [P,d]
        r, bin_state = self.traj_encoder.forward_tokens(
            x.reshape(P, n, F), bin_valid.reshape(P, n))
        z = r + t
        # The frozen V1 level-2 input is [CLS] + visible trajectory tokens.
        # Masked trajectories and padding are both absent from its key/value set.
        keep = (traj_valid & ~mae_mask).reshape(P)
        groups = torch.arange(B, device=x.device).repeat_interleave(M)[keep]
        h_cls, h_rows = self.level2.forward_tokens(z[keep], groups, B)
        state = x.new_zeros(P, h_cls.shape[-1])
        state[keep] = h_rows
        state = state.reshape(B, M, -1)

        # Keep the spatial axis alive across trajectories. Only visible,
        # non-padding trajectories contribute, so the reconstruction target
        # cannot leak through this path. A bin with no visible valid value gets
        # exactly zero context after projection.
        visible_bins = ((traj_valid & ~mae_mask).unsqueeze(-1) & bin_valid)
        w = visible_bins.to(x.dtype).unsqueeze(-1)
        bins = bin_state.reshape(B, M, n, -1)
        group_bins = (bins * w).sum(1) / w.sum(1).clamp_min(1.0)
        group_bin_valid = visible_bins.any(1)
        group_bins = self.profile_proj(group_bins)
        group_bins = group_bins * group_bin_valid.unsqueeze(-1).to(group_bins.dtype)

        base = h_cls.unsqueeze(1).expand(B, M, -1)
        if ablate_aggregate or ablate_cls:
            base = torch.zeros_like(base)
        if ablate_aggregate or ablate_group_bins:
            group_bins = torch.zeros_like(group_bins)
        pos = self.dec_pos_emb(torch.arange(n, device=x.device))
        query = self.dec_query(torch.cat([pos.unsqueeze(0).expand(P, -1, -1),
                                          t.unsqueeze(1).expand(P, n, -1)], -1))
        group_context = group_bins.unsqueeze(1).expand(B, M, n, -1).reshape(P, n, -1)
        raw = self.decoder(torch.cat([
            query, base.reshape(P, 1, -1).expand(P, n, -1), group_context], -1)
        ).squeeze(-1)                                      # [P,n]
        # Prediction lives in the target space. A softplus would bend a log1p
        # target, so it is only applied where the target is in seconds.
        pred = raw if self.target_transform == "log1p" else F.softplus(raw)
        target = x[..., 0].reshape(P, n)
        if self.target_transform == "log1p":
            target = torch.log1p(target.clamp_min(0.0))
        return {"representation": h_cls,
                "trajectory_state": state,
                "group_bin_state": group_bins,
                "group_bin_valid": group_bin_valid,
                "prediction": pred.reshape(B, M, n),
                "target": target.reshape(B, M, n)}


def reconstruction_loss_by_group(output, batch):
    """Return one Huber loss per group plus a has-supervision mask."""
    mask = batch["mae_mask"].unsqueeze(-1) & batch["bin_valid"]     # [B,M,50]
    err = F.huber_loss(output["prediction"], output["target"], reduction="none")
    cnt = mask.sum(-1)                                              # [B,M]
    per_traj = (err * mask).sum(-1) / cnt.clamp_min(1)
    rows = cnt > 0
    per_group = (per_traj * rows).sum(1) / rows.sum(1).clamp_min(1)
    return per_group, rows.any(1)


def masked_reconstruction_loss(output, batch):
    """Huber on the bins of MASKED trajectories, equal weight per group.

    A group's loss is the mean over the masked trajectories that have at least
    one bin with a known T_diff; groups with nothing to reconstruct are dropped
    rather than counted as a zero (with K_min=3 and the >=3-valid rule in
    collate_cells an unmasked group can still happen, and it must not dilute).
    """
    per_group, has = reconstruction_loss_by_group(output, batch)
    weight = has.to(per_group.dtype)
    return (per_group * weight).sum() / weight.sum().clamp_min(1)
