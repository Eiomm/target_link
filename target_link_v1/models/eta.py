"""Downstream ETA models (spec §9, §11): shared backbone, variant-dependent input.

All variants predict log(y_travel_s) — the label is heavily right-skewed
(median 12s, max 1660s) — and share the identical head, optimizer and data;
only the dynamic block differs:

  "speed"      x = [L_n, v_n]                  A0 production scalar
  "speed-mlp"  x = [L_n, lift(v_n)]            A1 scalar lifted to 128-d
  "ours"       x = [L_n, v_n, r_{l,t}]         A2 + trajectory representation

A1's lift mirrors the encoder input MLP in width, controlling feature
dimension and nonlinear capacity (spec §11.3). A2 trains the encoder
end-to-end through the two mean aggregations (spec §8):

  r_traj -> mean over K trajs per (sub-link, window) -> mean over subs per sample
"""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from target_link_v1.models.aggregation import scatter_mean
from target_link_v1.models.encoder import TrajectoryEncoder


class SpeedLift(nn.Module):
    """A1 control: lift the scalar mean speed to a 128-d vector."""

    def __init__(self, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, out_dim), nn.GELU(),
            nn.Linear(out_dim, out_dim), nn.LayerNorm(out_dim),
        )

    def forward(self, v_n: torch.Tensor) -> torch.Tensor:  # [B] -> [B, out]
        return self.net(v_n.unsqueeze(-1))


class ETAHead(nn.Module):
    """Shared downstream backbone — identical across all variants (spec §9)."""

    def __init__(self, in_dim: int, hidden: int = 256, depth: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(depth):
            layers += [
                nn.Linear(in_dim if i == 0 else hidden, hidden),
                nn.GELU(), nn.LayerNorm(hidden), nn.Dropout(dropout),
            ]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # -> [B] log(y)
        return self.net(x).squeeze(-1)


def encode_link_rep(
    encoder: TrajectoryEncoder,
    speeds: torch.Tensor, valid: torch.Tensor, lengths: torch.Tensor,
    prof_group: torch.Tensor, n_groups: int,
    edge_group: torch.Tensor, edge_sample: torch.Tensor, n_samples: int,
) -> torch.Tensor:
    """Profile rows -> per-(sub-link, window) mean -> per-sample subs-mean.

    Both steps are spec §8 mean aggregations via scatter_mean; the whole path
    is differentiable back into the encoder.
    """
    r = encoder(speeds, valid, lengths)                       # [P, d] per-trajectory
    r_group = scatter_mean(r, prof_group, n_groups)           # [G, d] (sub, window)
    r_edge = r_group[edge_group]                              # [E, d] group per edge
    return scatter_mean(r_edge, edge_sample, n_samples)       # [B, d] per-sample


class ETAModel(nn.Module):
    """ETA model with switchable dynamic representation (Ablation variants)."""

    VARIANTS = ("speed", "speed-mlp", "ours")

    def __init__(
        self,
        variant: str = "speed",
        hidden: int = 256, depth: int = 2, dropout: float = 0.1,
        encoder_kwargs: Dict | None = None,
    ) -> None:
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; expected one of {self.VARIANTS}")
        self.variant = variant
        if variant == "speed-mlp":
            self.lift = SpeedLift()
        if variant == "ours":
            self.encoder = TrajectoryEncoder(**(encoder_kwargs or {}))
        in_dim = {"speed": 2, "speed-mlp": 1 + 128, "ours": 2 + 128}[variant]
        self.head = ETAHead(in_dim, hidden=hidden, depth=depth, dropout=dropout)

    def forward(
        self, L_n: torch.Tensor, v_n: torch.Tensor, batch: Dict[str, torch.Tensor] | None = None
    ) -> torch.Tensor:
        """L_n/v_n [B]; batch carries the encoder tensors for variant "ours".
        Returns [B] predicted log(y_travel_s)."""
        feats = [L_n.unsqueeze(-1)]
        if self.variant == "speed-mlp":
            feats.append(self.lift(v_n))
        else:
            feats.append(v_n.unsqueeze(-1))
        if self.variant == "ours":
            feats.append(encode_link_rep(self.encoder, **batch))
        return self.head(torch.cat(feats, dim=-1))
