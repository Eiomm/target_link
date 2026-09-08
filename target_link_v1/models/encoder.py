"""Trajectory Encoder (spec §§5-7): 10m-bin motion profile -> r_traj in R^d.

Input per bin is x_i = [v_i / v_norm, m_i] (speed linearly scaled + valid flag),
passed through an MLP, plus a learnable position embedding (spec §5). A lightweight
TransformerEncoder (spec §6) contextualises the sequence; mean pooling over
valid bins yields the per-trajectory representation (spec §7, no CLS).

Masking semantics (spec §4 m_i):
  - pad bins (index >= length) are excluded from attention entirely;
  - invalid bins inside the profile (data missing / speed cap) DO attend, with
    v=0 and m=0 so the model can see "there is a hole here", but they never
    enter the pooling average.

input_mode:
  - "absolute": raw scaled speeds (Ablation 1 / B1)
  - "residual": v minus the per-trajectory valid mean (Ablation 2 / B2) —
    a uniform shift of all speeds leaves the output unchanged.

out_dim: if set (e.g. 128), a linear projection maps d_model -> out_dim so all
encoder sizes feed the downstream the same width (Ablation 4 fairness control).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TrajectoryEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        max_bins: int = 40,
        v_norm: float = 33.3,
        input_mode: str = "absolute",
        out_dim: int | None = None,
    ) -> None:
        super().__init__()
        if input_mode not in ("absolute", "residual"):
            raise ValueError(f"unknown input_mode: {input_mode}")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.v_norm = v_norm
        self.input_mode = input_mode

        self.input_mlp = nn.Sequential(
            nn.Linear(2, d_model), nn.GELU(),
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model),
        )
        self.pos_emb = nn.Embedding(max_bins, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * ffn_mult,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, out_dim) if out_dim is not None else nn.Identity()
        # r must enter the downstream at O(1) scale alongside other features:
        # a fresh Linear projection shrinks r ~7x (per-dim std 1.7 -> 0.24),
        # which starves the encoder of gradient until the head overfits the
        # scalar features first. LayerNorm restores the scale for every d.
        self.feature_norm = nn.LayerNorm(out_dim if out_dim is not None else d_model)

    def forward(
        self, speeds: torch.Tensor, valid: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """speeds [B,N] float32 (invalid/pad already zeroed), valid [B,N] bool,
        lengths [B] int. Returns r_traj [B, d_model or out_dim]."""
        B, N = speeds.shape
        device = speeds.device

        v = speeds / self.v_norm
        if self.input_mode == "residual":
            cnt = valid.sum(dim=1, keepdim=True).clamp(min=1).float()
            mean = torch.where(valid, v, torch.zeros_like(v)).sum(dim=1, keepdim=True) / cnt
            v = torch.where(valid, v - mean, torch.zeros_like(v))

        x = torch.stack([v, valid.float()], dim=-1)          # [B,N,2]
        pos = self.pos_emb(torch.arange(N, device=device))   # [N,d]
        h = self.input_mlp(x) + pos.unsqueeze(0)             # [B,N,d]

        pad = torch.arange(N, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
        h = self.encoder(h, src_key_padding_mask=pad)        # [B,N,d]

        m = valid.float().unsqueeze(-1)
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return self.feature_norm(self.out_proj(pooled))
