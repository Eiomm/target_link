"""Trajectory-level Transformer (v2 md/9_4.md §3, Level 2): the K trajectory
tokens sharing a (sub-link, time window) -> one dynamic link representation.

    r_{l,t} = h_CLS = Transformer([CLS, r_1, ..., r_K])[:, 0]

Trajectories within a group carry no meaningful order (which car passed the
sub-link first is arbitrary), so this is set attention: a learnable CLS token
is prepended and there is NO position embedding — the output is invariant to
the row order of the incoming profile batch. Variable K is handled by packing
each group into a [G, K_max, d] tensor with a key-padding mask, so padded
slots are excluded from attention entirely (same semantics as the Level-1 pad
mask, see encoder.py).

This replaces scatter_mean as the "ours" aggregation; scatter_mean itself
stays available as the aggregation ablation baseline (ETAModel
aggregation="mean"), so both arms of md/9_4.md §6 "Trajectory Aggregation"
run on identical data/heads.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def within_rank(group_idx: torch.Tensor, n_groups: int) -> torch.Tensor:
    """Position of each row among the rows sharing its group id, in row order."""
    counts = torch.bincount(group_idx, minlength=n_groups)
    order = torch.argsort(group_idx, stable=True)
    starts = torch.cumsum(counts, dim=0) - counts
    within = torch.empty_like(group_idx)
    within[order] = (torch.arange(len(group_idx), device=group_idx.device)
                     - starts[group_idx[order]])
    return within


def pack_groups(
    r: torch.Tensor, group_idx: torch.Tensor, n_groups: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ragged group rows -> dense [n_groups, K_max, d] + key-padding mask.

    r [P, d] float, group_idx [P] int64 in [0, n_groups), rows in ANY order
    (order-independence is what Level 2 promises, so packing must not assume
    group-sorted input). Returns (packed [G, K_max, d], pad [G, K_max] bool
    with True = padded slot). A group with no rows comes out all-zero and
    fully padded — its attention row sees the CLS token only.
    """
    P, d = r.shape
    device = r.device
    if P and int(group_idx.max().item()) >= n_groups:
        raise ValueError(
            f"group_idx max {int(group_idx.max().item())} outside [0, {n_groups})"
        )
    counts = torch.bincount(group_idx, minlength=n_groups)          # [G]
    k_max = max(int(counts.max().item()), 1) if n_groups else 1
    # within-group rank of each row: position among the rows sharing its group
    # id, in row order
    within = within_rank(group_idx, n_groups)
    packed = r.new_zeros(n_groups, k_max, d)
    packed[group_idx, within] = r
    pad = torch.arange(k_max, device=device).unsqueeze(0) >= counts.unsqueeze(1)
    return packed, pad


class TrajectoryLevelTransformer(nn.Module):
    """Set-attention over one group's trajectory tokens, read out at a CLS.

    Input is the ragged per-trajectory output of TrajectoryEncoder together
    with the same prof_group / n_groups pair scatter_mean consumes, so the two
    aggregations are drop-in replacements for each other.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        ffn_mult: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.cls_token = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * ffn_mult,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        # pre-LN stack ends unnormalised; h_CLS must also reach the downstream
        # at O(1) scale — same rationale as TrajectoryEncoder.feature_norm
        self.feature_norm = nn.LayerNorm(d_model)

    def forward_tokens(
        self, r_traj: torch.Tensor, group_idx: torch.Tensor, n_groups: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (h_CLS [G, d], h_rows [P, d]).

        h_rows is the transformer state of each trajectory token, aligned with
        the r_traj rows. A decoder may read h_rows instead of a level-1 skip
        connection, which is what keeps the CLS readout from being bypassed.
        """
        packed, pad = pack_groups(r_traj, group_idx, n_groups)      # [G, K, d]
        G = packed.shape[0]
        cls = self.cls_token.view(1, 1, -1).expand(G, 1, -1)
        x = torch.cat([cls, packed], dim=1)                         # [G, K+1, d]
        pad = torch.cat([pad.new_zeros((G, 1)), pad], dim=1)        # CLS never masked
        h = self.feature_norm(self.encoder(x, src_key_padding_mask=pad))
        within = within_rank(group_idx, n_groups)
        return h[:, 0], h[:, 1:, :][group_idx, within]

    def forward(
        self, r_traj: torch.Tensor, group_idx: torch.Tensor, n_groups: int
    ) -> torch.Tensor:
        """r_traj [P, d], group_idx [P] int64 in [0, n_groups).
        Returns h_CLS [n_groups, d] — one row per (sub-link, time window)."""
        return self.forward_tokens(r_traj, group_idx, n_groups)[0]
