"""Vehicle-to-(sub-)link aggregation (spec §8): fixed mean over trajectories.

For the K trajectory representations sharing a (sub-link, window):

    r_{l,t} = 1/K * sum_k r_traj^(k)

V1 fixes plain mean — Attention/Set pooling is explicitly out of scope for the
first stage, so this is a parameter-free differentiable scatter-mean. Gradients
flow back into every contributing encoder pass, scaled by 1/K each.
"""
from __future__ import annotations

import torch
from torch import nn


def scatter_mean(x: torch.Tensor, group_idx: torch.Tensor, n_groups: int) -> torch.Tensor:
    """Row i of the output = mean of the rows of ``x`` with ``group_idx == i``.

    x [B, D] float, group_idx [B] int64 in [0, n_groups). Groups with no
    members (n_groups larger than the ids present) come out as zeros.
    """
    if group_idx.dim() != 1 or group_idx.numel() != x.shape[0]:
        raise ValueError(
            f"group_idx {tuple(group_idx.shape)} does not index x {tuple(x.shape)}"
        )
    out = x.new_zeros(n_groups, x.shape[1])
    out.index_add_(0, group_idx, x)
    counts = torch.bincount(group_idx, minlength=n_groups).clamp(min=1)
    return out / counts.unsqueeze(1)


class LinkAggregator(nn.Module):
    """Mean aggregation of per-vehicle r_traj into per-(sub-link, window) r_{l,t}.

    Parameter-free on purpose (spec §8); kept as a module so the downstream
    trains against one stable interface should aggregation be revisited later.
    """

    def forward(
        self, r_traj: torch.Tensor, group_idx: torch.Tensor, n_groups: int
    ) -> torch.Tensor:
        return scatter_mean(r_traj, group_idx, n_groups)
