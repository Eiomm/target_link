"""Self-supervised curve MAE: pretrain the TrajectoryEncoder readout at a CLS.

Pretraining unit = one profile row (a sample's 10m speed curve on one sub-link
— the exact input TrajectoryEncoder eats downstream, repV2 §2.2). Contiguous
spans of VALID bins are declared "holes" (speed 0, valid flag 0 — the
encoder's native invalid-bin semantics, so no new input branch), and a small
decoder must reconstruct the masked speeds from the CLS summary ALONE: the
decoder attends only to mask queries conditioned on r, never to visible bins.
That is the repV2 §3.3 bottleneck — a structural guarantee that r carries the
curve's global shape, unlike a mean-pooled readout, which supervised training
can bypass through the scalar shortcut (the V1 bistability lesson).

Reuse contract (repV2 §5.4 — no second encoder):
  - CurveMAE drives the encoder through its PUBLIC submodules
    (input_mlp / pos_emb / encoder / out_proj / feature_norm), so a pretraining
    checkpoint loads into ETAModel's encoder with zero remapping;
  - curve_representation() is the single CLS readout, shared by pretraining
    and the stage-2 downstream (train_eta --init-from), so there is exactly
    one implementation of "curve -> r" to maintain.

CLS-liveness probe: forward(..., use_cls=False) feeds the decoder a zeroed r;
the reconstruction-loss increase it causes is the information the CLS carries
(the pretraining analogue of V1's branch-enable monitoring).
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from target_link_v1.models.encoder import TrajectoryEncoder


def _d_out(encoder: TrajectoryEncoder) -> int:
    p = encoder.out_proj
    return p.out_features if not isinstance(p, nn.Identity) else encoder.pos_emb.embedding_dim


def curve_representation(
    encoder: TrajectoryEncoder, cls_token: torch.Tensor,
    speeds: torch.Tensor, valid: torch.Tensor, lengths: torch.Tensor,
) -> torch.Tensor:
    """Curve -> r via a prepended CLS token through the encoder's own stack.

    speeds [B,N] raw m/s (invalid/pad zeroed), valid [B,N] bool, lengths [B].
    Returns r [B, d_out] — same output transform (out_proj + feature_norm) as
    TrajectoryEncoder.forward's mean-pooled readout, so r drops into every
    downstream consumer of encoder output unchanged.
    """
    B, N = speeds.shape
    device = speeds.device
    v = speeds / encoder.v_norm
    x = torch.stack([v, valid.float()], dim=-1)
    h = encoder.input_mlp(x) + encoder.pos_emb(torch.arange(N, device=device))
    cls = cls_token.view(1, 1, -1).expand(B, 1, -1)
    h = torch.cat([cls, h], dim=1)                          # [B, 1+N, d_model]
    pad = torch.arange(N, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
    pad = torch.cat([pad.new_zeros((B, 1)), pad], dim=1)    # CLS never padded
    h = encoder.encoder(h, src_key_padding_mask=pad)
    return encoder.feature_norm(encoder.out_proj(h[:, 0]))  # [B, d_out]


def _rand_subset(flags: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Random subset of the True positions in flags[b], at most k[b] of them.

    Single top-k on a random key (candidates get key in [0,1), the rest -1) —
    no argsort: sorting [B,N] bools cost 300-700ms at 64 torch threads on this
    pod (thread-contention pathology), top-k is one cheap pass.
    """
    B, N = flags.shape
    kmax = int(k.max()) if B else 0
    if kmax <= 0:
        return torch.zeros_like(flags)
    key = torch.where(flags, torch.rand(B, N, device=flags.device),
                      torch.full((B, N), -1.0, device=flags.device))
    _, idx = torch.topk(key, kmax, dim=1)
    sel = torch.zeros_like(flags)
    sel.scatter_(1, idx, torch.arange(kmax, device=flags.device).unsqueeze(0)
                 < k.unsqueeze(1))
    return flags & sel


def span_mask(
    valid: torch.Tensor, lengths: torch.Tensor,
    span_len: int = 3, ratio: float = 0.55,
) -> torch.Tensor:
    """Draw the to-mask spans. Returns bool [B,N], True = mask this bin.

    Contract: only VALID bins are masked (unobserved bins have no
    reconstruction target), every row keeps >= 1 valid bin visible, and each
    row masks exactly min(ceil(ratio * n_valid), n_valid - 1) bins — reached
    by drawing ceil(target / span_len) + 1 random contiguous spans per row
    (overlap margin), randomly FILLING any undershoot and randomly TRIMMING
    any overshoot (single-bin fill bins break span contiguity only there).
    Rows with <= 1 valid bin come out unmasked. Fully vectorised and
    device-agnostic; entry points should cap torch threads (this pod's 64
    threads thrash small CPU ops). Uses global torch RNG for repro.
    """
    B, N = valid.shape
    device = valid.device
    n_valid = valid.sum(dim=1)                                   # [B]
    target = (ratio * n_valid).ceil().long()
    target = target.clamp(max=n_valid - 1).clamp(min=0)          # keep >=1 visible

    n_starts_all = (lengths - span_len + 1).clamp(min=1)         # [B]
    n_spans = (target.float() / span_len).ceil().long() + 1      # overlap margin
    S = int(n_spans.max().clamp(min=1)) if B else 1
    starts = (torch.rand(B, S, device=device) * n_starts_all.unsqueeze(1)).long()
    offs = torch.arange(span_len, device=device)                 # [L]
    flat = starts.unsqueeze(-1) + offs                           # [B, S, L]
    flat = flat.clamp(max=N - 1)                                 # short-row safety
    mask = torch.zeros(B * N, dtype=torch.bool, device=device)
    mask[torch.arange(B, device=device).view(B, 1, 1) * N + flat] = True
    mask = mask.view(B, N) & valid                               # holes never masked

    deficit = (target - mask.sum(dim=1)).clamp(min=0)            # spans overlapped
    mask = mask | _rand_subset(valid & ~mask, deficit)           # -> fill to target
    over = mask.sum(dim=1) - target                              # spans overshot
    if bool((over > 0).any()):
        keep = _rand_subset(mask, target)                        # -> trim to target
        mask = torch.where(over.unsqueeze(1) > 0, keep, mask)
    return mask


class AttrHeads(nn.Module):
    """Free-label heads on r: y / v / len quantile buckets + Beijing hour."""

    def __init__(self, d_in: int, n_buckets: int = 16, n_hours: int = 24) -> None:
        super().__init__()
        def head(n: int) -> nn.Module:
            return nn.Sequential(nn.Linear(d_in, d_in), nn.GELU(), nn.Linear(d_in, n))
        self.heads = nn.ModuleDict({k: head(n_buckets) for k in ("y", "v", "len")})
        self.heads["hour"] = head(n_hours)

    def forward(self, r: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {k: head(r) for k, head in self.heads.items()}


class CurveMAE(nn.Module):
    """Span-masked pretrainer with a CLS-bottleneck decoder (repV2 §3)."""

    def __init__(
        self,
        encoder_kwargs: Dict | None = None,
        n_buckets: int = 16,
        n_hours: int = 24,
        dec_layers: int = 1,
        dec_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = TrajectoryEncoder(**(encoder_kwargs or {}))
        d_model, d_out = self.encoder.pos_emb.embedding_dim, _d_out(self.encoder)
        self.cls_token = nn.Parameter(torch.zeros(d_model))
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=dec_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=dec_layers)
        self.r_proj = nn.Linear(d_out, d_model)             # r -> decoder width
        self.rec_head = nn.Linear(d_model, 1)
        self.attr = AttrHeads(d_out, n_buckets=n_buckets, n_hours=n_hours)

    def forward(
        self, speeds: torch.Tensor, valid: torch.Tensor, lengths: torch.Tensor,
        mask: torch.Tensor, use_cls: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """speeds [B,N] raw m/s, valid [B,N], lengths [B], mask [B,N] (span_mask).

        Returns dict: r [B,d_out], rec [B,N] (predicted v/v_norm, meaningful at
        masked positions), attr logits dict. With use_cls=False the decoder is
        conditioned on a zeroed r — the CLS-liveness probe.
        """
        B, N = speeds.shape
        device = speeds.device
        speeds_in = speeds.masked_fill(mask, 0.0)
        valid_in = valid & ~mask                             # native hole semantics
        r = curve_representation(
            self.encoder, self.cls_token, speeds_in, valid_in, lengths)
        r_eff = r if use_cls else torch.zeros_like(r)

        # decoder input = mask queries conditioned on r ONLY (visible bins are
        # excluded from attention — the §3.3 bottleneck)
        pos = self.encoder.pos_emb(torch.arange(N, device=device))     # [N,d]
        q = self.mask_token.view(1, 1, -1) + pos.unsqueeze(0)          # [1,N,d]
        q = q.expand(B, N, -1) + self.r_proj(r_eff).unsqueeze(1)       # [B,N,d]
        dec_pad = ~mask                                     # True = not a mask query
        empty = mask.sum(dim=1) == 0                        # all-pad row guard
        dec_pad[empty, 0] = False                           # (NaN otherwise)
        hd = self.decoder(q, src_key_padding_mask=dec_pad)
        rec = self.rec_head(hd).squeeze(-1)                 # [B,N]
        return {"r": r, "rec": rec, "attr": self.attr(r), "mask": mask}

    def rec_targets(self, speeds: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Reconstruction targets: true speeds in the encoder's scaled units."""
        return (speeds / self.encoder.v_norm).masked_select(mask)
