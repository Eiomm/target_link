"""Whole-trajectory MAE with visible tokens and CLS in the decoder.

The model deliberately constructs visible trajectory features *after* selecting
``traj_valid & ~mae_mask``.  Thus a masked trajectory's profile and validity
bits are never read by the encoder. The decoder restores mask tokens to their
original slots and adds shared time embeddings and known hidden-row coverage
before joint self-attention. Hidden travel times and bin validity are never inputs.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TimeEmbedding(nn.Module):
    """Continuous, shared time embedding for visible rows and decoder queries."""

    def __init__(self, d_model: int, period: float = 600.0) -> None:
        super().__init__()
        self.period = float(period)
        self.mlp = nn.Sequential(
            nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        return self.mlp((delta_t / self.period).unsqueeze(-1))


class BucketTimeEmbedding(nn.Module):
    """Twenty 30-second bins; the inclusive 600s endpoint belongs to bin 19."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(20, d_model)

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        bucket = torch.floor(delta_t / 30).long().clamp(max=19)
        return self.embedding(bucket)


class TrajectoryMLPMAE(nn.Module):
    """Predict raw per-bin seconds from visible whole-trajectory profiles.

    ``x`` remains the corpus tensor ``[B, M, n_bins, 3]`` with columns
    ``(T_diff, ratio, observed)``.  Only its first two columns are read; the
    observed column is intentionally outside every model path.

    The input is flattened ``(T_clean, ratio, valid)``. Missing times are
    zeroed, but known coverage is retained independently of time validity.
    """

    def __init__(
        self,
        d_model: int = 256,
        heads: int = 8,
        layers: int = 4,
        dropout: float = 0.1,
        n_bins: int = 50,
        input_channels: int = 3,
        time_encoding: str = "seconds",
        decoder_layers: int = 2,
    ) -> None:
        super().__init__()
        if input_channels != 3:
            raise ValueError("input_channels must be 3: T_clean, ratio, valid")
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        if time_encoding not in ("seconds", "bucket30"):
            raise ValueError("time_encoding must be seconds or bucket30")
        if min(layers, decoder_layers) < 1:
            raise ValueError("encoder and decoder layers must be positive")
        self.d_model = int(d_model)
        self.n_bins = int(n_bins)
        self.input_channels = int(input_channels)

        self.trajectory_mlp = nn.Sequential(
            nn.Linear(self.n_bins * self.input_channels, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.time_embedding = (TimeEmbedding(d_model) if time_encoding == "seconds"
                               else BucketTimeEmbedding(d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.group_encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        self.decoder_embed = nn.Linear(d_model, d_model)
        self.ratio_embed = nn.Linear(self.n_bins, d_model)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_layers)
        self.decoder_norm = nn.LayerNorm(d_model)
        self.decoder_pred = nn.Linear(d_model, self.n_bins)
        parameter_count = sum(p.numel() for p in self.parameters())
        if parameter_count > 10_000_000:
            raise ValueError(f"Model has {parameter_count:,} parameters; limit is 10M")

    def _visible_features(
        self, x_visible: torch.Tensor, bin_valid_visible: torch.Tensor
    ) -> torch.Tensor:
        """Return flattened clean visible features without touching observed."""
        if x_visible.ndim != 3 or x_visible.shape[1:] != (self.n_bins, 3):
            raise ValueError(
                "visible x must have shape [P, %d, 3], got %s"
                % (self.n_bins, tuple(x_visible.shape))
            )
        valid = bin_valid_visible.to(dtype=x_visible.dtype).unsqueeze(-1)
        # ``where`` selects zero for invalid slots before values can enter an
        # MLP.  In particular NaN multiplied by a false mask is not used.
        clean = torch.where(
            bin_valid_visible.unsqueeze(-1),
            x_visible[..., :1],
            torch.zeros_like(x_visible[..., :1]),
        )
        if not torch.isfinite(clean).all() or (clean < 0).any():
            raise ValueError("visible valid T must be finite and nonnegative")
        ratio = x_visible[..., 1:2]
        self._check_ratio(ratio)
        return torch.cat([clean, ratio, valid], dim=-1).flatten(start_dim=1)

    @staticmethod
    def _check_ratio(ratio: torch.Tensor) -> None:
        # Known geometry is independent of T validity; unknown geometry must
        # not be silently converted to a zero-length observation.
        if not torch.isfinite(ratio).all() or (ratio < 0).any():
            raise ValueError("known ratio must be finite and nonnegative")

    def _checked_time(self, delta_t: torch.Tensor, traj_valid: torch.Tensor) -> torch.Tensor:
        """Reject bad real times while giving padded decoder rows a benign time."""
        real_time = delta_t[traj_valid]
        if not torch.isfinite(real_time).all():
            raise ValueError("delta_t must be finite for every traj_valid row")
        if ((real_time < 0) | (real_time > 600)).any():
            raise ValueError("delta_t must be in [0, 600] for every traj_valid row")
        # Do not use multiplication: padded NaN * 0 remains NaN.  The decoder
        # predicts all padded slots for shape compatibility, so it needs a
        # finite placeholder even though those predictions are never scored.
        return torch.where(traj_valid, delta_t, torch.zeros_like(delta_t))

    def _encode_visible(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack all groups into one Transformer pass and restore row states."""
        x = batch["x"]
        bin_valid = batch["bin_valid"]
        traj_valid = batch["traj_valid"]
        mae_mask = batch["mae_mask"]
        delta_t = batch["delta_t"]
        if x.ndim != 4 or x.shape[-2:] != (self.n_bins, 3):
            raise ValueError("x must have shape [B, M, %d, 3]" % self.n_bins)
        B, M = x.shape[:2]
        if bin_valid.shape != (B, M, self.n_bins):
            raise ValueError("bin_valid shape does not match x")
        if traj_valid.shape != (B, M) or mae_mask.shape != (B, M):
            raise ValueError("traj_valid and mae_mask must have shape [B, M]")
        if delta_t.shape != (B, M):
            raise ValueError("delta_t must have shape [B, M]")

        safe_time = self._checked_time(delta_t, traj_valid)
        visible = traj_valid & ~mae_mask
        trajectory_state = x.new_zeros(B, M, self.d_model)
        # Index first: hidden and padded x/bin_valid/delta_t are excluded
        # before their values are ever used to construct any feature or MLP.
        visible_index = visible.nonzero(as_tuple=False)  # [P, (group, row)]
        counts = visible.sum(dim=1)
        width = int(counts.max().item()) if B else 0
        # As in image MAE, broadcast one learned CLS parameter over samples.
        # Groups attend independently; their output CLS states are [B, d].
        cls_tokens = self.cls_token.to(dtype=x.dtype).expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, x.new_zeros(B, width, self.d_model)], dim=1)
        padding = torch.ones(B, width + 1, dtype=torch.bool, device=x.device)
        padding[:, 0] = False
        if visible_index.numel():
            groups, original_rows = visible_index.unbind(dim=1)
            rows_x = x[groups, original_rows]
            rows_valid = bin_valid[groups, original_rows]
            rows = self.trajectory_mlp(self._visible_features(rows_x, rows_valid))
            rows = rows + self.time_embedding(safe_time[groups, original_rows])
            starts = counts.cumsum(0) - counts
            packed_rows = torch.arange(len(groups), device=x.device) - starts[groups]
            tokens[groups, packed_rows + 1] = rows
            padding[groups, packed_rows + 1] = False

        # A group with no visible trajectories retains an unmasked CLS-only
        # sequence.  All padded tokens are literal zeros and key-masked.
        encoded = self.final_norm(
            self.group_encoder(tokens, src_key_padding_mask=padding)
        )
        representation = encoded[:, 0]
        if visible_index.numel():
            trajectory_state[groups, original_rows] = encoded[groups, packed_rows + 1]
        return representation, trajectory_state

    def encode(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return CLS and visible-row states using the same visibility rule."""
        representation, trajectory_state = self._encode_visible(batch)
        return {"representation": representation, "trajectory_state": trajectory_state}

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        representation, trajectory_state = self._encode_visible(batch)
        x = batch["x"]
        B, M = x.shape[:2]
        valid = batch["traj_valid"].bool()
        visible = valid & ~batch["mae_mask"].bool()
        # trajectory_state is already scattered back to original row slots.
        rows = torch.where(visible.unsqueeze(-1), self.decoder_embed(trajectory_state),
                           self.mask_token.expand(B, M, -1))
        hidden = valid & batch["mae_mask"].bool()
        # Select real hidden rows first: padded NaNs are never read.
        hidden_ratio = x[..., 1][hidden]
        self._check_ratio(hidden_ratio)
        ratio_condition = torch.zeros_like(rows)
        ratio_condition[hidden] = self.ratio_embed(hidden_ratio)
        rows = rows + ratio_condition
        rows = rows + self.time_embedding(self._checked_time(batch["delta_t"], valid))
        rows = torch.where(valid.unsqueeze(-1), rows, torch.zeros_like(rows))
        cls = self.decoder_embed(representation).unsqueeze(1)
        tokens = torch.cat([cls, rows], dim=1)
        padding = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), ~valid], dim=1)
        decoded = self.decoder_norm(self.decoder(tokens, src_key_padding_mask=padding))
        prediction_seconds = F.softplus(self.decoder_pred(decoded[:, 1:]))
        return {
            "prediction_seconds": prediction_seconds,
            "representation": representation,
            "trajectory_state": trajectory_state,
        }
