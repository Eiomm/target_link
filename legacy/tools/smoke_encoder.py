"""Smoke test for TrajectoryEncoder on real profiles (spec Step-7 style checks).

Checks: shapes, no-NaN, pad-invariance (output independent of padding length),
invalid-bin semantics, residual mode invariance to uniform speed shifts,
parameter counts, forward+backward on real data, d-sweep for Ablation 4.

Usage: python legacy/tools/smoke_encoder.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from target_link_v1.models import TrajectoryEncoder  # noqa: E402
from target_link_v1.utils import seed_everything  # noqa: E402


def n_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def main() -> None:
    seed_everything(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    d = np.load("data/processed/profiles_l200.npz")
    idx = np.arange(0, 4096)
    speeds = torch.from_numpy(d["speeds"][idx]).to(device)
    valid = torch.from_numpy(d["valid"][idx]).to(device)
    lengths = torch.from_numpy(d["lengths"][idx]).to(device)

    enc = TrajectoryEncoder().to(device).eval()
    print(f"[smoke] device={device} | default d=128 params={n_params(enc):,}")

    with torch.no_grad():
        r = enc(speeds, valid, lengths)
    assert r.shape == (len(idx), 128), r.shape
    assert torch.isfinite(r).all(), "NaN/inf in representation"
    print(f"[smoke] forward OK: r {tuple(r.shape)}, finite, "
          f"|r| mean={r.norm(dim=-1).mean():.3f}")

    # --- pad-invariance: trimming dead padding to a shorter N gives same output
    N2 = int(lengths.max().item())
    with torch.no_grad():
        r2 = enc(speeds[:, :N2], valid[:, :N2], lengths)
    assert torch.allclose(r, r2, atol=1e-5), (r - r2).abs().max()
    print(f"[smoke] pad-invariance OK: N 40 -> {N2}, r unchanged")

    # --- invalid-bin semantics: poking a hole changes r, but only via context
    s3, v3 = speeds.clone(), valid.clone()
    hole = v3 & (torch.arange(speeds.shape[1], device=device)[None] < lengths[:, None])
    pos0 = hole.float().argmax(dim=1)  # first valid bin of each row
    s3[torch.arange(len(idx), device=device), pos0] = 0.0
    v3[torch.arange(len(idx), device=device), pos0] = False
    with torch.no_grad():
        r3 = enc(s3, v3, lengths)
    changed = (r3 - r).norm(dim=-1) > 1e-6
    print(f"[smoke] hole-invalidation changes r for {int(changed.sum())}/{len(idx)} "
          f"profiles (context effect, expected > 0)")

    # --- residual mode: uniform speed shift must not change output
    enc_res = TrajectoryEncoder(input_mode="residual").to(device).eval()
    with torch.no_grad():
        ra = enc_res(speeds, valid, lengths)
        shift = 5.0
        s4 = torch.where(valid, speeds + shift, speeds)
        rb = enc_res(s4, valid, lengths)
    assert torch.allclose(ra, rb, atol=1e-4), (ra - rb).abs().max()
    print("[smoke] residual invariance OK: +5 m/s uniform shift -> same r")

    # --- absolute mode is NOT shift-invariant (contrast check)
    with torch.no_grad():
        shift = 5.0
        s4 = torch.where(valid, speeds + shift, speeds)
        rc = enc(s4, valid, lengths)
    assert not torch.allclose(r, rc, atol=1e-4)
    print("[smoke] absolute mode reacts to speed level (contrast OK)")

    # --- d sweep with out_proj=128 (Ablation 4 fairness control)
    for dm in (32, 64, 128, 256):
        e = TrajectoryEncoder(d_model=dm, out_dim=128).to(device).eval()
        with torch.no_grad():
            out = e(speeds, valid, lengths)
        assert out.shape == (len(idx), 128) and torch.isfinite(out).all()
        print(f"[smoke] d={dm:>3} params={n_params(e):>9,} -> out {tuple(out.shape)}")

    # --- forward + backward on GPU (training path)
    enc.train()
    r = enc(speeds, valid, lengths)
    loss = r.pow(2).mean()
    loss.backward()
    grads = [p.grad for p in enc.parameters() if p.grad is not None]
    assert all(torch.isfinite(g).all() for g in grads)
    print(f"[smoke] backward OK: loss={loss.item():.4f}, "
          f"{len(grads)} grad tensors all finite")


if __name__ == "__main__":
    main()
