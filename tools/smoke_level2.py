"""Smoke test for the v2 trajectory-level Transformer (md/9_4.md §3 Level 2).

Synthetic ragged groups (variable K, incl. K=1):
  1. forward: h_CLS shape [G, d], finite
  2. permutation invariance: shuffling profile rows leaves h_CLS unchanged
     (set semantics — no position embedding, trajectory order is arbitrary)
  3. batch independence: a group's h_CLS is identical computed alone or inside
     a bigger batch with a larger K_max (padding never leaks into attention)
  4. backward: loss on h_CLS reaches the CLS token and the trajectory encoder
Real profiles:
  5. encoder -> level2 over all (sub-link, window) groups (group-aligned chunks)
  6. ETAModel("ours") forward+backward with aggregation cls AND mean (both arms
     of the §6 ablation must be wired); the mean arm must equal the old
     two-step scatter_mean exactly (baseline intact)

Usage: python tools/smoke_level2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from target_link_v1.data import build_group_index, group_aligned_chunks, sort_by_group  # noqa: E402
from target_link_v1.models import (  # noqa: E402
    ETAModel, TrajectoryEncoder, TrajectoryLevelTransformer, encode_link_rep,
    pack_groups, scatter_mean,
)
from target_link_v1.utils import seed_everything  # noqa: E402

MAX_ROWS = 65536  # encoder forward chunk (group-aligned)


def main() -> None:
    seed_everything(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)

    # --- 1. forward on ragged groups (K=1 included) --------------------------
    ks = np.array([1, 1, 2, 3, 5, 8, 1, 3])
    group_idx = torch.from_numpy(np.repeat(np.arange(len(ks)), ks))
    r = torch.from_numpy(
        rng.standard_normal((len(group_idx), 128)).astype(np.float32))
    level2 = TrajectoryLevelTransformer(d_model=128).to(device).eval()
    with torch.no_grad():
        h = level2(r.to(device), group_idx.to(device), len(ks))
    assert h.shape == (len(ks), 128) and torch.isfinite(h).all()
    print(f"[l2] forward OK: h_CLS {tuple(h.shape)}, finite, "
          f"|h| mean={h.norm(dim=-1).mean():.3f}")

    # --- 2. permutation invariance: row order must not matter ----------------
    perm = rng.permutation(len(group_idx))
    with torch.no_grad():
        h_perm = level2(r[perm].to(device), group_idx[perm].to(device), len(ks))
    assert torch.allclose(h, h_perm, atol=1e-5), "h_CLS depends on row order"
    print("[l2] permutation invariance OK (no position embedding)")

    # --- 3. batch independence: alone vs inside a bigger padded batch --------
    gi_dev = group_idx.to(device)
    with torch.no_grad():
        h_alone = level2(r.to(device)[gi_dev < 3], gi_dev[gi_dev < 3], 3)
    assert torch.allclose(h_alone, h[:3], atol=1e-5), \
        "h_CLS leaks across groups / depends on batch K_max"
    print("[l2] batch independence OK (padding excluded from attention)")

    # --- pack_groups unit checks ----------------------------------------------
    packed, pad = pack_groups(r.to(device), gi_dev, len(ks))
    counts = torch.bincount(gi_dev, minlength=len(ks))
    assert packed.shape == (len(ks), int(ks.max()), 128)
    assert (pad.sum(dim=1) == int(ks.max()) - counts).all(), "mask width != K_max - K"
    assert (packed[pad]).abs().sum() == 0, "padded slots not zero"
    assert torch.allclose(packed[4, :5], r.to(device)[gi_dev == 4]), "rows misplaced"
    print("[l2] pack_groups OK: [G, K_max, d], padded slots zeroed+masked")

    # --- 4. backward reaches CLS token and the trajectory encoder -------------
    enc = TrajectoryEncoder(d_model=128, n_layers=2, out_dim=128).to(device)
    level2b = TrajectoryLevelTransformer(d_model=128).to(device)
    speeds = torch.from_numpy(rng.uniform(0, 30, (len(group_idx), 40)).astype(np.float32))
    valid = torch.ones_like(speeds, dtype=torch.bool)
    lengths = torch.full((len(group_idx),), 40, dtype=torch.int)
    r_enc = enc(speeds.to(device), valid.to(device), lengths.to(device))
    level2b(r_enc, gi_dev, len(ks)).pow(2).mean().backward()
    grads = [p.grad for p in list(enc.parameters()) + list(level2b.parameters())
             if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert level2b.cls_token.grad is not None and level2b.cls_token.grad.abs().sum() > 0
    assert any(g.abs().sum() > 0 for g in grads)
    print(f"[l2] backward OK: {len(grads)} grad tensors finite, CLS token + encoder nonzero")

    # --- 5. real profiles: encoder -> level2 over every group ----------------
    d = np.load("data/processed/profiles_l200.npz")
    gi = build_group_index(d["link_id"], d["sub_id"], d["window_id"])
    order = sort_by_group(gi.group_idx)
    gid_np = gi.group_idx[order]
    k = gi.keys.n_trajs.to_numpy()
    print(f"[l2] real data: {len(gid_np)} profiles -> {len(k)} groups, "
          f"K p50={np.median(k):.0f} p90={np.percentile(k, 90):.0f} max={k.max()}")
    speeds_t = torch.from_numpy(d["speeds"][order]).to(device)
    valid_t = torch.from_numpy(d["valid"][order]).to(device)
    lengths_t = torch.from_numpy(d["lengths"][order]).to(device)
    enc.eval()  # train-mode attention kernels cap the batch at 65535 rows
    with torch.no_grad():
        # per chunk the (sorted, group-aligned) group ids are one contiguous
        # range — renumber locally, exactly like ETAData.batch does per batch
        h_parts = []
        for rows in group_aligned_chunks(gid_np, MAX_ROWS):
            gids = torch.from_numpy(gid_np[rows]).to(device)
            base, n_loc = int(gids[0]), int(gids[-1]) - int(gids[0]) + 1
            h_parts.append(level2(
                enc(speeds_t[rows], valid_t[rows], lengths_t[rows]),
                gids - base, n_loc))
        h_all = torch.cat(h_parts)
    assert h_all.shape == (len(k), 128) and torch.isfinite(h_all).all()
    print(f"[l2] real data forward OK: h_CLS {tuple(h_all.shape)}, finite")

    # --- 6. ETAModel end-to-end, both aggregation arms -----------------------
    # small batch: 16 samples x 1 group each, drawn from real groups
    g_sel = np.sort(rng.choice(len(k), size=16, replace=False))  # searchsorted needs sorted
    prof_rows = np.concatenate([np.flatnonzero(gid_np == g) for g in g_sel])
    prof_of = np.searchsorted(g_sel, gid_np[prof_rows])
    batch = {
        "speeds": d["speeds"][order][prof_rows], "valid": d["valid"][order][prof_rows],
        "lengths": d["lengths"][order][prof_rows], "prof_group": prof_of,
        "n_groups": len(g_sel), "edge_group": np.arange(len(g_sel)),
        "edge_sample": np.arange(len(g_sel)), "n_samples": len(g_sel),
    }
    bt = {kk: (torch.from_numpy(v).to(device) if isinstance(v, np.ndarray) else v)
          for kk, v in batch.items()}
    L_n = torch.randn(len(g_sel), device=device)
    v_n = torch.rand(len(g_sel), device=device)
    for agg in ("cls", "mean"):
        model = ETAModel("ours", aggregation=agg,
                         encoder_kwargs={"d_model": 128, "n_layers": 2, "out_dim": 128},
                         level2_kwargs={"n_layers": 2}).to(device)
        out = model(L_n, v_n, bt)
        assert out.shape == (len(g_sel),) and torch.isfinite(out).all()
        out.pow(2).mean().backward()
        gs = [p.grad for p in model.parameters() if p.grad is not None]
        assert gs and any(g.abs().sum() > 0 for g in gs)
        if agg == "mean":  # baseline must equal the old two-step scatter_mean
            model.eval()  # fix dropout masks — both passes must be identical
            with torch.no_grad():
                manual = scatter_mean(
                    scatter_mean(model.encoder(**{kk: bt[kk] for kk in
                                                  ("speeds", "valid", "lengths")}),
                                 bt["prof_group"], len(g_sel))[bt["edge_group"]],
                    bt["edge_sample"], len(g_sel))
                wired = encode_link_rep(model.encoder, None, **bt)
            assert torch.allclose(manual, wired, atol=1e-6), "mean arm drifted"
        print(f"[l2] ETAModel ours-{agg}: fwd/bwd OK, "
              f"{sum(p.numel() for p in model.parameters()):,} params")

    print("[l2] all checks passed")


if __name__ == "__main__":
    main()
