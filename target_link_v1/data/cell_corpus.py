"""Read the V1 cell corpus as whole-trajectory-MAE batches.

Corpus contract (tools/build_corpus.py, spec md/最新讨论想法.md):

  observations/     one row per (cell, trajectory) observation, ragged
  training_groups_k3/ one row per training group: at most m_max=16 trajectories
                    of one cell. A cell with K>16 was split deterministically
                    and every trajectory appears in exactly one group, so a
                    group is the training unit and K_min/M_max are policy that
                    lives here, not in the corpus.

A cell is (map_version, target_link_id, seg_idx, 10-min window), i.e. one 500m
segment of one link in one time window. The model token is the 10m BIN of that
segment, so a batch is [B, m_max, 50, F] -- but the corpus is ragged in PIECES,
because a bin that straddles a link boundary is stored as several pieces. This
reader scatters the pieces back onto their real `bin_pos` (0..49); a piece
number must never be used as a spatial position, since a split bin would then
claim several of them.

Feature axis F = (T_diff, ratio, observed):
  T_diff    seconds to cross the bin, summed over the bin's pieces -- a split
            bin is one bin, its crossing time is the sum of its parts
  ratio     fraction of the bin the trajectory actually covered, summed over
            the pieces: 1.0 for an interior bin, <1 where the segment boundary
            cuts a bin (that is a real feature, not an error)
  observed  1.0 if any piece of the bin had a GPS fix
`valid` is NOT a feature: it is folded into the bin mask. Piece-level valid
means "T_diff is not NaN" (the corpus never imputes). A bin is valid only when
EVERY piece has a time, because T_diff is a SUM and dropping a NaN piece would
silently understate the bin. For an invalid bin T_diff is zero, while ratio
still sums all pieces and observed still uses OR over all pieces.
A bin with no piece at all (the trajectory skipped it, or the segment boundary
cut it away) is invalid and its features are zero -- the same "gap" convention
the pretrain encoder already uses for missing observations.

Whole-trajectory MAE masking (50%, pinned) is produced here, not in the model:
`mae_mask` marks trajectories excluded from Level 2 and the group-bin context.
The exact bins used by the reconstruction loss are derived through one shared
helper as `mae_mask[..., None] & bin_valid`. The mask is drawn per group from
`(epoch, group_id)`, so an epoch is
reproducible and a re-run of the same epoch is identical; call `set_epoch`
between epochs. Groups with fewer than three valid trajectories are left
unmasked because the policy requires at least one hidden and two visible
trajectories.
"""
from __future__ import annotations

import os
import zlib

import numpy as np
import pyarrow.compute as pc
import pyarrow.fs as pfs
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info

N_BINS = 50    # 500m segment / 10m bin -- the spatial grid, fixed by the spec
M_MAX = 16     # pinned 2026-09-09
FEATURES = ("T_diff", "ratio", "observed")

_OBS_COLS = ["cell_id", "sample_id", "dt", "T_diff", "ratio_pct",
             "observed", "valid", "bin_pos"]
_GRP_COLS = ["group_id", "cell_id", "K", "group_size", "sample_ids", "window"]
_PIECE_COLS = (("T", "T_diff"), ("R", "ratio_pct"), ("O", "observed"),
               ("V", "valid"), ("B", "bin_pos"))


class _Store:
    """List and read Parquet under one local or hdfs:// root with one API."""

    def __init__(self, root):
        uri = str(root)
        if "://" in uri:
            self.fs, base = pfs.FileSystem.from_uri(uri)
        else:
            self.fs, base = pfs.LocalFileSystem(), os.path.abspath(uri)
        base = (base or "").rstrip("/")
        self.base = "" if base.endswith(":") else base

    def _infos(self, rel=""):
        sel = pfs.FileSelector(("%s/%s" % (self.base, rel)).rstrip("/"),
                               recursive=False, allow_not_found=True)
        return self.fs.get_file_info(sel)

    def dirs(self, rel=""):
        return sorted(i.base_name for i in self._infos(rel)
                      if i.type == pfs.FileType.Directory)

    def files(self, rel=""):
        return sorted(i.path for i in self._infos(rel)
                      if i.type == pfs.FileType.File and i.path.endswith(".parquet"))

    def read(self, path, columns):
        return pq.read_table(path, filesystem=self.fs, columns=columns)


def _single(col):
    # a table column is a ChunkedArray (has combine_chunks), a slice is an Array
    return col.combine_chunks() if hasattr(col, "combine_chunks") else col


def _offsets(col):
    """Row offsets of a ragged column (n+1 int64)."""
    arr = _single(col)
    off = np.zeros(len(arr) + 1, dtype=np.int64)
    np.cumsum(arr.value_lengths().to_numpy(zero_copy_only=False), out=off[1:])
    return off


def _flat(col):
    return pc.list_flatten(_single(col))


class CellCorpusDataset(IterableDataset):
    """One yielded item = one training group (<= m_max trajectories of a cell).

    Yields fixed-width numpy so the worker ships no Python objects across the
    process boundary except the ids kept for debugging and joining backwards:

        x          float32 [m, 50, 3]   T_diff (s, raw), ratio, observed
        bin_valid  bool    [m, 50]      the spatial mask (the bin's T_diff is known)
        traj_valid bool    [m]          the trajectory has >= 1 valid bin
        delta_t    float32 [m]          t_seg_enter - window_start, in [0,600)
        cell_id, K, window, day, bucket, group_id, sample_ids
    """

    def __init__(self, directory, obs_dir="observations_v2",
                 groups_dir="training_groups_k3", days=None, seed=0, epoch=0,
                 shuffle_groups=True, max_groups=None, m_max=M_MAX,
                 n_bins=N_BINS, groups_per_partition=None):
        super().__init__()
        self.obs = _Store("%s/%s" % (str(directory).rstrip("/"), obs_dir))
        self.grp = _Store("%s/%s" % (str(directory).rstrip("/"), groups_dir))
        want = None if days is None else {"day=%s" % d for d in days}
        parts = []
        for day in sorted(set(self.grp.dirs()) & set(self.obs.dirs())):
            if not day.startswith("day=") or (want is not None and day not in want):
                continue
            for bucket in sorted(set(self.grp.dirs(day)) & set(self.obs.dirs(day))):
                parts.append((day, bucket))
        if not parts:
            raise ValueError("No (day, bucket) partition holds both observations "
                             "and the selected groups directory -- is the corpus built?")
        self.partitions = parts
        self.seed, self.epoch = int(seed), int(epoch)
        self.shuffle_groups = bool(shuffle_groups)
        self.max_groups = max_groups
        self.groups_per_partition = groups_per_partition
        self.m_max, self.n_bins = int(m_max), int(n_bins)
        if self.max_groups is not None and self.max_groups <= 0:
            raise ValueError("max_groups must be positive or None")
        if self.groups_per_partition is not None and self.groups_per_partition <= 0:
            raise ValueError("groups_per_partition must be positive or None")

    def set_epoch(self, epoch):
        """Reseed the group shuffle and the MAE mask; call once per epoch."""
        self.epoch = int(epoch)
        return self

    def _load(self, day, bucket):
        """One (day, bucket): group columns as numpy + a lookup into the rows.

        Both files are written cell_id-sorted, so a group's observations are a
        contiguous slice and one searchsorted per cell is enough.
        """
        gt = [self.grp.read(f, _GRP_COLS) for f in self.grp.files("%s/%s" % (day, bucket))]
        ot = [self.obs.read(f, _OBS_COLS) for f in self.obs.files("%s/%s" % (day, bucket))]
        if not gt or not ot:
            return None
        g = pc.concat_tables(gt) if len(gt) > 1 else gt[0]
        o = pc.concat_tables(ot) if len(ot) > 1 else ot[0]
        cid = o["cell_id"].to_numpy(zero_copy_only=False).astype(np.int64)
        # Do not use np.diff(cid) here: signed int64 subtraction can overflow
        # across the xxhash64 positive/negative boundary, hiding a real descent
        # or inventing one on an ascent. Direct comparison cannot overflow.
        if cid.size > 1 and np.any(cid[1:] < cid[:-1]):
            raise ValueError("observations partition is not sorted by cell_id")
        flat = {k: _flat(o[c]) for k, c in _PIECE_COLS}
        flat["off"] = _offsets(o["T_diff"])
        sid = g["sample_ids"]
        out = dict(
            cid=cid, flat=flat, obs=o,
            dt=o["dt"].to_numpy(zero_copy_only=False).astype(np.float32),
            sid_flat=_flat(sid), sid_off=_offsets(sid),
            g_cell=g["cell_id"].to_numpy(zero_copy_only=False).astype(np.int64),
            g_K=g["K"].to_numpy(zero_copy_only=False).astype(np.int64),
            g_win=g["window"].to_numpy(zero_copy_only=False).astype(np.int64),
            g_id=g["group_id"],
            n_groups=len(g))
        return out

    def _scatter(self, row, flat, x, valid):
        """Fold one observation's pieces into its 50 bins; returns has-any-bin."""
        a, b = int(flat["off"][row]), int(flat["off"][row + 1])
        bp = flat["B"][a:b].to_numpy(zero_copy_only=False).astype(np.int64)
        if bp.size == 0:
            return False
        if bp.min() < 0 or bp.max() >= self.n_bins:
            raise ValueError("bin_pos outside 0..%d; corpus is corrupt" % (self.n_bins - 1))
        t = flat["T"][a:b].to_numpy(zero_copy_only=False).astype(np.float64)
        r = flat["R"][a:b].to_numpy(zero_copy_only=False).astype(np.float64) / 10.0
        o = flat["O"][a:b].to_numpy(zero_copy_only=False).astype(bool)
        v = flat["V"][a:b].to_numpy(zero_copy_only=False).astype(bool)
        n = self.n_bins
        # a split bin is ONE bin: sum its parts rather than letting the last
        # piece overwrite the first
        cnt = np.bincount(bp, minlength=n)
        vcnt = np.bincount(bp[v], minlength=n)
        ocnt = np.bincount(bp[o], minlength=n)
        td = np.bincount(bp, weights=np.where(v, t, 0.0), minlength=n)
        rt = np.bincount(bp, weights=r, minlength=n)
        present = cnt > 0
        ok = present & (vcnt == cnt)
        x[:, 0] = np.where(ok, td, 0.0)
        x[:, 1] = np.where(present, rt, 0.0)
        x[:, 2] = np.where(present, ocnt > 0, 0.0)
        valid[:] = ok
        return bool(ok.any())

    def _pack(self, day, bucket, L, gi):
        cell = int(L["g_cell"][gi])
        lo = int(np.searchsorted(L["cid"], cell, "left"))
        hi = int(np.searchsorted(L["cid"], cell, "right"))
        if lo == hi:
            raise ValueError("group %d has no observations; corpus is corrupt" % cell)
        row_of = {s: lo + k
                  for k, s in enumerate(L["obs"]["sample_id"][lo:hi].to_pylist())}
        sids = L["sid_flat"][int(L["sid_off"][gi]):int(L["sid_off"][gi + 1])].to_pylist()
        m = len(sids)
        if m == 0 or m > self.m_max:
            raise ValueError("group size %d outside 1..%d" % (m, self.m_max))
        x = np.zeros((m, self.n_bins, len(FEATURES)), dtype=np.float32)
        bin_valid = np.zeros((m, self.n_bins), dtype=bool)
        traj_valid = np.zeros(m, dtype=bool)
        delta_t = np.zeros(m, dtype=np.float32)
        for j, sid in enumerate(sids):
            row = row_of.get(sid)
            if row is None:
                raise ValueError("group references an observation outside its "
                                 "partition: %s" % sid)
            traj_valid[j] = self._scatter(row, L["flat"], x[j], bin_valid[j])
            delta_t[j] = L["dt"][row]
        # dt is a float32 of a double difference, so it occasionally rounds up
        # to exactly 600.0 (measured 6 rows in 1,195,673 on one bucket) while
        # the contract is [0, 600). Those rows ARE the window edge, so clamp
        # them just inside instead of aborting -- a value past 600 by more than
        # the float32 spacing is still corruption and still raises.
        if delta_t.min() < 0.0 or delta_t.max() > 600.0:
            raise ValueError("delta_t outside [0, 600]")
        np.clip(delta_t, 0.0, np.nextafter(np.float32(600.0), np.float32(0)),
                out=delta_t)
        return dict(x=x, bin_valid=bin_valid, traj_valid=traj_valid, delta_t=delta_t,
                    cell_id=cell, K=int(L["g_K"][gi]), window=int(L["g_win"][gi]),
                    group_id=L["g_id"][gi].as_py(), sample_ids=sids,
                    day=day.split("=", 1)[1], bucket=bucket.split("=", 1)[1],
                    m_max=self.m_max, n_bins=self.n_bins, epoch=self.epoch)

    def __iter__(self):
        worker = get_worker_info()
        wid, nw = (worker.id, worker.num_workers) if worker else (0, 1)
        if worker is not None and self.max_groups is not None:
            raise ValueError("max_groups is only supported with num_workers=0")
        # every worker takes the SAME partition permutation, then disjoint
        # indices, so sharding stays balanced without a barrier
        # A capped epoch should not revisit the same leading partitions forever.
        # Validation passes a fixed epoch, so its order remains reproducible.
        order = np.random.default_rng([self.seed, self.epoch]).permutation(
            len(self.partitions))[wid::nw]
        seen = 0
        for pi in order:
            day, bucket = self.partitions[int(pi)]
            L = self._load(day, bucket)
            if L is None:
                continue
            idx = np.arange(L["n_groups"])
            if self.shuffle_groups:
                idx = np.random.default_rng([self.seed, self.epoch, int(pi)]).permutation(idx)
            if self.groups_per_partition is not None:
                idx = idx[:self.groups_per_partition]
            for gi in idx:
                yield self._pack(day, bucket, L, int(gi))
                seen += 1
                if self.max_groups is not None and seen >= self.max_groups:
                    return

    def n_partitions(self):
        return len(self.partitions)


def collate_cells(items, m_max=None, mae_ratio=0.5, epoch=None):
    """Pad groups to [B, m_max, 50, F] and draw the whole-trajectory MAE mask.

    Padding lives here and not in the corpus so M_max/K_min stay pure policy:
    slots past a group's size are all-zero with traj_valid=0, and the mask
    never covers them.
    """
    if not items:
        raise ValueError("empty batch")
    M = int(m_max or items[0]["m_max"])
    n_bins = int(items[0]["n_bins"])
    F = items[0]["x"].shape[-1]
    B = len(items)
    x = np.zeros((B, M, n_bins, F), dtype=np.float32)
    bin_valid = np.zeros((B, M, n_bins), dtype=bool)
    traj_valid = np.zeros((B, M), dtype=bool)
    delta_t = np.zeros((B, M), dtype=np.float32)
    mae_mask = np.zeros((B, M), dtype=bool)
    for i, it in enumerate(items):
        m = it["x"].shape[0]
        if m > M:
            raise ValueError("group of %d exceeds m_max=%d" % (m, M))
        x[i, :m], bin_valid[i, :m] = it["x"], it["bin_valid"]
        traj_valid[i, :m], delta_t[i, :m] = it["traj_valid"], it["delta_t"]
        rng = np.random.default_rng([zlib.crc32(it["group_id"].encode()),
                                     int(it["epoch"] if epoch is None else epoch)])
        nv = int(it["traj_valid"].sum())
        if nv >= 3:
            k = min(max(int(round(mae_ratio * nv)), 1), nv - 2)
            pick = rng.choice(np.flatnonzero(it["traj_valid"]), size=k, replace=False)
            mae_mask[i, pick] = True
    return dict(x=torch.from_numpy(x), bin_valid=torch.from_numpy(bin_valid),
                traj_valid=torch.from_numpy(traj_valid),
                delta_t=torch.from_numpy(delta_t), mae_mask=torch.from_numpy(mae_mask),
                cell_id=torch.tensor([it["cell_id"] for it in items], dtype=torch.int64),
                K=torch.tensor([it["K"] for it in items], dtype=torch.int64),
                group_id=[it["group_id"] for it in items],
                sample_ids=[it["sample_ids"] for it in items],
                m_max=M, n_bins=n_bins)
