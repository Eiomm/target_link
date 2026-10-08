"""Observation folding and stable trajectory grouping, shared by every data builder."""
from __future__ import annotations
import hashlib
import numpy as np
import pyarrow as pa
from .columns import _array, _flat, _offsets, _row_any

N_BINS = 50

def _seed(*parts: object) -> int:
    """Stable seed; unlike ``hash()``, this is unchanged between processes."""
    text = "\x1f".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(text, digest_size=8).digest(), "little")



class ObservationGroups:
    def __init__(self, m_max=64, seed=20260921, epoch=0):
        super().__init__()
        if m_max < 3:
            raise ValueError("m_max must be at least 3")
        self.m_max, self.seed, self.epoch = int(m_max), int(seed), int(epoch)

    def _load_partition(self, store, day, bucket):
        tables = [store.read(path) for path in store.files("day=%s/bucket=%s" % (day, bucket))]
        if not tables:
            return None
        table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
        original_cid = table["cell_id"].to_numpy(zero_copy_only=False).astype(np.int64)
        original_sid = np.asarray(table["sample_id"].to_pylist(), dtype=object)
        identity_order = np.lexsort((original_sid, original_cid))
        identity_cid, identity_sid = original_cid[identity_order], original_sid[identity_order]
        if (len(identity_cid) > 1 and
                np.any((identity_cid[1:] == identity_cid[:-1]) &
                       (identity_sid[1:] == identity_sid[:-1]))):
            raise ValueError("duplicate sample_id within a cell")
        raw_ids, raw_counts = np.unique(original_cid, return_counts=True)
        raw_count = dict(zip(raw_ids.tolist(), raw_counts.tolist()))
        lengths = {name: _array(table[name]).value_lengths().to_numpy(zero_copy_only=False)
                   for name in ("T_diff", "ratio_pct", "valid", "bin_pos")}
        if any(not np.array_equal(lengths["T_diff"], lengths[name])
               for name in ("ratio_pct", "valid", "bin_pos")):
            raise ValueError("ragged observation arrays have different per-row lengths")
        off = _offsets(table["T_diff"])
        valid_flat = _flat(table["valid"]).to_numpy(zero_copy_only=False).astype(bool)
        # Filter before Python-level cell/group construction.  This is important
        # for the 1M-row partitions: trajectories with no possible label cannot
        # participate in a group and need not allocate a 50-bin tensor.
        candidate = _row_any(valid_flat, off)
        if not candidate.any():
            return None
        if not candidate.all():
            table = table.filter(pa.array(candidate))
            off = _offsets(table["T_diff"])
            valid_flat = _flat(table["valid"]).to_numpy(zero_copy_only=False).astype(bool)

        cid = table["cell_id"].to_numpy(zero_copy_only=False).astype(np.int64)
        sid = original_sid[candidate]
        dt = table["dt"].to_numpy(zero_copy_only=False).astype(np.float32)
        if not np.isfinite(dt).all() or (dt < 0).any() or (dt > 600).any():
            raise ValueError("delta_t outside finite [0, 600]")
        flat = {
            "T": _flat(table["T_diff"]).to_numpy(zero_copy_only=False).astype(np.float64),
            "R": _flat(table["ratio_pct"]).to_numpy(zero_copy_only=False).astype(np.float64),
            "V": valid_flat,
            "B": _flat(table["bin_pos"]).to_numpy(zero_copy_only=False).astype(np.int64),
            "off": off,
            # A prefix sum is much cheaper than 50-wide bincounts for the
            # overwhelmingly common all-valid trajectory.
            "row_all_valid": None,
        }
        if flat["B"].size and ((flat["B"] < 0).any() or (flat["B"] >= N_BINS).any()):
            raise ValueError("bin_pos outside 0..49")
        if flat["R"].size and (not np.isfinite(flat["R"]).all() or (flat["R"] < 0).any()):
            raise ValueError("ratio_pct must be finite and nonnegative")
        times = flat["T"][flat["V"]]
        if not np.isfinite(times).all() or (times < 0).any():
            raise ValueError("valid T_diff must be finite and nonnegative")
        valid_prefix = np.empty(len(valid_flat) + 1, dtype=np.int64)
        valid_prefix[0] = 0
        np.cumsum(valid_flat.astype(np.int64), out=valid_prefix[1:])
        flat["row_all_valid"] = ((valid_prefix[off[1:]] - valid_prefix[off[:-1]]) ==
                                 np.diff(off))
        return dict(cid=cid, sid=sid, dt=dt, flat=flat, raw_count=raw_count,
                    partition_stats={"day": day, "bucket": bucket,
                                     "raw_rows": int(len(original_cid)),
                                     "raw_cells": int(len(raw_count)),
                                     "candidate_rows": int(candidate.sum()),
                                     "usable_rows": 0,
                                     "dropped_no_valid": int((~candidate).sum()),
                                     "dropped_tail": 0, "groups": 0,
                                     "full_groups": 0, "tail_groups": 0,
                                     "selected_groups": 0})


    @staticmethod
    def _cells(cid):
        # Stable sorting makes corpus physical ordering irrelevant.
        order = np.argsort(cid, kind="stable")
        sorted_cid = cid[order]
        starts = np.r_[0, np.flatnonzero(sorted_cid[1:] != sorted_cid[:-1]) + 1]
        return order, starts, np.r_[starts[1:], len(order)]


    def _scatter(self, row, flat):
        a, b = int(flat["off"][row]), int(flat["off"][row + 1])
        bp, t, r, v = flat["B"][a:b], flat["T"][a:b], flat["R"][a:b] / 10.0, flat["V"][a:b]
        count = np.bincount(bp, minlength=N_BINS)
        vcount = np.bincount(bp[v], minlength=N_BINS)
        present = count > 0
        bin_valid = present & (count == vcount)
        x = np.zeros((N_BINS, 3), dtype=np.float32)
        x[:, 0] = np.where(bin_valid, np.bincount(bp, weights=np.where(v, t, 0.0), minlength=N_BINS), 0.0)
        x[:, 1] = np.where(present, np.bincount(bp, weights=r, minlength=N_BINS), 0.0)
        # Channel 2 is intentionally zero.  ``observed`` is not even read from
        # parquet, so it cannot leak into this experiment by accident.
        return x, bin_valid


    @staticmethod
    def _has_valid_bin(row, flat):
        a, b = int(flat["off"][row]), int(flat["off"][row + 1])
        bp, v = flat["B"][a:b], flat["V"][a:b]
        count = np.bincount(bp, minlength=N_BINS)
        return bool(np.any((count > 0) & (np.bincount(bp[v], minlength=N_BINS) == count)))


    def _group_specs(self, loaded, day, bucket):
        stats = loaded["partition_stats"]
        order, starts, ends = self._cells(loaded["cid"])
        for lo, hi in zip(starts, ends):
            rows = order[lo:hi]
            cell_id = int(loaded["cid"][rows[0]])
            raw_k = int(loaded["raw_count"][cell_id])
            candidate_k = len(rows)
            packed = []
            for row in rows:
                if (loaded["flat"]["row_all_valid"][row] or
                        self._has_valid_bin(int(row), loaded["flat"])):
                    packed.append((str(loaded["sid"][row]), int(row)))
            usable_k = len(packed)
            stats["usable_rows"] += usable_k
            stats["dropped_no_valid"] += candidate_k - usable_k
            if usable_k < 3:
                # This includes a cell whose raw K was >=3 but became too small
                # after invalid labels were removed, as well as raw K<3 cells.
                stats["dropped_tail"] += usable_k
                continue
            # sample_id order is part of the construction contract; the separate
            # stable RNG is then what defines persistent group membership.
            packed.sort(key=lambda p: p[0])
            if any(a[0] == b[0] for a, b in zip(packed, packed[1:])):
                raise ValueError("duplicate sample_id within cell_id=%d" % cell_id)
            perm = np.random.default_rng(_seed(self.seed, day, bucket, cell_id, "members")).permutation(usable_k)
            packed = [packed[int(i)] for i in perm]
            if usable_k % self.m_max < 3:
                stats["dropped_tail"] += usable_k % self.m_max
            for group_index, begin in enumerate(range(0, usable_k, self.m_max)):
                members = packed[begin:begin + self.m_max]
                if len(members) < 3:  # discard only the undersized tail; never rebalance
                    continue
                stats["groups"] += 1
                if len(members) == self.m_max:
                    stats["full_groups"] += 1
                else:
                    stats["tail_groups"] += 1
                group_id = "groups-v1-m%d/%s/%s/%s/%d" % (self.m_max, day, bucket,
                                                            cell_id, group_index)
                yield dict(
                    cell_id=cell_id, K=usable_k, K_raw=raw_k, group_size=len(members),
                    group_id=group_id, rows=np.asarray([row for _, row in members], dtype=np.int64),
                    day=day, bucket=bucket,
                    dropped_trajectories=raw_k - usable_k,
                    dropped_no_valid=raw_k - usable_k,
                    dropped_tail=(usable_k % self.m_max if usable_k % self.m_max < 3 else 0),
                )


    def _pack(self, loaded, spec):
        if spec.get('tensor_ready'):
            return dict(spec, epoch=self.epoch)
        rows = spec["rows"]
        x, bv = zip(*(self._scatter(int(row), loaded["flat"]) for row in rows))
        result = {key: value for key, value in spec.items() if key != "rows"}
        return dict(result, x=np.stack(x), bin_valid=np.stack(bv),
                    traj_valid=np.ones(len(rows), dtype=bool),
                    delta_t=loaded["dt"][rows].astype(np.float32, copy=True),
                    sample_ids=[str(loaded["sid"][row]) for row in rows], m_max=self.m_max,
                    n_bins=N_BINS, epoch=self.epoch)
