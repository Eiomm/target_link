"""Canonical operations on Arrow table columns."""
import numpy as np
import pyarrow.compute as pc

def _array(column):
    return column.combine_chunks()


def _flat(column):
    return pc.list_flatten(_array(column))


def _offsets(column):
    values = _array(column).value_lengths().to_numpy(zero_copy_only=False)
    result = np.empty(len(values) + 1, dtype=np.int64)
    result[0] = 0
    np.cumsum(values, out=result[1:])
    return result


def _row_any(values: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Vectorized ``any`` for a ragged boolean array."""
    out = np.zeros(len(offsets) - 1, dtype=bool)
    if values.size:
        # Prefix sums also handles empty lists without an invalid reduceat.
        prefix = np.empty(values.size + 1, dtype=np.int64)
        prefix[0] = 0
        np.cumsum(values.astype(np.int64, copy=False), out=prefix[1:])
        out[:] = prefix[offsets[1:]] > prefix[offsets[:-1]]
    return out
