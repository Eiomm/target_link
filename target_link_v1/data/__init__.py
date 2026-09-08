"""Data utilities; legacy Pandas loaders are imported only when requested."""
from importlib import import_module

__all__ = ["ETAData", "GroupIndex", "build_eta_data", "build_group_index",
           "group_aligned_chunks", "sort_by_group"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    module = "eta_data" if name in ("ETAData", "build_eta_data") else "groups"
    value = getattr(import_module("target_link_v1.data." + module), name)
    globals()[name] = value
    return value
