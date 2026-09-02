"""Shared utilities: config loading, seeding, metric helpers."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML experiment config. All experiment knobs live here — never
    edit Python files to switch experiments."""
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if cfg is None:
        raise ValueError(f"Empty config file: {path}")
    return cfg


def save_config(cfg: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=True, allow_unicode=True)


def seed_everything(seed: int) -> None:
    """Fix all RNG sources used by training (cpu, cuda, numpy, python)."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mae(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mape(y_pred: np.ndarray, y_true: np.ndarray, eps: float = 1e-6) -> float:
    return float(np.mean(np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), eps)))


def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Classification accuracy (argmax on logits vs integer labels)."""
    return float(np.mean(np.argmax(y_pred, axis=-1) == y_true))


def dump_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False, default=str)
