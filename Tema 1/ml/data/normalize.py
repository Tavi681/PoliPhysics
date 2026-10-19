"""Normalization statistics (train split only)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

TYPE_NAMES = ("spider", "interior", "tip")


def running_mean_var():
    """Welford online mean/variance; update() accepts a batch [N, D]."""
    n = 0
    mean = None
    m2 = None

    def update(x: np.ndarray):
        nonlocal n, mean, m2
        x = np.asarray(x, dtype=np.float64).reshape(-1, x.shape[-1])
        if x.size == 0:
            return
        if mean is None:
            mean = np.zeros(x.shape[-1], dtype=np.float64)
            m2 = np.zeros(x.shape[-1], dtype=np.float64)
        # Parallel Welford merge of the batch into the running stats
        batch_n = x.shape[0]
        batch_mean = x.mean(axis=0)
        batch_m2 = ((x - batch_mean) ** 2).sum(axis=0)
        if n == 0:
            mean[:] = batch_mean
            m2[:] = batch_m2
            n = batch_n
            return
        delta = batch_mean - mean
        total = n + batch_n
        mean = (n * mean + batch_n * batch_mean) / total
        m2 = m2 + batch_m2 + delta ** 2 * n * batch_n / total
        n = total

    def finalize():
        var = m2 / max(n - 1, 1)
        return {
            "mean": mean.astype(np.float32),
            "std": np.sqrt(np.maximum(var, 1e-12)).astype(np.float32),
            "count": n,
        }

    return update, finalize


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def save_norm_stats(stats: dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_to_jsonable(stats), f, indent=2)


def _load_mean_std_block(v: dict) -> dict[str, Any]:
    return {
        "mean": np.asarray(v["mean"], dtype=np.float32),
        "std": np.asarray(v["std"], dtype=np.float32),
        "count": int(v.get("count", 0)),
    }


def load_norm_stats(path: Path) -> dict[str, Any]:
    with open(path) as f:
        raw = json.load(f)
    out: dict[str, Any] = {}
    for k, v in raw.items():
        if k == "a_by_type" and isinstance(v, dict):
            by = {}
            for name, entry in v.items():
                if isinstance(entry, dict) and "mean" in entry and "std" in entry:
                    by[name] = _load_mean_std_block(entry)
                else:
                    by[name] = entry
            out[k] = by
        elif isinstance(v, dict) and "mean" in v and "std" in v:
            out[k] = _load_mean_std_block(v)
        else:
            out[k] = v
    return out


def normalize(x: torch.Tensor | np.ndarray, mean, std):
    if isinstance(x, np.ndarray):
        return ((x - mean) / std).astype(np.float32)
    mean_t = torch.as_tensor(mean, device=x.device, dtype=x.dtype)
    std_t = torch.as_tensor(std, device=x.device, dtype=x.dtype)
    return (x - mean_t) / std_t


def denormalize(x: torch.Tensor | np.ndarray, mean, std):
    if isinstance(x, np.ndarray):
        return (x * std + mean).astype(np.float32)
    mean_t = torch.as_tensor(mean, device=x.device, dtype=x.dtype)
    std_t = torch.as_tensor(std, device=x.device, dtype=x.dtype)
    return x * std_t + mean_t
