"""Training noise scale from one-step velocity change statistics (GNS-style)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ml.data.frames import DT

# node_type: 0=spider, 1=interior, 2=tip
TYPE_NAMES = ("spider", "interior", "tip")


def dv_std_from_a_std(a_std: Sequence[float] | np.ndarray, dt: float = DT) -> float:
    """Scalar RMS std of dv = a*dt from per-component acceleration std."""
    a = np.asarray(a_std, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean((a * dt) ** 2)))


def compute_dv_std_by_type(
    store,
    run_ids: Sequence[str],
    *,
    dt: float = DT,
    max_frames_per_run: int = 200,
) -> dict[str, Any]:
    """Train-only std of per-component dv=a*dt, stratified by node type.

    Returns overall RMS scalar plus per-type RMS scalars and component stds.
    """
    # Welford per type for 3 components
    counts = {t: 0 for t in range(3)}
    means = {t: np.zeros(3, dtype=np.float64) for t in range(3)}
    M2 = {t: np.zeros(3, dtype=np.float64) for t in range(3)}

    for rid in run_ids:
        rec = store.get(rid)
        v = rec["v"]
        nt = rec["node_type"].astype(np.int64)
        T = int(v.shape[0])
        # sample evenly across the run
        idxs = np.linspace(0, max(T - 2, 0), num=min(max_frames_per_run, max(T - 1, 1)), dtype=int)
        for t in idxs:
            if t + 1 >= T:
                continue
            dv = (v[t + 1] - v[t]).astype(np.float64)  # already a*dt
            for typ in (0, 1, 2):
                mask = nt == typ
                if not np.any(mask):
                    continue
                chunk = dv[mask].reshape(-1, 3)
                for row in chunk:
                    counts[typ] += 1
                    delta = row - means[typ]
                    means[typ] += delta / counts[typ]
                    M2[typ] += delta * (row - means[typ])

    per_type: dict[str, Any] = {}
    rms_by_type: dict[str, float] = {}
    for typ, name in enumerate(TYPE_NAMES):
        n = counts[typ]
        if n < 2:
            std = np.ones(3, dtype=np.float64) * 1e-8
        else:
            std = np.sqrt(M2[typ] / (n - 1))
        rms = float(np.sqrt(np.mean(std ** 2)))
        per_type[name] = {
            "mean": means[typ].tolist(),
            "std": std.tolist(),
            "rms": rms,
            "count": int(n),
        }
        rms_by_type[name] = rms

    overall_vals = [rms_by_type[n] for n in TYPE_NAMES if counts[TYPE_NAMES.index(n)] > 0]
    overall = float(np.mean(overall_vals)) if overall_vals else 1e-8
    ratios = []
    active = [rms_by_type[n] for n in TYPE_NAMES if rms_by_type[n] > 0]
    if len(active) >= 2:
        ratios.append(max(active) / max(min(active), 1e-12))
    use_per_type = bool(ratios and ratios[0] > 3.0)

    return {
        "dt": float(dt),
        "overall_rms": overall,
        "by_type": per_type,
        "use_per_type": use_per_type,
        "ratio_max_min": float(ratios[0]) if ratios else 1.0,
    }


def resolve_noise(
    *,
    noise_k: float | None,
    noise_std: float | None,
    noise_stats: dict[str, Any] | None,
    a_std: Sequence[float] | None,
    dt: float = DT,
) -> tuple[float, np.ndarray | None]:
    """Return (scalar_noise_std, per_type_std_array[3] or None).

    If ``noise_k`` is set → ``k * std(dv)`` (from noise_stats, else RMS of a_std·dt).
    Else if ``noise_std`` is set → that scalar (legacy / explicit override).
    Else → no noise.
    """
    if noise_k is not None:
        k = float(noise_k)
        if noise_stats is not None:
            overall = float(noise_stats["overall_rms"])
            if noise_stats.get("use_per_type"):
                arr = np.array(
                    [float(noise_stats["by_type"][n]["rms"]) * k for n in TYPE_NAMES],
                    dtype=np.float32,
                )
                return float(overall * k), arr
            return float(overall * k), None
        if a_std is not None:
            return float(k * dv_std_from_a_std(a_std, dt)), None
        return 0.0, None
    if noise_std is not None:
        return float(noise_std), None
    return 0.0, None


def load_noise_stats(path: Path) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def save_noise_stats(stats: dict[str, Any], path: Path) -> None:
    Path(path).write_text(json.dumps(stats, indent=2))
