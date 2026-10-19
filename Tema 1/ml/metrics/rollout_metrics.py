"""Shared paper rollout metrics for PoliPhysics ML evaluation.

Error caps
----------
Position MSE (spider-relative or spider-absolute) is reported in **m²**.
Explosive / non-physical rollouts can produce arbitrarily large MSE; for table
aggregation each run's contribution is capped at:

* ``ERROR_CAP_REL_MSE = 100.0`` m²  (spider-relative coordinate MSE)
* ``ERROR_CAP_ABS_MSE = 100.0`` m²  (spider absolute position MSE)

Non-finite MSE is treated as capped (contributes the cap). 100 m² = 1e2 m²;
if working in mm², convert before comparing (1 m² = 1e6 mm²).

Physical edge tolerance
-----------------------
``PHYS_EDGE_TOL = 0.10`` means a segment is non-physical when
``|ℓ - ℓ0| / ℓ0 >= 10%``. Time-to-failure (TTF) is the first time (relative to
release) when the max edge relative deviation meets that threshold or the
state is non-finite. Never-failing rollouts have ``ttf_s = +inf``.

Per-run JSON schema
-------------------
Writers should set ``schema_version = 3`` and include ``ttf_s``, the
compact series fields documented in :mod:`ml.evaluate`, and ``z_0_series``
(spider absolute height on the same time grid as ``t_series``). Resume skips
files with ``schema_version < 3``.
"""
from __future__ import annotations

import math
import re
from typing import Any, Iterable, Sequence

import numpy as np

from ml.metrics.physics import wilson_interval

PHYS_EDGE_TOL = 0.10  # |Δℓ|/ℓ0 < 10% to remain physical
ERROR_CAP_REL_MSE = 100.0  # m²; spider-relative position MSE cap
ERROR_CAP_ABS_MSE = 100.0  # m²; spider absolute position MSE cap
SCHEMA_VERSION = 3

__all__ = [
    "PHYS_EDGE_TOL",
    "ERROR_CAP_REL_MSE",
    "ERROR_CAP_ABS_MSE",
    "SCHEMA_VERSION",
    "error_time_grid",
    "default_error_time_grid",
    "segment_rel_deviations",
    "time_to_failure",
    "is_physical_series",
    "is_physical_at_horizon",
    "cap_mse",
    "aggregate_horizon_errors",
    "aggregate_ttf",
    "physical_fraction_at_times",
    "aggregate_P_grid",
    "aggregate_error_vs_time",
    "delta_R_over_L_physical_only",
]


def error_time_grid(t_max: float = 60.0) -> list[float]:
    """Paper error-vs-time grid: 0.1 s steps to 10 s, then 1 s to floor(t_max).

    Returns ``[0.1, 0.2, ..., 10.0, 11.0, 12.0, ..., floor(t_max)]``.
    """
    fine = [i / 10.0 for i in range(1, 101)]  # 0.1 .. 10.0
    t_hi = int(math.floor(float(t_max)))
    coarse = [float(t) for t in range(11, t_hi + 1)]
    return fine + coarse


def default_error_time_grid(t_max: float = 60.0) -> list[float]:
    """Alias for :func:`error_time_grid` (YAML ``null`` → auto dense grid)."""
    return error_time_grid(t_max=t_max)


def segment_rel_deviations(
    X_frame: np.ndarray,
    edges: np.ndarray,
    L_seg: float,
) -> np.ndarray:
    """Per-edge relative length deviation ``|ℓ - L_seg| / L_seg`` for one frame.

    Parameters
    ----------
    X_frame :
        Absolute positions ``(n, 3)``.
    edges :
        Material edges ``(E, 2)`` integer endpoint indices.
    L_seg :
        Rest length ℓ0 (typically ``L / N_t``).

    Returns
    -------
    np.ndarray
        Shape ``(E,)``, non-negative relative deviations.
    """
    X = np.asarray(X_frame, dtype=np.float64)
    e = np.asarray(edges, dtype=np.int64)
    if e.ndim != 2 or e.shape[1] != 2:
        raise ValueError(f"edges must be (E, 2), got {e.shape}")
    i, j = e[:, 0], e[:, 1]
    lengths = np.linalg.norm(X[j] - X[i], axis=-1)
    denom = max(float(L_seg), 1e-12)
    return np.abs(lengths - float(L_seg)) / denom


def time_to_failure(
    t: np.ndarray | Sequence[float],
    max_edge_rel_series: np.ndarray | Sequence[float],
    finite_mask: np.ndarray | Sequence[bool],
) -> float:
    """First time where ``max_edge_rel >= PHYS_EDGE_TOL`` or state is non-finite.

    If the rollout never fails, returns ``+inf``.
    """
    tt = np.asarray(t, dtype=np.float64)
    mer = np.asarray(max_edge_rel_series, dtype=np.float64)
    fin = np.asarray(finite_mask, dtype=bool)
    if tt.shape != mer.shape or tt.shape != fin.shape:
        raise ValueError(
            f"t, max_edge_rel_series, finite_mask length mismatch: "
            f"{tt.shape}, {mer.shape}, {fin.shape}"
        )
    if tt.size == 0:
        return float("inf")
    fail = (mer >= PHYS_EDGE_TOL) | (~fin)
    idxs = np.flatnonzero(fail)
    if idxs.size == 0:
        return float("inf")
    return float(tt[int(idxs[0])])


def is_physical_series(
    max_edge_rel_series: np.ndarray | Sequence[float],
    finite_mask: np.ndarray | Sequence[bool] | None = None,
) -> np.ndarray:
    """Boolean mask: physical at each step (``max_edge_rel < tol`` and finite)."""
    mer = np.asarray(max_edge_rel_series, dtype=np.float64)
    phys = mer < PHYS_EDGE_TOL
    if finite_mask is None:
        phys = phys & np.isfinite(mer)
    else:
        phys = phys & np.asarray(finite_mask, dtype=bool)
    return phys


def is_physical_at_horizon(ttf_s: float, horizon_s: float) -> bool:
    """True if failure has not occurred by ``horizon_s`` (``ttf_s > horizon_s``)."""
    if ttf_s is None:
        return False
    ttf = float(ttf_s)
    if not np.isfinite(ttf):
        # +inf → never failed; NaN → unknown / not physical
        return bool(math.isinf(ttf) and ttf > 0)
    return ttf > float(horizon_s)


def cap_mse(
    value: float,
    cap: float = ERROR_CAP_REL_MSE,
) -> tuple[float, bool]:
    """Cap a MSE value; non-finite → ``(cap, True)``.

    Returns ``(capped_value, was_capped)``.
    """
    cap_f = float(cap)
    try:
        v = float(value)
    except (TypeError, ValueError):
        return cap_f, True
    if not np.isfinite(v) or v > cap_f:
        return cap_f, True
    return v, False


def _horizon_key(hs: float) -> str:
    return f"{float(hs):g}s"


def aggregate_horizon_errors(
    per_run_list: Sequence[dict[str, Any]],
    horizons: Sequence[float],
) -> dict[str, dict[str, float | int]]:
    """Mean capped horizon errors over **all** runs; physical fraction separate.

    Non-finite MSE → capped. Physical fraction uses :func:`is_physical_at_horizon`
    on each run's ``ttf_s`` (does not gate which errors enter the mean).
    """
    n = len(per_run_list)
    out: dict[str, dict[str, float | int]] = {}
    for hs in horizons:
        key = _horizon_key(hs)
        rel_vals: list[float] = []
        abs_vals: list[float] = []
        n_capped_rel = 0
        n_capped_abs = 0
        n_phys = 0
        for run in per_run_list:
            he = (run.get("horizon_errors") or {}).get(key, {})
            rel_c, capped_r = cap_mse(he.get("rel_mse", float("nan")), ERROR_CAP_REL_MSE)
            abs_c, capped_a = cap_mse(
                he.get("spider_abs_mse", float("nan")), ERROR_CAP_ABS_MSE
            )
            rel_vals.append(rel_c)
            abs_vals.append(abs_c)
            n_capped_rel += int(capped_r)
            n_capped_abs += int(capped_a)
            if is_physical_at_horizon(run.get("ttf_s", float("nan")), hs):
                n_phys += 1
        out[key] = {
            "rel_mse_mean": float(np.mean(rel_vals)) if rel_vals else float("nan"),
            "spider_abs_mse_mean": float(np.mean(abs_vals)) if abs_vals else float("nan"),
            "n_capped_rel": n_capped_rel,
            "n_capped_abs": n_capped_abs,
            "physical_fraction": n_phys / max(n, 1),
            "n_runs": n,
        }
    return out


def _quantile_sorted(sorted_a: np.ndarray, q: float) -> float:
    """Linear quantile on a pre-sorted 1-d array; safe with ``+inf``."""
    n = int(sorted_a.size)
    if n == 0:
        return float("nan")
    if n == 1:
        return float(sorted_a[0])
    idx = (q / 100.0) * (n - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(sorted_a[lo])
    w = idx - lo
    a, b = float(sorted_a[lo]), float(sorted_a[hi])
    if math.isinf(a) and math.isinf(b):
        return float("inf")
    if math.isinf(a):
        return float("inf")
    if math.isinf(b):
        return float("inf") if w > 0 else a
    return (1.0 - w) * a + w * b


def aggregate_ttf(per_run: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    """Median / IQR of ``ttf_s`` over runs (``+inf`` kept for never-fail)."""
    ttfs = np.asarray(
        [float(r.get("ttf_s", float("nan"))) for r in per_run],
        dtype=np.float64,
    )
    n = int(ttfs.size)
    if n == 0:
        return {
            "median": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
            "n_runs": 0,
            "n_failed": 0,
            "n_never_fail": 0,
        }
    # Drop NaN ttf; keep finite failures and +inf never-fail.
    valid = ttfs[np.isfinite(ttfs) | np.isposinf(ttfs)]
    n_failed = int(np.sum(np.isfinite(ttfs)))
    n_never = int(np.sum(np.isposinf(ttfs)))
    if valid.size == 0:
        med = q25 = q75 = float("nan")
    else:
        ordered = np.sort(valid)
        med = _quantile_sorted(ordered, 50.0)
        q25 = _quantile_sorted(ordered, 25.0)
        q75 = _quantile_sorted(ordered, 75.0)
    return {
        "median": med,
        "q25": q25,
        "q75": q75,
        "n_runs": n,
        "n_failed": n_failed,
        "n_never_fail": n_never,
    }


def physical_fraction_at_times(
    per_run: Sequence[dict[str, Any]],
    times: Sequence[float] = (1.0, 5.0, 10.0),
) -> dict[str, float]:
    """Fraction of runs still physical at each report time."""
    n = len(per_run)
    out: dict[str, float] = {}
    for t in times:
        n_phys = sum(
            1 for r in per_run if is_physical_at_horizon(r.get("ttf_s", float("nan")), t)
        )
        out[_horizon_key(t)] = n_phys / max(n, 1)
    return out


_RUN_X_RE = re.compile(r"_x([+-]?\d+)_i\d+")


def parse_x_from_run_id(run_id: str) -> int:
    """Grid coordinate encoded in ``N1_sw0.15_x+4_i0099``."""
    m = _RUN_X_RE.search(str(run_id))
    if not m:
        raise ValueError(f"run_id has no _x<int>_i field: {run_id}")
    return int(m.group(1))


def grid_x(run: dict[str, Any]) -> int:
    """Ballooning grid x from the run id, checked against ``x_grid``.

    ``run["q"]`` is the thread charge in coulombs (~1e-9). Rounding it to 6
    decimals yields 0 and pools every x together. The grid coordinate is the
    ``x`` token in the run id, which preprocess copies from the sweep / HDF5
    metadata into ``globals["x_grid"]`` (HDF5 ``params`` has no attribute named
    ``q`` for this coordinate; charge is ``Q_t``).
    """
    rid = run.get("run_id")
    if not rid:
        raise KeyError("run_id")
    x = parse_x_from_run_id(str(rid))
    if run.get("x_grid") is not None:
        x_attr = int(run["x_grid"])
        if x_attr != x:
            raise ValueError(
                f"run_id x={x} disagrees with x_grid={x_attr} for {rid}"
            )
    return x


def aggregate_P_grid(
    per_run: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rows for ``tab_P_grid.csv`` with Wilson intervals on sim and surrogate.

    A grid point is ``(N, x, sigma_w)``. ``x`` is parsed from ``run_id`` and
    cross-checked against ``x_grid`` when that field is present. Surrogate
    success is ``status == rise``; simulator success is ``outcome_sim`` /
    ``status_sim`` / ``outcome`` equal to ``rise``.
    """
    cells: dict[tuple[int, int, float], dict[str, int]] = {}
    for r in per_run:
        try:
            gkey = (
                int(r["N"]),
                grid_x(r),
                round(float(r["sigma_w"]), 4),
            )
        except (KeyError, TypeError):
            continue
        status_surr = str(r.get("status_surr", r.get("status", "")))
        status_sim = str(
            r.get("outcome_sim", r.get("status_sim", r.get("gt_status", r.get("outcome", ""))))
        )
        cell = cells.setdefault(gkey, {"n": 0, "sim_rise": 0, "surr_rise": 0})
        cell["n"] += 1
        cell["sim_rise"] += 1 if status_sim == "rise" else 0
        cell["surr_rise"] += 1 if status_surr == "rise" else 0

    rows: list[dict[str, Any]] = []
    for gkey, cell in sorted(cells.items()):
        n_cell = int(cell["n"])
        ks, km = int(cell["sim_rise"]), int(cell["surr_rise"])
        lo_s, hi_s = wilson_interval(ks, n_cell)
        lo_m, hi_m = wilson_interval(km, n_cell)
        rows.append({
            "N": gkey[0],
            "x": gkey[1],
            "sigma_w": gkey[2],
            "n_runs": n_cell,
            "P_sim": ks / max(n_cell, 1),
            "P_surr": km / max(n_cell, 1),
            "P_sim_wilson_lo": lo_s,
            "P_sim_wilson_hi": hi_s,
            "P_surr_wilson_lo": lo_m,
            "P_surr_wilson_hi": hi_m,
        })
    return rows


def aggregate_error_vs_time(
    per_run: Sequence[dict[str, Any]],
    grid: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    """Mean capped error-vs-time over all runs for each grid label."""
    labels: list[str]
    if grid is not None:
        labels = [f"{float(t):g}" for t in grid]
    else:
        keys: set[str] = set()
        for r in per_run:
            keys.update((r.get("error_vs_time") or {}).keys())
        labels = sorted(keys, key=lambda s: float(s))

    rows: list[dict[str, Any]] = []
    n = len(per_run)
    for lbl in labels:
        rel_vals: list[float] = []
        abs_vals: list[float] = []
        n_capped_rel = 0
        n_capped_abs = 0
        for r in per_run:
            err = (r.get("error_vs_time") or {}).get(lbl, {})
            rel_c, cr = cap_mse(err.get("rel_mse", float("nan")), ERROR_CAP_REL_MSE)
            abs_c, ca = cap_mse(
                err.get("spider_abs_mse", float("nan")), ERROR_CAP_ABS_MSE
            )
            rel_vals.append(rel_c)
            abs_vals.append(abs_c)
            n_capped_rel += int(cr)
            n_capped_abs += int(ca)
        rows.append({
            "t_s": float(lbl),
            "rel_mse_mean": float(np.mean(rel_vals)) if rel_vals else float("nan"),
            "spider_abs_mse_mean": float(np.mean(abs_vals)) if abs_vals else float("nan"),
            "n_runs": n,
            "n_capped_rel": n_capped_rel,
            "n_capped_abs": n_capped_abs,
        })
    return rows


def delta_R_over_L_physical_only(
    per_run: Sequence[dict[str, Any]],
    gt: Sequence[Any],
) -> dict[str, float | int]:
    """Mean |Δ(R/L)| / |R/L|_gt using physical-portion means only.

    ``gt`` may be a sequence of floats (GT ``R_over_L_physical``) or dicts with
    key ``R_over_L_physical``. Only pairs with finite, non-zero GT contribute.
    """
    if len(per_run) != len(gt):
        raise ValueError(
            f"per_run and gt length mismatch: {len(per_run)} vs {len(gt)}"
        )
    deltas: list[float] = []
    for r, g in zip(per_run, gt):
        rp = float(r.get("R_over_L_physical", float("nan")))
        if isinstance(g, dict):
            gp = float(g.get("R_over_L_physical", float("nan")))
        else:
            gp = float(g)
        if np.isfinite(rp) and np.isfinite(gp) and abs(gp) > 1e-12:
            deltas.append(abs(rp - gp) / abs(gp))
    return {
        "delta_R_over_L_mean": float(np.mean(deltas)) if deltas else float("nan"),
        "n_runs_used": len(deltas),
    }
