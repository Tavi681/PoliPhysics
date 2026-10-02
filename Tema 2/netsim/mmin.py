"""Algorithm 2: minimum-mass net by bisection on the cross-section scale.

All thread cross-sections are scaled uniformly, ``A_e = s * A_hat_e``, so the
net mass is linear in ``s``: ``m_net(s) = s * m_net(1)``. For a given impact
point the acceptance criterion is a function of ``s``:

* criterion A: the drone is *arrested* and *no* thread fails;
* criterion B: the drone is arrested, every failed segment lies within
  ``R_max`` of the impact point, and fewer than ``k_max`` segments fail.

For each impact point we find the smallest passing ``s`` by first bracketing it
(doubling ``s`` from the analytical guess ``s0`` with ``m_net = E_kin/e_mat``
until the criterion passes) and then bisecting to a relative tolerance ``tol``.
The minimum mass of the whole net is the worst (largest) of the per-point
minima, and the corresponding impact point is the worst case.

Per-point work is parallelised with ``multiprocessing`` and made resumable: each
single evaluation ``(point, s)`` is cached as a tiny HDF5 file and skipped if it
already exists.

Important: criterion B is not monotone in ``s`` in general, so
:func:`monotonicity_scan` should be run first to check the pass/fail pattern.
No physics or tolerances are tuned here.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field, replace
from typing import List, Optional, Tuple

import numpy as np

from .config import SimConfig
from .simulate import build_net, simulate_config

logger = logging.getLogger("netsim.mmin")

__all__ = ["MminConfig", "MminResult", "analytical_s0", "evaluate",
           "minimum_mass_point", "minimum_mass", "monotonicity_scan"]


@dataclass
class MminConfig:
    criterion: str = "A"              # "A" or "B"
    tol: float = 0.05                 # relative bisection tolerance on s
    impact_points: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0.0, 0.0)])
    max_doublings: int = 24
    max_bisect: int = 40
    s0_scale: float = 1.0             # safety factor on the analytical s0
    n_procs: int = 1
    cache_dir: Optional[str] = None   # for resumable per-evaluation HDF5 files

    def validate(self) -> None:
        if self.criterion not in ("A", "B"):
            raise ValueError(f"unknown criterion {self.criterion!r}")
        if self.tol <= 0:
            raise ValueError("tol must be positive")
        if not self.impact_points:
            raise ValueError("impact_points must be non-empty")


@dataclass
class MminResult:
    criterion: str
    s_min: float
    m_min: float
    worst_point: Tuple[float, float]
    n_failures: int
    broken_segments: List[int]
    broken_midpoints: list
    s0: float
    per_point: dict                   # point -> s_min
    evaluations: int


def analytical_s0(cfg: SimConfig) -> float:
    """Analytical scale guess so that ``m_net(s0) = E_kin / e_mat``."""
    material = cfg.material.resolve()
    net = build_net(cfg)
    E_kin = 0.5 * cfg.drone.M * cfg.drone.v0 ** 2
    e_mat = material.e_mat
    m1 = net.net_mass(material.rho, 1.0)
    if m1 <= 0 or e_mat <= 0:
        return 1.0
    return (E_kin / e_mat) / m1


def _criterion_pass(res, p, criterion, R_max, k_max) -> bool:
    if not res.arrested:
        return False
    if criterion == "A":
        return res.n_failures == 0
    # criterion B
    if res.n_failures >= k_max:
        return False
    mids = res.trajectory.failure_midpoints
    if mids.size:
        d = np.sqrt((mids[:, 0] - p[0]) ** 2 + (mids[:, 1] - p[1]) ** 2)
        if np.any(d > R_max):
            return False
    return True


def _cache_path(cache_dir, point_idx, s):
    return os.path.join(cache_dir, f"mmin_p{point_idx}_s{s:.6f}.h5")


def _read_cache(path):
    try:
        import h5py
        with h5py.File(path, "r") as h5:
            a = h5["eval"].attrs
            return {
                "passed": bool(a["passed"]),
                "arrested": bool(a["arrested"]),
                "n_failures": int(a["n_failures"]),
                "fail_segs": list(np.asarray(h5["fail_segs"])) if "fail_segs"
                in h5 else [],
                "fail_mids": np.asarray(h5["fail_mids"]) if "fail_mids" in h5
                else np.zeros((0, 3)),
            }
    except Exception:
        return None


def _write_cache(path, info):
    try:
        import h5py
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with h5py.File(path, "w") as h5:
            g = h5.create_group("eval")
            g.attrs["passed"] = bool(info["passed"])
            g.attrs["arrested"] = bool(info["arrested"])
            g.attrs["n_failures"] = int(info["n_failures"])
            h5.create_dataset("fail_segs",
                              data=np.asarray(info["fail_segs"], dtype=np.int64))
            h5.create_dataset("fail_mids",
                              data=np.asarray(info["fail_mids"], dtype=float))
    except Exception as exc:  # pragma: no cover
        logger.warning("could not write mmin cache %s: %s", path, exc)


def evaluate(cfg: SimConfig, s: float, point, mmincfg: MminConfig,
             point_idx: int = 0) -> dict:
    """Evaluate the criterion for one (impact point, scale) pair, with cache."""
    cache_dir = mmincfg.cache_dir
    if cache_dir is not None:
        path = _cache_path(cache_dir, point_idx, s)
        if os.path.exists(path):
            cached = _read_cache(path)
            if cached is not None:
                return cached

    cfg2 = replace(cfg)
    cfg2.numerics = replace(cfg.numerics, area_scale=float(s))
    cfg2.drone = replace(cfg.drone, p=(float(point[0]), float(point[1])))
    cfg2.output = replace(cfg.output, hdf5=None)
    res = simulate_config(cfg2, write=False)

    passed = _criterion_pass(res, point, mmincfg.criterion,
                             cfg.output.R_max, cfg.output.k_max)
    mids = res.trajectory.failure_midpoints
    segs = list(np.asarray(res.trajectory.failures[:, 0], dtype=np.int64)) \
        if res.trajectory.failures.size else []
    info = {"passed": bool(passed), "arrested": bool(res.arrested),
            "n_failures": int(res.n_failures), "fail_segs": segs,
            "fail_mids": np.asarray(mids, dtype=float)}

    if cache_dir is not None:
        _write_cache(_cache_path(cache_dir, point_idx, s), info)
    return info


def minimum_mass_point(args) -> dict:
    """Find the minimum passing scale for a single impact point (top-level so it
    is picklable for multiprocessing)."""
    cfg, point, mmincfg, point_idx, s0 = args
    n_eval = 0

    # Bracket: double s from s0 until the criterion passes.
    s_hi = s0
    hi_info = evaluate(cfg, s_hi, point, mmincfg, point_idx)
    n_eval += 1
    doublings = 0
    while not hi_info["passed"]:
        s_hi *= 2.0
        hi_info = evaluate(cfg, s_hi, point, mmincfg, point_idx)
        n_eval += 1
        doublings += 1
        if doublings > mmincfg.max_doublings:
            logger.warning("point %s: no passing s within %d doublings",
                           point, mmincfg.max_doublings)
            break

    # Lower bracket: last failing s (half of s_hi), or a small floor.
    s_lo = s_hi / 2.0 if doublings > 0 or s_hi > s0 else s_hi * 1e-3
    lo_info = evaluate(cfg, s_lo, point, mmincfg, point_idx)
    n_eval += 1
    if lo_info["passed"]:
        # Already passing at the floor; walk down to bracket.
        while lo_info["passed"] and s_lo > 1e-9:
            s_hi = s_lo
            hi_info = lo_info
            s_lo /= 2.0
            lo_info = evaluate(cfg, s_lo, point, mmincfg, point_idx)
            n_eval += 1

    # Bisection on [s_lo (fail), s_hi (pass)].
    it = 0
    while (s_hi - s_lo) / s_hi > mmincfg.tol and it < mmincfg.max_bisect:
        s_mid = 0.5 * (s_lo + s_hi)
        mid_info = evaluate(cfg, s_mid, point, mmincfg, point_idx)
        n_eval += 1
        if mid_info["passed"]:
            s_hi, hi_info = s_mid, mid_info
        else:
            s_lo = s_mid
        it += 1

    return {"point": tuple(point), "point_idx": point_idx, "s_min": s_hi,
            "info": hi_info, "evaluations": n_eval}


def minimum_mass(cfg: SimConfig, mmincfg: MminConfig) -> MminResult:
    """Run Algorithm 2 over all impact points; return the worst-case minimum."""
    mmincfg.validate()
    material = cfg.material.resolve()
    net = build_net(cfg)
    m1 = net.net_mass(material.rho, 1.0)
    s0 = analytical_s0(cfg) * mmincfg.s0_scale

    tasks = [(cfg, p, mmincfg, i, s0)
             for i, p in enumerate(mmincfg.impact_points)]

    if mmincfg.n_procs > 1 and len(tasks) > 1:
        import multiprocessing as mp
        with mp.Pool(mmincfg.n_procs) as pool:
            results = pool.map(minimum_mass_point, tasks)
    else:
        results = [minimum_mass_point(t) for t in tasks]

    per_point = {r["point"]: r["s_min"] for r in results}
    worst = max(results, key=lambda r: r["s_min"])
    s_min = worst["s_min"]
    evals = sum(r["evaluations"] for r in results)
    info = worst["info"]
    mids = info["fail_mids"]

    return MminResult(
        criterion=mmincfg.criterion,
        s_min=s_min,
        m_min=s_min * m1,
        worst_point=worst["point"],
        n_failures=info["n_failures"],
        broken_segments=[int(x) for x in info["fail_segs"]],
        broken_midpoints=mids.tolist() if hasattr(mids, "tolist") else list(mids),
        s0=s0,
        per_point=per_point,
        evaluations=evals,
    )


def monotonicity_scan(cfg: SimConfig, mmincfg: MminConfig, point, *,
                      n: int = 10, s_hi: Optional[float] = None) -> dict:
    """Coarse scan of the pass/fail pattern over s for one impact point.

    Scans ``n`` values in ``[0.25 s_hi, s_hi]`` (``s_hi`` defaults to a bracket
    found by doubling). Returns the pattern and whether it is monotone (once the
    criterion passes it stays passing for larger s).
    """
    if s_hi is None:
        s0 = analytical_s0(cfg) * mmincfg.s0_scale
        res = minimum_mass_point((cfg, point, mmincfg, 0, s0))
        s_hi = 2.0 * res["s_min"]

    s_values = np.linspace(0.25 * s_hi, s_hi, n)
    pattern = []
    for s in s_values:
        info = evaluate(cfg, float(s), point, mmincfg, 0)
        pattern.append(bool(info["passed"]))

    # Monotone if, after the first True, there is no False.
    monotone = True
    seen_true = False
    for ok in pattern:
        if ok:
            seen_true = True
        elif seen_true:
            monotone = False
            break

    return {"point": tuple(point), "s_values": s_values.tolist(),
            "pattern": pattern, "monotone": monotone, "s_hi": s_hi}
