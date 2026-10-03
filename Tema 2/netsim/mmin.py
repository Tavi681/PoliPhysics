"""Algorithm 2: minimum-mass net by bisection on the cross-section scale.

All thread cross-sections are scaled uniformly, ``A_e = s * A_hat_e``, so the
net mass is linear in ``s``: ``m_net(s) = s * m_net(1)``. For a given impact
point the acceptance criterion is a function of ``s``:

* criterion A: the drone is *arrested* and *no* thread fails;
* criterion B: the drone is arrested and failures are localized:
  ``n_failed == 0`` or ``(n_failed < k_max and R_d <= R_max)``.

Any run that passes A therefore passes B (zero failures is trivially localized).

For criterion A (monotone), each impact point is bracketed by doubling from the
analytical guess ``s0`` and then bisected. For criterion B (not monotone in
general) Algorithm 2 is a coarse scan over ``s`` (≥ 20 values) followed by a
local bisection around the smallest passing value; both
``s_min_all_pass`` (smallest s above which every scanned value passes) and
``s_min_first_pass`` (smallest scanned passing s) are reported.

Per-point work is parallelised with ``multiprocessing`` and made resumable: each
single evaluation ``(point, s)`` is cached as a tiny HDF5 file and skipped if it
already exists.

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
           "minimum_mass_point", "minimum_mass", "monotonicity_scan",
           "_criterion_pass", "_write_cache", "_read_cache"]


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
    # Criterion-B scan: number of s values in [s_lo, s_hi] before local bisection.
    n_scan: int = 24

    def validate(self) -> None:
        if self.criterion not in ("A", "B"):
            raise ValueError(f"unknown criterion {self.criterion!r}")
        if self.tol <= 0:
            raise ValueError("tol must be positive")
        if not self.impact_points:
            raise ValueError("impact_points must be non-empty")
        if self.n_scan < 5:
            raise ValueError("n_scan must be >= 5")


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
    # Criterion B only (optional diagnostics of non-monotonicity).
    s_min_first_pass: Optional[float] = None
    s_min_all_pass: Optional[float] = None
    scan_pattern: Optional[list] = None


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


def _fail_reason(res, p, criterion, R_max, k_max) -> str:
    """Human-readable reason a run fails the criterion (empty if it passes)."""
    if not res.arrested:
        outcome = getattr(res, "outcome", "unknown")
        return f"not_arrested({outcome})"
    if criterion == "A":
        if res.n_failures != 0:
            return f"n_failed={res.n_failures}"
        return ""
    # B: arrested and (n_failed == 0 or (n_failed < k_max and R_d <= R_max))
    if res.n_failures == 0:
        return ""
    if res.n_failures >= k_max:
        return f"n_failed={res.n_failures}>={k_max}"
    R_d = getattr(res, "R_d", None)
    if R_d is None:
        # Fall back to midpoints (unit tests / older caches).
        mids = res.trajectory.failure_midpoints
        if mids.size:
            d = np.sqrt((mids[:, 0] - p[0]) ** 2 + (mids[:, 1] - p[1]) ** 2)
            R_d = float(np.max(d))
        else:
            R_d = 0.0
    if R_d > R_max:
        return f"R_d={R_d:.4g}>{R_max}"
    return ""


def _criterion_pass(res, p, criterion, R_max, k_max) -> bool:
    """Acceptance: A = arrested & no failure; B = arrested & localized."""
    return _fail_reason(res, p, criterion, R_max, k_max) == ""


def _cache_path(cache_dir, point_idx, s, criterion="A"):
    return os.path.join(cache_dir,
                        f"mmin_p{point_idx}_c{criterion}_s{s:.6f}.h5")


def _read_cache(path):
    try:
        import h5py
        with h5py.File(path, "r") as h5:
            a = h5["eval"].attrs
            return {
                "passed": bool(a["passed"]),
                "arrested": bool(a["arrested"]),
                "n_failures": int(a["n_failures"]),
                "R_d": float(a["R_d"]) if "R_d" in a else float("nan"),
                "outcome": str(a["outcome"]) if "outcome" in a else "",
                "energy_error": (float(a["energy_error"])
                                 if "energy_error" in a else float("nan")),
                "fail_reason": (str(a["fail_reason"])
                                if "fail_reason" in a else ""),
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
            g.attrs["R_d"] = float(info.get("R_d", float("nan")))
            g.attrs["outcome"] = str(info.get("outcome", ""))
            g.attrs["energy_error"] = float(info.get("energy_error",
                                                     float("nan")))
            g.attrs["fail_reason"] = str(info.get("fail_reason", ""))
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
        path = _cache_path(cache_dir, point_idx, s, mmincfg.criterion)
        if os.path.exists(path):
            cached = _read_cache(path)
            if cached is not None:
                return cached

    cfg2 = replace(cfg)
    cfg2.numerics = replace(cfg.numerics, area_scale=float(s))
    cfg2.drone = replace(cfg.drone, p=(float(point[0]), float(point[1])))
    cfg2.output = replace(cfg.output, hdf5=None)
    res = simulate_config(cfg2, write=False)

    reason = _fail_reason(res, point, mmincfg.criterion,
                          cfg.output.R_max, cfg.output.k_max)
    passed = reason == ""
    mids = res.trajectory.failure_midpoints
    segs = list(np.asarray(res.trajectory.failures[:, 0], dtype=np.int64)) \
        if res.trajectory.failures.size else []
    info = {
        "passed": bool(passed),
        "arrested": bool(res.arrested),
        "n_failures": int(res.n_failures),
        "R_d": float(res.R_d),
        "outcome": str(res.outcome),
        "energy_error": float(res.energy_error),
        "fail_reason": reason,
        "fail_segs": segs,
        "fail_mids": np.asarray(mids, dtype=float),
    }

    if cache_dir is not None:
        _write_cache(_cache_path(cache_dir, point_idx, s, mmincfg.criterion),
                     info)
    return info


def _bisect_bracket(cfg, point, mmincfg, point_idx, s_lo, s_hi, hi_info):
    """Bisection on [s_lo (fail), s_hi (pass)]; returns (s_hi, hi_info, n_eval)."""
    n_eval = 0
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
    return s_hi, hi_info, n_eval


def _minimum_mass_point_A(cfg, point, mmincfg, point_idx, s0) -> dict:
    """Monotone bisection (criterion A)."""
    n_eval = 0
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

    s_lo = s_hi / 2.0 if doublings > 0 or s_hi > s0 else s_hi * 1e-3
    lo_info = evaluate(cfg, s_lo, point, mmincfg, point_idx)
    n_eval += 1
    if lo_info["passed"]:
        while lo_info["passed"] and s_lo > 1e-9:
            s_hi = s_lo
            hi_info = lo_info
            s_lo /= 2.0
            lo_info = evaluate(cfg, s_lo, point, mmincfg, point_idx)
            n_eval += 1

    s_hi, hi_info, n_bis = _bisect_bracket(
        cfg, point, mmincfg, point_idx, s_lo, s_hi, hi_info)
    n_eval += n_bis
    return {"point": tuple(point), "point_idx": point_idx, "s_min": s_hi,
            "info": hi_info, "evaluations": n_eval,
            "s_min_first_pass": s_hi, "s_min_all_pass": s_hi,
            "scan_pattern": None}


def _minimum_mass_point_B(cfg, point, mmincfg, point_idx, s0) -> dict:
    """Non-monotone scan (≥ n_scan values) + local bisection for criterion B.

    Reports:
    * ``s_min_first_pass``: smallest scanned s that passes;
    * ``s_min_all_pass``: smallest s above which *all* larger scanned values
      pass (or None if the scan never settles);
    * ``s_min``: local bisection around the first-pass bracket (primary report),
      falling back to ``s_min_all_pass`` when no isolated pass exists.
    """
    n_eval = 0
    # Upper end of the scan: double until A-like arrest with no cascade, or cap.
    s_hi = max(s0, 1e-6)
    info_hi = evaluate(cfg, s_hi, point, mmincfg, point_idx)
    n_eval += 1
    doublings = 0
    # Keep doubling while not arrested; once arrested keep going a bit so the
    # scan covers the stiff-net regime where B may turn False again.
    while doublings < mmincfg.max_doublings:
        if info_hi["arrested"] and info_hi["passed"] and doublings >= 2:
            break
        if info_hi["arrested"] and doublings >= 4 and info_hi["n_failures"] == 0:
            break
        s_hi *= 2.0
        info_hi = evaluate(cfg, s_hi, point, mmincfg, point_idx)
        n_eval += 1
        doublings += 1

    s_lo = 0.25 * s_hi
    s_values = np.linspace(s_lo, s_hi, mmincfg.n_scan)
    pattern = []
    infos = []
    for s in s_values:
        info = evaluate(cfg, float(s), point, mmincfg, point_idx)
        n_eval += 1
        pattern.append(bool(info["passed"]))
        infos.append(info)
        if not info["passed"]:
            logger.info(
                "B-scan F at s=%.4g: reason=%s n_failed=%d R_d=%.4g "
                "outcome=%s energy_error=%.3e",
                float(s), info["fail_reason"], info["n_failures"],
                info["R_d"], info["outcome"], info["energy_error"])

    # Smallest scanned passing s.
    first_pass = None
    for s, ok in zip(s_values, pattern):
        if ok:
            first_pass = float(s)
            break

    # Smallest s above which all larger scanned values pass.
    all_pass = None
    for i in range(len(pattern)):
        if pattern[i] and all(pattern[i:]):
            all_pass = float(s_values[i])
            break

    if first_pass is None and all_pass is None:
        logger.warning("point %s: criterion B never passed in the scan", point)
        return {"point": tuple(point), "point_idx": point_idx, "s_min": s_hi,
                "info": info_hi, "evaluations": n_eval,
                "s_min_first_pass": None, "s_min_all_pass": None,
                "scan_pattern": pattern}

    # Local bisection around the first pass: last failing s below it, then
    # bisect up to the first pass.
    i_pass = next(i for i, ok in enumerate(pattern) if ok)
    s_pass = float(s_values[i_pass])
    info_pass = infos[i_pass]
    if i_pass > 0:
        s_fail = float(s_values[i_pass - 1])
    else:
        s_fail = 0.5 * s_pass

    s_bis, info_bis, n_bis = _bisect_bracket(
        cfg, point, mmincfg, point_idx, s_fail, s_pass, info_pass)
    n_eval += n_bis

    # Primary s_min: report the all-pass threshold when the scan is
    # non-monotone (honest); otherwise the bisected first-pass value.
    # Always keep both numbers in the result.
    if (all_pass is not None and first_pass is not None
            and all_pass > first_pass * (1.0 + mmincfg.tol)):
        s_min = all_pass
        info_bis = evaluate(cfg, s_min, point, mmincfg, point_idx)
        n_eval += 1
    else:
        s_min = s_bis

    return {"point": tuple(point), "point_idx": point_idx, "s_min": s_min,
            "info": info_bis, "evaluations": n_eval,
            "s_min_first_pass": first_pass, "s_min_all_pass": all_pass,
            "scan_pattern": pattern}


def minimum_mass_point(args) -> dict:
    """Find the minimum passing scale for a single impact point (picklable)."""
    cfg, point, mmincfg, point_idx, s0 = args
    if mmincfg.criterion == "B":
        return _minimum_mass_point_B(cfg, point, mmincfg, point_idx, s0)
    return _minimum_mass_point_A(cfg, point, mmincfg, point_idx, s0)


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
        s_min_first_pass=worst.get("s_min_first_pass"),
        s_min_all_pass=worst.get("s_min_all_pass"),
        scan_pattern=worst.get("scan_pattern"),
    )


def monotonicity_scan(cfg: SimConfig, mmincfg: MminConfig, point, *,
                      n: int = 10, s_hi: Optional[float] = None) -> dict:
    """Coarse scan of the pass/fail pattern over s for one impact point.

    Scans ``n`` values in ``[0.25 s_hi, s_hi]`` (``s_hi`` defaults to a bracket
    found by doubling under criterion A so A and B share the same absolute
    scale — otherwise relative ``s/s_hi`` patterns are not comparable).
    Returns the pattern, per-s diagnostics, and whether it is monotone.
    """
    if s_hi is None:
        # Use criterion-A doubling so the scan covers the stiff-net regime
        # where zero-failure arrest is expected; B is then evaluated on the
        # same absolute s grid.
        cfg_A = replace(mmincfg, criterion="A")
        s0 = analytical_s0(cfg) * mmincfg.s0_scale
        res = _minimum_mass_point_A(cfg, point, cfg_A, 0, s0)
        s_hi = 2.0 * res["s_min"]

    s_values = np.linspace(0.25 * s_hi, s_hi, n)
    pattern = []
    details = []
    for s in s_values:
        info = evaluate(cfg, float(s), point, mmincfg, 0)
        pattern.append(bool(info["passed"]))
        details.append({
            "s": float(s),
            "passed": bool(info["passed"]),
            "fail_reason": info.get("fail_reason", ""),
            "n_failures": info["n_failures"],
            "R_d": info.get("R_d", float("nan")),
            "outcome": info.get("outcome", ""),
            "arrested": info["arrested"],
            "energy_error": info.get("energy_error", float("nan")),
        })
        if not info["passed"]:
            logger.info(
                "scan F s=%.4g reason=%s n_failed=%d R_d=%.4g outcome=%s "
                "Eerr=%.3e",
                float(s), info.get("fail_reason", ""), info["n_failures"],
                info.get("R_d", float("nan")), info.get("outcome", ""),
                info.get("energy_error", float("nan")))

    monotone = True
    seen_true = False
    for ok in pattern:
        if ok:
            seen_true = True
        elif seen_true:
            monotone = False
            break

    return {"point": tuple(point), "s_values": s_values.tolist(),
            "pattern": pattern, "monotone": monotone, "s_hi": s_hi,
            "details": details}
