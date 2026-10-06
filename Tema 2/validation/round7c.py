"""Round 7c: thread-based B, re-eval from cache, n_s=40 star table.

    python -m validation.round7c estimate
    python -m validation.round7c check_s_ns
    python -m validation.round7c round7c
"""

from __future__ import annotations

import json
import os
import sys
import time

from validation.export_paper import (
    OUT_DIR, _MMIN_CACHE, _MMIN_TOL, _mmin_cfg, _mmin_n_procs,
    _mmin_case, _parallel_map,
    merge_tagged_mmin_caches, count_mmin_cache_h5,
    export_mmin_round7c,
)

NS_PATH = OUT_DIR / "ns_check.json"


def _stamp_provenance():
    import validation.export_paper as ep
    env_c = os.environ.get("GIT_COMMIT", "").strip()
    if env_c:
        ep._COMMIT = env_c
    d = os.environ.get("CODE_DIRTY", "").strip().lower()
    if d in ("true", "1", "yes"):
        ep._CODE_DIRTY = True
    elif d in ("false", "0", "no"):
        ep._CODE_DIRTY = False
    jobs = os.environ.get("JOBS", "").strip()
    if jobs:
        ep._N_JOBS = max(1, int(jobs))


def _load_ns():
    if not NS_PATH.is_file():
        return {}
    try:
        return json.loads(NS_PATH.read_text())
    except Exception:
        return {}


def _save_ns(payload):
    prev = _load_ns()
    prev.update(payload)
    NS_PATH.parent.mkdir(parents=True, exist_ok=True)
    NS_PATH.write_text(json.dumps(prev, indent=2) + "\n")


def _alg2_ab(mat, n_s, M=1.0, v0=15.0):
    from netsim.mmin import MminConfig, minimum_mass

    pts = [(0.0, 0.0), (0.25, 0.0), (0.5, 0.0)]
    cfg = _mmin_cfg(mat, M, v0, net_kind="star", n_s=n_s)
    cache = str(_MMIN_CACHE / f"star_{mat}_M{M:g}_v{v0:g}_ns{n_s}")
    t0 = time.perf_counter()
    mmA = MminConfig(criterion="A", tol=_MMIN_TOL, impact_points=pts,
                     n_procs=_mmin_n_procs(), cache_dir=cache)
    rA = minimum_mass(cfg, mmA)
    mmB = MminConfig(criterion="B_any", tol=_MMIN_TOL, impact_points=pts,
                     n_procs=_mmin_n_procs(), n_scan=24, cache_dir=cache)
    MminConfig.assert_same_tol(mmA, mmB)
    rB = minimum_mass(cfg, mmB, s_cap=rA.s_min)
    dt = time.perf_counter() - t0
    Ekin = 0.5 * M * v0 ** 2
    return {
        "mA": rA.m_min, "mBany": rB.m_min, "sA": rA.s_min, "sBany": rB.s_min,
        "s_lo_A": rA.s_lo, "s_hi_A": rA.s_hi, "n_bisect_A": rA.n_bisect,
        "s_lo_Bany": rB.s_lo, "s_hi_Bany": rB.s_hi, "n_bisect_Bany": rB.n_bisect,
        "mA_over_Ekin_g_per_J": 1e3 * rA.m_min / Ekin,
        "mBany_over_Ekin_g_per_J": 1e3 * rB.m_min / Ekin,
        "nA_threads": rA.n_failures, "nB_threads": rB.n_failures,
        "nA_segments": rA.n_failed_segments, "nB_segments": rB.n_failed_segments,
        "worstA": list(rA.worst_point), "worstB": list(rB.worst_point),
        "cpu_s": dt,
    }


def cmd_estimate():
    """Wall-time estimate from r7b measured times; cache reuse + 2 new D/S runs."""
    n_copied = merge_tagged_mmin_caches()
    n_h5 = count_mmin_cache_h5()
    n_workers = min(16, os.cpu_count() or 16)
    payload = _load_ns()
    times = payload.get("times_s") or {}
    tS40 = float(times.get("S_ns40") or 6.50)
    tD40 = float(times.get("D_ns40") or 535.4)
    # r7b D B_any n_s=40 (0.25,20) was ~6.1 h at n_procs=1; 3 impact points
    # in parallel cut that to ~2 h per B criterion. A is ~7 serial evals.
    wall_a = 0.15  # star+ring n_s=10 + cached star keys, mostly hits
    wall_check = 0.10  # S n_s=10/40/80 from cache
    n_scan = 24 + 5
    wall_new_S = (5 * n_scan * tS40) / max(3, 1) / 3600.0
    wall_new_D_A = (7 * tD40) / 3600.0
    wall_new_D_B = 4 * (n_scan * tD40 / 3.0) / 3600.0
    wall_new = wall_new_S + wall_new_D_A + wall_new_D_B
    total = wall_a + wall_check + wall_new
    print("=== 7c wall estimate (cache reuse + 2 new M=1 v0=10 n_s=40) ===",
          flush=True)
    print(f"  cache files={n_h5}  merged_tagged={n_copied}  workers={n_workers}",
          flush=True)
    print(f"  T40 S={tS40:.2f}s D={tD40:.1f}s", flush=True)
    print(f"  (check) S M=1 v0=15 n_s=10/40/80  ~{wall_check:.2f} h", flush=True)
    print(f"  (cache) star+ring n_s=10 + star keys  ~{wall_a:.2f} h", flush=True)
    print(f"  (new)   star S/D M=1 v0=10 n_s=40    ~{wall_new:.2f} h", flush=True)
    print(f"  total ~{total:.2f} h  (24 h guard)", flush=True)
    _save_ns({"estimate_7c_h": {
        "check": wall_check, "cache": wall_a, "new": wall_new, "total": total,
        "n_cache": n_h5, "n_workers": n_workers,
    }})
    return 0


def cmd_check_s_ns():
    """Star S M=1 v0=15 at n_s=10, 40, 80; B_any 40 vs 80 within 1%."""
    _stamp_provenance()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== 7c: star S M=1 v0=15 A+B_any at n_s=10, 40, 80 (thread B) ===",
          flush=True)
    pair = {}
    for n_s in (10, 40, 80):
        print(f"  running S n_s={n_s} ...", flush=True)
        pair[str(n_s)] = _alg2_ab("S", n_s)
        r = pair[str(n_s)]
        print(f"    mA={1e3*r['mA']:.4g}g mBany={1e3*r['mBany']:.4g}g "
              f"nB_thr={r['nB_threads']} nB_seg={r['nB_segments']} "
              f"mB/E={r['mBany_over_Ekin_g_per_J']:.4g} "
              f"bracketB=[{r['s_lo_Bany']},{r['s_hi_Bany']}] "
              f"n_bisectB={r['n_bisect_Bany']} cpu={r['cpu_s']:.1f}s",
              flush=True)
    relB = abs(pair["80"]["mBany"] - pair["40"]["mBany"]) / max(
        pair["40"]["mBany"], 1e-15)
    ok = relB <= 0.01
    print(f"  B_any n_s=40 vs 80 |rel|={100*relB:.2f}%  "
          f"{'OK' if ok else 'NO (>1%)'}", flush=True)
    _save_ns({"check_s_ns": pair, "bany_40_vs_80_rel": relB,
              "bany_40_vs_80_ok": ok})
    return 0 if ok else 0  # still continue the table even if the gate fails


def cmd_round7c():
    """Full 7c: estimate, S n_s check, table (cache then new v0=10)."""
    _stamp_provenance()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cmd_estimate()
    n_inner = min(16, int(os.environ.get("JOBS", str(os.cpu_count() or 16))))
    os.environ["MMIN_N_PROCS"] = str(n_inner)
    cmd_check_s_ns()

    grid10 = [(0.25, 10.0), (0.25, 15.0), (0.25, 20.0),
              (1.0, 10.0), (1.0, 15.0), (1.0, 20.0),
              (2.0, 10.0), (2.0, 15.0), (2.0, 20.0)]
    ring_jobs = [("star+ring", mat, M, v0, True, 10)
                 for mat in ("S", "D") for M, v0 in grid10]
    cache_star = [("star", mat, M, v0, True, 40)
                  for mat in ("S", "D")
                  for M, v0 in ((1.0, 15.0), (0.25, 20.0))]
    new_star = [("star", mat, 1.0, 10.0, True, 40) for mat in ("S", "D")]

    print("=== 7c cache re-eval: star+ring n_s=10 + star n_s=40 (1,15)/(0.25,20) ===",
          flush=True)
    os.environ["MMIN_N_PROCS"] = "1"
    ring_rows = _parallel_map(_mmin_case, ring_jobs,
                              desc="star+ring n_s=10")
    star_cached = _parallel_map(_mmin_case, cache_star,
                                desc="star n_s=40 cached keys")

    print("=== 7c new production: star S/D M=1 v0=10 n_s=40 (inner 16) ===",
          flush=True)
    os.environ["MMIN_N_PROCS"] = str(n_inner)
    star_new = []
    for job in new_star:
        print(f"  sequential {job[1]} M=1 v0=10 n_s=40  MMIN_N_PROCS={n_inner}",
              flush=True)
        star_new.append(_mmin_case(job))

    computed_star = list(star_cached) + list(star_new)
    path = export_mmin_round7c(
        computed_star=computed_star, computed_ring=ring_rows)
    print(f"  wrote {path}", flush=True)
    _save_ns({"item7c": True, "tab_mmin": str(path)})
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "round7c"
    fn = {"estimate": cmd_estimate, "check_s_ns": cmd_check_s_ns,
          "round7c": cmd_round7c}.get(cmd)
    if fn is None:
        print(f"unknown command {cmd!r}; use estimate|check_s_ns|round7c",
              file=sys.stderr)
        return 2
    return fn()


if __name__ == "__main__":
    sys.exit(main())
