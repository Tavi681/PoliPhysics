"""Round-7 n_s timing and the 3% gate for star S/D (item 3).

Run on the VM, in order:

    python -m validation.round7_ns timing
    python -m validation.round7_ns check31
    python -m validation.round7_ns maybe32
"""

from __future__ import annotations

import json
import sys
import time

from validation.export_paper import OUT_DIR, _mmin_cfg

NS_PATH = OUT_DIR / "ns_check.json"


def _time_star(mat, n_s, s=1.0):
    from dataclasses import replace
    from netsim.simulate import simulate_config

    cfg = _mmin_cfg(mat, 1.0, 15.0, net_kind="star", n_s=n_s)
    cfg = replace(
        cfg,
        drone=replace(cfg.drone, p=(0.5, 0.0)),
        numerics=replace(cfg.numerics, area_scale=s, t_end=0.25),
    )
    t0 = time.perf_counter()
    res = simulate_config(cfg, write=False)
    dt = time.perf_counter() - t0
    return dt, res.outcome, res.n_failures


def cmd_timing():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    times = {}
    print("=== n_s timing: star S/D  M=1 v0=15 a=R/2 ===", flush=True)
    for mat in ("S", "D"):
        for n_s in (10, 40):
            dt, outcome, nfail = _time_star(mat, n_s)
            times[f"{mat}_ns{n_s}"] = dt
            print(f"  {mat} n_s={n_s:2d}  {dt:8.2f}s  "
                  f"outcome={outcome} n_fail={nfail}", flush=True)
    t10 = 0.5 * (times["S_ns10"] + times["D_ns10"])
    t40 = 0.5 * (times["S_ns40"] + times["D_ns40"])
    ratio = t40 / max(t10, 1e-9)
    # Rough Alg.2 cost: 3 points × ~25 evaluations.
    ev_per_crit = 75
    extra_31 = 4 * ev_per_crit * t40          # star S/D × A + B_any
    extra_32 = 18 * 5 * ev_per_crit * t40     # star S/D × 9 (M,v0) × 5 crit
    print(f"  mean T10={t10:.2f}s T40={t40:.2f}s  T40/T10={ratio:.2f}",
          flush=True)
    print(f"  extrapolated CPU  3.1 (4 Alg.2 @ n_s=40) ~ {extra_31/3600:.2f} h",
          flush=True)
    print(f"  extrapolated CPU  3.2 (18 star rows @ n_s=40) ~ {extra_32/3600:.2f} h",
          flush=True)
    payload = {
        "times_s": times, "T10": t10, "T40": t40, "ratio": ratio,
        "extrapolated_h_31": extra_31 / 3600.0,
        "extrapolated_h_32": extra_32 / 3600.0,
    }
    prev = {}
    if NS_PATH.is_file():
        try:
            prev = json.loads(NS_PATH.read_text())
        except Exception:
            prev = {}
    prev.update(payload)
    NS_PATH.write_text(json.dumps(prev, indent=2) + "\n")
    print(f"  wrote {NS_PATH}", flush=True)
    return 0


def _alg2_ab(mat, n_s):
    from netsim.mmin import MminConfig, minimum_mass
    from validation.export_paper import _MMIN_CACHE

    pts = [(0.0, 0.0), (0.25, 0.0), (0.5, 0.0)]
    cfg = _mmin_cfg(mat, 1.0, 15.0, net_kind="star", n_s=n_s)
    cache = str(_MMIN_CACHE / f"star_{mat}_M1_v15_ns{n_s}")
    t0 = time.perf_counter()
    mmA = MminConfig(criterion="A", tol=0.08, impact_points=pts, n_procs=1,
                     cache_dir=cache)
    rA = minimum_mass(cfg, mmA)
    mmB = MminConfig(criterion="B_any", tol=0.01, impact_points=pts,
                     n_procs=1, n_scan=24, cache_dir=cache)
    rB = minimum_mass(cfg, mmB, s_cap=rA.s_min)
    dt = time.perf_counter() - t0
    return {
        "mA": rA.m_min, "mBany": rB.m_min, "sA": rA.s_min, "sBany": rB.s_min,
        "worstA": list(rA.worst_point), "worstB": list(rB.worst_point),
        "nA": rA.n_failures, "nB": rB.n_failures, "cpu_s": dt,
    }


def cmd_check31():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== item 3.1: star S/D Alg.2 A & B_any at n_s=10 vs 40 ===",
          flush=True)
    results = {}
    need_32 = False
    for mat in ("S", "D"):
        pair = {}
        for n_s in (10, 40):
            print(f"  running {mat} n_s={n_s} ...", flush=True)
            pair[str(n_s)] = _alg2_ab(mat, n_s)
            r = pair[str(n_s)]
            print(f"    mA={1e3*r['mA']:.4g}g mBany={1e3*r['mBany']:.4g}g "
                  f"cpu={r['cpu_s']:.1f}s", flush=True)
        relA = abs(pair["40"]["mA"] - pair["10"]["mA"]) / max(pair["10"]["mA"], 1e-15)
        relB = abs(pair["40"]["mBany"] - pair["10"]["mBany"]) / max(
            pair["10"]["mBany"], 1e-15)
        pair["rel_mA"] = relA
        pair["rel_mBany"] = relB
        if relA > 0.03 or relB > 0.03:
            need_32 = True
        print(f"  {mat}: |mA_40/mA_10-1|={100*relA:.2f}%  "
              f"|mB_40/mB_10-1|={100*relB:.2f}%", flush=True)
        results[mat] = pair
    payload = {}
    if NS_PATH.is_file():
        try:
            payload = json.loads(NS_PATH.read_text())
        except Exception:
            payload = {}
    payload["check31"] = results
    payload["need_32"] = need_32
    payload["keep_ns10"] = not need_32
    NS_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    if need_32:
        print("  GATE FAIL: n_s=40 differs by >3% → will run item 3.2",
              flush=True)
    else:
        print("  GATE PASS: within 3% → keep n_s=10", flush=True)
    return 0


def cmd_maybe32():
    need = False
    if NS_PATH.is_file():
        try:
            need = bool(json.loads(NS_PATH.read_text()).get("need_32"))
        except Exception:
            need = False
    if not need:
        print("=== item 3.2 skipped (n_s=10 vs 40 within 3%) ===", flush=True)
        return 0
    print("=== item 3.2: full star S/D tab_mmin at n_s=40 ===", flush=True)
    from validation.export_paper import export_mmin
    export_mmin(True, n_s=40, nets=("star",), name="tab_mmin_star_ns40.csv")
    return 0


def main(argv=None):
    argv = list(argv if argv is not None else sys.argv[1:])
    cmd = argv[0] if argv else "timing"
    fn = {"timing": cmd_timing, "check31": cmd_check31,
          "maybe32": cmd_maybe32}.get(cmd)
    if fn is None:
        print(f"unknown command {cmd!r}; use timing|check31|maybe32",
              file=sys.stderr)
        return 2
    return fn()


if __name__ == "__main__":
    raise SystemExit(main())
