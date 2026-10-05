"""Round-7 n_s timing and the 3% gate for star S/D (item 3).

Run on the VM, in order:

    python -m validation.round7_ns timing
    python -m validation.round7_ns check31
    python -m validation.round7_ns maybe32
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from validation.export_paper import OUT_DIR, _mmin_cfg, _MMIN_TOL, _mmin_n_procs

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
    mmA = MminConfig(criterion="A", tol=_MMIN_TOL, impact_points=pts,
                     n_procs=_mmin_n_procs(), cache_dir=cache)
    rA = minimum_mass(cfg, mmA)
    mmB = MminConfig(criterion="B_any", tol=_MMIN_TOL, impact_points=pts,
                     n_procs=_mmin_n_procs(), n_scan=24, cache_dir=cache)
    MminConfig.assert_same_tol(mmA, mmB)
    rB = minimum_mass(cfg, mmB, s_cap=rA.s_min)
    dt = time.perf_counter() - t0
    Ekin = 0.5 * 1.0 * 15.0 ** 2
    return {
        "mA": rA.m_min, "mBany": rB.m_min, "sA": rA.s_min, "sBany": rB.s_min,
        "s_lo_A": rA.s_lo, "s_hi_A": rA.s_hi, "n_bisect_A": rA.n_bisect,
        "s_lo_Bany": rB.s_lo, "s_hi_Bany": rB.s_hi, "n_bisect_Bany": rB.n_bisect,
        "mA_over_Ekin_g_per_J": 1e3 * rA.m_min / Ekin,
        "mBany_over_Ekin_g_per_J": 1e3 * rB.m_min / Ekin,
        "worstA": list(rA.worst_point), "worstB": list(rB.worst_point),
        "nA": rA.n_failures, "nB": rB.n_failures, "cpu_s": dt,
    }


def cmd_check31():
    _stamp_provenance()
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
                  f"mA/E={r['mA_over_Ekin_g_per_J']:.4g} "
                  f"bracketA=[{r['s_lo_A']},{r['s_hi_A']}] n_bisectA={r['n_bisect_A']} "
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


def _seed_cache(dst, seed):
    import shutil
    dst = Path(dst)
    seed = Path(seed)
    dst.mkdir(parents=True, exist_ok=True)
    if not seed.is_dir():
        return
    for p in seed.glob("*.h5"):
        t = dst / p.name
        if not t.exists():
            shutil.copy2(p, t)


def _one_crit(job):
    """One (mat, n_s, M, v0, criterion[, R_max, s_cap]) Alg.2 — pool worker."""
    from netsim.mmin import MminConfig, minimum_mass
    from validation.export_paper import _MMIN_CACHE, _mmin_cfg

    mat, n_s, M, v0, criterion, rmax, s_cap = job
    pts = [(0.0, 0.0), (0.25, 0.0), (0.5, 0.0)]
    cfg = _mmin_cfg(mat, M, v0, net_kind="star", n_s=int(n_s))
    tag = criterion if rmax is None else f"Bloc{int(round(100 * rmax)):03d}"
    shared = _MMIN_CACHE / f"star_{mat}_M{M:g}_v{v0:g}_ns{int(n_s)}"
    cache = _MMIN_CACHE / f"star_{mat}_M{M:g}_v{v0:g}_ns{int(n_s)}_{tag}"
    _seed_cache(cache, shared)
    mm = MminConfig(
        criterion=criterion,
        tol=_MMIN_TOL,
        impact_points=pts, n_procs=_mmin_n_procs(),
        n_scan=24, R_max=rmax, cache_dir=str(cache),
    )
    t0 = time.perf_counter()
    r = minimum_mass(cfg, mm, s_cap=s_cap)
    dt = time.perf_counter() - t0
    Ekin = 0.5 * M * v0 ** 2
    row = {
        "net": "star", "material": mat, "n_s": int(n_s),
        "M": M, "v0": v0, "Ekin": Ekin, "criterion": tag,
        "s_min": r.s_min, "m_min": r.m_min,
        "m_over_Ekin": r.m_min / Ekin if Ekin else float("nan"),
        "m_over_Ekin_g_per_J": 1e3 * r.m_min / Ekin if Ekin else float("nan"),
        "worst_p": list(r.worst_point), "n_broken": r.n_failures,
        "s_lo": r.s_lo, "s_hi": r.s_hi, "n_bisect": r.n_bisect,
        "cpu_s": dt,
    }
    print(f"    [3.2] {mat} n_s={n_s} M={M:g} v0={v0:g} {tag} "
          f"m={1e3*r.m_min:.4g}g m/E={row['m_over_Ekin_g_per_J']:.4g} g/J "
          f"[{r.s_lo},{r.s_hi}] n_bisect={r.n_bisect} cpu={dt:.1f}s",
          flush=True)
    return row


def cmd_maybe32():
    """Reduced 3.2: two (M,v0) at n_s=40, plus S n_s=80 A/B_any. No 18-row grid."""
    _stamp_provenance()
    import concurrent.futures
    from validation.export_paper import _write

    print("=== item 3.2 (reduced): NOT the 18-row star@40 table ===", flush=True)
    payload = {}
    if NS_PATH.is_file():
        try:
            payload = json.loads(NS_PATH.read_text())
        except Exception:
            payload = {}

    t_s = float((((payload.get("check31") or {}).get("S") or {}).get("40")
                 or {}).get("cpu_s") or 2519.5)
    t_d = float((((payload.get("check31") or {}).get("D") or {}).get("40")
                 or {}).get("cpu_s") or 3.0 * t_s)
    # Split A+B_any: A ~40%, B ~60%; n_s=80 ~4× n_s=40; B_loc ~ B_any.
    tA_S, tB_S = 0.40 * t_s, 0.60 * t_s
    tA_D, tB_D = 0.40 * t_d, 0.60 * t_d
    tA_S80, tB_S80 = 4.0 * tA_S, 4.0 * tB_S
    # Wave 1: 5 A jobs (two (1,15) A likely cached). Wave 2: 17 B jobs.
    n_workers = min(16, os.cpu_count() or 16)
    wall1 = max(tA_S, tA_D, tA_S80, 30.0) / 3600.0
    wall2 = max(tB_S, tB_D, tB_S80, 30.0) / 3600.0
    print(f"  measured 3.1 A+B_any: S n_s=40 {t_s/60:.1f} min, "
          f"D n_s=40 {t_d/60:.1f} min", flush=True)
    print(f"  estimate before start ({n_workers} workers): "
          f"wave1(A) ~{wall1:.2f} h  wave2(B) ~{wall2:.2f} h  "
          f"total ~{wall1+wall2:.2f} h", flush=True)
    print("  jobs: star S/D n_s=40, A+B_any+B_loc×3, (M,v0)=(1,15) and "
          "(0.25,20); plus star S n_s=80 M=1 v0=15 A+B_any", flush=True)

    cases40 = (
        ("S", 40, 1.0, 15.0),
        ("D", 40, 1.0, 15.0),
        ("S", 40, 0.25, 20.0),
        ("D", 40, 0.25, 20.0),
    )
    b_specs = (("B_any", None), ("B_loc", 0.25), ("B_loc", 0.5), ("B_loc", 0.75))

    def _run(jobs):
        if not jobs:
            return []
        n = min(n_workers, len(jobs))
        print(f"    [{len(jobs)} jobs × {n} workers]", flush=True)
        if n == 1:
            return [_one_crit(j) for j in jobs]
        with concurrent.futures.ProcessPoolExecutor(max_workers=n) as pool:
            return list(pool.map(_one_crit, jobs))

    wave1 = [(mat, ns, M, v0, "A", None, None)
             for mat, ns, M, v0 in cases40]
    wave1.append(("S", 80, 1.0, 15.0, "A", None, None))
    rows = _run(wave1)
    a_cap = {}
    for r in rows:
        a_cap[(r["material"], r["n_s"], r["M"], r["v0"])] = r["s_min"]

    wave2 = []
    for mat, ns, M, v0 in cases40:
        cap = a_cap.get((mat, ns, M, v0))
        for crit, rmax in b_specs:
            wave2.append((mat, ns, M, v0, crit, rmax, cap))
    cap80 = a_cap.get(("S", 80, 1.0, 15.0))
    wave2.append(("S", 80, 1.0, 15.0, "B_any", None, cap80))
    rows.extend(_run(wave2))

    # Persist a compact table.
    csv_rows = []
    for r in rows:
        csv_rows.append({
            "net": "star", "material": r["material"], "n_s": r["n_s"],
            "M": f"{r['M']:.6g}", "v0": f"{r['v0']:.6g}",
            "Ekin": f"{r['Ekin']:.6g}", "criterion": r["criterion"],
            "m_min_g": f"{1e3*r['m_min']:.6g}",
            "m_over_Ekin_g_per_J": f"{r['m_over_Ekin_g_per_J']:.4g}",
            "s_lo": f"{r['s_lo']:.6g}" if r.get("s_lo") is not None else "",
            "s_hi": f"{r['s_hi']:.6g}" if r.get("s_hi") is not None else "",
            "n_bisect": r.get("n_bisect", ""),
            "worst_p_x": f"{r['worst_p'][0]:.6g}",
            "worst_p_y": f"{r['worst_p'][1]:.6g}",
            "n_broken": r["n_broken"],
            "cpu_s": f"{r['cpu_s']:.4g}",
        })
    _write("tab_mmin_ns40_sample.csv",
           ["net", "material", "n_s", "M", "v0", "Ekin", "criterion",
            "m_min_g", "m_over_Ekin_g_per_J", "s_lo", "s_hi", "n_bisect",
            "worst_p_x", "worst_p_y", "n_broken", "cpu_s"],
           csv_rows, config="mmin_ns40_sample")

    print("=== m/E_kin at n_s=40: (1,15) E=112.5 J vs (0.25,20) E=50 J ===",
          flush=True)
    scale_ok = True
    for mat in ("S", "D"):
        for tag in ("A", "B_any", "Bloc025", "Bloc050", "Bloc075"):
            pair = [r for r in rows
                    if r["material"] == mat and r["n_s"] == 40
                    and r["criterion"] == tag]
            by_mv = {(r["M"], r["v0"]): r for r in pair}
            a = by_mv.get((1.0, 15.0))
            b = by_mv.get((0.25, 20.0))
            if not a or not b:
                print(f"  {mat} {tag}: missing row", flush=True)
                scale_ok = False
                continue
            ra, rb = a["m_over_Ekin_g_per_J"], b["m_over_Ekin_g_per_J"]
            rel = abs(ra - rb) / max(abs(ra), 1e-15)
            ok = rel <= 0.01
            scale_ok = scale_ok and ok
            print(f"  {mat} {tag}: m/E (1,15)={ra:.4g}  (0.25,20)={rb:.4g}  "
                  f"|rel|={100*rel:.2f}%  {'OK' if ok else 'NO'}", flush=True)
    if scale_ok:
        print("  SCALE: m/E_kin agrees within 1% → n_s=40 table from "
              "E_kin scaling of these two points.", flush=True)
    else:
        print("  SCALE: m/E_kin differs by >1% on at least one criterion; "
              "do not claim E_kin scaling for the n_s=40 table.", flush=True)

    print("=== n_s=40 vs 80, star S M=1 v0=15 A and B_any ===", flush=True)
    for tag in ("A", "B_any"):
        a40 = next((r for r in rows if r["material"] == "S" and r["n_s"] == 40
                    and r["M"] == 1.0 and r["v0"] == 15.0
                    and r["criterion"] == tag), None)
        a80 = next((r for r in rows if r["material"] == "S" and r["n_s"] == 80
                    and r["criterion"] == tag), None)
        if not a40 or not a80:
            print(f"  {tag}: missing", flush=True)
            continue
        rel = abs(a80["m_min"] - a40["m_min"]) / max(a40["m_min"], 1e-15)
        print(f"  {tag}: m40={1e3*a40['m_min']:.4g}g m80={1e3*a80['m_min']:.4g}g "
              f"|rel|={100*rel:.2f}%", flush=True)

    payload["item32_reduced"] = True
    payload["scale_ok"] = scale_ok
    ser = []
    for r in rows:
        ser.append({
            "material": r["material"], "n_s": int(r["n_s"]),
            "M": float(r["M"]), "v0": float(r["v0"]),
            "criterion": r["criterion"],
            "m_min": float(r["m_min"]),
            "m_over_Ekin_g_per_J": float(r["m_over_Ekin_g_per_J"]),
            "n_broken": int(r["n_broken"]),
            "cpu_s": float(r["cpu_s"]),
            "worst_p": [float(r["worst_p"][0]), float(r["worst_p"][1])],
        })
    payload["item32_rows"] = ser
    NS_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    return 0


def cmd_estimate():
    """Wall-time estimate from measured r7 per-run times and cache reuse."""
    _stamp_provenance()
    payload = {}
    if NS_PATH.is_file():
        try:
            payload = json.loads(NS_PATH.read_text())
        except Exception:
            payload = {}
    t = payload.get("times_s") or {}
    tS10 = float(t.get("S_ns10", 1.46))
    tS40 = float(t.get("S_ns40", 6.50))
    tD10 = float(t.get("D_ns10", 77.8))
    tD40 = float(t.get("D_ns40", 535.0))
    c31 = payload.get("check31") or {}
    cpuS40 = float(((c31.get("S") or {}).get("40") or {}).get("cpu_s") or 2519.5)
    cpuD40 = float(((c31.get("D") or {}).get("40") or {}).get("cpu_s") or 24221.5)
    n_cache = 0
    from validation.export_paper import _MMIN_CACHE
    if _MMIN_CACHE.is_dir():
        n_cache = sum(1 for _ in _MMIN_CACHE.rglob("*.h5"))
    n_workers = min(16, os.cpu_count() or 16)
    extra = 3  # ~log2(0.08/0.01) extra A bisection steps
    # (a) 36 rows, 16 case-workers; new A midpoints, B mostly cache hits.
    nA = 36 * 3 * extra
    t_miss10 = 0.5 * (tS10 + min(tD10, 30.0))
    wall_a = (nA * t_miss10) / n_workers / 3600.0
    # (b) 4 cases; n_s=10 cache; n_s=40 extra A bisect. Inner 16 on scan/points.
    wall_b = (3 * extra * (tS40 + min(tD40 * 0.2, 120.0))) / n_workers / 3600.0
    wall_b = max(wall_b, 0.15)
    # (c) (1,15) n_s=40 cached; (0.25,20) extra A + B cache; S n_s=80 A+B_any
    #     S80 ~ 4× S40 A+Bany if cold, but many n_s=40 scales differ.
    wall_c_s80 = (cpuS40 * 4.0) / n_workers / 3600.0
    wall_c_d025 = (cpuD40 * 0.15) / 3600.0  # extra A only, one case
    wall_c = max(wall_c_s80, wall_c_d025, 0.5)
    total = wall_a + wall_b + wall_c
    print("=== 7b wall estimate (measured r7 times, cache reuse) ===", flush=True)
    print(f"  cache files={n_cache}  workers={n_workers}  A extra bisect~{extra}",
          flush=True)
    print(f"  single-sim T10 S={tS10:.2f}s D={tD10:.1f}s  "
          f"T40 S={tS40:.2f}s D={tD40:.1f}s", flush=True)
    print(f"  3.1 A+Bany n_s=40 from scratch: S={cpuS40/60:.1f} min  "
          f"D={cpuD40/3600:.2f} h", flush=True)
    print(f"  (a) tab_mmin n_s=10 36 rows  ~{wall_a:.2f} h", flush=True)
    print(f"  (b) 3.1 S/D n_s=10 vs 40     ~{wall_b:.2f} h", flush=True)
    print(f"  (c) reduced 3.2              ~{wall_c:.2f} h", flush=True)
    print(f"  total ~{total:.2f} h  (24 h guard)", flush=True)
    payload["estimate_7b_h"] = {
        "a": wall_a, "b": wall_b, "c": wall_c, "total": total,
        "n_cache": n_cache, "n_workers": n_workers,
    }
    NS_PATH.parent.mkdir(parents=True, exist_ok=True)
    NS_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    return 0


def cmd_round7b():
    """a → b → c. Prefer the on-VM shell which also stamps export_paper."""
    _stamp_provenance()
    cmd_estimate()
    print("=== 7b (a) tab_mmin n_s=10, 36 rows, tol=0.01 ===", flush=True)
    from validation.export_paper import export_mmin
    export_mmin(full=True, n_s=10)
    cmd_check31()
    cmd_maybe32()
    return 0


def main(argv=None):
    argv = list(argv if argv is not None else sys.argv[1:])
    cmd = argv[0] if argv else "timing"
    fn = {"timing": cmd_timing, "check31": cmd_check31,
          "maybe32": cmd_maybe32, "estimate": cmd_estimate,
          "round7b": cmd_round7b}.get(cmd)
    if fn is None:
        print(f"unknown command {cmd!r}; "
              f"use timing|check31|maybe32|estimate|round7b",
              file=sys.stderr)
        return 2
    return fn()


if __name__ == "__main__":
    raise SystemExit(main())
