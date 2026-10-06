"""Item 6: export every paper CSV listed in ``paper_results/SPEC.md``.

Each CSV is written to ``paper_results/`` with the exact column names from the
spec and a header line ``# commit=<hash> date=<ISO> config=<file>``. One function
per CSV; each is wrapped so a single failure does not abort the whole export and
a per-file status line is printed. The very expensive grids (``tab_mmin``,
``dyn_runs``, ``fig_phase_maps``) run on a reduced grid by default and the full
grid with ``--full``; the reduction is reported in the run log.

Run:  python -m validation.export_paper [--full] [--out DIR]

No physics or tolerances are tuned; all numbers are measured or analytic.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime
import math
import os
import sys
from pathlib import Path
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/netsim_mpl")

import numpy as np

# ref/ is import-only for riemann and offc (never tables.py, which prints).
_REF = str(Path(__file__).resolve().parent.parent / "ref")
if _REF not in sys.path:
    sys.path.insert(0, _REF)

from netsim.io_hdf5 import git_commit
from netsim.materials import get_material

OUT_DIR = Path(__file__).resolve().parent.parent / "paper_results"
_ROOT = Path(__file__).resolve().parent.parent
_COMMIT = git_commit()
_DATE = datetime.datetime.now().isoformat(timespec="seconds")
_CODE_DIRTY: Optional[bool] = None  # set once at export start

# Process-pool width for independent export cases (overridden by --jobs).
_N_JOBS = max(1, os.cpu_count() or 1)
_ALLOW_DIRTY = False
_MMIN_CACHE = _ROOT / ".cache" / "mmin"


def merge_tagged_mmin_caches(cache_root=None) -> int:
    """Copy round-7b tagged dirs (…_ns40_B_any) into the untagged sibling."""
    import shutil
    root = Path(cache_root or _MMIN_CACHE)
    if not root.is_dir():
        return 0
    n = 0
    tags = ("_A", "_B_any", "_Bloc025", "_Bloc050", "_Bloc075")
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        stem = None
        for tag in tags:
            if d.name.endswith(tag):
                stem = d.name[:-len(tag)]
                break
        if stem is None:
            continue
        dst = root / stem
        dst.mkdir(parents=True, exist_ok=True)
        for p in d.glob("mmin_p*.h5"):
            t = dst / p.name
            if not t.exists():
                shutil.copy2(p, t)
                n += 1
    return n


def count_mmin_cache_h5(cache_root=None) -> int:
    root = Path(cache_root or _MMIN_CACHE)
    if not root.is_dir():
        return 0
    return sum(1 for p in root.rglob("mmin_p*.h5") if p.is_file())


def _repo_root() -> Path:
    return _ROOT.parent if (_ROOT.parent / ".git").exists() else _ROOT


def _code_paths_dirty() -> bool:
    """True if netsim/validation/tests/configs differ from HEAD (scoped)."""
    import subprocess
    try:
        repo = _repo_root()
        prefix = "Tema 2/" if repo != _ROOT else ""
        paths = [f"{prefix}{p}" for p in
                 ("netsim", "validation", "tests", "configs")]
        r = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", *paths],
            cwd=str(repo), check=False)
        if r.returncode != 0:
            return True
        # Untracked files under those paths also count as dirty.
        r2 = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "--", *paths],
            cwd=str(repo), capture_output=True, text=True, check=False)
        return bool(r2.stdout.strip())
    except Exception:
        return True


def _working_tree_dirty() -> bool:
    """Legacy full-tree dirty check (export refuse / --allow-dirty)."""
    import subprocess
    try:
        repo = _repo_root()
        pathspec = "Tema 2" if repo != _ROOT else "."
        r = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal",
             "--", pathspec],
            cwd=str(repo), capture_output=True, text=True, check=False)
        return bool(r.stdout.strip())
    except Exception:
        return True


def _header_dirty_flag() -> str:
    """``code_dirty`` recorded at export start (scoped paths), not mid-write."""
    dirty = _CODE_DIRTY if _CODE_DIRTY is not None else _code_paths_dirty()
    return "true" if dirty else "false"


def _parallel_map(fn, jobs, desc=""):
    """Map ``fn`` over ``jobs`` with a process pool when it pays off."""
    if not jobs:
        return []
    n = min(int(_N_JOBS), len(jobs))
    if n <= 1:
        return [fn(j) for j in jobs]
    print(f"    [{desc}] {len(jobs)} jobs × {n} workers", flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=n) as pool:
        return list(pool.map(fn, jobs))


def _write(name, fieldnames, rows, config="default"):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    with open(path, "w", newline="") as fh:
        fh.write(f"# commit={_COMMIT} date={_DATE} config={config} "
                 f"code_dirty={_header_dirty_flag()}\n")
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def _pct(num, ana):
    return 100.0 * abs(num - ana) / abs(ana) if ana else float("nan")


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
def export_params_used():
    rows = []
    for name in ("S", "D"):
        m = get_material(name)
        for sym, val, unit, com in (
            (f"E0_{name}", m.E0, "Pa", f"material {name} small-strain modulus"),
            (f"b_{name}", m.b, "Pa", f"material {name} cubic coefficient"),
            (f"eps_b_{name}", m.eps_b, "-", f"material {name} breaking strain"),
            (f"rho_{name}", m.rho, "kg/m^3", f"material {name} density"),
            (f"e_mat_{name}", m.e_mat, "J/kg", f"material {name} specific energy"),
        ):
            rows.append({"symbol": sym, "value": f"{val:.6g}", "unit": unit,
                         "comment": com})
    for sym, val, unit, com in (
        ("r_d", 0.15, "m", "drone radius (working value; '?' in paper)"),
        ("k_c", 1e7, "N/m^1.5", "absolute penalty stiffness (working value)"),
        ("k_c_factor", 8.0, "-",
         "D relative k_c: k_c = k_c_factor^2 (E0 A / l_s) / "
         "((3/2) sqrt(delta_ref)); not 8e7 absolute"),
        ("delta_ref", 0.01 * 0.15, "m",
         "contact reference penetration = delta_ref_frac * r_d "
         "(delta_ref_frac=0.01)"),
        ("delta_ref_frac", 0.01, "-", "delta_ref / r_d"),
        ("damping_qs", 400.0, "1/s",
         "viscous damping per unit mass in quasi-static Tests 2/5 "
         "(off-centre pull / hub F(w)); dynamic Alg.1/2 uses 0"),
        ("dt_out_dyn", 5e-3, "s",
         "output interval of Alg.2 / production dynamic runs "
         "(item-1 diagnostics use 1e-4 s)"),
        ("R_max", 0.5, "m", "cascade radius (working value)"),
        ("k_max", 10, "-", "cascade failure count (working value)"),
        ("N", 8, "-", "reference number of radials"),
        ("R", 1.0, "m", "frame radius"),
        ("eps_p_frac", 0.1, "-", "reference prestress eps_p/eps_b"),
        ("n_s", 40, "-", "reference segments per thread"),
        ("C", 0.5, "-", "CFL factor"),
    ):
        rows.append({"symbol": sym, "value": f"{val}", "unit": unit,
                     "comment": com})
    return _write("params_used.csv", ["symbol", "value", "unit", "comment"],
                  rows)


# --------------------------------------------------------------------------- #
# Smith (Test 1)
# --------------------------------------------------------------------------- #
def export_smith():
    from validation.test1_smith import run_smith, strain_analytic, MATERIAL_V0
    rows = []
    for mat in ("S", "D"):
        for v0 in MATERIAL_V0[mat]:
            if strain_analytic(get_material(mat), v0) is None:
                continue  # v0 exceeds breaking strain for this material
            r = run_smith(v0, material_name=mat, n_resolved=800)
            cL_num = r.get("cL_num", r["front_num"] / r["t"])
            cT_num = r.get("cT_num", r["kink_num"] / r["t"])
            rows.append({
                "material": mat, "v0": f"{v0:.6g}",
                "eps_an": f"{r['eps_ana']:.6g}", "eps_num": f"{r['eps_num']:.6g}",
                "eps_err_pct": f"{100*r['eps_relerr']:.4g}",
                "cL_an": f"{r['cL']:.6g}", "cL_num": f"{cL_num:.6g}",
                "cL_err_pct": f"{_pct(cL_num, r['cL']):.4g}",
                "cT_an": f"{r['cT']:.6g}", "cT_num": f"{cT_num:.6g}",
                "cT_err_pct": f"{_pct(cT_num, r['cT']):.4g}",
                "n_seg": int(round(r["n_resolved"])),
            })
    cols = ["material", "v0", "eps_an", "eps_num", "eps_err_pct", "cL_an",
            "cL_num", "cL_err_pct", "cT_an", "cT_num", "cT_err_pct", "n_seg"]
    return _write("tab_smith.csv", cols, rows, config="test1_smith")


def export_smith_conv():
    from validation.test1_smith import run_smith
    rows = []
    for n_res in (200, 400, 800):
        r = run_smith(500.0, material_name="S", n_resolved=n_res)
        cL_num = r.get("cL_num", r["front_num"] / r["t"])
        cT_num = r.get("cT_num", r["kink_num"] / r["t"])
        rows.append({
            "n_seg": int(round(r["n_resolved"])),
            "eps_err_pct": f"{100*r['eps_relerr']:.4g}",
            "cL_err_pct": f"{_pct(cL_num, r['cL']):.4g}",
            "cT_err_pct": f"{_pct(cT_num, r['cT']):.4g}",
        })
    return _write("tab_smith_conv.csv",
                  ["n_seg", "eps_err_pct", "cL_err_pct", "cT_err_pct"], rows,
                  config="test1_smith")


def export_fig_smith():
    from validation.test1_smith import run_smith
    rows = []
    # S@300 and D@500; frac_time keeps both fronts inside the half-thread.
    for mat, v0 in (("S", 300.0), ("D", 500.0)):
        for frac in (0.35, 0.45, 0.55):
            r = run_smith(v0, material_name=mat, n_resolved=400, frac_time=frac)
            tt = r["t"]
            xf = r["traj"].x[-1]
            edges = r["edges"]
            xs = r["xs"]
            Lx = r["Lx"]
            Xseg = 0.5 * (xs[:-1] + xs[1:]) - Lx / 2
            right = Xseg > 0
            mids = 0.5 * (xf[edges[:, 0]] + xf[edges[:, 1]])[right]
            for X, eps, mid in zip(r["X"], r["eps_profile"], mids):
                rows.append({
                    "material": mat, "v0": f"{v0:.6g}",
                    "t": f"{tt:.6g}", "X": f"{X:.6g}", "eps": f"{eps:.6g}",
                    "x": f"{mid[0]:.6g}", "y": f"{mid[2]:.6g}",
                })
    return _write("fig_smith.csv",
                  ["material", "v0", "t", "X", "eps", "x", "y"], rows,
                  config="test1_smith")


# --------------------------------------------------------------------------- #
# Junction (Test 3)
# --------------------------------------------------------------------------- #
def export_junction():
    from validation.test3_junction import run_junction_sim
    rows = []
    for mat in ("S", "D"):
        for N in (2, 3, 4, 6, 8, 12, 16):
            r = run_junction_sim(name=mat, N=N, ep_frac=0.1)
            rows.append({
                "material": mat, "N": N,
                "SN": f"{r['SN']:.6g}", "SN2d": f"{r['SN2d_wo']:.6g}",
                "T0_1d": f"{r['T0_lin']:.6g}", "T0_2d": f"{r['T0_2d_wo']:.6g}",
                "T0_num": f"{r['T0_sim']:.6g}",
                "dTmax_1d": f"{r['dTmax_1d']:.6g}",
                "dTmax_2d": f"{r['dTmax_2d_wo']:.6g}",
                "dTmax_num": f"{r['dTmax_sim']:.6g}",
                "dTmin_1d": f"{r['dTmin_1d']:.6g}",
                "dTmin_2d": f"{r['dTmin_2d_wo']:.6g}",
                "dTmin_num": f"{r['dTmin_sim']:.6g}",
            })
    cols = ["material", "N", "SN", "SN2d", "T0_1d", "T0_2d", "T0_num",
            "dTmax_1d", "dTmax_2d", "dTmax_num", "dTmin_1d", "dTmin_2d",
            "dTmin_num"]
    return _write("tab_junction.csv", cols, rows, config="test3_junction")


def export_junction_scaling():
    from validation.test3_junction import run_junction_sim
    rows = []
    for ep_frac in (0.02, 0.1, 0.3):
        r = run_junction_sim(name="D", N=8, ep_frac=ep_frac)
        rows.append({
            "material": "D", "ep_frac": f"{ep_frac:.6g}",
            "ZT_over_ZL": f"{r['ZT_over_ZL_wo']:.6g}",
            "T0_1d": f"{r['T0_lin']:.6g}", "T0_2d": f"{r['T0_2d_wo']:.6g}",
            "T0_num": f"{r['T0_sim']:.6g}",
        })
    return _write("junction_scaling.csv",
                  ["material", "ep_frac", "ZT_over_ZL", "T0_1d", "T0_2d",
                   "T0_num"], rows, config="test3_junction")


# --------------------------------------------------------------------------- #
# Convergence (Test 4) and energy
# --------------------------------------------------------------------------- #
def export_conv():
    from validation.test4_convergence import run_case
    rows = []
    # Gripped is the baseline; keep previously produced frictionless rows as a
    # low-priority sensitivity (cheap; no new frictionless cases beyond this).
    for mode in ("gripped", "frictionless"):
        cases = [run_case(n_s, mode) for n_s in (10, 20, 40, 80)]
        w_ref = cases[-1]["w_max_over_R"]
        e_ref = cases[-1]["eta"]
        for c in cases:
            rows.append({
                "contact": mode, "n_s": c["n_s"],
                "wmax_over_R": f"{c['w_max_over_R']:.6g}",
                "d_wmax_pct": f"{_pct(c['w_max_over_R'], w_ref):.4g}",
                "n_failed": c["n_failures"],
                "eta": f"{c['eta']:.6g}",
                "d_eta_pct": f"{_pct(c['eta'], e_ref):.4g}",
                "R_d": f"{c['R_d']:.6g}", "cpu_s": f"{c['cpu']:.4g}",
                "drone_x_arrest": f"{c.get('drone_x_arrest', float('nan')):.6g}",
                "drone_y_arrest": f"{c.get('drone_y_arrest', float('nan')):.6g}",
                "m_net_g": f"{1e3 * c.get('m_net', float('nan')):.6g}",
                "Kd_arrest_J": f"{c.get('Kd_arrest', float('nan')):.6g}",
                "Knet_arrest_J": f"{c.get('Knet_arrest', float('nan')):.6g}",
                "Uel_arrest_J": f"{c.get('Uel_arrest', float('nan')):.6g}",
                "Ufail_arrest_J": f"{c.get('Ufail_arrest', float('nan')):.6g}",
                "Ucontact_arrest_J": f"{c.get('Ucontact_arrest', float('nan')):.6g}",
                "energy_error_J": f"{c.get('energy_error_J', float('nan')):.6g}",
            })
    cols = ["contact", "n_s", "wmax_over_R", "d_wmax_pct", "n_failed", "eta",
            "d_eta_pct", "R_d", "cpu_s", "drone_x_arrest", "drone_y_arrest",
            "m_net_g", "Kd_arrest_J", "Knet_arrest_J", "Uel_arrest_J",
            "Ufail_arrest_J", "Ucontact_arrest_J", "energy_error_J"]
    return _write("tab_conv.csv", cols, rows, config="test4_convergence")


def export_energy_conv():
    from netsim.config import (SimConfig, MaterialConfig, NetConfig,
                               DroneConfig, NumericsConfig, ContactConfig,
                               OutputConfig)
    from netsim.simulate import simulate_config
    rows = []
    for mode in ("gripped", "frictionless"):
        for C in (0.5, 0.25, 0.125):
            cfg = SimConfig(
                material=MaterialConfig(name="S"),
                net=NetConfig(kind="star", N=8, R=1.0, eps_p=0.0, A_hat=1e-6),
                drone=DroneConfig(M=0.2, r_d=0.15, v0=5.0, p=(0.5, 0.0),
                                  gravity=False),
                numerics=NumericsConfig(n_s=20, C=C, t_end=0.06, dt_out=2e-4,
                                        use_numba=False),
                contact=ContactConfig(mode=mode, k_c=1e7),
                output=OutputConfig(hdf5=None, R_max=0.5, k_max=10),
            )
            res = simulate_config(cfg, write=False)
            rows.append({"contact": mode, "C": f"{C:.6g}",
                         "dt": f"{res.trajectory.dt:.6g}",
                         "energy_error": f"{res.energy_error:.6g}"})
    return _write("energy_conv.csv", ["contact", "C", "dt", "energy_error"],
                  rows, config="reference")


# --------------------------------------------------------------------------- #
# Dynamic overload (5a)
# --------------------------------------------------------------------------- #
def export_daf():
    from netsim.overload import run_overload, analytic_eps_m_over_eps0
    regimes = {"D": 0.01, "S": round(0.8 * get_material("S").eps_b, 4)}
    rows = []
    for mat_name, eps0 in regimes.items():
        mat = get_material(mat_name)
        for N in (4, 8, 16):
            r = run_overload(N, mat, eps0=eps0, m_hub=0.01, n_s=2,
                             constrain_hub_z=True)
            # Analytical eps_m/eps0 for S at the measured n_eff (energy identity).
            eps_an = ""
            if mat_name == "S":
                eps_an = f"{analytic_eps_m_over_eps0(N, r.n_eff):.6g}"
            rows.append({
                "material": mat_name, "N": N,
                "n_eff_num": f"{r.n_eff:.6g}",
                "eps_s_over_eps0": f"{r.eps_s_over_eps0:.6g}",
                "eps_m_over_eps0": f"{r.eps_m_over_eps0:.6g}",
                "eps_m_over_eps0_an": eps_an,
                "m_hub": f"{0.01:.6g}",
            })
    return _write("tab_daf.csv",
                  ["material", "N", "n_eff_num", "eps_s_over_eps0",
                   "eps_m_over_eps0", "eps_m_over_eps0_an", "m_hub"],
                  rows, config="test_daf")


def export_daf_ramp():
    from netsim.overload import run_overload, daf_sdof_nonlinear
    mat = get_material("D")
    rows = []
    r0 = run_overload(8, mat, eps0=0.01, m_hub=0.01, n_s=2,
                      constrain_hub_z=True)
    n_eff = r0.n_eff
    eps_s_ratio = r0.eps_s_over_eps0
    for tf in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0):
        r = run_overload(8, mat, eps0=0.01, m_hub=0.01, n_s=2, t_f_over_Tn=tf,
                         dyn_periods=16.0, constrain_hub_z=True)
        x = math.pi * tf
        daf_an = 2.0 if tf == 0.0 else 1.0 + abs(math.sin(x)) / x
        daf_num = r.eps_m / r.eps_s if r.eps_s > 0 else float("nan")
        daf_nl = daf_sdof_nonlinear(n_eff, tf, eps_s_over_eps0=eps_s_ratio)
        rows.append({"N": 8, "material": "D", "tf_over_Tn": f"{tf:.6g}",
                     "eps_m_over_eps0": f"{r.eps_m/r.eps0_target:.6g}",
                     "daf_an": f"{daf_an:.6g}",
                     "daf_num": f"{daf_num:.6g}",
                     "daf_sdof_nl": f"{daf_nl:.6g}"})
    return _write("fig_daf_ramp.csv",
                  ["N", "material", "tf_over_Tn", "eps_m_over_eps0", "daf_an",
                   "daf_num", "daf_sdof_nl"],
                  rows, config="test_daf")


# --------------------------------------------------------------------------- #
# Off-centre eta (results)
# --------------------------------------------------------------------------- #
def _etaa_case(job):
    """One (material, ep_frac, a) row for fig_etaa — process-pool worker."""
    mat, ep_frac, a = job
    from validation.offcentre import run_offcentre
    try:
        import offc
    except Exception:
        offc = None
    try:
        import offc_free
    except Exception:
        offc_free = None

    r = run_offcentre(material_name=mat, a_over_R=a, eps_p_frac=ep_frac,
                      n_s=20, continue_to_B=True)
    etaA_num = r.get("eta_A", r.get("eta_ff", float("nan")))
    etaB_num = r.get("eta_B", float("nan"))
    etaA_ref = etaB_ref = ""
    if offc is not None and a > 0:
        try:
            ref = offc.run(mat, 8, a, ep_frac)
            etaA_ref = f"{ref[0][2]:.6g}"
            if len(ref) > 1:
                etaB_ref = f"{ref[1][2]:.6g}"
        except Exception:
            pass
    etaB_str = f"{etaB_num:.6g}" if etaB_num == etaB_num else ""
    etaA_free_num = etaA_free_ref = px_fail = ""
    if a > 0:
        rf = run_offcentre(material_name=mat, a_over_R=a, eps_p_frac=ep_frac,
                           n_s=20, free_lateral=True, continue_to_B=False)
        if rf["eta_A"] == rf["eta_A"]:
            etaA_free_num = f"{rf['eta_A']:.6g}"
        if rf["px_fail"] == rf["px_fail"]:
            px_fail = f"{rf['px_fail']:.6g}"
        if offc_free is not None:
            try:
                fr = offc_free.run(mat, 8, a, ep_frac)
                etaA_free_ref = f"{fr[3]:.6g}"
            except Exception:
                pass
    etaB_free_num = px_second = ""
    if a > 0:
        rb = run_offcentre(material_name=mat, a_over_R=a, eps_p_frac=ep_frac,
                           n_s=20, continue_to_B=True,
                           free_lateral_after_failures=1)
        if rb["eta_B"] == rb["eta_B"]:
            etaB_free_num = f"{rb['eta_B']:.6g}"
        if rb.get("px_at_second_fail", float("nan")) == rb.get(
                "px_at_second_fail", float("nan")):
            pxs = rb.get("px_at_second_fail", float("nan"))
            if pxs == pxs:
                px_second = f"{pxs:.6g}"
    return {
        "net": "star", "material": mat, "ep_frac": f"{ep_frac:.6g}",
        "a_over_R": f"{a:.6g}",
        "etaA_num": f"{etaA_num:.6g}",
        "etaB_num": etaB_str,
        "etaA_ref": etaA_ref, "etaB_ref": etaB_ref,
        "first_failure": r.get("first_kind", "none"),
        "etaA_free_num": etaA_free_num,
        "etaA_free_ref": etaA_free_ref,
        "px_fail": px_fail,
        "etaB_free_num": etaB_free_num,
        "px_at_second_fail": px_second,
    }


def export_etaa():
    a_list = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5)
    jobs = [(mat, ep, a)
            for mat in ("S", "D")
            for ep in (0.0, 0.1)
            for a in a_list]
    rows = _parallel_map(_etaa_case, jobs, desc="fig_etaa")
    cols = ["net", "material", "ep_frac", "a_over_R", "etaA_num", "etaB_num",
            "etaA_ref", "etaB_ref", "first_failure",
            "etaA_free_num", "etaA_free_ref", "px_fail",
            "etaB_free_num", "px_at_second_fail"]
    return _write("fig_etaa.csv", cols, rows, config="offcentre")


def export_etaA_quasistatic():
    # Centre hub pull: all N radials stretch and reach eps_b together, so
    # analytically eta_an = 1 - Phi(eps_p)/Phi(eps_b) and the breaking deflection
    # is wb_an = R sqrt(((1+eps_b)/(1+eps_p))^2 - 1).
    from validation.offcentre import run_offcentre
    speed = 0.3  # matches run_offcentre default kinematic pull speed
    rows = []
    for mat in ("S", "D"):
        m = get_material(mat)
        for ep_frac in (0.0, 0.1, 0.3):
            eps_p = ep_frac * m.eps_b
            eta_an = 1.0 - float(m.Phi(eps_p)) / float(m.Phi(m.eps_b))
            wb_an = 1.0 * math.sqrt(((1 + m.eps_b) / (1 + eps_p)) ** 2 - 1.0)
            r = run_offcentre(material_name=mat, a_over_R=0.0,
                              eps_p_frac=ep_frac, n_s=20,
                              continue_to_B=False)
            wb_num = speed * r["t_first"] if "t_first" in r else float("nan")
            rows.append({
                "material": mat, "ep_frac": f"{ep_frac:.6g}",
                "eta_an": f"{eta_an:.6g}",
                "eta_num": f"{r.get('eta_A', r.get('eta_ff', float('nan'))):.6g}",
                "wb_an": f"{wb_an:.6g}", "wb_num": f"{wb_num:.6g}",
            })
    return _write("tab_etaA.csv",
                  ["material", "ep_frac", "eta_an", "eta_num", "wb_an",
                   "wb_num"], rows, config="offcentre")


def export_Fw():
    # Hub force-deflection: analytic + numerical (segment reaction on hub).
    from validation.offcentre import hub_force_deflection
    rows = []
    for mat_name in ("S", "D"):
        for r in hub_force_deflection(mat_name, eps_p_frac=0.1, n_s=20):
            rows.append({
                "material": r["material"],
                "w_over_R": f"{r['w_over_R']:.6g}",
                "F_an": f"{r['F_an']:.6g}",
                "F_num": f"{r['F_num']:.6g}",
                "failed": bool(r["failed"]),
            })
    return _write("fig_Fw.csv",
                  ["material", "w_over_R", "F_an", "F_num", "failed"], rows,
                  config="overload")


# --------------------------------------------------------------------------- #
# Algorithm 2 (5b)
# --------------------------------------------------------------------------- #
def _mmin_cfg(material, M, v0, net_kind="star", n_s=10):
    from netsim.config import (SimConfig, MaterialConfig, NetConfig,
                               DroneConfig, NumericsConfig, ContactConfig,
                               OutputConfig)
    # Comparable prestress for star and star+ring (round 5).
    ep = 0.1 * get_material(material).eps_b
    net = (NetConfig(kind="star_with_rings", N=8, R=1.0, eps_p=ep, A_hat=1e-6,
                     radii=[0.5, 1.0], q_ratio=1.0, fix_radii=True)
           if net_kind == "star+ring"
           else NetConfig(kind="star", N=8, R=1.0, eps_p=ep, A_hat=1e-6))
    return SimConfig(
        material=MaterialConfig(name=material), net=net,
        drone=DroneConfig(M=M, r_d=0.15, v0=v0, p=(0.0, 0.0)),
        numerics=NumericsConfig(n_s=n_s, C=0.5, t_end=0.5, dt_out=5e-3,
                                use_numba=True),
        contact=ContactConfig(mode="gripped", k_c=1e7,
                              k_c_mode=("relative" if material == "D"
                                        else "absolute"),
                              k_c_factor=8.0,
                              penetration_guard="reduce_dt"),
        output=OutputConfig(hdf5=None, R_max=0.5, k_max=10),
    )


def export_mmin_monotone():
    from netsim.mmin import MminConfig, monotonicity_scan
    rows = []
    for mat in ("S", "D"):
        cfg = _mmin_cfg(mat, 1.0, 15.0, net_kind="star+ring")
        for crit in ("A", "B_any", "B_loc"):
            mm = MminConfig(criterion=crit, impact_points=[(0.5, 0.0)],
                            R_max=0.5)
            sc = monotonicity_scan(cfg, mm, (0.5, 0.0), n=10)
            for s, ok in zip(sc["s_values"], sc["pattern"]):
                rows.append({"material": mat, "criterion": crit,
                             "s_over_shi": f"{s/sc['s_hi']:.6g}",
                             "pass": bool(ok)})
    return _write("mmin_monotone.csv",
                  ["material", "criterion", "s_over_shi", "pass"], rows,
                  config="mmin")


_A_WORST = (0.5, 0.0)
_MMIN_CRIT_KEYS = ("A", "Bany", "Bloc025", "Bloc050", "Bloc075")
_MMIN_TOL = 0.01


def _mmin_n_procs():
    """Inner Alg.2 workers; default 1 when the outer export pool is wide."""
    return max(1, int(os.environ.get("MMIN_N_PROCS", "1")))


def _mmin_row_cols():
    cols = ["net", "material", "n_s", "M", "v0", "Ekin", "m_lower_g",
            "mA_min_g", "mBany_min_g", "mBloc025_min_g", "mBloc050_min_g",
            "mBloc075_min_g", "ratio_Bany", "ratio_Bloc025", "ratio_Bloc050",
            "ratio_Bloc075", "mA_over_Ekin_g_per_J",
            "mBany_over_Ekin_g_per_J", "mBloc025_over_Ekin_g_per_J",
            "mBloc050_over_Ekin_g_per_J", "mBloc075_over_Ekin_g_per_J",
            "n_broken_A", "n_broken_Bany", "n_broken_Bloc025",
            "n_broken_Bloc050", "n_broken_Bloc075",
            "n_failed_segments_A", "n_failed_segments_Bany",
            "n_failed_segments_Bloc025", "n_failed_segments_Bloc050",
            "n_failed_segments_Bloc075",
            "worst_p_x", "worst_p_y", "m_source"]
    for key in _MMIN_CRIT_KEYS:
        cols += [f"worst_p_x_{key}", f"worst_p_y_{key}",
                 f"n_broken_{key}_at_Aworst", f"outcome_below_{key}",
                 f"s_lo_{key}", f"s_hi_{key}", f"n_bisect_{key}"]
    return cols


def _mmin_case(job):
    """One (net, material, M, v0, full[, n_s]) row — process-pool worker."""
    if len(job) == 6:
        net_kind, mat, M, v0, full, n_s = job
    else:
        net_kind, mat, M, v0, full = job
        n_s = 10 if full else 8
    from netsim.mmin import MminConfig, minimum_mass, evaluate
    from netsim.simulate import build_net

    m = get_material(mat)
    pts = [(0.0, 0.0), (0.25, 0.0), (0.5, 0.0)]
    R = 1.0
    cfg = _mmin_cfg(mat, M, v0, net_kind=net_kind, n_s=int(n_s))
    net = build_net(cfg)
    Ekin = 0.5 * M * v0 ** 2
    m_lower = Ekin / m.e_mat
    n_inner = _mmin_n_procs()
    cache = str(_MMIN_CACHE / f"{net_kind}_{mat}_M{M:g}_v{v0:g}_ns{int(n_s)}")
    mmA = MminConfig(criterion="A", tol=_MMIN_TOL, impact_points=pts,
                     n_procs=n_inner, cache_dir=cache)
    mm_any = MminConfig(criterion="B_any", tol=_MMIN_TOL, impact_points=pts,
                        n_procs=n_inner, n_scan=24, cache_dir=cache)
    mm_locs = {
        frac: MminConfig(criterion="B_loc", tol=_MMIN_TOL, impact_points=pts,
                         n_procs=n_inner, n_scan=24, R_max=frac * R,
                         cache_dir=cache)
        for frac in (0.25, 0.5, 0.75)
    }
    MminConfig.assert_same_tol(mmA, mm_any, *mm_locs.values())
    rA = minimum_mass(cfg, mmA)
    r_any = minimum_mass(cfg, mm_any, s_cap=rA.s_min)
    blocs = {frac: minimum_mass(cfg, mm, s_cap=rA.s_min)
             for frac, mm in mm_locs.items()}

    named = {
        "A": (mmA, rA),
        "Bany": (mm_any, r_any),
        "Bloc025": (mm_locs[0.25], blocs[0.25]),
        "Bloc050": (mm_locs[0.5], blocs[0.5]),
        "Bloc075": (mm_locs[0.75], blocs[0.75]),
    }

    extra = {}
    for key, (mm, rx) in named.items():
        wp = rx.worst_point
        extra[f"worst_p_x_{key}"] = f"{wp[0]:.6g}"
        extra[f"worst_p_y_{key}"] = f"{wp[1]:.6g}"
        info_aw = evaluate(cfg, float(rx.s_min), _A_WORST, mm, 2)
        extra[f"n_broken_{key}_at_Aworst"] = int(
            info_aw.get("n_failed_threads", info_aw.get("n_failures", 0)))
        s_lo = rx.s_below if rx.s_below is not None else 0.99 * rx.s_min
        if s_lo >= rx.s_min:
            s_lo = 0.99 * rx.s_min
        wp_idx = next((i for i, q in enumerate(pts)
                       if abs(q[0] - wp[0]) < 1e-12
                       and abs(q[1] - wp[1]) < 1e-12), 0)
        info_lo = evaluate(cfg, float(s_lo), wp, mm, wp_idx)
        extra[f"outcome_below_{key}"] = info_lo.get("outcome", "")
        extra[f"s_lo_{key}"] = f"{(rx.s_lo if rx.s_lo is not None else s_lo):.6g}"
        extra[f"s_hi_{key}"] = f"{(rx.s_hi if rx.s_hi is not None else rx.s_min):.6g}"
        extra[f"n_bisect_{key}"] = int(rx.n_bisect)

    def g(x):
        return 1e3 * x

    def ratio(mb):
        return rA.m_min / mb if mb > 0 else float("nan")

    mA = rA.m_min
    worst = rA.worst_point
    row = {
        "net": net_kind, "material": mat, "M": f"{M:.6g}", "v0": f"{v0:.6g}",
        "Ekin": f"{Ekin:.6g}",
        "m_lower_g": f"{g(m_lower):.6g}",
        "mA_min_g": f"{g(mA):.6g}",
        "mBany_min_g": f"{g(r_any.m_min):.6g}",
        "mBloc025_min_g": f"{g(blocs[0.25].m_min):.6g}",
        "mBloc050_min_g": f"{g(blocs[0.5].m_min):.6g}",
        "mBloc075_min_g": f"{g(blocs[0.75].m_min):.6g}",
        "ratio_Bany": f"{ratio(r_any.m_min):.6g}",
        "ratio_Bloc025": f"{ratio(blocs[0.25].m_min):.6g}",
        "ratio_Bloc050": f"{ratio(blocs[0.5].m_min):.6g}",
        "ratio_Bloc075": f"{ratio(blocs[0.75].m_min):.6g}",
        "mA_over_Ekin_g_per_J": f"{(g(mA)/Ekin):.4g}",
        "mBany_over_Ekin_g_per_J": f"{(g(r_any.m_min)/Ekin):.4g}",
        "mBloc025_over_Ekin_g_per_J": f"{(g(blocs[0.25].m_min)/Ekin):.4g}",
        "mBloc050_over_Ekin_g_per_J": f"{(g(blocs[0.5].m_min)/Ekin):.4g}",
        "mBloc075_over_Ekin_g_per_J": f"{(g(blocs[0.75].m_min)/Ekin):.4g}",
        "n_broken_A": rA.n_failures,
        "n_broken_Bany": r_any.n_failures,
        "n_broken_Bloc025": blocs[0.25].n_failures,
        "n_broken_Bloc050": blocs[0.5].n_failures,
        "n_broken_Bloc075": blocs[0.75].n_failures,
        "n_failed_segments_A": rA.n_failed_segments,
        "n_failed_segments_Bany": r_any.n_failed_segments,
        "n_failed_segments_Bloc025": blocs[0.25].n_failed_segments,
        "n_failed_segments_Bloc050": blocs[0.5].n_failed_segments,
        "n_failed_segments_Bloc075": blocs[0.75].n_failed_segments,
        "worst_p_x": f"{worst[0]:.6g}",
        "worst_p_y": f"{worst[1]:.6g}",
        "n_s": str(int(n_s)),
        "m_source": "computed",
    }
    row.update(extra)
    print(
        f"    [tab_mmin] {net_kind} {mat} M={M:g} v0={v0:g} n_s={n_s} "
        f"mA={g(mA):.4g}g A@({worst[0]:.2f},{worst[1]:.2f}) "
        f"Bany@({r_any.worst_point[0]:.2f},{r_any.worst_point[1]:.2f}) "
        f"nB={r_any.n_failures} nB_Aworst={extra['n_broken_Bany_at_Aworst']}",
        flush=True)
    return row


_STAR40_COMPUTED = (
    (1.0, 15.0),   # re-eval from cache
    (0.25, 20.0),  # re-eval from cache; scale m ∝ M for other M at v0=20
    (1.0, 10.0),   # only new production runs
)


def _scale_mmin_row(src, M_new, note):
    """Fill a sibling M at the same v0 by m ∝ M (s and m_min scale with M)."""
    M_src = float(src["M"])
    fac = float(M_new) / M_src
    row = dict(src)
    row["M"] = f"{M_new:.6g}"
    row["Ekin"] = f"{fac * float(src['Ekin']):.6g}"
    for k in ("m_lower_g", "mA_min_g", "mBany_min_g", "mBloc025_min_g",
              "mBloc050_min_g", "mBloc075_min_g"):
        row[k] = f"{fac * float(src[k]):.6g}"
    for key in _MMIN_CRIT_KEYS:
        for pref in ("s_lo", "s_hi"):
            col = f"{pref}_{key}"
            if src.get(col) not in (None, ""):
                row[col] = f"{fac * float(src[col]):.6g}"
    row["m_source"] = note
    return row


def _assemble_star40(computed):
    by = {(r["material"], float(r["M"]), float(r["v0"])): r
          for r in computed}
    out = []
    plan = (
        (15.0, 1.0, (0.25, 2.0), "scaled_m_prop_M from M=1 v0=15"),
        (20.0, 0.25, (1.0, 2.0), "scaled_m_prop_M from M=0.25 v0=20"),
        (10.0, 1.0, (0.25, 2.0), "scaled_m_prop_M from M=1 v0=10"),
    )
    for mat in ("S", "D"):
        for v0, Msrc, others, note in plan:
            src = by[(mat, Msrc, v0)]
            out.append(src)
            for M in others:
                out.append(_scale_mmin_row(src, M, note))
    out.sort(key=lambda r: (r["material"], float(r["M"]), float(r["v0"])))
    return out


def export_mmin_round7c(name="tab_mmin.csv", computed_star=None,
                        computed_ring=None):
    """Star n_s=40 (3 computed (M,v0) + m∝M fill) and star+ring n_s=10."""
    n_copied = merge_tagged_mmin_caches()
    if n_copied:
        print(f"    merged {n_copied} tagged cache files into untagged dirs",
              flush=True)
    grid = [(0.25, 10.0), (0.25, 15.0), (0.25, 20.0),
            (1.0, 10.0), (1.0, 15.0), (1.0, 20.0),
            (2.0, 10.0), (2.0, 15.0), (2.0, 20.0)]
    if computed_ring is None:
        ring_jobs = [("star+ring", mat, M, v0, True, 10)
                     for mat in ("S", "D") for M, v0 in grid]
        computed_ring = _parallel_map(
            _mmin_case, ring_jobs, desc="tab_mmin star+ring n_s=10")
    if computed_star is None:
        star_jobs = [("star", mat, M, v0, True, 40)
                     for mat in ("S", "D") for M, v0 in _STAR40_COMPUTED]
        computed_star = _parallel_map(
            _mmin_case, star_jobs, desc="tab_mmin star n_s=40 keys")
    star_rows = _assemble_star40(computed_star)
    rows = list(star_rows) + list(computed_ring)
    return _write(name, _mmin_row_cols(), rows, config="mmin")


def export_mmin(full=False, n_s=None, nets=None, name="tab_mmin.csv"):
    # Star n_s=40 + star+ring n_s=10 (round 7c) on --full.
    if full and n_s is None and nets is None:
        return export_mmin_round7c(name=name)
    if full:
        grid = [(0.25, 10.0), (0.25, 15.0), (0.25, 20.0),
                (1.0, 10.0), (1.0, 15.0), (1.0, 20.0),
                (2.0, 10.0), (2.0, 15.0), (2.0, 20.0)]
    else:
        grid = [(0.25, 20.0), (2.0, 20.0)]
    if n_s is None:
        n_s = 10 if full else 8
    if nets is None:
        nets = ("star", "star+ring")
    jobs = [(net, mat, M, v0, full, n_s)
            for net in nets
            for mat in ("S", "D")
            for M, v0 in grid]
    rows = _parallel_map(_mmin_case, jobs, desc=name)
    return _write(name, _mmin_row_cols(), rows, config="mmin")


# --------------------------------------------------------------------------- #
# Dynamic runs and phase maps (reduced by default)
# --------------------------------------------------------------------------- #
def _dyn_case(job):
    """One (material, M, v0, a, s) row for dyn_runs — process-pool worker."""
    mat, M, v0, a, s = job
    from netsim.config import (SimConfig, MaterialConfig, NetConfig,
                               DroneConfig, NumericsConfig, ContactConfig,
                               OutputConfig)
    from netsim.simulate import simulate_config, build_net

    m = get_material(mat)
    ep = 0.1 * m.eps_b
    cfg = SimConfig(
        material=MaterialConfig(name=mat),
        net=NetConfig(kind="star_with_rings", N=8, R=1.0,
                      eps_p=ep, A_hat=1e-6,
                      radii=[0.5, 1.0], q_ratio=1.0,
                      fix_radii=True),
        drone=DroneConfig(M=M, r_d=0.15, v0=v0, p=(a, 0.0)),
        numerics=NumericsConfig(
            n_s=10, C=0.5, t_end=0.25, dt_out=5e-3,
            area_scale=s, use_numba=True,
            damping=0.0),
        contact=ContactConfig(
            mode="gripped", k_c=1e7,
            k_c_mode=("relative" if mat == "D" else "absolute"),
            k_c_factor=8.0,
            penetration_guard="reduce_dt"),
        output=OutputConfig(hdf5=None, R_max=0.5, k_max=10),
    )
    net = build_net(cfg)
    m_net = net.net_mass(m.rho, s)
    res = simulate_config(cfg, write=False)
    PiE = (0.5 * M * v0 ** 2) / (m_net * m.e_mat) if m_net > 0 else float("nan")
    return {
        "material": mat, "net": "star+ring",
        "M": f"{M:.6g}", "v0": f"{v0:.6g}",
        "a_over_R": f"{a:.6g}", "s": f"{s:.6g}",
        "m_net": f"{m_net:.6g}",
        "PiE_M_over_m": f"{PiE:.6g}",
        "arrested": bool(res.arrested),
        "wmax_over_R": f"{res.w_max:.6g}",
        "eta": f"{res.eta:.6g}",
        "n_failed": res.n_failures,
        "R_d": f"{res.R_d:.6g}",
        "cascade": bool(res.cascade),
        "energy_error": f"{res.energy_error:.6g}",
        "drone_x_arrest": f"{res.drone_x_arrest:.6g}",
        "drone_y_arrest": f"{res.drone_y_arrest:.6g}",
    }


def export_dyn_runs(full=False):
    Ms = (0.25, 1.0, 2.0) if full else (1.0,)
    v0s = (10.0, 15.0, 20.0) if full else (15.0,)
    a_list = (0.0, 0.25, 0.5) if full else (0.0, 0.5)
    s_list = (0.5, 1.0, 1.5) if full else (1.0,)
    jobs = [(mat, M, v0, a, s)
            for mat in ("S", "D")
            for M in Ms
            for v0 in v0s
            for a in a_list
            for s in s_list]
    rows = _parallel_map(_dyn_case, jobs, desc="dyn_runs")
    cols = ["material", "net", "M", "v0", "a_over_R", "s", "m_net",
            "PiE_M_over_m", "arrested", "wmax_over_R", "eta", "n_failed",
            "R_d", "cascade", "energy_error",
            "drone_x_arrest", "drone_y_arrest"]
    return _write("dyn_runs.csv", cols, rows, config="dyn")


def _phase_break_rows(res):
    traj = res.trajectory
    break_info = {}
    if traj.failures.size:
        order = np.argsort(traj.failures[:, 2])
        dxy = traj.failure_drone_xy
        for k, idx in enumerate(order):
            parent = int(traj.failures[idx, 1])
            t_b = float(traj.failures[idx, 2])
            if parent not in break_info:
                dx = float(dxy[idx, 0]) if dxy.shape[0] > idx else float("nan")
                dy = float(dxy[idx, 1]) if dxy.shape[0] > idx else float("nan")
                break_info[parent] = (t_b, k + 1, dx, dy)
    net = res.net
    seg_rows = []
    for e in range(net.n_e):
        i, j = int(net.edges[e, 0]), int(net.edges[e, 1])
        x1, y1 = net.nodes[i]
        x2, y2 = net.nodes[j]
        info = break_info.get(e)
        seg_rows.append({
            "seg": e, "parent": e, "x1": f"{x1:.6g}",
            "y1": f"{y1:.6g}", "x2": f"{x2:.6g}", "y2": f"{y2:.6g}",
            "broken": bool(info is not None),
            "t_break": (f"{info[0]:.6g}" if info else ""),
            "break_order": (f"{info[1]}" if info else ""),
            "drone_x_break": (f"{info[2]:.6g}" if info else ""),
            "drone_y_break": (f"{info[3]:.6g}" if info else ""),
        })
    return seg_rows, break_info


def _s_bany_for(net_kind, mat, M=1.0, v0=20.0, n_s=10):
    from netsim.mmin import MminConfig, minimum_mass
    pts = [(0.0, 0.0), (0.25, 0.0), (0.5, 0.0)]
    cfg = _mmin_cfg(mat, M, v0, net_kind=net_kind, n_s=n_s)
    cache = str(_MMIN_CACHE / f"{net_kind}_{mat}_M{M:g}_v{v0:g}_ns{n_s}")
    mmA = MminConfig(criterion="A", tol=_MMIN_TOL, impact_points=pts,
                     n_procs=_mmin_n_procs(), cache_dir=cache)
    rA = minimum_mass(cfg, mmA)
    mmB = MminConfig(criterion="B_any", tol=_MMIN_TOL, impact_points=pts,
                     n_procs=_mmin_n_procs(), n_scan=24, cache_dir=cache)
    MminConfig.assert_same_tol(mmA, mmB)
    rB = minimum_mass(cfg, mmB, s_cap=rA.s_min)
    return rB, rA


def export_phase_maps():
    """Maps at global ``s_Bany`` and the A-worst impact (0.5, 0)."""
    from netsim.config import (SimConfig, MaterialConfig, NetConfig,
                               DroneConfig, NumericsConfig, ContactConfig,
                               OutputConfig)
    from netsim.simulate import simulate_config
    d = OUT_DIR / "fig_phase_maps"
    d.mkdir(parents=True, exist_ok=True)
    notes = []
    for net_kind in ("star", "star+ring"):
        for mat in ("S", "D"):
            run_name = f"phase_{net_kind.replace('+', '_')}_{mat}"
            h5 = str(d / f"{run_name}.h5")
            rB, rA = _s_bany_for(net_kind, mat, M=1.0, v0=20.0, n_s=10)
            s_map = rB.s_min
            p = _A_WORST
            ep = 0.1 * get_material(mat).eps_b
            net_cfg = (NetConfig(kind="star_with_rings", N=8, R=1.0, eps_p=ep,
                                 A_hat=1e-6, radii=[0.5, 1.0], q_ratio=1.0,
                                 fix_radii=True)
                       if net_kind == "star+ring"
                       else NetConfig(kind="star", N=8, R=1.0, eps_p=ep,
                                      A_hat=1e-6))
            cfg = SimConfig(
                material=MaterialConfig(name=mat),
                net=net_cfg,
                drone=DroneConfig(M=1.0, r_d=0.15, v0=20.0, p=p),
                numerics=NumericsConfig(n_s=10, C=0.5, t_end=0.25, dt_out=5e-3,
                                        area_scale=float(s_map),
                                        use_numba=True),
                contact=ContactConfig(mode="gripped", k_c=1e7,
                                      k_c_mode=("relative" if mat == "D"
                                                else "absolute"),
                                      k_c_factor=8.0,
                                      penetration_guard="reduce_dt"),
                output=OutputConfig(hdf5=h5, hdf5_light=True,
                                    hdf5_max_frames=50, R_max=0.5, k_max=10),
            )
            res = simulate_config(cfg, write=True)
            seg_rows, break_info = _phase_break_rows(res)
            n_broken = sum(1 for r in seg_rows if r["broken"] in (True, "True"))
            why = ""
            if n_broken == 0:
                why = (f"no broken segs at A-worst {p} with global s_Bany="
                       f"{s_map:.6g} (B-bind {rB.worst_point}, "
                       f"A-bind {rA.worst_point}, n_fail={res.n_failures}, "
                       f"outcome={res.outcome}); s_Bany is set by a different "
                       f"impact than (0.5, 0), so this thicker net can pass A")
                notes.append(f"{run_name}: {why}")
            print(f"    [phase] {run_name} s_Bany={s_map:.4g} "
                  f"B@({rB.worst_point[0]:.2f},{rB.worst_point[1]:.2f}) "
                  f"broken={n_broken} outcome={res.outcome}", flush=True)
            with open(d / f"segments_{run_name}.csv", "w", newline="") as fh:
                fh.write(f"# commit={_COMMIT} date={_DATE} config=phase "
                         f"net={net_kind} s_Bany={s_map:.6g} "
                         f"impact=({p[0]:.2f},{p[1]:.2f}) "
                         f"B_bind=({rB.worst_point[0]:.4g},"
                         f"{rB.worst_point[1]:.4g}) "
                         f"n_broken={n_broken} code_dirty="
                         f"{_header_dirty_flag()}\n")
                if why:
                    fh.write(f"# note={why}\n")
                w = csv.DictWriter(
                    fh, fieldnames=["seg", "parent", "x1", "y1", "x2", "y2",
                                    "broken", "t_break", "break_order",
                                    "drone_x_break", "drone_y_break"])
                w.writeheader()
                w.writerows(seg_rows)
    if notes:
        (d / "phase_maps_note.txt").write_text("\n".join(notes) + "\n")
    else:
        (d / "phase_maps_note.txt").write_text(
            "broken segments present at A-worst (0.5, 0) for all four maps\n")
    return d


def _t_first_contact(traj):
    energy = getattr(traj, "energy", None)
    if energy is None or np.asarray(energy).size == 0:
        return float("nan")
    uc = np.asarray(energy)[:, 4]
    hit = np.nonzero(uc > 0)[0]
    return float(traj.t[int(hit[0])]) if hit.size else float("nan")


def _t_arrest(traj):
    drone = getattr(traj, "drone", None)
    if drone is None or not np.isfinite(drone).any():
        return float("nan")
    vz = np.asarray(drone)[:, 5]
    seen_neg = False
    for i, vzi in enumerate(vz):
        if not np.isfinite(vzi):
            continue
        if vzi < 0:
            seen_neg = True
        elif seen_neg:
            return float(traj.t[i])
    return float("nan")


def export_runs_round7():
    """Light HDF5 pilot: phase-map cases plus a few dyn rows."""
    import time
    from netsim.config import (SimConfig, MaterialConfig, NetConfig,
                               DroneConfig, NumericsConfig, ContactConfig,
                               OutputConfig)
    from netsim.simulate import simulate_config
    d = OUT_DIR / "runs_round7"
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    write_s = 0.0
    sim_s = 0.0
    skip_dyn = False

    def _record(run_id, path, net_kind, mat, M, v0, a, s, n_s, res, n_full):
        traj = res.trajectory
        t_fail = (float(traj.failures[0, 2])
                  if traj.failures.size else float("nan"))
        n_kept = int(min(50, n_full))
        rows.append({
            "run_id": run_id, "path": str(Path(path).relative_to(OUT_DIR)),
            "net": net_kind, "material": mat,
            "M": f"{M:.6g}", "v0": f"{v0:.6g}", "a": f"{a:.6g}",
            "s": f"{s:.6g}", "n_s": n_s,
            "n_failed": res.n_failures,
            "arrested": bool(res.arrested),
            "outcome": res.outcome,
            "t_first_contact": f"{_t_first_contact(traj):.6g}",
            "t_first_fail": f"{t_fail:.6g}" if t_fail == t_fail else "",
            "t_arrest": f"{_t_arrest(traj):.6g}",
            "n_frames_kept": n_kept,
            "n_frames_full": n_full,
        })

    # Phase-map cases already written by export_phase_maps; copy index only
    # if those files exist, otherwise re-run a thin set.
    maps = OUT_DIR / "fig_phase_maps"
    for net_kind, mat in (("star", "S"), ("star", "D"),
                          ("star+ring", "S"), ("star+ring", "D")):
        name = f"phase_{net_kind.replace('+', '_')}_{mat}"
        src = maps / f"{name}.h5"
        if src.is_file():
            rows.append({
                "run_id": name, "path": f"fig_phase_maps/{name}.h5",
                "net": net_kind, "material": mat,
                "M": "1", "v0": "20", "a": "0.5", "s": "", "n_s": 10,
                "n_failed": "", "arrested": "", "outcome": "",
                "t_first_contact": "", "t_first_fail": "", "t_arrest": "",
                "n_frames_kept": "", "n_frames_full": "",
            })

    pilots = [(mat, a) for mat in ("S", "D") for a in (0.0, 0.25, 0.5)]
    for mat, a in pilots:
        if skip_dyn:
            break
        run_id = f"dyn_{mat}_a{a:g}"
        h5 = str(d / f"{run_id}.h5")
        m = get_material(mat)
        ep = 0.1 * m.eps_b
        cfg = SimConfig(
            material=MaterialConfig(name=mat),
            net=NetConfig(kind="star_with_rings", N=8, R=1.0, eps_p=ep,
                          A_hat=1e-6, radii=[0.5, 1.0], q_ratio=1.0,
                          fix_radii=True),
            drone=DroneConfig(M=1.0, r_d=0.15, v0=15.0, p=(a, 0.0)),
            numerics=NumericsConfig(n_s=10, C=0.5, t_end=0.25, dt_out=5e-3,
                                    area_scale=1.0, use_numba=True),
            contact=ContactConfig(mode="gripped", k_c=1e7,
                                  k_c_mode=("relative" if mat == "D"
                                            else "absolute"),
                                  k_c_factor=8.0,
                                  penetration_guard="reduce_dt"),
            output=OutputConfig(hdf5=None, R_max=0.5, k_max=10),
        )
        t0 = time.perf_counter()
        res = simulate_config(cfg, write=False)
        sim_s += time.perf_counter() - t0
        n_full = int(res.trajectory.t.size)
        cfg.output.hdf5 = h5
        cfg.output.hdf5_light = True
        cfg.output.hdf5_max_frames = 50
        t1 = time.perf_counter()
        from netsim.io_hdf5 import write_run
        from netsim.discretize import discretize
        disc = discretize(res.net, res.material, 10, area_scale=1.0, r_d=0.15)
        write_run(h5, res.net, disc, res.material, cfg, res.trajectory,
                  eta=res.eta, cascade=res.cascade, light=True, max_frames=50)
        dt_w = time.perf_counter() - t1
        write_s += dt_w
        overhead = write_s / max(sim_s, 1e-9)
        print(f"    [runs_round7] {run_id} write={dt_w:.3f}s "
              f"overhead={100*overhead:.2f}%", flush=True)
        if overhead > 0.05 and len(rows) >= 2:
            skip_dyn = True
            print("    [runs_round7] write overhead > 5%; skipping rest",
                  flush=True)
        _record(run_id, h5, "star+ring", mat, 1.0, 15.0, a, 1.0, 10,
                res, n_full)

    path = _write("runs_round7_index.csv",
                  ["run_id", "path", "net", "material", "M", "v0", "a", "s",
                   "n_s", "n_failed", "arrested", "outcome",
                   "t_first_contact", "t_first_fail", "t_arrest",
                   "n_frames_kept", "n_frames_full"],
                  rows, config="runs_round7")
    return path


def _etaa_ring_case(job):
    """One (material, q_ratio, a) row for fig_etaa_ring — process-pool worker."""
    mat, q_ratio, a = job
    from netsim.config import (SimConfig, MaterialConfig, NetConfig,
                               DroneConfig, NumericsConfig, ContactConfig,
                               OutputConfig, KinematicConfig)
    from netsim.simulate import simulate, build_net
    from netsim.discretize import discretize
    from validation.offcentre import _absorbed

    m = get_material(mat)
    ep = 0.1 * m.eps_b
    speed = 0.3
    cfg = SimConfig(
        material=MaterialConfig(name=mat),
        net=NetConfig(kind="star_with_rings", N=8, R=1.0,
                      eps_p=ep, A_hat=1e-6, radii=[0.5, 1.0],
                      q_ratio=q_ratio, fix_radii=True),
        drone=DroneConfig(M=1.0, r_d=0.15, v0=0.0),
        numerics=NumericsConfig(n_s=10, C=0.5, t_end=5.0 / speed,
                                dt_out=2e-3, damping=400.0,
                                use_numba=False),
        contact=ContactConfig(mode="gripped", k_c=1e7),
        output=OutputConfig(hdf5=None, R_max=10.0, k_max=1e4),
    )
    net = build_net(cfg)
    disc = discretize(net, m, 10, r_d=0.15)
    target = np.array([a, 0.0])
    free = np.nonzero(~disc.anchored)[0]
    d2 = np.sum((disc.x0[free, :2] - target) ** 2, axis=1)
    pnode = int(free[int(np.argmin(d2))])
    cfg.kinematic = KinematicConfig(
        enabled=True, node=pnode, mode="displacement",
        direction=(0.0, 0.0, -1.0),
        func=lambda t, s=speed: s * t)
    # Stop after two time-separated failure events (A then B); do not run to t_end.
    res = simulate(net, m, cfg.drone, cfg.numerics, cfg.contact,
                   cfg.output, kinematic=cfg.kinematic,
                   stop_on_failure=False, stop_after_n_failures=2)
    traj = res.trajectory
    etaA = etaB = float("nan")
    first_kind = "none"
    if traj.failures.shape[0]:
        order = np.argsort(traj.failures[:, 2])
        tA = float(traj.failures[order[0], 2])
        before = np.nonzero(traj.t < tA)[0]
        iA = before[-1] if before.size else 0
        etaA = (float(traj.energy[iA, 2]) - traj.U_prestress) / (
            res.m_net * m.e_mat)
        parent0 = int(traj.failures[order[0], 1])
        first_kind = "rad" if parent0 != 0 else "in"
        gap = 1e-4
        for j in order[1:]:
            tB = float(traj.failures[j, 2])
            if tB > tA + gap:
                beforeB = np.nonzero(traj.t < tB)[0]
                iB = beforeB[-1] if beforeB.size else iA
                etaB = _absorbed(traj, iB, res.m_net, m.e_mat)
                break
    return {
        "net": "star+ring", "material": mat,
        "ep_frac": f"{ep/m.eps_b:.6g}", "a_over_R": f"{a:.6g}",
        "etaA_num": f"{etaA:.6g}" if etaA == etaA else "",
        "etaB_num": f"{etaB:.6g}" if etaB == etaB else "",
        "etaA_ref": "", "etaB_ref": "",
        "first_failure": first_kind,
        "q_ratio": f"{q_ratio:.6g}",
    }


def export_etaa_ring():
    # Ring-net off-centre eta with the ring fixed at R/2 (fix_radii=True).
    a_list = (0.0, 0.1, 0.25, 0.5)
    jobs = [(mat, q, a)
            for mat in ("S", "D")
            for q in (0.5, 1.0)
            for a in a_list]
    rows = _parallel_map(_etaa_ring_case, jobs, desc="fig_etaa_ring")
    cols = ["net", "material", "ep_frac", "a_over_R", "etaA_num", "etaB_num",
            "etaA_ref", "etaB_ref", "first_failure", "q_ratio"]
    return _write("fig_etaa_ring.csv", cols, rows, config="offcentre_ring")


EXPORTERS = [
    ("params_used.csv", lambda full: export_params_used()),
    ("tab_smith.csv", lambda full: export_smith()),
    ("tab_smith_conv.csv", lambda full: export_smith_conv()),
    ("fig_smith.csv", lambda full: export_fig_smith()),
    ("tab_junction.csv", lambda full: export_junction()),
    ("junction_scaling.csv", lambda full: export_junction_scaling()),
    ("tab_conv.csv", lambda full: export_conv()),
    ("energy_conv.csv", lambda full: export_energy_conv()),
    ("tab_daf.csv", lambda full: export_daf()),
    ("fig_daf_ramp.csv", lambda full: export_daf_ramp()),
    ("fig_etaa.csv", lambda full: export_etaa()),
    ("fig_etaa_ring.csv", lambda full: export_etaa_ring()),
    ("tab_etaA.csv", lambda full: export_etaA_quasistatic()),
    ("fig_Fw.csv", lambda full: export_Fw()),
    ("mmin_monotone.csv", lambda full: export_mmin_monotone()),
    ("tab_mmin.csv", lambda full: export_mmin(full)),
    ("dyn_runs.csv", lambda full: export_dyn_runs(full)),
    ("fig_phase_maps/", lambda full: export_phase_maps()),
    ("runs_round7/", lambda full: export_runs_round7()),
]


def main(argv=None):
    global _N_JOBS, _ALLOW_DIRTY, _COMMIT, _DATE, _CODE_DIRTY
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true",
                    help="use the full (expensive) grids for mmin/dyn_runs")
    ap.add_argument("--only", default=None,
                    help="comma-separated list of CSV names to export")
    ap.add_argument("--jobs", type=int, default=0,
                    help="parallel workers for independent cases "
                         "(default: all CPUs)")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="export even with a dirty working tree "
                         "(header still records code_dirty from the "
                         "scoped netsim/validation/tests/configs check)")
    args = ap.parse_args(argv)

    _N_JOBS = max(1, args.jobs) if args.jobs > 0 else max(1, os.cpu_count() or 1)
    _ALLOW_DIRTY = bool(args.allow_dirty)
    # Provenance frozen at start. Env wins so a VM without .git can still
    # stamp the local commit / dirty flag that packed the tarball.
    import subprocess
    env_commit = os.environ.get("GIT_COMMIT", "").strip()
    if env_commit:
        _COMMIT = env_commit
    else:
        try:
            _COMMIT = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=str(_repo_root()),
                capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            _COMMIT = git_commit()
    _DATE = datetime.datetime.now().isoformat(timespec="seconds")
    env_dirty = os.environ.get("CODE_DIRTY", "").strip().lower()
    if env_dirty in ("true", "1", "yes"):
        _CODE_DIRTY = True
    elif env_dirty in ("false", "0", "no"):
        _CODE_DIRTY = False
    else:
        _CODE_DIRTY = _code_paths_dirty()
    dirty_tree = _working_tree_dirty()
    if dirty_tree and not _ALLOW_DIRTY:
        print("Refusing to export: Tema 2 working tree is dirty. "
              "Commit first, or pass --allow-dirty "
              "(CSV headers record code_dirty from the scoped check).",
              flush=True)
        sys.exit(2)
    only = set(args.only.split(",")) if args.only else None
    print(f"Exporting paper CSVs to {OUT_DIR} (commit {_COMMIT[:8]}, "
          f"jobs={_N_JOBS}, code_dirty="
          f"{'true' if _CODE_DIRTY else 'false'})")
    for name, fn in EXPORTERS:
        if only and name not in only:
            continue
        try:
            path = fn(args.full)
            print(f"  [ok]   {name:22s} -> {path}", flush=True)
        except Exception as exc:  # pragma: no cover
            import traceback
            print(f"  [FAIL] {name:22s} : {exc}", flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()
