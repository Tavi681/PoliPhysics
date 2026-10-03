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

# Process-pool width for independent export cases (overridden by --jobs).
_N_JOBS = max(1, os.cpu_count() or 1)
_ALLOW_DIRTY = False


def _working_tree_dirty() -> bool:
    """True if Tema 2 has uncommitted changes (tracked or untracked)."""
    import subprocess
    try:
        repo = _ROOT.parent if (_ROOT.parent / ".git").exists() else _ROOT
        pathspec = "Tema 2" if repo != _ROOT else "."
        r = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal",
             "--", pathspec],
            cwd=str(repo), capture_output=True, text=True, check=False)
        return bool(r.stdout.strip())
    except Exception:
        return True


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
    dirty = _working_tree_dirty()
    with open(path, "w", newline="") as fh:
        fh.write(f"# commit={_COMMIT} date={_DATE} config={config} "
                 f"dirty={'true' if dirty else 'false'}\n")
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
        ("k_c_factor", 8.0, "-", "relative-k_c factor for material D (round 4)"),
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
    for frac in (0.35, 0.45, 0.55):
        r = run_smith(500.0, material_name="S", n_resolved=400, frac_time=frac)
        tt = r["t"]
        xf = r["traj"].x[-1]
        edges = r["edges"]
        xs = r["xs"]
        Lx = r["Lx"]
        Xseg = 0.5 * (xs[:-1] + xs[1:]) - Lx / 2
        right = Xseg > 0
        mids = 0.5 * (xf[edges[:, 0]] + xf[edges[:, 1]])[right]
        for X, eps, mid in zip(r["X"], r["eps_profile"], mids):
            rows.append({"t": f"{tt:.6g}", "X": f"{X:.6g}", "eps": f"{eps:.6g}",
                         "x": f"{mid[0]:.6g}", "y": f"{mid[2]:.6g}"})
    return _write("fig_smith.csv", ["t", "X", "eps", "x", "y"], rows,
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
            })
    cols = ["contact", "n_s", "wmax_over_R", "d_wmax_pct", "n_failed", "eta",
            "d_eta_pct", "R_d", "cpu_s", "drone_x_arrest", "drone_y_arrest"]
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
            "etaA_free_num", "etaA_free_ref", "px_fail"]
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


def _mmin_case(job):
    """One (net, material, M, v0, full) row for tab_mmin — process-pool worker."""
    net_kind, mat, M, v0, full = job
    from netsim.mmin import MminConfig, minimum_mass
    from netsim.simulate import build_net

    m = get_material(mat)
    pts = [(0.0, 0.0), (0.25, 0.0), (0.5, 0.0)]
    R = 1.0
    cfg = _mmin_cfg(mat, M, v0, net_kind=net_kind, n_s=10 if full else 8)
    net = build_net(cfg)
    Ekin = 0.5 * M * v0 ** 2
    m_lower = Ekin / m.e_mat
    tol = 0.08
    mmA = MminConfig(criterion="A", tol=tol, impact_points=pts, n_procs=1)
    rA = minimum_mass(cfg, mmA)
    # Same absolute scale for all B variants; cap each at s^A so mB <= mA.
    mm_any = MminConfig(criterion="B_any", tol=tol, impact_points=pts,
                        n_procs=1, n_scan=24)
    r_any = minimum_mass(cfg, mm_any, s_cap=rA.s_min)
    blocs = {}
    n_broken = {"B_any": r_any.n_failures}
    worst = r_any.worst_point
    for frac in (0.25, 0.5, 0.75):
        mm = MminConfig(criterion="B_loc", tol=tol, impact_points=pts,
                        n_procs=1, n_scan=24, R_max=frac * R)
        rb = minimum_mass(cfg, mm, s_cap=rA.s_min)
        blocs[frac] = rb
        n_broken[f"B_loc{int(100*frac):03d}"] = rb.n_failures
        if rb.s_min >= r_any.s_min:
            worst = rb.worst_point

    def g(x):
        return 1e3 * x

    def ratio(mb):
        return rA.m_min / mb if mb > 0 else float("nan")

    mA = rA.m_min
    return {
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
        "mA_over_Ekin_g_per_J": f"{(g(mA)/Ekin):.6g}",
        "n_broken_Bany": n_broken["B_any"],
        "n_broken_Bloc025": n_broken["B_loc025"],
        "n_broken_Bloc050": n_broken["B_loc050"],
        "n_broken_Bloc075": n_broken["B_loc075"],
        "worst_p_x": f"{worst[0]:.6g}",
        "worst_p_y": f"{worst[1]:.6g}",
    }


def export_mmin(full=False):
    # Star and star+ring at eps_p = 0.1 eps_b; full (M,v0) grid with --full.
    if full:
        grid = [(0.25, 10.0), (0.25, 15.0), (0.25, 20.0),
                (1.0, 10.0), (1.0, 15.0), (1.0, 20.0),
                (2.0, 10.0), (2.0, 15.0), (2.0, 20.0)]
    else:
        grid = [(0.25, 20.0), (2.0, 20.0)]
    jobs = [(net, mat, M, v0, full)
            for net in ("star", "star+ring")
            for mat in ("S", "D")
            for M, v0 in grid]
    rows = _parallel_map(_mmin_case, jobs, desc="tab_mmin")
    cols = ["net", "material", "M", "v0", "Ekin", "m_lower_g",
            "mA_min_g", "mBany_min_g", "mBloc025_min_g", "mBloc050_min_g",
            "mBloc075_min_g", "ratio_Bany", "ratio_Bloc025", "ratio_Bloc050",
            "ratio_Bloc075", "mA_over_Ekin_g_per_J",
            "n_broken_Bany", "n_broken_Bloc025", "n_broken_Bloc050",
            "n_broken_Bloc075", "worst_p_x", "worst_p_y"]
    return _write("tab_mmin.csv", cols, rows, config="mmin")


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


def export_phase_maps():
    """Write maps at ``m^{B_any}_min`` for S and D (same net / impact)."""
    from netsim.config import (SimConfig, MaterialConfig, NetConfig,
                               DroneConfig, NumericsConfig, ContactConfig,
                               OutputConfig)
    from netsim.mmin import MminConfig, minimum_mass
    from netsim.simulate import simulate_config
    d = OUT_DIR / "fig_phase_maps"
    d.mkdir(parents=True, exist_ok=True)
    written = []
    for mat in ("S", "D"):
        run_name = f"phase_{mat}"
        h5 = str(d / f"{run_name}.h5")
        # Find mBany at the paper impact (M=1, v0=20, a=0.25) on star+ring.
        from dataclasses import replace as dc_replace
        cfg0 = _mmin_cfg(mat, 1.0, 20.0, net_kind="star+ring", n_s=10)
        cfg0 = dc_replace(cfg0, drone=dc_replace(cfg0.drone, p=(0.25, 0.0)))
        mmA = MminConfig(criterion="A", tol=0.08, impact_points=[(0.25, 0.0)])
        rA = minimum_mass(cfg0, mmA)
        mmB = MminConfig(criterion="B_any", tol=0.08,
                         impact_points=[(0.25, 0.0)], n_scan=24)
        rB = minimum_mass(cfg0, mmB, s_cap=rA.s_min)
        s_map = rB.s_min
        ep = 0.1 * get_material(mat).eps_b
        cfg = SimConfig(
            material=MaterialConfig(name=mat),
            net=NetConfig(kind="star_with_rings", N=8, R=1.0, eps_p=ep,
                          A_hat=1e-6, radii=[0.5, 1.0], q_ratio=1.0,
                          fix_radii=True),
            drone=DroneConfig(M=1.0, r_d=0.15, v0=20.0, p=(0.25, 0.0)),
            numerics=NumericsConfig(n_s=10, C=0.5, t_end=0.25, dt_out=5e-3,
                                    area_scale=float(s_map),
                                    use_numba=True),
            contact=ContactConfig(mode="gripped", k_c=1e7,
                                  k_c_mode=("relative" if mat == "D"
                                            else "absolute"),
                                  k_c_factor=8.0,
                                  penetration_guard="reduce_dt"),
            output=OutputConfig(hdf5=h5, R_max=0.5, k_max=10),
        )
        res = simulate_config(cfg, write=True)
        traj = res.trajectory
        # Per coarse edge: earliest break among its discrete segments.
        break_info = {}  # parent -> (t, order, drone_x, drone_y)
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
        dirty = _working_tree_dirty()
        with open(d / f"segments_{run_name}.csv", "w", newline="") as fh:
            fh.write(f"# commit={_COMMIT} date={_DATE} config=phase "
                     f"s_Bany={s_map:.6g} dirty="
                     f"{'true' if dirty else 'false'}\n")
            w = csv.DictWriter(
                fh, fieldnames=["seg", "parent", "x1", "y1", "x2", "y2",
                                "broken", "t_break", "break_order",
                                "drone_x_break", "drone_y_break"])
            w.writeheader()
            w.writerows(seg_rows)
        written.append(h5)
    return d


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
]


def main(argv=None):
    global _N_JOBS, _ALLOW_DIRTY
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
                         "(header still records dirty=true)")
    args = ap.parse_args(argv)

    _N_JOBS = max(1, args.jobs) if args.jobs > 0 else max(1, os.cpu_count() or 1)
    _ALLOW_DIRTY = bool(args.allow_dirty)
    dirty = _working_tree_dirty()
    if dirty and not _ALLOW_DIRTY:
        print("Refusing to export: Tema 2 working tree is dirty. "
              "Commit first, or pass --allow-dirty "
              "(CSV headers will then include dirty=true).", flush=True)
        sys.exit(2)
    only = set(args.only.split(",")) if args.only else None
    print(f"Exporting paper CSVs to {OUT_DIR} (commit {_COMMIT[:8]}, "
          f"jobs={_N_JOBS}, dirty={'true' if dirty else 'false'})")
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
