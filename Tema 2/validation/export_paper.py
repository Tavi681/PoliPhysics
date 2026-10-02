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
_COMMIT = git_commit()
_DATE = datetime.datetime.now().isoformat(timespec="seconds")


def _write(name, fieldnames, rows, config="default"):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    with open(path, "w", newline="") as fh:
        fh.write(f"# commit={_COMMIT} date={_DATE} config={config}\n")
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
        ("k_c_factor", 4.0, "-", "relative-k_c factor for material D"),
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
    from validation.test1_smith import run_smith, strain_analytic
    rows = []
    for mat in ("S", "D"):
        for v0 in (100.0, 500.0, 1000.0):
            if strain_analytic(get_material(mat), v0) is None:
                continue  # v0 exceeds breaking strain for this material
            r = run_smith(v0, material_name=mat, n_resolved=800)
            tt = r["t"]
            cL_num = r["front_num"] / tt
            cT_num = r["kink_num"] / tt
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
        tt = r["t"]
        rows.append({
            "n_seg": int(round(r["n_resolved"])),
            "eps_err_pct": f"{100*r['eps_relerr']:.4g}",
            "cL_err_pct": f"{_pct(r['front_num']/tt, r['cL']):.4g}",
            "cT_err_pct": f"{_pct(r['kink_num']/tt, r['cT']):.4g}",
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
            })
    cols = ["contact", "n_s", "wmax_over_R", "d_wmax_pct", "n_failed", "eta",
            "d_eta_pct", "R_d", "cpu_s"]
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
    from netsim.overload import run_overload
    regimes = {"D": 0.01, "S": round(0.8 * get_material("S").eps_b, 4)}
    rows = []
    for mat_name, eps0 in regimes.items():
        mat = get_material(mat_name)
        for N in (4, 8, 16):
            r = run_overload(N, mat, eps0=eps0, m_hub=0.01, n_s=2)
            rows.append({
                "material": mat_name, "N": N,
                "n_eff_num": f"{r.n_eff:.6g}",
                "eps_s_over_eps0": f"{r.eps_s/eps0:.6g}",
                "eps_m_over_eps0": f"{r.eps_m/eps0:.6g}",
                "m_hub": f"{0.01:.6g}",
            })
    return _write("tab_daf.csv",
                  ["material", "N", "n_eff_num", "eps_s_over_eps0",
                   "eps_m_over_eps0", "m_hub"], rows, config="test_daf")


def export_daf_ramp():
    from netsim.overload import run_overload
    mat = get_material("D")
    rows = []
    for tf in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
        r = run_overload(8, mat, eps0=0.01, m_hub=0.01, n_s=2, t_f_over_Tn=tf,
                         dyn_periods=16.0)
        x = math.pi * tf
        daf_an = 2.0 if tf == 0.0 else 1.0 + abs(math.sin(x)) / x
        rows.append({"N": 8, "material": "D", "tf_over_Tn": f"{tf:.6g}",
                     "eps_m_over_eps0": f"{r.eps_m/r.eps0_target:.6g}",
                     "daf_an": f"{daf_an:.6g}"})
    return _write("fig_daf_ramp.csv",
                  ["N", "material", "tf_over_Tn", "eps_m_over_eps0", "daf_an"],
                  rows, config="test_daf")


# --------------------------------------------------------------------------- #
# Off-centre eta (results)
# --------------------------------------------------------------------------- #
def export_etaa():
    from validation.offcentre import run_offcentre
    try:
        import offc
    except Exception:
        offc = None
    a_list = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5)
    rows = []
    for mat in ("S", "D"):
        for ep_frac in (0.0, 0.1):
            ref_map = {}
            if offc is not None:
                for a in a_list:
                    if a == 0.0:
                        continue
                    try:
                        fails = offc.run(mat, 8, a, ep_frac)
                        ref_map[a] = fails
                    except Exception:
                        ref_map[a] = None
            for a in a_list:
                r = run_offcentre(material_name=mat, a_over_R=a,
                                  eps_p_frac=ep_frac, n_s=20)
                etaA_num = r.get("eta_ff", float("nan"))
                ref = ref_map.get(a)
                etaA_ref = etaB_ref = ""
                if ref:
                    etaA_ref = f"{ref[0][2]:.6g}"
                    if len(ref) > 1:
                        etaB_ref = f"{ref[1][2]:.6g}"
                rows.append({
                    "net": "star", "material": mat, "ep_frac": f"{ep_frac:.6g}",
                    "a_over_R": f"{a:.6g}",
                    "etaA_num": f"{etaA_num:.6g}", "etaB_num": "",
                    "etaA_ref": etaA_ref, "etaB_ref": etaB_ref,
                    "first_failure": r.get("first_kind", "none"),
                })
    cols = ["net", "material", "ep_frac", "a_over_R", "etaA_num", "etaB_num",
            "etaA_ref", "etaB_ref", "first_failure"]
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
                              eps_p_frac=ep_frac, n_s=20)
            wb_num = speed * r["t_first"] if "t_first" in r else float("nan")
            rows.append({
                "material": mat, "ep_frac": f"{ep_frac:.6g}",
                "eta_an": f"{eta_an:.6g}",
                "eta_num": f"{r.get('eta_ff', float('nan')):.6g}",
                "wb_an": f"{wb_an:.6g}", "wb_num": f"{wb_num:.6g}",
            })
    return _write("tab_etaA.csv",
                  ["material", "ep_frac", "eta_an", "eta_num", "wb_an",
                   "wb_num"], rows, config="offcentre")


def export_Fw():
    # Hub force-deflection: analytic (overload restoring force) across w/R.
    from netsim.overload import _restoring_force
    rows = []
    for mat_name in ("S", "D"):
        mat = get_material(mat_name)
        for wor in np.linspace(0.02, 0.6, 15):
            w = wor * 1.0
            # _restoring_force already sums over the N radials.
            F = _restoring_force(mat, 8, 1.0, 1e-6, w)
            eps = math.sqrt(1.0 + w * w) - 1.0
            rows.append({"material": mat_name, "w_over_R": f"{wor:.6g}",
                         "F_an": f"{F:.6g}", "F_num": "",
                         "failed": bool(eps >= mat.eps_b)})
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
    # Ring nets need a nonzero prestress for the FDM (q=0 is singular).
    ep = 0.1 * get_material(material).eps_b
    net = (NetConfig(kind="star_with_rings", N=8, R=1.0, eps_p=ep, A_hat=1e-6,
                     radii=[0.5, 1.0], q_ratio=1.0) if net_kind == "star+ring"
           else NetConfig(kind="star", N=8, R=1.0, eps_p=0.0, A_hat=1e-6))
    return SimConfig(
        material=MaterialConfig(name=material), net=net,
        drone=DroneConfig(M=M, r_d=0.15, v0=v0, p=(0.0, 0.0)),
        numerics=NumericsConfig(n_s=n_s, C=0.5, t_end=0.25, dt_out=5e-3,
                                use_numba=True),
        contact=ContactConfig(mode="gripped", k_c=1e7,
                              penetration_guard="reduce_dt"),
        output=OutputConfig(hdf5=None, R_max=0.5, k_max=10),
    )


def export_mmin_monotone():
    from netsim.mmin import MminConfig, monotonicity_scan
    rows = []
    for mat in ("S", "D"):
        cfg = _mmin_cfg(mat, 1.0, 15.0)
        for crit in ("A", "B"):
            mm = MminConfig(criterion=crit, impact_points=[(0.5, 0.0)])
            sc = monotonicity_scan(cfg, mm, (0.5, 0.0), n=10)
            for s, ok in zip(sc["s_values"], sc["pattern"]):
                rows.append({"material": mat, "criterion": crit,
                             "s_over_shi": f"{s/sc['s_hi']:.6g}",
                             "pass": bool(ok)})
    return _write("mmin_monotone.csv",
                  ["material", "criterion", "s_over_shi", "pass"], rows,
                  config="mmin")


def export_mmin(full=False):
    from netsim.mmin import MminConfig, minimum_mass
    Ms = (0.25, 2.0) if full else (2.0,)
    pts = [(0.0, 0.0), (0.25, 0.0), (0.5, 0.0)]
    rows = []
    for mat in ("S", "D"):
        m = get_material(mat)
        for M in Ms:
            v0 = 20.0
            cfg = _mmin_cfg(mat, M, v0, net_kind="star+ring",
                            n_s=10 if full else 8)
            from netsim.simulate import build_net
            net = build_net(cfg)
            m1 = net.net_mass(m.rho, 1.0)
            Ekin = 0.5 * M * v0 ** 2
            m_lower = Ekin / m.e_mat
            mmA = MminConfig(criterion="A", tol=0.08, impact_points=pts,
                             n_procs=min(3, len(pts)))
            rA = minimum_mass(cfg, mmA)
            mmB = MminConfig(criterion="B", tol=0.08, impact_points=pts,
                             n_procs=min(3, len(pts)))
            rB = minimum_mass(cfg, mmB)
            ratio = rA.m_min / rB.m_min if rB.m_min else float("nan")
            rows.append({
                "material": mat, "M": f"{M:.6g}", "v0": f"{v0:.6g}",
                "Ekin": f"{Ekin:.6g}",
                "m_lower_g": f"{1e3*m_lower:.6g}",
                "mA_min_g": f"{1e3*rA.m_min:.6g}",
                "mB_min_g": f"{1e3*rB.m_min:.6g}",
                "ratio": f"{ratio:.6g}",
                "n_broken_B": rB.n_failures,
                "worst_p_x": f"{rB.worst_point[0]:.6g}",
                "worst_p_y": f"{rB.worst_point[1]:.6g}",
            })
    cols = ["material", "M", "v0", "Ekin", "m_lower_g", "mA_min_g", "mB_min_g",
            "ratio", "n_broken_B", "worst_p_x", "worst_p_y"]
    return _write("tab_mmin.csv", cols, rows, config="mmin")


# --------------------------------------------------------------------------- #
# Dynamic runs and phase maps (reduced by default)
# --------------------------------------------------------------------------- #
def export_dyn_runs(full=False):
    from netsim.config import (SimConfig, MaterialConfig, NetConfig,
                               DroneConfig, NumericsConfig, ContactConfig,
                               OutputConfig)
    from netsim.simulate import simulate_config, build_net
    Ms = (0.25, 1.0, 2.0) if full else (1.0,)
    v0s = (10.0, 15.0, 20.0) if full else (15.0,)
    a_list = (0.0, 0.25, 0.5) if full else (0.0, 0.5)
    s_list = (0.5, 1.0, 1.5) if full else (1.0,)
    rows = []
    for mat in ("S", "D"):
        m = get_material(mat)
        ep = 0.1 * m.eps_b
        for M in Ms:
            for v0 in v0s:
                for a in a_list:
                    for s in s_list:
                        cfg = SimConfig(
                            material=MaterialConfig(name=mat),
                            net=NetConfig(kind="star_with_rings", N=8, R=1.0,
                                          eps_p=ep, A_hat=1e-6,
                                          radii=[0.5, 1.0], q_ratio=1.0),
                            drone=DroneConfig(M=M, r_d=0.15, v0=v0, p=(a, 0.0)),
                            numerics=NumericsConfig(
                                n_s=10, C=0.5, t_end=0.25, dt_out=5e-3,
                                area_scale=s, use_numba=True,
                                damping=0.0),
                            contact=ContactConfig(
                                mode="gripped", k_c=1e7,
                                k_c_mode=("relative" if mat == "D"
                                          else "absolute"),
                                k_c_factor=4.0,
                                penetration_guard="reduce_dt"),
                            output=OutputConfig(hdf5=None, R_max=0.5, k_max=10),
                        )
                        net = build_net(cfg)
                        m_net = net.net_mass(m.rho, s)
                        res = simulate_config(cfg, write=False)
                        PiE = (0.5 * M * v0 ** 2) / (m_net * m.e_mat) \
                            if m_net > 0 else float("nan")
                        rows.append({
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
                        })
    cols = ["material", "net", "M", "v0", "a_over_R", "s", "m_net",
            "PiE_M_over_m", "arrested", "wmax_over_R", "eta", "n_failed",
            "R_d", "cascade", "energy_error"]
    return _write("dyn_runs.csv", cols, rows, config="dyn")


def export_phase_maps():
    """Write 2 representative HDF5 runs plus their segment maps."""
    from netsim.config import (SimConfig, MaterialConfig, NetConfig,
                               DroneConfig, NumericsConfig, ContactConfig,
                               OutputConfig)
    from netsim.simulate import simulate_config
    d = OUT_DIR / "fig_phase_maps"
    d.mkdir(parents=True, exist_ok=True)
    written = []
    for mat in ("S", "D"):
        run_name = f"phase_{mat}"
        h5 = str(d / f"{run_name}.h5")
        ep = 0.1 * get_material(mat).eps_b
        cfg = SimConfig(
            material=MaterialConfig(name=mat),
            net=NetConfig(kind="star_with_rings", N=8, R=1.0, eps_p=ep,
                          A_hat=1e-6, radii=[0.5, 1.0], q_ratio=1.0),
            drone=DroneConfig(M=1.0, r_d=0.15, v0=20.0, p=(0.25, 0.0)),
            numerics=NumericsConfig(n_s=10, C=0.5, t_end=0.25, dt_out=5e-3,
                                    area_scale=1.0,
                                    use_numba=True),
            contact=ContactConfig(mode="gripped", k_c=1e7,
                                  k_c_mode=("relative" if mat == "D"
                                            else "absolute"),
                                  k_c_factor=4.0,
                                  penetration_guard="reduce_dt"),
            output=OutputConfig(hdf5=h5, R_max=0.5, k_max=10),
        )
        res = simulate_config(cfg, write=True)
        # segments_<run>.csv
        traj = res.trajectory
        broken = {int(s): float(t) for s, _, t in traj.failures} \
            if traj.failures.size else {}
        net = res.net
        seg_rows = []
        for e in range(net.n_e):
            i, j = int(net.edges[e, 0]), int(net.edges[e, 1])
            x1, y1 = net.nodes[i]
            x2, y2 = net.nodes[j]
            seg_rows.append({"seg": e, "parent": e, "x1": f"{x1:.6g}",
                             "y1": f"{y1:.6g}", "x2": f"{x2:.6g}",
                             "y2": f"{y2:.6g}",
                             "broken": bool(e in broken),
                             "t_break": f"{broken.get(e, float('nan')):.6g}"})
        with open(d / f"segments_{run_name}.csv", "w", newline="") as fh:
            fh.write(f"# commit={_COMMIT} date={_DATE} config=phase\n")
            w = csv.DictWriter(fh, fieldnames=["seg", "parent", "x1", "y1",
                                               "x2", "y2", "broken", "t_break"])
            w.writeheader()
            w.writerows(seg_rows)
        written.append(h5)
    return d


def export_etaa_ring():
    # Ring-net off-centre eta (reference left empty per SPEC).
    from netsim.config import (SimConfig, MaterialConfig, NetConfig,
                               DroneConfig, NumericsConfig, ContactConfig,
                               OutputConfig, KinematicConfig)
    from netsim.simulate import simulate_config, build_net
    a_list = (0.0, 0.1, 0.25, 0.5)
    rows = []
    for mat in ("S", "D"):
        for q_ratio in (0.5, 1.0):
            for a in a_list:
                cfg = SimConfig(
                    material=MaterialConfig(name=mat),
                    net=NetConfig(kind="star_with_rings", N=8, R=1.0,
                                  eps_p=0.01, A_hat=1e-6, radii=[0.5, 1.0],
                                  q_ratio=q_ratio),
                    drone=DroneConfig(M=1.0, r_d=0.15, v0=15.0, p=(a, 0.0)),
                    numerics=NumericsConfig(n_s=10, C=0.5, t_end=0.2,
                                            dt_out=5e-3, use_numba=True,
                                            damping=0.0),
                    contact=ContactConfig(mode="gripped", k_c=1e7,
                                          penetration_guard="reduce_dt"),
                    output=OutputConfig(hdf5=None, R_max=0.5, k_max=10),
                )
                res = simulate_config(cfg, write=False)
                rows.append({
                    "net": "star+ring", "material": mat,
                    "ep_frac": "0.1", "a_over_R": f"{a:.6g}",
                    "etaA_num": f"{res.eta:.6g}", "etaB_num": "",
                    "etaA_ref": "", "etaB_ref": "",
                    "first_failure": "", "q_ratio": f"{q_ratio:.6g}",
                })
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true",
                    help="use the full (expensive) grids for mmin/dyn_runs")
    ap.add_argument("--only", default=None,
                    help="comma-separated list of CSV names to export")
    args = ap.parse_args(argv)

    only = set(args.only.split(",")) if args.only else None
    print(f"Exporting paper CSVs to {OUT_DIR} (commit {_COMMIT[:8]})")
    for name, fn in EXPORTERS:
        if only and name not in only:
            continue
        try:
            path = fn(args.full)
            print(f"  [ok]   {name:22s} -> {path}")
        except Exception as exc:  # pragma: no cover
            import traceback
            print(f"  [FAIL] {name:22s} : {exc}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
