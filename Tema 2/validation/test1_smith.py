"""Validation Test 1: single thread transverse impact (Smith's problem).

A long straight thread (zero prestress), anchored at both ends, has its middle
node given a constant transverse velocity v0. Smith's analytical solution gives:

* the strain behind the longitudinal front, eps, from
  v0^2 = (T(eps)/mu0) [2 sqrt(eps(1+eps)) - eps], mu0 = rho A;
* the longitudinal front at Lagrangian coordinate c_L t, c_L = sqrt(T/(mu0 eps));
* the transverse kink at c_T t, c_T = sqrt(T/(mu0 (1+eps))).

Fronts are located where eps crosses half the jump. This module writes a CSV
table (analytical vs numerical, with relative errors) and a figure eps(X) at
three times.
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

os.environ.setdefault("MPLCONFIGDIR", "/tmp/netsim_mpl")

from netsim.materials import get_material
from netsim.topology import from_arrays
from netsim.discretize import discretize
from netsim.config import (DroneConfig, NumericsConfig, ContactConfig,
                           OutputConfig, KinematicConfig)
from netsim.integrator import integrate


def strain_analytic(material, v0):
    """Solve v0^2 = (sigma/rho)[2 sqrt(eps(1+eps)) - eps] for eps."""
    def f(eps):
        return (float(material.sigma(eps)) / material.rho) * (
            2 * math.sqrt(eps * (1 + eps)) - eps) - v0 ** 2
    # bracket in (0, eps_b)
    hi = material.eps_b * 0.999
    if f(hi) < 0:
        return None  # would exceed eps_b (thread breaks)
    return brentq(f, 1e-9, hi, xtol=1e-12)


def _fit_step(X, prof, amp, x_guess, rising=False):
    """Locate a step in ``prof(X)`` by fitting an error-function profile.

    The field steps between ``amp`` (plateau) and 0 across the front. For a
    falling step (``rising=False``) the model is
    ``f(X) = 0.5 amp (1 - erf((X - x0)/w))``. The fit is restricted to a window
    around ``x_guess`` (the analytical front) that still contains plateau on one
    side and zero on the other, which makes the estimate robust to the ringing
    that a bare half-crossing detector picks up on refined grids.
    """
    from scipy.special import erf
    from scipy.optimize import curve_fit

    lo, hi = 0.45 * x_guess, 1.55 * x_guess
    m = (X >= lo) & (X <= hi)
    if m.sum() < 5:
        m = np.ones_like(X, dtype=bool)
    Xw, Pw = X[m], prof[m]

    if rising:
        def model(x, x0, w):
            return 0.5 * amp * (1.0 + erf((x - x0) / w))
    else:
        def model(x, x0, w):
            return 0.5 * amp * (1.0 - erf((x - x0) / w))

    dx = float(np.median(np.diff(X))) if X.size > 1 else x_guess * 1e-3
    try:
        popt, _ = curve_fit(model, Xw, Pw, p0=[x_guess, 3 * dx],
                            maxfev=20000)
        x0 = float(popt[0])
        if np.isfinite(x0):
            return x0
    except Exception:
        pass
    # Fallback: half-crossing.
    above = np.nonzero(prof > 0.5 * amp)[0]
    return float(X[above[-1]]) if above.size else float("nan")


def nodes_for_resolution(n_resolved, Lx=2.0, frac_time=0.55):
    """Number of chain nodes so that ~n_resolved segments lie within the
    longitudinal front at the end of the run.

    The longitudinal front sits at X = c_L t = frac_time * (Lx/2), independent
    of c_L, so the number of segments inside it is
    n_resolved = frac_time * (n_nodes - 1) / 2.
    """
    return int(round(2 * n_resolved / frac_time)) + 1


def run_smith(v0=300.0, material_name="S", Lx=2.0, n_nodes=801, n_resolved=None,
              frac_time=0.55, A_hat=1e-6):
    if n_resolved is not None:
        n_nodes = nodes_for_resolution(n_resolved, Lx, frac_time)
    material = get_material(material_name)
    eps_a = strain_analytic(material, v0)
    if eps_a is None:
        raise ValueError(f"v0={v0} exceeds breaking strain for {material_name}")
    cL = float(material.c_L(eps_a))
    cT = float(material.c_T(eps_a))

    # Straight chain along x, zero prestress, anchored ends.
    xs = np.linspace(0.0, Lx, n_nodes)
    nodes = np.c_[xs, np.zeros(n_nodes)]
    edges = np.c_[np.arange(n_nodes - 1), np.arange(1, n_nodes)]
    anchored = np.zeros(n_nodes, dtype=bool)
    anchored[0] = anchored[-1] = True
    h = Lx / (n_nodes - 1)
    rest = np.full(n_nodes - 1, h)
    q = np.zeros(n_nodes - 1)
    A = np.full(n_nodes - 1, A_hat)
    net = from_arrays(nodes, edges, anchored, q, A, rest_length=rest, R=Lx)

    mid = n_nodes // 2
    disc = discretize(net, material, 1, r_d=1.0, warn_ratio=1e9)

    # Time so the longitudinal front stays away from the anchors.
    t_end = frac_time * (Lx / 2) / cL

    drone = DroneConfig(M=1.0, r_d=1.0, v0=0.0)
    numerics = NumericsConfig(n_s=1, C=0.4, t_end=t_end, dt_out=t_end / 3,
                              use_numba=False)
    contact = ContactConfig(mode="frictionless", k_c=1e7)
    output = OutputConfig(hdf5=None, R_max=1e9, k_max=10 ** 9)
    kin = KinematicConfig(enabled=True, node=mid, mode="velocity",
                          direction=(0.0, 0.0, 1.0), amplitude=v0)

    traj = integrate(disc, material, drone, numerics, contact, output,
                     kinematic=kin, net_R=Lx)

    # Strain vs Lagrangian coordinate (segment midpoints, distance from centre).
    xf = traj.x[-1]
    d = xf[edges[:, 1]] - xf[edges[:, 0]]
    eps_num_seg = np.linalg.norm(d, axis=1) / rest - 1.0
    Xseg = (0.5 * (xs[:-1] + xs[1:]) - Lx / 2)  # Lagrangian (undeformed) coord

    # Right half.
    right = Xseg > 0
    Xr = Xseg[right]
    er = eps_num_seg[right]

    tt = traj.t[-1]
    cLt = cL * tt
    cTt = cT * tt

    # Plateau strain: median in the interval (0.15, 0.6) c_L t (behind kink).
    band = (Xr > 0.15 * cLt) & (Xr < 0.6 * cLt)
    eps_num = float(np.median(er[band])) if np.any(band) else float(np.max(er))

    # Longitudinal front: fit an error-function step (plateau -> 0) to the
    # strain profile. A sharp half-crossing detector overshoots on refined grids
    # (post-front ringing), so the error would grow with resolution; the erf fit
    # is a sub-cell-accurate, ringing-robust estimate of the front location.
    front_num = _fit_step(Xr, er, eps_num, cLt, rising=False)

    # Kink (transverse front): |v_z| drops from ~v0 (behind) to 0 (ahead) at
    # X = c_T t; fit the same error-function step to |v_z|.
    vz = traj.v[-1][:, 2]
    vzt_seg = np.abs(0.5 * (vz[edges[:, 0]] + vz[edges[:, 1]])[right])
    kink_num = _fit_step(Xr, vzt_seg, v0, cTt, rising=False)

    front_ana = cL * tt
    kink_ana = cT * tt
    n_res = frac_time * (n_nodes - 1) / 2
    return {
        "v0": v0, "material": material_name, "n_nodes": n_nodes,
        "n_resolved": n_res,
        "eps_num": eps_num, "eps_ana": eps_a,
        "eps_relerr": abs(eps_num - eps_a) / eps_a,
        "front_num": front_num, "front_ana": front_ana,
        "front_relerr": abs(front_num - front_ana) / front_ana,
        "kink_num": kink_num, "kink_ana": kink_ana,
        "kink_relerr": abs(kink_num - kink_ana) / kink_ana,
        "X": Xr, "eps_profile": er, "t": tt, "cL": cL, "cT": cT,
        "traj": traj, "edges": edges, "xs": xs, "Lx": Lx, "rest": rest,
    }


# v0 chosen comfortably below the breaking strain of each material (S breaks
# near 936 m/s, D near 1094 m/s; at those limits Smith's steady profile
# degenerates and both strain and kink detection become meaningless).
MATERIAL_V0 = {"S": (100.0, 200.0, 300.0), "D": (100.0, 300.0, 500.0)}
RESOLUTIONS = (200, 400, 800)


def main(out_dir=None):
    out_dir = Path(out_dir or Path(__file__).resolve().parent / "output")
    out_dir.mkdir(exist_ok=True)
    rows = []
    print("Test 1: single thread (Smith), materials S and D")
    print(f"{'mat':>3} {'v0':>6} {'n_seg':>6} {'eps num':>9} {'eps ana':>9} "
          f"{'eps e%':>7} {'front e%':>8} {'kink e%':>8}")
    for mat in ("S", "D"):
        for v0 in MATERIAL_V0[mat]:
            # Skip (material, v0) combinations that would break the thread.
            try:
                if strain_analytic(get_material(mat), v0) is None:
                    print(f"{mat:>3} {v0:>6.0f}  (exceeds eps_b -> skipped)")
                    continue
            except Exception as exc:
                print(f"{mat:>3} {v0:>6.0f}  (skipped: {exc})")
                continue
            for n_res in RESOLUTIONS:
                r = run_smith(v0, material_name=mat, n_resolved=n_res)
                print(f"{mat:>3} {v0:>6.0f} {int(r['n_resolved']):>6} "
                      f"{r['eps_num']:>9.4f} {r['eps_ana']:>9.4f} "
                      f"{100*r['eps_relerr']:>7.2f} "
                      f"{100*r['front_relerr']:>8.2f} "
                      f"{100*r['kink_relerr']:>8.2f}")
                rows.append({
                    "material": mat, "v0": v0,
                    "n_resolved": int(round(r["n_resolved"])),
                    "n_nodes": r["n_nodes"],
                    "eps_num": r["eps_num"], "eps_ana": r["eps_ana"],
                    "eps_relerr": r["eps_relerr"],
                    "front_num": r["front_num"], "front_ana": r["front_ana"],
                    "front_relerr": r["front_relerr"],
                    "kink_num": r["kink_num"], "kink_ana": r["kink_ana"],
                    "kink_relerr": r["kink_relerr"],
                })

    csv_path = out_dir / "test1_smith.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", csv_path)

    # Figure eps(X) at three times for v0 = 300.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        for frac in (0.35, 0.45, 0.55):
            r = run_smith(300.0, frac_time=frac)
            ax.plot(r["X"], r["eps_profile"], label=f"t = {r['t']*1e3:.2f} ms")
            ax.axhline(r["eps_ana"], ls=":", color="k", lw=0.6)
        ax.set_xlabel("Lagrangian coordinate X [m]")
        ax.set_ylabel("strain eps")
        ax.set_title("Smith single thread, v0 = 300 m/s (S)")
        ax.legend()
        fig.tight_layout()
        fig_path = out_dir / "test1_smith_eps.png"
        fig.savefig(fig_path, dpi=120)
        print("wrote", fig_path)
    except Exception as exc:  # pragma: no cover
        print("figure skipped:", exc)


if __name__ == "__main__":
    main()
