"""Round-5 diagnostics: B variants, DAF normalization, failure-energy vs C.

Run:  python -m validation.round5_checks
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/netsim_mpl")

import numpy as np

from netsim.materials import get_material
from netsim.topology import from_arrays
from netsim.discretize import discretize
from netsim.config import (NumericsConfig, ContactConfig, OutputConfig,
                           DroneConfig, KinematicConfig)
from netsim.integrator import integrate
from netsim._kernels import segment_forces
from netsim.overload import run_overload, daf_sdof_nonlinear, analytic_eps_m_over_eps0
from netsim.mmin import pass_from_info, R_d_loc


def item_daf():
    print("\n=== Item 4: DAF normalization (eps_m/eps_s) ===")
    mat = get_material("D")
    r = run_overload(8, mat, eps0=0.01, m_hub=0.01, n_s=2, t_f_over_Tn=0.0,
                     constrain_hub_z=True)
    daf_num = r.eps_m / r.eps_s
    daf_nl = daf_sdof_nonlinear(r.n_eff, 0.0, eps_s_over_eps0=r.eps_s_over_eps0)
    print(f"  sim eps_m/eps_s = {daf_num:.6g}")
    print(f"  sdof eps_m/eps_s = {daf_nl:.6g}")
    print(f"  |diff| = {abs(daf_num - daf_nl):.3g}")
    matS = get_material("S")
    rS = run_overload(8, matS, eps0=0.8 * matS.eps_b, m_hub=0.01, n_s=2,
                      constrain_hub_z=True)
    an = analytic_eps_m_over_eps0(8, rS.n_eff)
    print(f"  S n_eff={rS.n_eff:.4g} eps_m/eps0_an={an:.6g} "
          f"num={rS.eps_m_over_eps0:.6g}")
    return abs(daf_num - daf_nl) < 0.02


def _pull_residual(mat_name, C):
    mat = get_material(mat_name)
    L = 1.0
    n_nodes = 21
    xs = np.linspace(0.0, L, n_nodes)
    nodes = np.c_[xs, np.zeros(n_nodes)]
    edges = np.c_[np.arange(n_nodes - 1), np.arange(1, n_nodes)]
    anchored = np.zeros(n_nodes, dtype=bool)
    anchored[0] = True
    h = L / (n_nodes - 1)
    rest = np.full(n_nodes - 1, h)
    A = np.full(n_nodes - 1, 1e-6)
    net = from_arrays(nodes, edges, anchored, np.zeros(n_nodes - 1), A,
                      rest_length=rest, R=L)
    disc = discretize(net, mat, 1, r_d=1.0, warn_ratio=1e9)
    c = mat.c_tan_max
    t_transit = L / c
    speed = 0.5 * mat.eps_b * c / 5.0
    t_b_est = mat.eps_b * L / speed
    t_end = t_b_est + 10.0 * t_transit
    drone = DroneConfig(M=1.0, r_d=1.0, v0=0.0)
    numerics = NumericsConfig(n_s=1, C=C, t_end=t_end,
                              dt_out=min(t_transit / 20.0, t_end / 2000),
                              use_numba=False)
    contact = ContactConfig(mode="frictionless", k_c=1e7)
    output = OutputConfig(hdf5=None, R_max=1e9, k_max=10 ** 9)
    end = n_nodes - 1
    kin = KinematicConfig(enabled=True, node=end, mode="velocity",
                          direction=(1.0, 0.0, 0.0), amplitude=speed)
    traj = integrate(disc, mat, drone, numerics, contact, output,
                     kinematic=kin, net_R=L)
    W_acc = 0.0
    f_prev = None
    for i in range(traj.t.size):
        x = traj.x[i]
        intact = traj.intact[i]
        f = segment_forces(x, disc.seg_edges, disc.seg_rest_length,
                           disc.seg_A, intact, mat.E0, mat.b)
        if f_prev is not None:
            dt_i = float(traj.t[i] - traj.t[i - 1])
            F_avg = -0.5 * (f_prev[end, 0] + f[end, 0])
            W_acc += float(F_avg * speed * dt_i)
        f_prev = f
    e0 = traj.energy[0]
    e1 = traj.energy[-1]
    U0 = float(e0[2])
    KE = float(e1[1])
    Uel = float(e1[2])
    Ufail = float(e1[3])
    residual = W_acc - (KE + Uel + Ufail - U0)
    rel = abs(residual) / max(abs(W_acc), 1e-30)
    return rel, int(traj.failures.shape[0])


def item_booking_vs_C():
    print("\n=== Item 5: failure-energy residual vs C (Phi at eps_b) ===")
    for mat_name in ("S", "D"):
        rels = []
        for C in (0.5, 0.25, 0.125):
            rel, n_fail = _pull_residual(mat_name, C)
            rels.append(rel)
            print(f"  {mat_name} C={C}: n_fail={n_fail} rel={rel:.3e}")
        if rels[0] > 0:
            print(f"  {mat_name}: rel(0.25)/rel(0.5)={rels[1]/rels[0]:.3g}, "
                  f"rel(0.125)/rel(0.5)={rels[2]/rels[0]:.3g}")
        if rels[0] < 1e-3:
            print(f"  {mat_name}: PASS (<1e-3 at C=0.5)")
        else:
            print(f"  {mat_name}: residual stays O(1e-3); leave as reported")


def item_B_logic():
    print("\n=== Item 1: B_any / B_loc smoke ===")
    info = {
        "arrested": True, "n_failures": 2, "outcome": "arrested",
        "R_d": 0.9,
        "fail_mids": np.array([[0.7, 0.0, 0.0], [0.75, 0.0, 0.0]]),
        "fail_drone_xy": np.array([[0.6, 0.0], [0.62, 0.0]]),
    }
    ok_any, _ = pass_from_info(info, "B_any", 0.5, 10)
    ok_loc, reason = pass_from_info(info, "B_loc", 0.5, 10)
    Rd = R_d_loc(info["fail_mids"], info["fail_drone_xy"])
    print(f"  R_d vs p = {info['R_d']:.3g}, R_d_loc = {Rd:.3g}")
    print(f"  B_any={ok_any}, B_loc(0.5)={ok_loc} ({reason or 'ok'})")
    # Farther break: R_d_loc ≈ 0.4 → fails at 0.25, passes at 0.5.
    info2 = dict(info)
    info2["fail_mids"] = np.array([[1.0, 0.0, 0.0], [1.05, 0.0, 0.0]])
    info2["fail_drone_xy"] = np.array([[0.6, 0.0], [0.6, 0.0]])
    Rd2 = R_d_loc(info2["fail_mids"], info2["fail_drone_xy"])
    ok25, _ = pass_from_info(info2, "B_loc", 0.25, 10)
    ok50, _ = pass_from_info(info2, "B_loc", 0.5, 10)
    print(f"  far case R_d_loc={Rd2:.3g}: B_loc(0.25)={ok25}, B_loc(0.5)={ok50}")
    return ok_any and ok_loc and (not ok25) and ok50


def main():
    ok_b = item_B_logic()
    ok_d = item_daf()
    item_booking_vs_C()
    print("\n=== done ===")
    print(f"B logic {'PASS' if ok_b else 'FAIL'}; "
          f"DAF match {'PASS' if ok_d else 'CHECK'}")
    return 0 if ok_b and ok_d else 1


if __name__ == "__main__":
    sys.exit(main())
