"""Validation Test 3: wave transmission/reflection at an N-thread junction.

A longitudinal tension step is launched along thread 0 (by prescribing an
outward longitudinal velocity at its far anchor). After the step reaches the hub
we measure, a few cells from the hub, the strain on thread 0 (e0*) and on the
other threads, and form the tension ratios

    T0/Tinc      = (sigma(e0*) - sigma(ep)) / Tinc,
    dTopp/Tinc   = (sigma(e_opp) - sigma(ep)) / Tinc,   Tinc = sigma(e1)-sigma(ep).

These are compared with:

* the linear theory: with S_N = 1 (N=2) or N/2-1 (N>=3),
  T0/Tinc = 2 S_N/(1+S_N), dT_j/Tinc = -2 cos(phi_j)/(1+S_N);
* the finite-amplitude reference ``ref/riemann.junction`` (imported, not copied).

Long threads are used so that reflections from the far anchors do not return
during the measurement window. The prestress ep is large enough that same-side
neighbours do not go slack at the amplitudes tested.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/netsim_mpl")
_REF = Path(__file__).resolve().parent.parent / "ref"
if str(_REF) not in sys.path:
    sys.path.insert(0, str(_REF))

import riemann  # ref/riemann.py

from netsim.config import (SimConfig, MaterialConfig, NetConfig, DroneConfig,
                           NumericsConfig, ContactConfig, OutputConfig,
                           KinematicConfig)
from netsim.materials import get_material
from netsim.topology import star
from netsim.discretize import discretize
from netsim.simulate import simulate
from validation.junction_linear_2d import (S_N, junction_linear_2d,
                                           linear_ratios_1d)

# Measurement window (documented in README): the strain plateau is sampled
# `cells` segments from the hub, over t in [tarr + 0.6 L/c, tarr + 1.5 L/c],
# i.e. after the incident front has reached the hub and the junction has reached
# its steady state, but before reflections from the far anchors return (~2 L/c
# after arrival).
WIN_LO = 0.6
WIN_HI = 1.5


def linear_ratios(N):
    """Backward-compatible 1D ratios (T0, dTopp)."""
    T0, dT = linear_ratios_1d(N)
    phi = 2 * np.pi * np.arange(N) / N
    j_opp = 1 + int(np.argmin(np.cos(phi[1:])))
    return T0, dT[j_opp]


def _thread_segment_order(disc, N):
    """Return, per thread, the segment indices ordered from the hub outward."""
    order = {k: [] for k in range(N)}
    for s in range(disc.n_seg):
        order[int(disc.seg_parent[s])].append(s)
    return order


def run_junction_sim(name="S", N=4, ep_frac=0.1, e1_frac=None, L=2.0,
                     n_seg=400, cells=5, A_hat=1e-6, d_frac=0.02):
    ref_mat = getattr(riemann, name)
    material = get_material(name)
    ep = ep_frac * material.eps_b
    # Small pulse: increment d_frac*eps_b over the prestress (linear regime),
    # unless an explicit e1_frac is given (finite-amplitude comparison).
    if e1_frac is None:
        e1 = ep + d_frac * material.eps_b
    else:
        e1 = e1_frac * material.eps_b
    v1 = ref_mat.Phi(e1, ep)  # incident particle velocity (finite amplitude)

    net = star(N, L, ep, material=material, A_hat=A_hat)
    disc = discretize(net, material, n_seg, r_d=1.0, warn_ratio=1e9)
    seg_order = _thread_segment_order(disc, N)

    # Drive the far anchor of thread 0 outward (+x) at v1.
    anchor0 = 1  # coarse node index of thread 0's anchor
    cfg = SimConfig(
        material=MaterialConfig(name=name),
        net=NetConfig(kind="star", N=N, R=L, eps_p=ep, A_hat=A_hat),
        drone=DroneConfig(M=1.0, r_d=1.0, v0=0.0),
        numerics=NumericsConfig(n_s=n_seg, C=0.4, t_end=0.0, dt_out=1e-6,
                                use_numba=False),
        contact=ContactConfig(mode="frictionless", k_c=1e7),
        output=OutputConfig(hdf5=None, R_max=1e9, k_max=10 ** 9),
        kinematic=KinematicConfig(enabled=True, node=anchor0, mode="velocity",
                                  direction=(1.0, 0.0, 0.0), amplitude=v1),
    )
    # Measurement window: after the hub settles to the steady junction state,
    # but before reflections from the far anchors return (~2 L/c after arrival).
    c_inc = ref_mat.c(e1) if e1 > 0 else material.c_L0
    tarr = L / c_inc
    cfg.numerics.t_end = tarr + 1.6 * L / c_inc
    cfg.numerics.dt_out = cfg.numerics.t_end / 120

    material = cfg.material.resolve()
    res = simulate(net, material, cfg.drone, cfg.numerics, cfg.contact,
                   cfg.output, kinematic=cfg.kinematic)
    traj = res.trajectory

    # Strain per thread `cells` segments from the hub, averaged over the last
    # frames of the measurement window (steady junction state after arrival).
    win = (traj.t >= (tarr + WIN_LO * L / c_inc)) & (traj.t <= (tarr + WIN_HI * L / c_inc))
    frames = np.nonzero(win)[0]
    if frames.size == 0:
        frames = np.array([traj.t.size - 1])
    strains = np.zeros(N)
    for k in range(N):
        s = seg_order[k][cells]
        i, j = disc.seg_edges[s]
        vals = []
        for fr in frames:
            d = traj.x[fr, j] - traj.x[fr, i]
            vals.append(np.linalg.norm(d) / disc.seg_rest_length[s] - 1.0)
        strains[k] = float(np.mean(vals))

    Tinc = float(material.sigma(e1) - material.sigma(ep))
    sig = lambda e: float(material.sigma(e))
    phi = 2 * np.pi * np.arange(N) / N
    # Per-thread transmitted tension ratio (index 0 = incident thread 0).
    dT_sim = np.array([(sig(strains[k]) - sig(ep)) / Tinc for k in range(N)])
    T0_sim = dT_sim[0]
    off = dT_sim[1:]
    dTmax_sim = float(off.max())
    dTmin_sim = float(off.min())
    j_opp = 1 + int(np.argmin(np.cos(phi[1:])))
    dTopp_sim = float(dT_sim[j_opp])

    # Reference finite-amplitude junction (1D Riemann).
    r = riemann.junction(ref_mat, N, ep, ep, e1)
    T0_ref = (float(ref_mat.sig(r["e0s"])) - float(ref_mat.sig(ep))) / Tinc
    dT_ref = np.array([(float(ref_mat.sig(e)) - float(ref_mat.sig(ep))) / Tinc
                       for e in r["ej"]])
    dTopp_ref = float(dT_ref[j_opp - 1])

    # Linear predictions.
    T0_lin, dT_lin = linear_ratios_1d(N)
    j2w = junction_linear_2d(material, N, ep, variant="with")
    j2o = junction_linear_2d(material, N, ep, variant="wo")

    return {
        "N": N, "material": name, "ep_frac": ep_frac, "ep": ep, "e1": e1,
        "T0_sim": T0_sim, "T0_ref": T0_ref, "T0_lin": T0_lin,
        "T0_2d_with": j2w["T0"], "T0_2d_wo": j2o["T0"],
        "dTopp_sim": dTopp_sim, "dTopp_ref": dTopp_ref,
        "dTopp_lin": float(dT_lin[j_opp]),
        "dTopp_2d_with": float(j2w["dT"][j_opp]),
        "dTopp_2d_wo": float(j2o["dT"][j_opp]),
        "dTmax_sim": dTmax_sim, "dTmin_sim": dTmin_sim,
        "dTmax_2d_with": j2w["dTmax"], "dTmin_2d_with": j2w["dTmin"],
        "dTmax_2d_wo": j2o["dTmax"], "dTmin_2d_wo": j2o["dTmin"],
        "dTmax_1d": float(dT_lin[1:].max()), "dTmin_1d": float(dT_lin[1:].min()),
        "ZT_over_ZL_with": j2w["ZT_over_ZL"], "ZT_over_ZL_wo": j2o["ZT_over_ZL"],
        "SN": S_N(N), "SN2d_with": j2w["SN2d"], "SN2d_wo": j2o["SN2d"],
        "strains": strains,
    }


def best_variant(name="S", ep_frac=0.3, Ns=(3, 4, 5, 6, 8, 12, 16), L=3.0,
                 n_seg=500):
    """Return which c_T variant the simulator matches, aggregating |err| over N.

    Uses material S at a large prestress (ep_frac=0.3), where the (1+eps_p)
    factor most separates the two variants.
    """
    err_with = err_wo = 0.0
    for N in Ns:
        r = run_junction_sim(name, N=N, ep_frac=ep_frac, L=L, n_seg=n_seg)
        err_with += abs(r["T0_sim"] - r["T0_2d_with"])
        err_wo += abs(r["T0_sim"] - r["T0_2d_wo"])
    return ("with" if err_with <= err_wo else "wo"), err_with, err_wo


def main():
    Ns = (2, 3, 4, 5, 6, 8, 12, 16)
    print("Test 3: junction ratios vs 1D and transverse-corrected 2D theory")
    print("(small pulse, increment 0.02 eps_b; measurement window "
          f"t in [tarr+{WIN_LO} L/c, tarr+{WIN_HI} L/c], {5} cells from hub)\n")
    for name in ("S", "D"):
        for ep_frac in (0.02, 0.1, 0.3):
            print(f"--- material {name}, eps_p/eps_b = {ep_frac} ---")
            print(f"{'N':>3} | {'T0 num':>7} {'T0 1d':>6} {'T0 2dW':>7} "
                  f"{'T0 2dO':>7} | {'dTo num':>7} {'dTo 1d':>6} "
                  f"{'dTo 2dW':>7}")
            for N in Ns:
                r = run_junction_sim(name, N=N, ep_frac=ep_frac, L=3.0,
                                     n_seg=500)
                print(f"{N:>3} | {r['T0_sim']:>7.3f} {r['T0_lin']:>6.3f} "
                      f"{r['T0_2d_with']:>7.3f} {r['T0_2d_wo']:>7.3f} | "
                      f"{r['dTopp_sim']:>7.3f} {r['dTopp_lin']:>6.3f} "
                      f"{r['dTopp_2d_with']:>7.3f}")
            print()

    # Variant selection (aggregated residual over N, S at ep=0.3 eb).
    var, ew, eo = best_variant("S", ep_frac=0.3)
    print(f"Chosen c_T variant (S, ep=0.3 eb, sum over N): '{var}' "
          f"(sum|err_with|={ew:.3f}, sum|err_wo|={eo:.3f})\n")

    # Scaling check (item 1): discrepancy ~ Z_T/Z_L ~ sqrt(eps_p) for D.
    print("Scaling check (D, N=8): 1D-numerical T0 discrepancy vs Z_T/Z_L")
    print(f"{'ep_frac':>7} {'ZT/ZL':>7} {'T0 1d':>6} {'T0 num':>7} "
          f"{'num-1d':>7} {'T0 2dW':>7} {'num-2dW':>8}")
    for ep_frac in (0.02, 0.1, 0.3):
        r = run_junction_sim("D", N=8, ep_frac=ep_frac, L=3.0, n_seg=500)
        print(f"{ep_frac:>7} {r['ZT_over_ZL_with']:>7.3f} {r['T0_lin']:>6.3f} "
              f"{r['T0_sim']:>7.3f} {r['T0_sim']-r['T0_lin']:>7.3f} "
              f"{r['T0_2d_with']:>7.3f} {r['T0_sim']-r['T0_2d_with']:>8.3f}")


if __name__ == "__main__":
    main()
