"""Validation Test 4: convergence in n_s on the reference case.

Reference case: star N=8, material S, M=1 kg, v0=15 m/s, off-centre impact at
a=R/2, both contact modes. Reports w_max/R, number and radial positions of
failed segments, R_d, eta, the energy error and CPU time for n_s in
{10, 20, 40, 80}, with the relative difference of w_max from n_s=80. Convergence
of w_max alone is not sufficient, so eta and R_d are reported alongside it.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/netsim_mpl")

from netsim.config import (SimConfig, MaterialConfig, NetConfig, DroneConfig,
                           NumericsConfig, ContactConfig, OutputConfig)
from netsim.simulate import simulate_config


def run_case(n_s, mode, v0=15.0, t_end=0.2):
    cfg = SimConfig(
        material=MaterialConfig(name="S"),
        net=NetConfig(kind="star", N=8, R=1.0, eps_p=0.0, A_hat=1e-6),
        drone=DroneConfig(M=1.0, r_d=0.15, v0=v0, p=(0.5, 0.0)),
        numerics=NumericsConfig(n_s=n_s, C=0.5, t_end=t_end, dt_out=1e-3,
                                use_numba=True),
        # A convergence sweep deliberately includes coarse grids where the
        # penetration can graze delta_ref; use reduce_dt so the contact stays
        # resolved (dt shrinks from the actual delta) instead of hard-aborting.
        contact=ContactConfig(mode=mode, k_c=1e7, penetration_guard="reduce_dt"),
        output=OutputConfig(hdf5=None, R_max=0.5, k_max=10),
    )
    t0 = time.time()
    res = simulate_config(cfg, write=False)
    cpu = time.time() - t0
    # Radial positions (from the impact point) of the failed segment midpoints.
    mids = res.trajectory.failure_midpoints
    p = cfg.drone.p
    if mids.size:
        fail_radii = [float((((m[0] - p[0]) ** 2 + (m[1] - p[1]) ** 2) ** 0.5))
                      for m in mids]
    else:
        fail_radii = []
    return {"n_s": n_s, "mode": mode, "w_max_over_R": res.w_max / cfg.net.R,
            "n_failures": res.n_failures, "eta": res.eta, "R_d": res.R_d,
            "cascade": res.cascade, "fail_radii": fail_radii,
            "outcome": res.outcome, "cpu": cpu,
            "energy_error": res.energy_error}


def main():
    for mode in ("gripped", "frictionless"):
        print(f"\nConvergence (mode = {mode})")
        print(f"{'n_s':>4} {'w_max/R':>9} {'n_fail':>7} {'R_d':>8} "
              f"{'eta':>8} {'e_err':>9} {'outcome':>11} {'cpu[s]':>7} "
              f"{'d(w)%':>7}")
        rows = []
        for n_s in (10, 20, 40, 80):
            rows.append(run_case(n_s, mode))
        ref = rows[-1]["w_max_over_R"]
        for r in rows:
            dw = 100 * abs(r["w_max_over_R"] - ref) / ref if ref else 0.0
            print(f"{r['n_s']:>4} {r['w_max_over_R']:>9.4f} "
                  f"{r['n_failures']:>7} {r['R_d']:>8.4f} {r['eta']:>8.4f} "
                  f"{r['energy_error']:>9.2e} {r['outcome']:>11} "
                  f"{r['cpu']:>7.2f} {dw:>7.2f}")
        # Report failure positions (radial distance from impact) at the finest
        # grid, where they are most resolved.
        fine = rows[-1]
        if fine["fail_radii"]:
            radii = ", ".join(f"{x:.3f}" for x in sorted(fine["fail_radii"]))
            print(f"  failed-segment radii @ n_s={fine['n_s']} "
                  f"(cascade={fine['cascade']}): [{radii}]")
        else:
            print(f"  no segment failures @ n_s={fine['n_s']} "
                  f"(cascade={fine['cascade']})")


if __name__ == "__main__":
    main()
