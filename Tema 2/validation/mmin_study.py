"""Validation 5b: Algorithm 2 minimum-mass study.

Runs the coarse monotonicity scan (required before trusting the bisection,
because criterion B is not monotone in general) for one case per material, then
reports the worst-case minimum mass for both criteria A and B over a list of
impact points.

Run:  python -m validation.mmin_study
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/netsim_mpl")

from netsim.config import (SimConfig, MaterialConfig, NetConfig, DroneConfig,
                           NumericsConfig, ContactConfig, OutputConfig)
from netsim.mmin import MminConfig, minimum_mass, monotonicity_scan

IMPACTS = [(0.0, 0.0), (0.25, 0.0), (0.5, 0.0), (0.25, 0.25)]


def _cfg(material, n_s=10, t_end=0.25):
    return SimConfig(
        material=MaterialConfig(name=material),
        net=NetConfig(kind="star", N=8, R=1.0, eps_p=0.0, A_hat=1e-6),
        drone=DroneConfig(M=1.0, r_d=0.15, v0=15.0, p=(0.0, 0.0)),
        numerics=NumericsConfig(n_s=n_s, C=0.5, t_end=t_end, dt_out=5e-3,
                                use_numba=True),
        contact=ContactConfig(mode="gripped", k_c=1e7,
                              penetration_guard="reduce_dt"),
        output=OutputConfig(hdf5=None, R_max=0.5, k_max=10),
    )


def main():
    for material in ("S", "D"):
        cfg = _cfg(material)
        print(f"\n=== Material {material} ===")

        # Monotonicity scan (criterion B) for the hardest listed point.
        for crit in ("A", "B"):
            mm = MminConfig(criterion=crit, impact_points=[(0.5, 0.0)])
            sc = monotonicity_scan(cfg, mm, (0.5, 0.0), n=10)
            pat = "".join("T" if p else "F" for p in sc["pattern"])
            print(f"monotonicity scan (crit {crit}, p=(0.5,0)): {pat} "
                  f"monotone={sc['monotone']}")

        # Worst-case minimum mass for both criteria.
        for crit in ("A", "B"):
            mm = MminConfig(criterion=crit, tol=0.05, impact_points=IMPACTS,
                            n_procs=min(4, len(IMPACTS)))
            t0 = time.time()
            r = minimum_mass(cfg, mm)
            dt = time.time() - t0
            print(f"criterion {crit}: m_min={r.m_min:.4e} kg "
                  f"(s_min={r.s_min:.4f}) worst={r.worst_point} "
                  f"n_fail={r.n_failures} evals={r.evaluations} [{dt:.1f}s]")


if __name__ == "__main__":
    main()
