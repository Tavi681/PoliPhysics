"""Validation Test 7 (script): energy conservation vs the CFL factor C.

On the reference case without failures (small M, small v0) the relative energy
error must be < 1e-3 for C = 0.5 and must decrease with C.
"""

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/netsim_mpl")

from netsim.config import (SimConfig, MaterialConfig, NetConfig, DroneConfig,
                           NumericsConfig, ContactConfig, OutputConfig)
from netsim.simulate import simulate_config


def run(C):
    # Frictionless contact is fully conservative, so velocity-Verlet conserves
    # energy to second order in dt. (The gripped mode adds a one-time inelastic
    # capture loss at first contact, a dt-independent floor.)
    cfg = SimConfig(
        material=MaterialConfig(name="S"),
        net=NetConfig(kind="star", N=8, R=1.0, eps_p=0.0, A_hat=1e-6),
        drone=DroneConfig(M=0.2, r_d=0.15, v0=5.0, p=(0.5, 0.0)),
        numerics=NumericsConfig(n_s=20, C=C, t_end=0.06, dt_out=2e-4,
                                use_numba=False),
        contact=ContactConfig(mode="frictionless", k_c=1e7),
        output=OutputConfig(hdf5=None, R_max=0.5, k_max=10),
    )
    return simulate_config(cfg, write=False)


def main():
    print("Test 7: energy conservation vs C")
    print(f"{'C':>6} {'n_fail':>7} {'energy_error':>14}")
    for C in (0.5, 0.25, 0.125):
        res = run(C)
        print(f"{C:>6.3f} {res.n_failures:>7} {res.energy_error:>14.3e}")


if __name__ == "__main__":
    main()
