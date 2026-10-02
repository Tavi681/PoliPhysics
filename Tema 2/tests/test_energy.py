"""Validation 7: energy conservation without failures.

On the reference case (small M, small v0 so no thread fails), the relative energy
error must be < 1e-3 for C = 0.5 and must decrease with C.
"""

import pytest

from netsim.config import (SimConfig, MaterialConfig, NetConfig, DroneConfig,
                           NumericsConfig, ContactConfig, OutputConfig)
from netsim.simulate import simulate_config


def _cfg(C, mode="frictionless"):
    # Frictionless contact is conservative -> clean second-order energy behavior.
    return SimConfig(
        material=MaterialConfig(name="S"),
        net=NetConfig(kind="star", N=8, R=1.0, eps_p=0.0, A_hat=1e-6),
        drone=DroneConfig(M=0.2, r_d=0.15, v0=5.0, p=(0.5, 0.0), gravity=False),
        numerics=NumericsConfig(n_s=20, C=C, t_end=0.06, dt_out=2e-4,
                                use_numba=False),
        contact=ContactConfig(mode=mode, k_c=1e7),
        output=OutputConfig(hdf5=None, R_max=0.5, k_max=10),
    )


def test_energy_conservation_C_half():
    res = simulate_config(_cfg(0.5), write=False)
    assert res.n_failures == 0, "test case should not break threads"
    assert res.energy_error < 1e-3, f"energy_error={res.energy_error:.3e}"


def test_energy_error_decreases_with_C():
    e_half = simulate_config(_cfg(0.5), write=False).energy_error
    e_quarter = simulate_config(_cfg(0.25), write=False).energy_error
    # Smaller C (smaller dt) must improve conservation.
    assert e_quarter < e_half, (e_quarter, e_half)


def test_energy_conservation_gripped_with_capture_column():
    # With the inelastic-capture column booked, the gripped no-failure run must
    # also conserve energy to < 1e-3. Without the column the residual equals the
    # inelastic loss dE_cap = 1/2 (M m)/(M+m) v_rel^2.
    res = simulate_config(_cfg(0.5, mode="gripped"), write=False)
    assert res.n_failures == 0, "test case should not break threads"
    assert res.energy_error < 1e-3, f"energy_error={res.energy_error:.3e}"
    # The capture column must be strictly positive after grip (energy dissipated).
    assert float(res.trajectory.energy[-1, 5]) > 0.0
