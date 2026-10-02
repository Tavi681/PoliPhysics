"""Item 2: contact resolution for the stiff material D.

The absolute penalty stiffness under-resolves the D contact (delta reaches
delta_ref), so the abort guard must fire; the relative k_c mode keeps the
contact resolved and gives a well-behaved energy balance.
"""

import math

import pytest

from netsim.config import (SimConfig, MaterialConfig, NetConfig, DroneConfig,
                           NumericsConfig, ContactConfig, OutputConfig)
from netsim.simulate import build_net, simulate
from netsim.integrator import PenetrationError, resolve_k_c


def _cfg(mode, k_c_mode, guard, factor=4.0, t_end=0.03, n_s=20):
    return SimConfig(
        material=MaterialConfig(name="D"),
        net=NetConfig(kind="star", N=8, R=1.0, eps_p=0.0, A_hat=1e-6),
        drone=DroneConfig(M=1.0, r_d=0.15, v0=15.0, p=(0.5, 0.0)),
        numerics=NumericsConfig(n_s=n_s, C=0.5, t_end=t_end, dt_out=1e-3,
                                use_numba=False),
        contact=ContactConfig(mode=mode, k_c=1e7, k_c_mode=k_c_mode,
                              k_c_factor=factor, penetration_guard=guard),
        output=OutputConfig(hdf5=None, R_max=0.5, k_max=10),
    )


def _run(cfg):
    mat = cfg.material.resolve()
    net = build_net(cfg)
    return simulate(net, mat, cfg.drone, cfg.numerics, cfg.contact, cfg.output)


def test_resolve_k_c_relative_formula():
    # k_c = factor^2 * (E0 A / l_s) / (1.5 sqrt(delta_ref)); m_node cancels.
    from netsim.materials import get_material
    mat = get_material("D")
    seg_A = [1e-6, 1e-6]
    seg_rest = [0.025, 0.05]
    r_d = 0.15
    c = ContactConfig(k_c_mode="relative", k_c_factor=4.0, delta_ref_frac=0.01)
    k_axial = mat.E0 * 1e-6 / 0.025  # stiffest (shortest) segment
    delta_ref = 0.01 * r_d
    expect = 4.0 ** 2 * k_axial / (1.5 * math.sqrt(delta_ref))
    got = resolve_k_c(c, mat, seg_A, seg_rest, r_d)
    assert got == pytest.approx(expect, rel=1e-12)
    # Absolute mode returns k_c verbatim.
    ca = ContactConfig(k_c_mode="absolute", k_c=1e7)
    assert resolve_k_c(ca, mat, seg_A, seg_rest, r_d) == 1e7


def test_abort_guard_fires_for_absolute_D():
    cfg = _cfg("frictionless", "absolute", "abort")
    with pytest.raises(PenetrationError) as ei:
        _run(cfg)
    assert ei.value.delta > ei.value.delta_ref


def test_relative_k_c_resolves_D_contact():
    cfg = _cfg("frictionless", "relative", "abort", factor=4.0)
    res = _run(cfg)  # must not raise
    tr = res.trajectory
    # Contact stays well resolved and k_c is much stiffer than absolute 1e7.
    assert tr.delta_max < tr.delta_ref
    assert tr.k_c > 1e8
    # Energy balance is orders of magnitude better than the absolute case.
    assert res.energy_error < 0.05
