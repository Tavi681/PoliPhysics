"""Round 8b: piston default is off (bit-identical); T_frac uses Z(ε0)."""

import numpy as np

from netsim.config import (
    SimConfig, MaterialConfig, NetConfig, DroneConfig, NumericsConfig,
    ContactConfig, OutputConfig, KinematicConfig,
)
from netsim.simulate import simulate
from netsim.materials import get_material
from netsim.topology import star
from netsim.integrator import _piston_segment_mask

from validation.round8 import energy_T_frac, _Et


def _kin_cfg(piston_n_seg=0, n_s=6, t_end=0.004):
    mat = get_material("S")
    ep = 0.1 * mat.eps_b
    net = star(4, 1.0, ep, material=mat, A_hat=1e-6)
    cfg = SimConfig(
        material=MaterialConfig(name="S"),
        net=NetConfig(kind="star", N=4, R=1.0, eps_p=ep, A_hat=1e-6),
        drone=DroneConfig(M=1.0, r_d=1.0, v0=0.0),
        numerics=NumericsConfig(n_s=n_s, C=0.4, t_end=t_end, dt_out=5e-4,
                                use_numba=False),
        contact=ContactConfig(mode="frictionless", k_c=1e7),
        output=OutputConfig(hdf5=None, R_max=1e9, k_max=10 ** 9),
        kinematic=KinematicConfig(
            enabled=True, node=1, mode="velocity",
            direction=(1.0, 0.0, 0.0), amplitude=5.0,
            piston_n_seg=piston_n_seg,
        ),
    )
    return net, mat, cfg


def test_piston_default_zero_and_bit_identical():
    net, mat, cfg0 = _kin_cfg()
    assert cfg0.kinematic.piston_n_seg == 0
    a = simulate(net, mat, cfg0.drone, cfg0.numerics, cfg0.contact,
                 cfg0.output, kinematic=cfg0.kinematic)
    net2, mat2, cfg1 = _kin_cfg(piston_n_seg=0)
    b = simulate(net2, mat2, cfg1.drone, cfg1.numerics, cfg1.contact,
                 cfg1.output, kinematic=cfg1.kinematic)
    np.testing.assert_array_equal(a.trajectory.energy, b.trajectory.energy)
    np.testing.assert_array_equal(a.trajectory.t, b.trajectory.t)
    assert a.outcome == b.outcome


def test_piston_mask_walks_from_driven_node():
    edges = np.array([[0, 1], [1, 2], [2, 3], [0, 4]])
    mask = _piston_segment_mask(edges, forced_node=3, n_seg=2)
    assert mask[2] and mask[1]
    assert not mask[0] and not mask[3]


def test_energy_T_frac_uses_Z_eps0_not_Z_ep():
    mat = get_material("S")
    ep = 0.1 * mat.eps_b
    eps0 = 0.83 * mat.eps_b
    r = np.sqrt(_Et(mat, eps0) / _Et(mat, ep))
    # Symmetric 3-neighbour split of a 1D-like increment.
    dT = np.array([0.73, 0.35, 0.35, 0.35])
    Tf = energy_T_frac(mat, eps0, ep, dT)
    # Incident at Z(ε0): T_frac = r * sum_{j>0} dT_j^2, which is < r * old
    # T0*(2-T0)*r form that produced T_frac > 1.
    old_wrong = dT[0] * (2.0 - dT[0]) * r
    assert Tf < 1.05
    assert Tf < old_wrong
    np.testing.assert_allclose(Tf, r * np.sum(dT[1:] ** 2), rtol=1e-12)
