"""Coulomb source flag and the contact entanglement criterion."""

import numpy as np

from ballooning.params import Params
from ballooning.geometry import initial_state
from ballooning.integrator import _entangled
from ballooning import forces


def _coulomb_force(P, xi, topo):
    f = np.zeros(P.n_dof)
    forces._add_coulomb_force(P, topo, topo.positions(xi), forces.node_charges(P, topo), f)
    return f


def test_coulomb_excludes_spider_as_source_by_default():
    P = Params(N=1, N_t=4, L=0.4, Q_s=1e-9, Q_t=1e-9, charge_model="tip",
               coulomb_include_spider=False)
    xi, _, topo = initial_state(P)
    tip = int(topo.tip_nodes[0])
    f = _coulomb_force(P, xi, topo)
    # Threads do not feel the spider; the spider does feel the tip.
    assert np.linalg.norm(f[3 * tip: 3 * tip + 3]) == 0.0
    assert np.linalg.norm(f[0:3]) > 0.0

    P.coulomb_include_spider = True
    f_on = _coulomb_force(P, xi, topo)
    assert np.linalg.norm(f_on[3 * tip: 3 * tip + 3]) > 0.0


def test_entangled_iff_closer_than_two_radii():
    P = Params(N=2, N_t=4, L=0.4, r=300e-9)
    xi, _, topo = initial_state(P)
    assert _entangled(P, topo, xi) is False

    X = topo.positions(xi)
    a = int(topo.thread_nodes[0][3])
    b = int(topo.thread_nodes[1][3])
    X[b] = X[a] + np.array([1.5 * P.r, 0.0, 0.0])  # closer than 2r
    assert _entangled(P, topo, xi) is True

    X[b] = X[a] + np.array([2.5 * P.r, 0.0, 0.0])
    assert _entangled(P, topo, xi) is False
