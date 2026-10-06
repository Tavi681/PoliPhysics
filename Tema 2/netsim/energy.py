"""Energy-balance components (mandatory at every output frame).

The six components stored per output frame are

    [KE_drone, KE_net, U_elastic, U_failure, U_contact, dE_capture]

where

* KE_drone   = (1/2) M_eff |v_d|^2 (M_eff includes the gripped node's mass);
* KE_net     = (1/2) sum_i m_i |v_i|^2 over net nodes not slaved to the drone;
* U_elastic  = sum over intact segments of A_e (ell_e/n_s) Phi(eps);
* U_failure  = accumulated elastic energy booked at thread failures;
* U_contact  = (2/5) k_c sum delta^{5/2} over contacting pairs;
* dE_capture = energy dissipated in the inelastic capture of the gripped node,
               (1/2) (M m)/(M+m) |v_node - v_drone|^2, booked once at grip time
               (zero before capture and in frictionless mode).

The relative energy error reported at the end is normalized by
``E_kin,0 + U_prestress``.
"""

from __future__ import annotations

import numpy as np

from ._kernels import segment_strains

__all__ = ["elastic_energy", "elastic_energy_per_segment",
           "kinetic_energy_net", "kinetic_energy_drone"]


def elastic_energy_per_segment(x, seg_edges, seg_rest, seg_A, intact,
                               material) -> np.ndarray:
    """Elastic strain energy of each intact segment [J]."""
    eps, _ = segment_strains(x, seg_edges, seg_rest)
    phi = material.Phi(eps)
    return np.where(intact, seg_A * seg_rest * phi, 0.0)


def elastic_energy(x, seg_edges, seg_rest, seg_A, intact, material) -> float:
    """Elastic strain energy of the intact segments [J]."""
    eps, _ = segment_strains(x, seg_edges, seg_rest)
    phi = material.Phi(eps)  # zero for eps <= 0
    e = np.where(intact, seg_A * seg_rest * phi, 0.0)
    return float(np.sum(e))


def kinetic_energy_net(v, mass, exclude_mask=None) -> float:
    """Kinetic energy of the net nodes [J], excluding ``exclude_mask`` nodes."""
    ke = 0.5 * mass * np.sum(v * v, axis=1)
    if exclude_mask is not None:
        ke = np.where(exclude_mask, 0.0, ke)
    return float(np.sum(ke))


def kinetic_energy_drone(M_eff, vd) -> float:
    """Kinetic energy of the drone (with any gripped mass) [J]."""
    return float(0.5 * M_eff * np.dot(vd, vd))
