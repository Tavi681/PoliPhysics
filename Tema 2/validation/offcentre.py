"""Off-centre impact reference (quasi-static, gripped): compare with ref/offc.py.

A point ``p`` at distance ``a`` from the hub on radial 0 is displaced slowly
along -z (the gripped drone pulling the thread at a point). We compare, at first
failure:

* eta_A(a) with ``ref/offc.run`` (a/R=0.5: ~0.068 S, 0.069 D; a/R=0.05: 0.176 S,
  0.224 D);
* which segment fails first (inner for a/R<=0.1, outer for a/R>=0.2).

The ``frictionless`` mode is also run and reported without correction (radial 0
is expected to fail as a whole there).

Light viscous damping is used only to reach quasi-static equilibrium; it removes
kinetic energy but not stored strain energy, so the elastic-energy-based eta is
the quasi-static value.
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

from netsim.config import (SimConfig, MaterialConfig, NetConfig, DroneConfig,
                           NumericsConfig, ContactConfig, OutputConfig,
                           KinematicConfig)
from netsim.materials import get_material
from netsim.discretize import discretize
from netsim.simulate import build_net, simulate


def _p_node_index(disc, a_over_R):
    if np.isclose(a_over_R, 0.0, atol=1e-12):
        return 0  # the hub (coarse node 0) is the centre impact point
    mask = (disc.node_parent_thread == 0) & np.isclose(
        disc.node_thread_pos, a_over_R, atol=1e-9)
    idx = np.nonzero(mask)[0]
    if idx.size != 1:
        raise ValueError(
            f"expected one node at fraction {a_over_R}, found {idx.size}; "
            "choose n_s so that a/R is a grid point")
    return int(idx[0])


def run_offcentre(material_name="S", a_over_R=0.5, eps_p_frac=0.0,
                  mode="gripped", N=8, R=1.0, n_s=20, speed=0.3, A_hat=1e-6,
                  damping=400.0):
    material = get_material(material_name)
    eps_p = eps_p_frac * material.eps_b
    net = star(material_name, N, R, eps_p, A_hat)
    disc = discretize(net, material, n_s, r_d=0.15)
    pnode = _p_node_index(disc, a_over_R)

    cfg = SimConfig(
        material=MaterialConfig(name=material_name),
        net=NetConfig(kind="star", N=N, R=R, eps_p=eps_p, A_hat=A_hat),
        drone=DroneConfig(M=1.0, r_d=0.15, v0=0.0),
        numerics=NumericsConfig(n_s=n_s, C=0.5, t_end=3.0 * R / speed,
                                dt_out=1e-3, damping=damping, use_numba=False),
        contact=ContactConfig(mode=mode, k_c=1e7),
        output=OutputConfig(hdf5=None, R_max=10.0, k_max=10 * N * n_s),
        kinematic=KinematicConfig(enabled=True, node=pnode, mode="displacement",
                                  direction=(0.0, 0.0, -1.0),
                                  func=lambda t, s=speed: s * t),
    )
    material = cfg.material.resolve()
    net = build_net(cfg)
    res = simulate(net, material, cfg.drone, cfg.numerics, cfg.contact,
                   cfg.output, kinematic=cfg.kinematic, stop_on_failure=True)
    traj = res.trajectory

    if traj.failures.shape[0] == 0:
        return {"eta_ff": float("nan"), "first_kind": "none", "result": res}

    # First failure: earliest time.
    order = np.argsort(traj.failures[:, 2])
    first = traj.failures[order[0]]
    t_first = float(first[2])
    seg_idx = int(first[0])
    parent = int(first[1])

    # Classify inner / outer / radial.
    mid = traj.failure_midpoints[order[0]]
    if parent == 0:
        first_kind = "in" if mid[0] < a_over_R * R else "out"
    else:
        first_kind = "rad"

    # eta at first failure from the last frame strictly before t_first.
    e_mat = material.e_mat
    denom = res.m_net * e_mat
    before = np.nonzero(traj.t < t_first)[0]
    i = before[-1] if before.size else 0
    u_el = traj.energy[i, 2]
    absorbed = u_el - traj.U_prestress
    eta_ff = absorbed / denom if denom > 0 else float("nan")
    return {"eta_ff": eta_ff, "first_kind": first_kind, "t_first": t_first,
            "seg_idx": seg_idx, "parent": parent, "result": res}


# star builder that matches ref A_hat convention (import here to avoid cycle).
from netsim.topology import star as _star_topo


def star(material_name, N, R, eps_p, A_hat):
    return _star_topo(N, R, eps_p, material=get_material(material_name),
                      A_hat=A_hat)


def main():
    import offc  # ref/offc.py

    print("Off-centre quasi-static (gripped): eta_A and first-failure segment")
    print(f"{'mat':>3} {'a/R':>5} {'eta sim':>8} {'eta ref':>8} "
          f"{'first sim':>9} {'first ref':>9}")
    for name in ("S", "D"):
        for a in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
            sim = run_offcentre(name, a_over_R=a)
            ref_fails = offc.run(name, 8, a, 0.0)
            ref_first_kind, _, ref_eta = ref_fails[0]
            print(f"{name:>3} {a:>5} {sim['eta_ff']:>8.3f} {ref_eta:>8.3f} "
                  f"{sim['first_kind']:>9} {ref_first_kind:>9}")

    print("\nNote: the frictionless/gripped distinction is a property of the "
          "drone contact model and is exercised by the dynamic drop\n"
          "(configs/reference_star_N8_S*.yaml), not by this single-node "
          "kinematic quasi-static pull. In the dynamic frictionless drop\n"
          "radial 0 fails as a whole (the drone slides / perforates), as "
          "reported in the run summary.")


if __name__ == "__main__":
    main()
