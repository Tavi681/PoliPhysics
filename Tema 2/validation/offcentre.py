"""Off-centre impact reference (quasi-static, gripped): compare with ref/offc.py.

A point ``p`` at distance ``a`` from the hub on radial 0 is displaced slowly
along -z (the gripped drone pulling the thread at a point). We compare, at first
failure and at the next failure after re-equilibration:

* eta_A(a), eta_B(a) with ``ref/offc.run``;
* which segment fails first (inner for a/R<=0.1, outer for a/R>=0.2).

``eta`` is absorbed work ``W/(m_net e_mat)`` with prestress subtracted, matching
the updated ``ref/offc.py``.
"""

from __future__ import annotations

import math
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
from netsim._kernels import segment_forces


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


def _classify_failure(disc, seg_idx, parent, mid, a_over_R, R):
    if parent == 0:
        return "in" if mid[0] < a_over_R * R else "out"
    return "rad"


def _absorbed(traj, i, m_net, e_mat):
    u_el = float(traj.energy[i, 2])
    u_fail = float(traj.energy[i, 3])
    absorbed = (u_el + u_fail) - traj.U_prestress
    return absorbed / (m_net * e_mat) if (m_net * e_mat) > 0 else float("nan")


def run_offcentre(material_name="S", a_over_R=0.5, eps_p_frac=0.0,
                  mode="gripped", N=8, R=1.0, n_s=20, speed=0.3, A_hat=1e-6,
                  damping=400.0, continue_to_B=True, free_lateral=False):
    """Quasi-static off-centre pull. Returns eta_A, eta_B and failure kinds.

    With ``continue_to_B=True`` the run does not stop at first failure: the
    failed segments stay permanently removed, the prescribed displacement
    continues, and ``eta_B`` is the absorbed work at the next failure (same
    definition as ``ref/offc.py``).

    With ``free_lateral=True`` only the vertical component of the gripped node
    is prescribed; the in-plane coordinates are free (matches
    ``ref/offc_free.py``). Then only criterion A is computed.
    """
    material = get_material(material_name)
    eps_p = eps_p_frac * material.eps_b
    net = star(material_name, N, R, eps_p, A_hat)
    disc = discretize(net, material, n_s, r_d=0.15)
    pnode = _p_node_index(disc, a_over_R)

    cfg = SimConfig(
        material=MaterialConfig(name=material_name),
        net=NetConfig(kind="star", N=N, R=R, eps_p=eps_p, A_hat=A_hat),
        drone=DroneConfig(M=1.0, r_d=0.15, v0=0.0),
        numerics=NumericsConfig(n_s=n_s, C=0.5, t_end=5.0 * R / speed,
                                dt_out=1e-3, damping=damping, use_numba=False),
        contact=ContactConfig(mode=mode, k_c=1e7),
        output=OutputConfig(hdf5=None, R_max=10.0, k_max=10 * N * n_s),
        kinematic=KinematicConfig(enabled=True, node=pnode, mode="displacement",
                                  direction=(0.0, 0.0, -1.0),
                                  free_lateral=free_lateral,
                                  func=lambda t, s=speed: s * t),
    )
    material = cfg.material.resolve()
    net = build_net(cfg)
    # Free-lateral runs only need first failure (offc_free is criterion A only).
    # Fixed continue_to_B needs two failures for eta_A and eta_B, then can stop.
    if free_lateral or not continue_to_B:
        stop_on_failure = True
        stop_after = None
    else:
        stop_on_failure = False
        stop_after = 2
    res = simulate(net, material, cfg.drone, cfg.numerics, cfg.contact,
                   cfg.output, kinematic=cfg.kinematic,
                   stop_on_failure=stop_on_failure,
                   stop_after_n_failures=stop_after)
    traj = res.trajectory

    empty = {
        "eta_ff": float("nan"), "eta_A": float("nan"), "eta_B": float("nan"),
        "eta_A_free": float("nan"), "px_fail": float("nan"),
        "first_kind": "none", "second_kind": "none",
        "t_first": float("nan"), "t_second": float("nan"),
        "result": res,
    }
    if traj.failures.shape[0] == 0:
        return empty

    order = np.argsort(traj.failures[:, 2])
    fails = traj.failures[order]
    mids = traj.failure_midpoints[order]
    e_mat = material.e_mat
    denom = res.m_net * e_mat

    # --- First failure (eta_A) ------------------------------------------------
    t_first = float(fails[0, 2])
    seg0 = int(fails[0, 0])
    parent0 = int(fails[0, 1])
    first_kind = _classify_failure(disc, seg0, parent0, mids[0], a_over_R, R)
    before = np.nonzero(traj.t < t_first)[0]
    iA = before[-1] if before.size else 0
    u_el = traj.energy[iA, 2]
    eta_A = (u_el - traj.U_prestress) / denom if denom > 0 else float("nan")
    # Lateral position of the gripped node at failure.
    px_fail = float(traj.x[iA, pnode, 0])

    eta_B = float("nan")
    second_kind = "none"
    t_second = float("nan")
    if (not free_lateral) and fails.shape[0] >= 2:
        t0 = t_first
        # Skip co-cascade segments (same event within ~0.1 ms); criterion B is
        # the next failure after load redistribution (cf. ref/offc.py).
        gap = 1e-4
        jB = None
        for j in range(1, fails.shape[0]):
            if float(fails[j, 2]) > t0 + gap:
                jB = j
                break
        if jB is not None:
            t_second = float(fails[jB, 2])
            parentB = int(fails[jB, 1])
            second_kind = _classify_failure(
                disc, int(fails[jB, 0]), parentB, mids[jB], a_over_R, R)
            beforeB = np.nonzero(traj.t < t_second)[0]
            iB = beforeB[-1] if beforeB.size else iA
            eta_B = _absorbed(traj, iB, res.m_net, e_mat)

    return {
        "eta_ff": eta_A, "eta_A": eta_A, "eta_B": eta_B,
        "eta_A_free": eta_A if free_lateral else float("nan"),
        "px_fail": px_fail,
        "first_kind": first_kind, "second_kind": second_kind,
        "t_first": t_first, "t_second": t_second,
        "seg_idx": seg0, "parent": parent0, "result": res,
    }


def hub_force_deflection(material_name="S", N=8, R=1.0, eps_p_frac=0.1,
                         n_s=20, speed=0.3, A_hat=1e-6, damping=400.0,
                         w_over_R=None):
    """Quasi-static hub pull: vertical restoring force vs deflection.

    Returns rows ``(w/R, F_an, F_num, failed)``. ``F_num`` is the upward
    restoring force on the hub from the segments (equals the downward load).
    ``F_an`` uses the prestressed rest length ``R/(1+eps_p)``.
    """
    material = get_material(material_name)
    eps_p = eps_p_frac * material.eps_b
    if w_over_R is None:
        w_over_R = np.linspace(0.02, 0.6, 15)

    rest_len = R / (1.0 + eps_p)

    def F_analytic(w):
        L = math.sqrt(R * R + w * w)
        eps = L / rest_len - 1.0
        T = A_hat * float(material.sigma(max(eps, 0.0)))
        return N * T * (w / L) if L > 0 else 0.0

    net = star(material_name, N, R, eps_p, A_hat)
    disc = discretize(net, material, n_s, r_d=0.15)
    hub = 0

    w_max = float(np.max(w_over_R)) * R
    t_end = w_max / speed + 0.05
    cfg = SimConfig(
        material=MaterialConfig(name=material_name),
        net=NetConfig(kind="star", N=N, R=R, eps_p=eps_p, A_hat=A_hat),
        drone=DroneConfig(M=1.0, r_d=0.15, v0=0.0),
        numerics=NumericsConfig(n_s=n_s, C=0.5, t_end=t_end, dt_out=5e-4,
                                damping=damping, use_numba=False),
        contact=ContactConfig(mode="gripped", k_c=1e7),
        output=OutputConfig(hdf5=None, R_max=10.0, k_max=10 * N * n_s),
        kinematic=KinematicConfig(enabled=True, node=hub, mode="displacement",
                                  direction=(0.0, 0.0, -1.0),
                                  func=lambda t, s=speed: s * t),
    )
    res = simulate(build_net(cfg), material, cfg.drone, cfg.numerics,
                   cfg.contact, cfg.output, kinematic=cfg.kinematic,
                   stop_on_failure=False)
    traj = res.trajectory

    rows = []
    for wor in w_over_R:
        w_target = wor * R
        wz = -traj.x[:, hub, 2]  # positive downward deflection
        i = int(np.argmin(np.abs(wz - w_target)))
        x = traj.x[i]
        intact = traj.intact[i]
        f = segment_forces(x, disc.seg_edges, disc.seg_rest_length, disc.seg_A,
                           intact, material.E0, material.b)
        # Threads pull the depressed hub upward => f[hub,2] > 0 = restoring.
        F_num = float(f[hub, 2])
        F_an = F_analytic(w_target)
        eps_max = float(np.max(
            np.linalg.norm(x[disc.seg_edges[:, 1]] - x[disc.seg_edges[:, 0]],
                           axis=1) / disc.seg_rest_length - 1.0))
        failed = bool(np.any(~intact)) or bool(eps_max >= material.eps_b * 0.999)
        rows.append({
            "material": material_name,
            "w_over_R": float(wor),
            "F_an": float(F_an),
            "F_num": float(F_num),
            "failed": bool(failed),
        })
    return rows


# star builder that matches ref A_hat convention (import here to avoid cycle).
from netsim.topology import star as _star_topo


def star(material_name, N, R, eps_p, A_hat):
    return _star_topo(N, R, eps_p, material=get_material(material_name),
                      A_hat=A_hat)


def main():
    import offc  # ref/offc.py

    print("Off-centre quasi-static (gripped): eta_A / eta_B vs ref/offc")
    print(f"{'mat':>3} {'ep':>4} {'a/R':>5} {'etaA_sim':>8} {'etaA_ref':>8} "
          f"{'etaB_sim':>8} {'etaB_ref':>8} {'1st':>5}")
    for name in ("S", "D"):
        for epf in (0.0, 0.1):
            for a in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
                sim = run_offcentre(name, a_over_R=a, eps_p_frac=epf)
                ref = offc.run(name, 8, a, epf)
                etaA_ref = ref[0][2]
                etaB_ref = ref[1][2] if len(ref) > 1 else float("nan")
                print(f"{name:>3} {epf:>4.1f} {a:>5} {sim['eta_A']:>8.4f} "
                      f"{etaA_ref:>8.4f} {sim['eta_B']:>8.4f} "
                      f"{etaB_ref:>8.4f} {sim['first_kind']:>5}")


if __name__ == "__main__":
    main()
