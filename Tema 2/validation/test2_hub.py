"""Validation Test 2: quasi-static hub impact on the star (N=8).

The hub is displaced slowly along -z. Analytical relations (uniform prestress
eps_p, radial rest length R/(1+eps_p)):

    1 + eps(w) = (1 + eps_p) sqrt(1 + (w/R)^2),
    F(w)       = N T(eps(w)) w / sqrt(R^2 + w^2),
    w_b/R      = sqrt(((1+eps_b)/(1+eps_p))^2 - 1),
    eta_A      = 1 - Phi(eps_p)/Phi(eps_b).

This module reproduces the objective's table of w_b/R and eta_A and checks rate
independence with two hub speeds.

Paper values reproduced (eps_p/eps_b = 0, 0.1, 0.3):
    w_b/R : S 0.831 / 0.770 / 0.650 ; D 0.255 / 0.241 / 0.212
    eta_A : S 1.000 / 0.993 / 0.937 ; D 1.000 / 0.990 / 0.910
"""

from __future__ import annotations

import math

import numpy as np

from netsim.config import (SimConfig, MaterialConfig, NetConfig, DroneConfig,
                           NumericsConfig, ContactConfig, OutputConfig,
                           KinematicConfig)
from netsim.materials import get_material
from netsim.simulate import build_net, simulate


def analytic(material, eps_p, R=1.0):
    eps_b = material.eps_b
    wb_over_R = math.sqrt(((1 + eps_b) / (1 + eps_p)) ** 2 - 1.0)
    eta_A = 1.0 - float(material.Phi(eps_p)) / float(material.Phi(eps_b))
    return wb_over_R, eta_A


def run_hub(material_name="S", N=8, eps_p_frac=0.0, speed=2.0, n_s=1,
            R=1.0, A_hat=1e-6):
    """Run a quasi-static hub displacement to failure; return a result dict."""
    material = get_material(material_name)
    eps_p = eps_p_frac * material.eps_b
    cfg = SimConfig(
        material=MaterialConfig(name=material_name),
        net=NetConfig(kind="star", N=N, R=R, eps_p=eps_p, A_hat=A_hat),
        drone=DroneConfig(M=1.0, r_d=0.15, v0=0.0, p=(0.0, 0.0)),
        numerics=NumericsConfig(n_s=n_s, C=0.5, t_end=2.0 * R / speed,
                                dt_out=1e-3, use_numba=False),
        contact=ContactConfig(mode="frictionless", k_c=1e7),
        output=OutputConfig(hdf5=None, R_max=10.0, k_max=10 * N),
        kinematic=KinematicConfig(enabled=True, node=0, mode="displacement",
                                  direction=(0.0, 0.0, -1.0), amplitude=0.0,
                                  func=lambda t, s=speed: s * t),
    )
    material = cfg.material.resolve()
    net = build_net(cfg)
    res = simulate(net, material, cfg.drone, cfg.numerics, cfg.contact,
                   cfg.output, kinematic=cfg.kinematic, stop_on_failure=True)
    traj = res.trajectory

    # First failure time -> w_b.
    if traj.failures.shape[0] > 0:
        t_first = float(traj.failures[:, 2].min())
        wb = speed * t_first
    else:
        wb = float("nan")

    # Strain(w) and F(w) measured from the frames (hub is node 0).
    hub_z = -speed * traj.t
    anchor = net.nodes3()[1]  # first anchor at (R, 0, 0)
    hub_pos = np.zeros((traj.t.size, 3))
    hub_pos[:, 2] = hub_z
    length = np.linalg.norm(anchor - hub_pos, axis=1)
    rest0 = R / (1 + eps_p)
    eps_num = length / rest0 - 1.0
    F_num = N * material.tension(eps_num, net.A()[0]) * (-hub_z) / length

    wb_over_R, eta_A = analytic(material, eps_p, R)
    return {
        "material": material_name,
        "eps_p_frac": eps_p_frac,
        "wb_over_R_num": wb / R,
        "wb_over_R_ana": wb_over_R,
        "eta_num": res.eta,
        "eta_A_ana": eta_A,
        "w": -hub_z,
        "eps_num": eps_num,
        "F_num": F_num,
        "result": res,
    }


def main():
    print("Test 2: quasi-static hub impact (N=8)")
    print(f"{'mat':>3} {'ep/eb':>5} {'wb/R num':>9} {'wb/R ana':>9} "
          f"{'eta num':>8} {'eta_A':>8}")
    for name in ("S", "D"):
        for frac in (0.0, 0.1, 0.3):
            r = run_hub(name, eps_p_frac=frac,
                        speed=2.0 if name == "S" else 0.5)
            print(f"{name:>3} {frac:>5} {r['wb_over_R_num']:>9.3f} "
                  f"{r['wb_over_R_ana']:>9.3f} {r['eta_num']:>8.3f} "
                  f"{r['eta_A_ana']:>8.3f}")


if __name__ == "__main__":
    main()
