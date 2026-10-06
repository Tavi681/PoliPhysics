"""Algorithm 1 entry point: :func:`simulate` and :func:`simulate_config`.

``simulate`` takes the dataclasses (net, material, drone, numerics, contact,
output, and optional kinematic forcing) and returns a :class:`Result`. It can be
called with all cross-sections scaled by a factor ``s`` (A_e = s * A_hat_e)
without rebuilding the topology, in preparation for Algorithm 2 (not implemented
here).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .discretize import discretize
from .integrator import integrate, Trajectory
from .topology import Net, star, star_with_rings

__all__ = ["Result", "simulate", "simulate_config", "build_net"]


@dataclass
class Result:
    """Outcome and trajectory of a run."""

    outcome: str
    arrested: bool
    w_max: float
    eta: float
    n_failures: int
    R_d: float
    cascade: bool
    energy_error: float
    trajectory: Trajectory
    net: Net
    material: object
    m_net: float
    drone_x_arrest: float = float("nan")
    drone_y_arrest: float = float("nan")
    drone_x_first_fail: float = float("nan")
    drone_y_first_fail: float = float("nan")

    def summary(self) -> dict:
        return {
            "outcome": self.outcome,
            "arrested": self.arrested,
            "w_max": self.w_max,
            "eta": self.eta,
            "n_failures": self.n_failures,
            "R_d": self.R_d,
            "cascade": self.cascade,
            "energy_error": self.energy_error,
            "drone_x_arrest": self.drone_x_arrest,
            "drone_y_arrest": self.drone_y_arrest,
        }


def simulate(net, material, drone, numerics, contact, output, kinematic=None, *,
             area_scale: Optional[float] = None,
             stop_on_failure: bool = False,
             stop_after_n_failures: Optional[int] = None) -> Result:
    """Run Algorithm 1 for the given configuration objects."""
    if area_scale is None:
        area_scale = numerics.area_scale

    disc = discretize(net, material, numerics.n_s, area_scale=area_scale,
                      r_d=drone.r_d)

    traj = integrate(disc, material, drone, numerics, contact, output,
                     kinematic=kinematic, net_R=net.R, impact_point=drone.p,
                     stop_on_failure=stop_on_failure,
                     stop_after_n_failures=stop_after_n_failures)

    m_net = net.net_mass(material.rho, area_scale)
    e_mat = material.e_mat

    # eta: energy absorbed by the net (elastic + failure - prestress) / (m_net e_mat).
    u_el = float(traj.energy[-1, 2])
    u_fail = float(traj.energy[-1, 3])
    absorbed = (u_el + u_fail) - traj.U_prestress
    eta = absorbed / (m_net * e_mat) if (m_net * e_mat) > 0 else 0.0

    n_failures = int(traj.failures.shape[0])
    cascade = (n_failures >= output.k_max) or (traj.R_d > output.R_max)

    return Result(
        outcome=traj.outcome,
        arrested=traj.arrested,
        w_max=traj.w_max,
        eta=eta,
        n_failures=n_failures,
        R_d=traj.R_d,
        cascade=cascade,
        energy_error=traj.energy_error,
        trajectory=traj,
        net=net,
        material=material,
        m_net=m_net,
        drone_x_arrest=traj.drone_x_arrest,
        drone_y_arrest=traj.drone_y_arrest,
        drone_x_first_fail=traj.drone_x_first_fail,
        drone_y_first_fail=traj.drone_y_first_fail,
    )


def build_net(cfg) -> Net:
    """Construct the :class:`Net` described by ``cfg.net``."""
    material = cfg.material.resolve()
    nc = cfg.net
    if nc.kind == "star":
        return star(nc.N, nc.R, nc.eps_p, material=material, A_hat=nc.A_hat,
                    eps_p_thread0=getattr(nc, "eps_p_thread0", None))
    if nc.kind == "star_with_rings":
        return star_with_rings(nc.N, nc.R, nc.radii, nc.eps_p,
                               material=material, A_hat=nc.A_hat,
                               q_ratio=nc.q_ratio,
                               fix_radii=getattr(nc, "fix_radii", True))
    raise ValueError(f"unknown net kind {nc.kind!r}")


def simulate_config(cfg, *, write: bool = True) -> Result:
    """Build the net from ``cfg`` and run; optionally write HDF5."""
    cfg.validate()
    material = cfg.material.resolve()
    net = build_net(cfg)

    result = simulate(net, material, cfg.drone, cfg.numerics, cfg.contact,
                      cfg.output, kinematic=cfg.kinematic)

    if write and cfg.output.hdf5:
        from .io_hdf5 import write_run
        from .discretize import discretize as _discretize

        disc = _discretize(net, material, cfg.numerics.n_s,
                           area_scale=cfg.numerics.area_scale, r_d=cfg.drone.r_d)
        write_run(cfg.output.hdf5, net, disc, material, cfg, result.trajectory,
                  eta=result.eta, cascade=result.cascade,
                  light=bool(getattr(cfg.output, "hdf5_light", False)),
                  max_frames=int(getattr(cfg.output, "hdf5_max_frames", 50)))
    return result
