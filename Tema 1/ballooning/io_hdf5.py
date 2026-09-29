"""HDF5 output, one file per run (io_hdf5.py).

Layout (per the objective):
  /params  (attrs): all physical and numerical parameters, field model, charge
                    model, z0, h, git commit hash.
  /turb    (attrs, reserved): sigma_w, ell, U_h, lambda, seed, k_n, a_n, b_n, omega_n.
  /t       (n_t,)
  /x       (n_t, n, 3)
  /v       (n_t, n, 3)
  /theta   (n_t, N N_t)
  /edges   (N N_t, 2)
  /thread_id (n,)     -1 for the spider
  /q_node  (n,)
  /outcome (attrs): steady/rise/fall/timeout, exit time, entangled flag, V, R, theta_L.
"""

from __future__ import annotations

import subprocess

import h5py
import numpy as np

from .params import Params
from .geometry import Topology
from .integrator import Trajectory
from . import observables


def git_commit_hash() -> str:
    """Return the current git commit hash, or '' if unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return ""


_PARAM_ATTRS = [
    "N", "N_t", "L", "r", "d0", "z0", "m", "r_s", "Q_s", "rho_t", "Y", "nu",
    "mu", "g", "k_e", "Q_t", "charge_model", "field_model", "flow_model",
    "E_constant", "dt0", "dt_max", "dt_min", "eps", "K", "t_end", "output_dt",
    "delta", "t_w", "use_alg2_stopping", "h", "lag_tangent",
    "coulomb_include_spider", "entangle_contact_factor",
]


def write_trajectory(path: str, P: Params, traj: Trajectory) -> None:
    """Write a full run to a single HDF5 file."""
    topo = Topology(P)
    with h5py.File(path, "w") as f:
        # /params
        gp = f.create_group("params")
        for name in _PARAM_ATTRS:
            gp.attrs[name] = getattr(P, name)
        gp.attrs["flow_velocity"] = np.asarray(P.flow_velocity, dtype=float)
        gp.attrs["first_integral_beta"] = P.beta_first_integral
        gp.attrs["git_commit"] = git_commit_hash()

        # /turb (reserved)
        gturb = f.create_group("turb")
        for key in ("sigma_w", "ell", "U_h", "lambda", "seed",
                    "k_n", "a_n", "b_n", "omega_n"):
            if key in P.turb:
                gturb.attrs[key] = P.turb[key]

        # datasets
        f.create_dataset("t", data=traj.t)
        f.create_dataset("x", data=traj.x)
        f.create_dataset("v", data=traj.v)
        f.create_dataset("theta", data=traj.theta)
        f.create_dataset("edges", data=traj.edges)
        f.create_dataset("thread_id", data=traj.thread_id)
        f.create_dataset("q_node", data=traj.q_node)

        # /outcome
        go = f.create_group("outcome")
        Xf = traj.x[-1]
        Vf = traj.v[-1]
        obs = observables.summary(P, topo, Xf, Vf)
        go.attrs["status"] = traj.outcome.get("status", "timeout")
        exit_time = traj.outcome.get("exit_time")
        go.attrs["exit_time"] = -1.0 if exit_time is None else float(exit_time)
        go.attrs["entangled"] = bool(traj.outcome.get("entangled", False))
        go.attrs["steady_state"] = bool(traj.outcome.get("steady_state", False))
        go.attrs["V"] = obs["V"]
        go.attrs["R"] = obs["R"]
        go.attrs["theta_L"] = obs["theta_L"]
