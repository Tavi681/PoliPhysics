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
    "adaptive_dt", "delta", "t_w", "stop_on_steady", "use_alg2_stopping", "h",
    "release_mode",
    "sigma_w", "ell", "U_h", "turb_N_k", "turb_seed", "turb_lambda",
    "turb_renormalize",
    "lag_tangent",
    "coulomb_include_spider", "entangle_contact_factor",
]


def write_trajectory(path: str, P: Params, traj: Trajectory,
                     float32: bool = False) -> None:
    """Write a full run to a single HDF5 file.

    ``float32=True`` stores the large trajectory arrays (t, x, v, theta) in single
    precision for compact ML datasets; metadata and outcome are unchanged.
    """
    fdtype = np.float32 if float32 else np.float64
    topo = Topology(P)
    with h5py.File(path, "w") as f:
        # /params
        gp = f.create_group("params")
        for name in _PARAM_ATTRS:
            gp.attrs[name] = getattr(P, name)
        gp.attrs["flow_velocity"] = np.asarray(P.flow_velocity, dtype=float)
        gp.attrs["first_integral_beta"] = P.beta_first_integral
        gp.attrs["git_commit"] = git_commit_hash()

        # /turb (populated for the kinematic flow model: sigma_w, ell, U_h,
        # lambda, seed, N_k, k_min_used, k_max, energy_fraction, k_n, a_n,
        # b_n, omega_n).
        gturb = f.create_group("turb")
        for key, val in P.turb.items():
            gturb.attrs[key] = val

        # datasets
        f.create_dataset("t", data=traj.t.astype(fdtype))
        f.create_dataset("x", data=traj.x.astype(fdtype))
        f.create_dataset("v", data=traj.v.astype(fdtype))
        f.create_dataset("theta", data=traj.theta.astype(fdtype))
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
        diag = traj.outcome.get("diag", {})
        for key in ("newton_failures", "max_abs_u", "mean_dt", "min_dt", "n_steps"):
            if key in diag:
                go.attrs[key] = diag[key]
