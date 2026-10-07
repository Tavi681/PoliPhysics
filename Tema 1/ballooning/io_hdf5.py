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

import os
import subprocess
from pathlib import Path

import h5py
import numpy as np

from .params import Params
from .geometry import Topology
from .integrator import Trajectory
from . import forces
from . import observables


def git_provenance() -> dict:
    """Commit hash + dirty-tree flag.

    Prefers a live ``git`` checkout (repo root = parent of ``Tema 1/``). On a
    packed VM tree without ``.git``, falls back to ``Tema 1/COMMIT`` and
    ``Tema 1/COMMIT.dirty`` written by the GCP packers.
    """
    tema1 = Path(__file__).resolve().parents[1]
    repo = tema1.parent
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty_out = subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        ).decode()
        return {"commit": commit, "dirty": bool(dirty_out.strip()),
                "source": "git"}
    except Exception:
        pass
    commit = ""
    cf = tema1 / "COMMIT"
    if cf.exists():
        commit = cf.read_text().strip()
    dirty = False
    df = tema1 / "COMMIT.dirty"
    if df.exists():
        dirty = df.read_text().strip() not in ("", "clean", "0", "false")
    elif commit:
        dirty = True
    return {"commit": commit, "dirty": dirty,
            "source": "COMMIT" if commit else "none"}


def git_commit_hash() -> str:
    """Return the current git commit hash, or '' if unavailable."""
    return git_provenance()["commit"]


def git_dirty() -> bool:
    return bool(git_provenance()["dirty"])


def _write_git_attrs(group) -> None:
    prov = git_provenance()
    group.attrs["git_commit"] = prov["commit"]
    group.attrs["git_dirty"] = bool(prov["dirty"])


ML_T6_DATASETS = {
    "/t", "/x", "/v", "/twist", "/u_air", "/charge",
    "/topology/thread_id", "/topology/node_type", "/topology/edges",
}


def _atomic_replace(path: Path, tmp: Path) -> None:
    """Rename ``tmp`` → ``path`` (same filesystem)."""
    os.replace(tmp, path)


def write_hdf5_atomic(path: str, writer) -> None:
    """Write HDF5 via ``writer(tmp_path)`` then rename (crash-safe)."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    try:
        writer(str(tmp))
        _atomic_replace(dest, tmp)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


_PARAM_ATTRS = [
    "N", "N_t", "L", "r", "d0", "z0", "m", "r_s", "Q_s", "rho_t", "Y", "nu",
    "mu", "g", "k_e", "Q_t", "charge_model", "field_model", "flow_model",
    "E_constant", "dt0", "dt_max", "dt_min", "eps", "K", "t_end", "output_dt",
    "adaptive_dt", "delta", "t_w", "stop_on_steady", "use_alg2_stopping", "h",
    "release_mode", "clamp_relax_time",
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
            val = getattr(P, name)
            if val is None:
                continue
            gp.attrs[name] = val
        gp.attrs["flow_velocity"] = np.asarray(P.flow_velocity, dtype=float)
        gp.attrs["first_integral_beta"] = P.beta_first_integral
        _write_git_attrs(gp)

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


def write_ml_hdf5(path: str, P: Params, traj: Trajectory,
                  u_air: np.ndarray | None = None,
                  float32: bool = True) -> None:
    """Stage B ML record (Table tab:hdf5). Additive; does not touch ``/theta``.

    Layout:
      /t (T), /x (T, n, 3), /v (T, n, 3), /twist (T, n_edges),
      /u_air (T, n, 3) recomputed from ``KinematicSimulation`` at saved frames,
      /topology (thread_id, node_type, edges), /charge (n,),
      /params, /turb (incl. renorm_factor), /outcome.

    Frame 0 is the clamped-release state (``t = 0`` of phase 2). Default
    storage is float32. The legacy ``write_trajectory`` / ``/theta`` API is
    unchanged for ``produce_results.py``.
    """
    fdtype = np.float32 if float32 else np.float64
    topo = Topology(P)
    if u_air is None:
        from .fields import make_flow
        flow = make_flow(P)
        u_air = np.stack([
            flow.u(traj.x[k], float(traj.t[k])) for k in range(len(traj.t))
        ])
    u_air = np.asarray(u_air)

    node_type = np.full(topo.n_nodes, "interior", dtype=object)
    node_type[0] = "spider"
    for tip in topo.tip_nodes:
        node_type[int(tip)] = "tip"
    str_dt = h5py.string_dtype(encoding="utf-8")

    def _write(path_out: str) -> None:
        with h5py.File(path_out, "w") as f:
            gp = f.create_group("params")
            for name in _PARAM_ATTRS:
                val = getattr(P, name)
                if val is None:
                    continue
                gp.attrs[name] = val
            gp.attrs["flow_velocity"] = np.asarray(P.flow_velocity, dtype=float)
            gp.attrs["first_integral_beta"] = P.beta_first_integral
            _write_git_attrs(gp)

            gturb = f.create_group("turb")
            for key, val in P.turb.items():
                arr = np.asarray(val)
                if arr.ndim == 0:
                    gturb.attrs[key] = val
                else:
                    data = arr.astype(fdtype) if arr.dtype.kind == "f" else arr
                    gturb.create_dataset(key, data=data)

            f.create_dataset("t", data=np.asarray(traj.t, dtype=fdtype))
            f.create_dataset("x", data=np.asarray(traj.x, dtype=fdtype))
            f.create_dataset("v", data=np.asarray(traj.v, dtype=fdtype))
            f.create_dataset("twist", data=np.asarray(traj.theta, dtype=fdtype))
            f.create_dataset("u_air", data=np.asarray(u_air, dtype=fdtype))
            f.create_dataset("charge", data=np.asarray(traj.q_node, dtype=fdtype))

            gtop = f.create_group("topology")
            gtop.create_dataset("thread_id", data=np.asarray(topo.thread_id))
            gtop.create_dataset("node_type", data=np.asarray(node_type, dtype=object),
                                dtype=str_dt)
            gtop.create_dataset("edges", data=np.asarray(topo.edges))

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

    write_hdf5_atomic(path, _write)


def write_ml_hdf5_solver_fail(
    path: str,
    P: Params,
    *,
    t_fail: float | None,
    dt_fail: float | None,
    message: str,
    traj: Trajectory | None = None,
    u_air: np.ndarray | None = None,
    float32: bool = True,
) -> None:
    """Minimal ML record when the integrator raises (future runs).

    Uses ``traj`` when the worker captured a partial trajectory; otherwise a
    single placeholder frame so T6 datasets exist.
    """
    fdtype = np.float32 if float32 else np.float64
    topo = Topology(P)
    if traj is None or len(traj.t) == 0:
        from .geometry import initial_state

        xi0, xi_dot0, _ = initial_state(P)
        X = topo.positions(xi0)
        V = topo.positions(xi_dot0)
        th = topo.thetas(xi0)
        traj = Trajectory(
            t=np.array([0.0], dtype=float),
            x=X.reshape(1, *X.shape),
            v=V.reshape(1, *V.shape),
            theta=th.reshape(1, *th.shape),
            edges=topo.edges.copy(),
            thread_id=topo.thread_id.copy(),
            q_node=forces.node_charges(P, topo),
            outcome={"status": "solver_fail", "exit_time": t_fail, "entangled": False,
                     "diag": {"newton_failures": 0, "n_steps": 0}},
        )
    if u_air is None:
        from .fields import make_flow

        flow = make_flow(P)
        u_air = np.stack([flow.u(traj.x[k], float(traj.t[k])) for k in range(len(traj.t))])
    write_ml_hdf5(path, P, traj, u_air=u_air, float32=float32)
    with h5py.File(path, "a") as f:
        go = f["outcome"]
        go.attrs["status"] = "solver_fail"
        if t_fail is not None:
            go.attrs["t_fail"] = float(t_fail)
            go.attrs["exit_time"] = float(t_fail)
        if dt_fail is not None:
            go.attrs["dt_fail"] = float(dt_fail)
        go.attrs["fail_message"] = str(message)[:4000]
