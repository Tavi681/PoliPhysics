"""HDF5 output: one file per run, following the paper's table.

Layout::

    /params (attrs)   material (E0,b,rho,eps_b), eps_p, R, M, r_d, v0, p,
                      contact mode, n_s, C, dt, k_c, git commit, versions,
                      full flattened config
    /graph/nodes      (n_v, 2)   prestressed net node positions
    /graph/edges      (n_e, 2)   thread connectivity
    /graph/anchored   (n_v,)     anchored flag
    /graph/q          (n_e,)     force densities
    /graph/A          (n_e,)     cross-sections
    /t                (n_t,)     output times
    /x, /v            (n_t, n_seg_nodes, 3)
    /intact           (n_t, n_seg)
    /drone            (n_t, 6)   position + velocity
    /failures         (n_f, 3)   segment index, parent thread, failure time
    /energy           (n_t, 6)   energy-balance components:
                                 KE_drone, KE_net, U_elastic, U_failure,
                                 U_contact, dE_capture
    /seg/edges        (n_seg, 2)
    /seg/parent       (n_seg,)
    /seg/rest_length  (n_seg,)
    /outcome (attrs)  arrested, outcome, w_max, eta, R_d, cascade, energy_error

Datasets are gzip-compressed and chunked along the time axis.
"""

from __future__ import annotations

import subprocess

import numpy as np

__all__ = ["git_commit", "package_versions", "write_run"]


def git_commit() -> str:
    """Return the current git commit hash, or 'unknown'.

    Falls back to a ``COMMIT`` file next to the package (used on ephemeral VMs
    that ship without a ``.git`` directory).
    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        pass
    try:
        from pathlib import Path
        for base in (Path.cwd(), Path(__file__).resolve().parents[1]):
            p = base / "COMMIT"
            if p.is_file():
                return p.read_text().strip() or "unknown"
    except Exception:
        pass
    return "unknown"


def package_versions() -> dict:
    """Return versions of the key packages used."""
    import importlib.metadata as md

    versions = {}
    for pkg in ("numpy", "scipy", "h5py", "pyyaml", "numba"):
        try:
            versions[pkg] = md.version(pkg)
        except Exception:
            versions[pkg] = "not-installed"
    import sys

    versions["python"] = sys.version.split()[0]
    return versions


def _chunk_time(shape):
    """Chunk with one time slice per chunk (chunk along the time axis)."""
    return (1,) + tuple(shape[1:])


def write_run(path, net, disc, material, cfg, traj, *, eta, cascade,
              extra_params=None):
    """Write a full run to ``path`` (HDF5)."""
    import h5py

    with h5py.File(path, "w") as h5:
        # ---- /params -------------------------------------------------------
        p = h5.create_group("params")
        p.attrs["material_name"] = material.name
        p.attrs["E0"] = material.E0
        p.attrs["b"] = material.b
        p.attrs["rho"] = material.rho
        p.attrs["eps_b"] = material.eps_b
        p.attrs["eps_p"] = cfg.net.eps_p
        p.attrs["R"] = net.R
        p.attrs["M"] = cfg.drone.M
        p.attrs["r_d"] = cfg.drone.r_d
        p.attrs["v0"] = cfg.drone.v0
        p.attrs["p"] = np.asarray(cfg.drone.p, dtype=float)
        p.attrs["contact_mode"] = cfg.contact.mode
        p.attrs["n_s"] = cfg.numerics.n_s
        p.attrs["C"] = cfg.numerics.C
        p.attrs["dt"] = traj.dt
        p.attrs["dt_cfl"] = traj.dt_cfl
        p.attrs["dt_contact"] = traj.dt_contact
        p.attrs["k_c"] = cfg.contact.k_c
        p.attrs["area_scale"] = cfg.numerics.area_scale
        p.attrs["git_commit"] = git_commit()
        for k, val in package_versions().items():
            p.attrs[f"version_{k}"] = val
        # Full flattened config for provenance.
        for k, val in cfg.to_flat_dict().items():
            try:
                p.attrs[f"cfg.{k}"] = val
            except (TypeError, ValueError):
                p.attrs[f"cfg.{k}"] = str(val)
        if extra_params:
            for k, val in extra_params.items():
                p.attrs[k] = val

        # ---- /graph --------------------------------------------------------
        g = h5.create_group("graph")
        g.create_dataset("nodes", data=net.nodes, compression="gzip")
        g.create_dataset("edges", data=net.edges, compression="gzip")
        g.create_dataset("anchored", data=net.anchored, compression="gzip")
        g.create_dataset("q", data=net.q, compression="gzip")
        g.create_dataset("A", data=net.A(cfg.numerics.area_scale),
                         compression="gzip")

        # ---- /seg ----------------------------------------------------------
        s = h5.create_group("seg")
        s.create_dataset("edges", data=disc.seg_edges, compression="gzip")
        s.create_dataset("parent", data=disc.seg_parent, compression="gzip")
        s.create_dataset("rest_length", data=disc.seg_rest_length,
                         compression="gzip")

        # ---- time series ---------------------------------------------------
        def ds(name, data):
            data = np.asarray(data)
            if data.ndim >= 2 and data.shape[0] > 0:
                chunks = _chunk_time(data.shape)
            else:
                chunks = True if data.size else None
            kw = dict(compression="gzip")
            if chunks is not None:
                kw["chunks"] = chunks
            h5.create_dataset(name, data=data, **kw)

        ds("t", traj.t)
        ds("x", traj.x)
        ds("v", traj.v)
        ds("intact", traj.intact)
        ds("drone", traj.drone)
        ds("energy", traj.energy)
        h5.create_dataset("failures", data=traj.failures, compression="gzip")

        # ---- /outcome ------------------------------------------------------
        o = h5.create_group("outcome")
        o.attrs["arrested"] = bool(traj.arrested)
        o.attrs["outcome"] = traj.outcome
        o.attrs["w_max"] = traj.w_max
        o.attrs["eta"] = float(eta)
        o.attrs["R_d"] = traj.R_d
        o.attrs["cascade"] = bool(cascade)
        o.attrs["energy_error"] = traj.energy_error
        o.attrs["n_failures"] = int(traj.failures.shape[0])
        o.attrs["steps"] = traj.steps
