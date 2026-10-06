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

import os
import subprocess

import numpy as np

__all__ = ["git_commit", "package_versions", "write_run", "select_light_frames",
           "select_fail_dense_frames", "write_diagnostic"]


def git_commit() -> str:
    """Return the current git commit hash, or 'unknown'.

    Order: ``GIT_COMMIT`` env (ephemeral VM), live ``git rev-parse``, then a
    ``COMMIT`` file next to the package.
    """
    env = os.environ.get("GIT_COMMIT", "").strip()
    if env:
        return env
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


def select_light_frames(traj, max_frames=50):
    """Indices that keep first/last, contact, failures, arrest; cap at ``max_frames``."""
    n = int(getattr(traj.t, "size", 0))
    if n <= 0:
        return np.zeros(0, dtype=int)
    if n <= max_frames:
        return np.arange(n, dtype=int)

    special = {0, n - 1}
    if getattr(traj, "failures", None) is not None and traj.failures.size:
        t = np.asarray(traj.t)
        for tf in np.asarray(traj.failures)[:, 2]:
            special.add(int(np.argmin(np.abs(t - float(tf)))))
    energy = getattr(traj, "energy", None)
    if energy is not None and np.asarray(energy).size:
        uc = np.asarray(energy)[:, 4] if np.asarray(energy).ndim == 2 else None
        if uc is not None:
            hit = np.nonzero(uc > 0)[0]
            if hit.size:
                special.add(int(hit[0]))
    drone = getattr(traj, "drone", None)
    if drone is not None and np.asarray(drone).size and np.isfinite(drone).any():
        vz = np.asarray(drone)[:, 5]
        seen_neg = False
        for i, vzi in enumerate(vz):
            if not np.isfinite(vzi):
                continue
            if vzi < 0:
                seen_neg = True
            elif seen_neg:
                special.add(i)
                break
    special = sorted(i for i in special if 0 <= i < n)
    if len(special) >= max_frames:
        return np.asarray(special[:max_frames], dtype=int)
    fill = np.linspace(0, n - 1, max_frames).astype(int)
    keep = sorted(set(special) | set(fill.tolist()))
    if len(keep) > max_frames:
        # Prefer specials; thin the uniform fill.
        extra = [i for i in keep if i not in special]
        need = max_frames - len(special)
        if need <= 0:
            keep = special[:max_frames]
        else:
            step = max(1, len(extra) / need)
            picked = [extra[int(k * step)] for k in range(need)]
            keep = sorted(set(special) | set(picked))[:max_frames]
    return np.asarray(keep, dtype=int)


def write_run(path, net, disc, material, cfg, traj, *, eta, cascade,
              extra_params=None, light=False, max_frames=50):
    """Write a full (or frame-decimated light) run to ``path`` (HDF5)."""
    import h5py

    t_idx = select_light_frames(traj, max_frames) if light else None
    n_full = int(getattr(traj.t, "size", 0))

    def _take(data):
        arr = np.asarray(data)
        if t_idx is None or arr.ndim == 0 or arr.shape[0] != n_full:
            return arr
        return arr[t_idx]

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
        p.attrs["light"] = bool(light)
        p.attrs["n_frames_full"] = n_full
        p.attrs["n_frames_kept"] = int(t_idx.size) if t_idx is not None else n_full
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

        ds("t", _take(traj.t))
        ds("x", _take(traj.x))
        ds("v", _take(traj.v))
        ds("intact", _take(traj.intact))
        ds("drone", _take(traj.drone))
        ds("energy", _take(traj.energy))
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


def select_fail_dense_frames(t, fail_times, dt_fine=1e-4, dt_coarse=5e-4,
                             pre=0.005, post=0.010):
    """Keep 0.1 ms frames near failures and 0.5 ms frames otherwise."""
    t = np.asarray(t, dtype=float)
    fail_times = np.asarray(fail_times, dtype=float).ravel()
    n = t.size
    if n == 0:
        return np.zeros(0, dtype=int)
    keep = np.zeros(n, dtype=bool)
    last = -1e99
    for i, ti in enumerate(t):
        fine = bool(fail_times.size) and np.any(
            (fail_times - pre <= ti) & (ti <= fail_times + post))
        step = dt_fine if fine else dt_coarse
        if ti - last >= 0.5 * step:
            keep[i] = True
            last = ti
    keep[0] = True
    keep[-1] = True
    return np.nonzero(keep)[0]


def write_diagnostic(path, cfg, traj, *, eta, extra=None, g1=None, g2=None,
                     g3=None, disc=None):
    """Compact HDF5: drone, energy, groups, failures; no full mesh."""
    import h5py

    fail_t = (traj.failures[:, 2] if traj.failures.size
              else np.zeros(0))
    idx = select_fail_dense_frames(traj.t, fail_t)

    def _take(data):
        arr = np.asarray(data)
        if arr.ndim == 0 or arr.shape[0] != traj.t.size:
            return arr
        return arr[idx]

    with h5py.File(path, "w") as h5:
        p = h5.create_group("params")
        p.attrs["git_commit"] = git_commit()
        p.attrs["n_frames_full"] = int(traj.t.size)
        p.attrs["n_frames_kept"] = int(idx.size)
        p.attrs["lock_xy"] = bool(getattr(cfg.drone, "lock_xy", False))
        p.attrs["contact_mode"] = cfg.contact.mode
        p.attrs["n_s"] = cfg.numerics.n_s
        p.attrs["dt_out"] = cfg.numerics.dt_out
        if extra:
            for k, val in extra.items():
                try:
                    p.attrs[k] = val
                except (TypeError, ValueError):
                    p.attrs[k] = str(val)
        def ds(name, data):
            data = np.asarray(data)
            kw = dict(compression="gzip")
            if data.ndim >= 2 and data.shape[0] > 0:
                kw["chunks"] = _chunk_time(data.shape)
            h5.create_dataset(name, data=data, **kw)
        ds("t", _take(traj.t))
        ds("drone", _take(traj.drone))
        ds("energy", _take(traj.energy))
        if g1 is not None:
            ds("uel_g1", _take(g1))
            ds("uel_g2", _take(g2))
            ds("uel_g3", _take(g3))
        h5.create_dataset("failures", data=traj.failures, compression="gzip")
        o = h5.create_group("outcome")
        o.attrs["arrested"] = bool(traj.arrested)
        o.attrs["outcome"] = traj.outcome
        o.attrs["w_max"] = traj.w_max
        o.attrs["eta"] = float(eta)
        o.attrs["energy_error"] = traj.energy_error
        o.attrs["n_failures"] = int(traj.failures.shape[0])
