"""Stage B production sweep (Algorithm alg:sweep).

Grid, seed hash, per-realization rows, Wilson CI aggregates, and P_dd
(Eq. (eq:Pdd)). Orchestration / resume / parallelism live in
``scripts/stage_b.py``. This module must not be imported by
``scripts/produce_results.py``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .params import Params
from .geometry import Topology
from .integrator import simulate
from . import observables
from .studies import WC_E, charge_for_lift_ratio


# ---------------------------------------------------------------------------
# Grid (Stage B production -- final spec B1/B2)
# ---------------------------------------------------------------------------
SWEEP_N = (1, 2, 4, 8)
SWEEP_SIGMA_W = (0.15, 0.30)
SWEEP_ELL = 1.0
SWEEP_U_H = 1.0
SWEEP_M_KG = 1e-6  # 1 mg
SWEEP_Z0 = 0.5
SWEEP_H = 2.0
SWEEP_T_END = 60.0
SWEEP_X = (-4, -2, -1, 0, 1, 2, 4)
SWEEP_X_CORE = (-1, 0, 1)      # M=200 (dense near the transition)
SWEEP_M_CORE = 200            # realizations for x in SWEEP_X_CORE
SWEEP_M_WING = 100            # realizations for the other x
SWEEP_M_REALIZATIONS = 200   # legacy uniform-M default (cost probe / smoke / tests)
SWEEP_N_K = 200
SWEEP_N_T = 50               # Stage B production resolution (B0/T2 validated)
SWEEP_OUTPUT_DT = 0.01       # HDF5 / ML cadence (B3)
SWEEP_SNAPSHOT_DT = 0.25     # fig:snapshot cadence (subsampled from HDF5 frames)
SWEEP_DT_MAX = 1e-2          # default; overridden by B0/T3 via results/b0_config.json
SWEEP_E = WC_E

# --- B2 generalization groups -------------------------------------------
SWEEP_GEN_N = (3, 6)          # gen_N: new thread counts
SWEEP_GEN_N_SIGMA = 0.30
SWEEP_GEN_SIGMA = 0.22        # gen_sigma: new sigma_w
SWEEP_GEN_SIGMA_N = (2, 4)
SWEEP_GEN_X = (-1, 0, 1)
SWEEP_M_GEN = 50

# Relative per-run cost weights for shard balancing. Seeded from the stored
# cost probe (mean wall s at N_t=100, N in 1/2/4/8); N=3/6 linearly
# interpolated/extrapolated. Only relative magnitudes matter, so the N_t/dt
# rescaling cancels and these are safe to bake in for deterministic sharding.
SWEEP_COST_WEIGHTS = {
    1: 675.71, 2: 1754.44, 3: 2165.30, 4: 2576.15, 6: 4298.90, 8: 6021.62,
}


def eddy_diffusivity(sigma_w: float, ell: float = SWEEP_ELL) -> float:
    """K = sigma_w * ell."""
    return float(sigma_w) * float(ell)


def settling_speed(P: Params) -> float:
    """w_s = m g / (N eta_par L + zeta_s)."""
    return P.m * P.g / (P.N * P.eta_par * P.L + P.zeta_s)


def fbar_from_x(N: int, sigma_w: float, x: int, m: float = SWEEP_M_KG,
                L: float = 0.5, N_t: int = 100) -> float:
    """Fbar_l = 1 + x * K / (w_s h), with K = sigma_w * ell."""
    P = Params(N=N, N_t=N_t, L=L, m=m)
    w_s = settling_speed(P)
    K = eddy_diffusivity(sigma_w)
    return 1.0 + float(x) * K / (w_s * SWEEP_H)


def sweep_seed(N: int, sigma_w: float, x: int, i: int) -> int:
    """Deterministic seed from the grid key (N, sigma_w, x, i).

    Formula (documented):
        key = f\"{N}|{sigma_w:.6g}|{int(x)}|{int(i)}\"   # ASCII
        seed = int.from_bytes(blake2b(key, digest_size=8), \"little\") % (2**31 - 1)

    Uses blake2b so the mapping is stable across processes and independent of
    Python's salted ``hash()``.
    """
    key = f"{int(N)}|{float(sigma_w):.6g}|{int(x)}|{int(i)}".encode("ascii")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**31 - 1)


def P_dd(U0: float, K: float, z0: float = SWEEP_Z0, h: float = SWEEP_H) -> float:
    """Analytic rise probability, Eq. (eq:Pdd).

    Brownian motion with drift ``U0`` and diffusivity ``K`` between absorbing
    barriers at 0 and ``h``, starting at ``z0``:

        P_dd = (exp(-U0 z0 / K) - 1) / (exp(-U0 h / K) - 1)

    with the zero-drift limit ``P_dd -> z0 / h``. Here
    ``U0 = w_s (Fbar_l - 1)`` and ``K = sigma_w * ell``.
    """
    if K <= 0.0:
        return float("nan")
    if abs(U0) < 1e-30 * (abs(K) + 1.0):
        return z0 / h
    a = U0 / K
    num = math.exp(-a * z0) - 1.0
    den = math.exp(-a * h) - 1.0
    if abs(den) < 1e-300:
        return z0 / h
    return num / den


def wilson_ci(n_success: int, n_total: int, z: float = 1.959963984540054):
    """Wilson score 95% CI for a binomial proportion.

    Returns ``(P, P_lo, P_hi)``. If ``n_total == 0`` all NaN.
    """
    if n_total <= 0:
        return float("nan"), float("nan"), float("nan")
    p = n_success / n_total
    z2 = z * z
    den = 1.0 + z2 / n_total
    centre = (p + z2 / (2.0 * n_total)) / den
    margin = (z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n_total)) / n_total)
              / den)
    return p, max(0.0, centre - margin), min(1.0, centre + margin)


@dataclass(frozen=True)
class SweepPoint:
    N: int
    sigma_w: float
    x: int
    i: int  # realization index in 0 .. M-1
    split: str = "main"  # "main" (B1), "gen_N" or "gen_sigma" (B2)

    @property
    def seed(self) -> int:
        return sweep_seed(self.N, self.sigma_w, self.x, self.i)

    @property
    def Fbar_l(self) -> float:
        return fbar_from_x(self.N, self.sigma_w, self.x)

    @property
    def K(self) -> float:
        return eddy_diffusivity(self.sigma_w)

    @property
    def key(self) -> tuple:
        """Resume / dedup key (N, sigma_w, x, seed)."""
        return (self.N, self.sigma_w, self.x, self.seed)


def realizations_for(x: int, split: str = "main") -> int:
    """M (number of realizations) for a (x, split) group."""
    if split == "main":
        return SWEEP_M_CORE if x in SWEEP_X_CORE else SWEEP_M_WING
    return SWEEP_M_GEN


def iter_grid_points(M: int = SWEEP_M_REALIZATIONS,
                     N_values=SWEEP_N) -> list[SweepPoint]:
    """Uniform-M grid (cost probe / smoke / tests): N=1,2,4 then N=8.

    This helper keeps a single M for every (sigma_w, x); the production grid with
    per-x M and the B2 generalization set is built by :func:`iter_production_points`.
    """
    ordered = [n for n in (1, 2, 4, 8) if n in set(N_values)]
    out: list[SweepPoint] = []
    for N in ordered:
        for sigma_w in SWEEP_SIGMA_W:
            for x in SWEEP_X:
                for i in range(M):
                    out.append(SweepPoint(N=N, sigma_w=sigma_w, x=x, i=i))
    return out


def b1_groups() -> list[tuple]:
    """(N, sigma_w, x, M, split) for the B1 main grid (N outer, 1/2/4/8)."""
    out = []
    for N in (1, 2, 4, 8):
        for sigma_w in SWEEP_SIGMA_W:
            for x in SWEEP_X:
                out.append((N, sigma_w, x, realizations_for(x, "main"), "main"))
    return out


def b2_groups() -> list[tuple]:
    """(N, sigma_w, x, M, split) for the B2 generalization set."""
    out = []
    for N in SWEEP_GEN_N:                       # gen_N: N in {3,6}, sigma_w=0.30
        for x in SWEEP_GEN_X:
            out.append((N, SWEEP_GEN_N_SIGMA, x, SWEEP_M_GEN, "gen_N"))
    for N in SWEEP_GEN_SIGMA_N:                 # gen_sigma: N in {2,4}, sigma_w=0.22
        for x in SWEEP_GEN_X:
            out.append((N, SWEEP_GEN_SIGMA, x, SWEEP_M_GEN, "gen_sigma"))
    return out


def iter_production_points() -> list[SweepPoint]:
    """Full B1+B2 grid in run order: B2 first, then B1 N in {1,2,4}, then N=8.

    B1: N in {1,2,4,8} x sigma_w in {0.15,0.30} x x in {-4,-2,-1,0,1,2,4};
        M=200 for x in {-1,0,1}, else M=100  ->  8000 runs (split="main").
    B2: gen_N (N in {3,6}, sigma_w=0.30) + gen_sigma (N in {2,4}, sigma_w=0.22),
        x in {-1,0,1}, M=50  ->  300+300 runs.
    """
    out: list[SweepPoint] = []
    for (N, sigma_w, x, M, split) in b2_groups():
        for i in range(M):
            out.append(SweepPoint(N, sigma_w, x, i, split))
    b1 = b1_groups()
    for N_sel in ((1, 2, 4), (8,)):
        for (N, sigma_w, x, M, split) in b1:
            if N not in N_sel:
                continue
            for i in range(M):
                out.append(SweepPoint(N, sigma_w, x, i, split))
    return out


def load_b0_config(path=None) -> dict:
    """Read ``results/b0_config.json`` (T3 ``dt_max`` / factor). Empty if absent."""
    p = Path(path) if path else Path(__file__).resolve().parents[1] / "results" / "b0_config.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def load_dt_max(path=None) -> float:
    """``dt_max`` chosen by B0/T3, else the Stage B default 0.01."""
    cfg = load_b0_config(path)
    if "dt_max" in cfg:
        return float(cfg["dt_max"])
    return float(SWEEP_DT_MAX)


def cost_weight(N: int, cost_means: dict | None = None) -> float:
    """Relative per-run cost weight for shard balancing."""
    if cost_means:
        cm = {int(k): float(v) for k, v in cost_means.items()}
        if N in cm:
            return cm[N]
    return SWEEP_COST_WEIGHTS.get(int(N), float(N) * 1000.0)


def shard_assignment(points: list[SweepPoint], num_shards: int,
                     cost_means: dict | None = None) -> list[int]:
    """Deterministic LPT (longest-processing-time) balanced shard index per point.

    Depends only on the (ordered) point list, ``num_shards`` and the cost
    weights, so every VM derives the same assignment and resume is stable.
    """
    if num_shards <= 1:
        return [0] * len(points)
    load = [0.0] * num_shards
    assign = [0] * len(points)
    order = sorted(range(len(points)),
                   key=lambda j: cost_weight(points[j].N, cost_means),
                   reverse=True)
    for j in order:
        s = min(range(num_shards), key=lambda k: load[k])
        assign[j] = s
        load[s] += cost_weight(points[j].N, cost_means)
    return assign


def sweep_params(pt: SweepPoint, N_t: int = SWEEP_N_T, t_end: float = SWEEP_T_END,
                 output_dt: float = SWEEP_OUTPUT_DT,
                 dt_max: float = SWEEP_DT_MAX,
                 clamp_relax_time: float | None = None) -> Params:
    """Params for one Stage B realization (clamped release into turbulence).

    ``clamp_relax_time`` is optional: when set (>0), phase 1 holds the spider
    for that fixed still-air duration instead of the default steady-window stop.
    Default ``None`` preserves commit 507637e clamp-to-steady behaviour.
    """
    fbar = pt.Fbar_l
    P = Params(
        N=pt.N, N_t=N_t, L=0.5, m=SWEEP_M_KG, Q_s=0.0, charge_model="tip",
        field_model="constant", E_constant=SWEEP_E,
        flow_model="kinematic",
        sigma_w=pt.sigma_w, ell=SWEEP_ELL, U_h=SWEEP_U_H,
        turb_N_k=SWEEP_N_K, turb_seed=pt.seed, turb_renormalize=True,
        z0=SWEEP_Z0, h=SWEEP_H, t_end=t_end,
        release_mode="clamped",
        use_alg2_stopping=True, stop_on_steady=False,
        adaptive_dt=True, dt0=1e-4, dt_max=dt_max, output_dt=output_dt,
        delta=1e-6, t_w=0.05,
        clamp_relax_time=clamp_relax_time,
    )
    P.Q_t = charge_for_lift_ratio(P, SWEEP_E, fbar)
    return P


def _subsample_indices(t: np.ndarray, dt: float) -> np.ndarray:
    """Frame indices closest to multiples of ``dt`` (for snapshot subsampling)."""
    t = np.asarray(t, dtype=float)
    if t.size == 0:
        return np.array([], dtype=int)
    out: list[int] = []
    target = 0.0
    tmax = float(t[-1])
    while target <= tmax + 1e-9:
        j = int(np.searchsorted(t, target - 1e-12))
        if j >= t.size:
            j = t.size - 1
        if not out or out[-1] != j:
            out.append(j)
        target += dt
    if out[-1] != t.size - 1:
        out.append(t.size - 1)
    return np.asarray(out, dtype=int)


def _trajectory_observables(P: Params, traj) -> dict:
    """Time averages of R/L and theta_L after release; min d_min over the run.

    Vectorized over frames (same numerics as the per-frame loop for N>=1).
    """
    topo = Topology(P)
    X = np.asarray(traj.x, dtype=float)
    if X.ndim != 3 or X.shape[0] == 0:
        return {
            "R_over_L_mean": float("nan"),
            "R_over_L_std": float("nan"),
            "theta_L_mean": float("nan"),
            "dmin_min_um": float("nan"),
        }
    tips = np.asarray(topo.tip_nodes, dtype=int)
    axis_xy = X[:, 0, :2]
    tip_xy = X[:, tips, :2]
    R = np.linalg.norm(tip_xy - axis_xy[:, None, :], axis=-1).mean(axis=1) / P.L

    th = np.empty(X.shape[0], dtype=float)
    for k in range(X.shape[0]):
        th[k] = observables.tip_angle(topo, X[k])

    if topo.N < 2:
        dmin_um = float("nan")
    else:
        dmin = np.full(X.shape[0], np.inf)
        for ja in range(topo.N):
            na = np.asarray(topo.thread_nodes[ja][2:], dtype=int)
            if na.size == 0:
                continue
            for jb in range(ja + 1, topo.N):
                nb = np.asarray(topo.thread_nodes[jb][2:], dtype=int)
                if nb.size == 0:
                    continue
                A = X[:, na, :]
                B = X[:, nb, :]
                d = np.linalg.norm(A[:, :, None, :] - B[:, None, :, :], axis=-1)
                dmin = np.minimum(dmin, d.reshape(X.shape[0], -1).min(axis=1))
        finite = dmin[np.isfinite(dmin)]
        dmin_um = float(np.min(finite) * 1e6) if finite.size else float("nan")
    return {
        "R_over_L_mean": float(np.mean(R)),
        "R_over_L_std": float(np.std(R)),
        "theta_L_mean": float(np.mean(th)),
        "dmin_min_um": dmin_um,
    }


def _parse_solver_fail(msg: str) -> tuple[float | None, float | None]:
    import re

    t_fail = dt_fail = None
    m = re.search(r"at t=([\d.eE+-]+)", msg)
    if m:
        t_fail = float(m.group(1))
    m = re.search(r"dt=([\d.eE+-]+)", msg)
    if m:
        dt_fail = float(m.group(1))
    return t_fail, dt_fail


def run_sweep_job(spec: dict) -> dict:
    """Pickable worker: run one realization and return a sweep.csv row dict.

    ``spec`` keys: N, sigma_w, x, i, N_t, t_end, output_dt, write_hdf5,
    hdf5_dir, snapshot (bool), snapshot_dir.
    """
    pt = SweepPoint(N=int(spec["N"]), sigma_w=float(spec["sigma_w"]),
                    x=int(spec["x"]), i=int(spec["i"]),
                    split=str(spec.get("split", "main")))
    N_t = int(spec.get("N_t", SWEEP_N_T))
    t_end = float(spec.get("t_end", SWEEP_T_END))
    output_dt = float(spec.get("output_dt", SWEEP_OUTPUT_DT))
    dt_max = float(spec.get("dt_max", SWEEP_DT_MAX))
    crt = spec.get("clamp_relax_time", None)
    clamp_relax_time = float(crt) if crt not in (None, "", False) else None
    P = sweep_params(pt, N_t=N_t, t_end=t_end, output_dt=output_dt, dt_max=dt_max,
                     clamp_relax_time=clamp_relax_time)

    t0 = time.perf_counter()
    traj = None
    fail_msg = ""
    try:
        traj = simulate(P)
    except Exception as exc:
        fail_msg = f"{type(exc).__name__}: {exc}"
        t_fail, dt_fail = _parse_solver_fail(fail_msg)
        wall = time.perf_counter() - t0
        hdf5_path = ""
        if spec.get("write_hdf5"):
            from pathlib import Path
            from .io_hdf5 import write_ml_hdf5_solver_fail

            out_dir = Path(spec["hdf5_dir"])
            out_dir.mkdir(parents=True, exist_ok=True)
            h5 = out_dir / (
                f"N{pt.N}_sw{pt.sigma_w:.2f}_x{pt.x:+d}_i{pt.i:04d}.h5"
            )
            write_ml_hdf5_solver_fail(
                str(h5), P, t_fail=t_fail, dt_fail=dt_fail, message=fail_msg,
            )
            hdf5_path = str(h5)
        return {
            "N": pt.N,
            "N_t": N_t,
            "sigma_w": pt.sigma_w,
            "ell": SWEEP_ELL,
            "x": pt.x,
            "Fbar_l": pt.Fbar_l,
            "q_nC": P.Q_t * 1e9,
            "seed": pt.seed,
            "split": pt.split,
            "outcome": "solver_fail",
            "exit_time": float(t_fail) if t_fail is not None else float("nan"),
            "R_over_L_mean": float("nan"),
            "R_over_L_std": float("nan"),
            "theta_L_mean": float("nan"),
            "dmin_min_um": float("nan"),
            "entangled": False,
            "newton_failures": -1,
            "wall_time_s": wall,
            "hdf5_path": hdf5_path,
            "i": pt.i,
            "renorm_factor": float(P.turb.get("renorm_factor", 1.0)),
            "solver_fail_t": t_fail,
            "solver_fail_dt": dt_fail,
            "solver_fail_message": fail_msg,
        }
    wall = time.perf_counter() - t0

    obs = _trajectory_observables(P, traj)
    diag = traj.outcome.get("diag", {})
    status = traj.outcome.get("status", "timeout")
    exit_time = traj.outcome.get("exit_time")
    if exit_time is None:
        exit_time = float(traj.t[-1])
    topo = Topology(P)
    R_release = float(observables.tip_radius(topo, traj.x[0]) / P.L)

    from .io_hdf5 import git_provenance
    prov = git_provenance()

    row = {
        "N": pt.N,
        "N_t": N_t,
        "sigma_w": pt.sigma_w,
        "ell": SWEEP_ELL,
        "x": pt.x,
        "Fbar_l": pt.Fbar_l,
        "q_nC": P.Q_t * 1e9,
        "seed": pt.seed,
        "split": pt.split,
        "outcome": status,
        "exit_time": float(exit_time),
        "R_over_L_release": R_release,
        "R_over_L_mean": obs["R_over_L_mean"],
        "R_over_L_std": obs["R_over_L_std"],
        "theta_L_mean": obs["theta_L_mean"],
        "dmin_min_um": obs["dmin_min_um"],
        "entangled": bool(traj.outcome.get("entangled", False)),
        "newton_failures": int(diag.get("newton_failures", 0)),
        "wall_time_s": wall,
        "hdf5_path": "",
        "git_commit": prov["commit"],
        "git_dirty": bool(prov["dirty"]),
        "clamp_relax_time": clamp_relax_time if clamp_relax_time else "",
        "phase1_t_exit": traj.outcome.get("phase1_t_exit", ""),
        # extras (not in the B3 sweep.csv header, kept for callers/aggregation)
        "i": pt.i,
        "renorm_factor": float(P.turb.get("renorm_factor", 1.0)),
    }

    # Air velocity at the recorded node positions (same field / seed), reused by
    # both the ML HDF5 (/u_air) and the snapshot subsample.
    u_air = None
    if spec.get("write_hdf5") or spec.get("snapshot"):
        from .fields import make_flow
        flow = make_flow(P)
        u_air = np.stack([flow.u(traj.x[k], float(traj.t[k]))
                          for k in range(len(traj.t))])

    if spec.get("write_hdf5"):
        from pathlib import Path
        from .io_hdf5 import write_ml_hdf5
        out_dir = Path(spec["hdf5_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / (
            f"N{pt.N}_sw{pt.sigma_w:.2f}_x{pt.x:+d}_i{pt.i:04d}.h5"
        )
        write_ml_hdf5(str(path), P, traj, u_air=u_air, float32=True)
        row["hdf5_path"] = str(path)
        row["hdf5_bytes"] = path.stat().st_size

    if spec.get("snapshot"):
        from pathlib import Path
        out_dir = Path(spec["snapshot_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        idx = _subsample_indices(traj.t, SWEEP_SNAPSHOT_DT)
        path = out_dir / f"N{pt.N}_sw{pt.sigma_w:.2f}_x{pt.x:+d}_seed{pt.seed}.npz"
        np.savez_compressed(
            path,
            t=traj.t[idx], positions=traj.x[idx], u=u_air[idx],
            N=pt.N, sigma_w=pt.sigma_w, x_grid=pt.x, seed=pt.seed, i=pt.i,
            Fbar_l=pt.Fbar_l, outcome=np.array(status),
        )
        row["snapshot_path"] = str(path)

    return row


def aggregate_phase(rows: list[dict]) -> list[dict]:
    """Build tab_phase.csv rows from completed sweep.csv rows."""
    from collections import defaultdict
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (int(r["N"]), float(r["sigma_w"]), int(r["x"]))
        groups[key].append(r)

    out = []
    for (N, sigma_w, x), grp in sorted(groups.items()):
        fbar = float(grp[0]["Fbar_l"])
        n_up = sum(1 for r in grp if r["outcome"] == "rise")
        n_down = sum(1 for r in grp if r["outcome"] == "fall")
        n_timeout = sum(1 for r in grp if r["outcome"] == "timeout")
        M = len(grp)
        n_decided = n_up + n_down
        P, P_lo, P_hi = wilson_ci(n_up, n_decided)
        P_base = Params(N=N, N_t=100, L=0.5, m=SWEEP_M_KG)
        w_s = settling_speed(P_base)
        U0 = w_s * (fbar - 1.0)
        K = eddy_diffusivity(sigma_w)
        mean_exit = float(np.mean([float(r["exit_time"]) for r in grp]))
        out.append({
            "N": N,
            "sigma_w": sigma_w,
            "x": x,
            "Fbar_l": fbar,
            "M": M,
            "n_up": n_up,
            "n_down": n_down,
            "n_timeout": n_timeout,
            "P": P,
            "P_lo": P_lo,
            "P_hi": P_hi,
            "P_dd": P_dd(U0, K),
            "mean_exit_time": mean_exit,
        })
    return out


_H5_NAME = re.compile(
    r"^N(?P<N>\d+)_sw(?P<sw>[\d.]+)_x(?P<x>[+-]?\d+)_i(?P<i>\d+)\.h5$"
)


def production_point_lookup() -> dict[tuple, SweepPoint]:
    return {(p.N, p.sigma_w, p.x, p.i): p for p in iter_production_points()}


def sweep_row_from_hdf5(path: Path | str, *, pt: SweepPoint | None = None) -> dict:
    """Rebuild one ``sweep.csv`` row from an ML HDF5 file (source of truth)."""
    import h5py
    from .integrator import Trajectory

    path = Path(path)
    m = _H5_NAME.match(path.name)
    if m is None:
        raise ValueError(f"unexpected HDF5 name: {path.name}")
    N = int(m.group("N"))
    sigma_w = float(m.group("sw"))
    x = int(m.group("x"))
    i = int(m.group("i"))
    if pt is None:
        pt = production_point_lookup().get((N, sigma_w, x, i))
        if pt is None:
            pt = SweepPoint(N=N, sigma_w=sigma_w, x=x, i=i)

    with h5py.File(path, "r") as f:
        N_t = int(f["params"].attrs.get("N_t", SWEEP_N_T))
        q_t = float(f["params"].attrs.get("Q_t", 0.0))
        t = np.asarray(f["t"][:], dtype=float)
        xtr = np.asarray(f["x"][:], dtype=float)
        vtr = np.asarray(f["v"][:], dtype=float)
        twist = np.asarray(f["twist"][:], dtype=float)
        charge = np.asarray(f["charge"][:], dtype=float)
        edges = np.asarray(f["topology/edges"][:])
        thread_id = np.asarray(f["topology/thread_id"][:])
        oc = f["outcome"]
        status = str(oc.attrs.get("status", "timeout"))
        exit_time = oc.attrs.get("exit_time", -1.0)
        if exit_time is not None and float(exit_time) < 0:
            exit_time = float(t[-1]) if t.size else float("nan")
        else:
            exit_time = float(exit_time)
        entangled = bool(oc.attrs.get("entangled", False))
        nf = int(oc.attrs.get("newton_failures", 0))

    traj = Trajectory(
        t=t, x=xtr, v=vtr, theta=twist,
        edges=edges, thread_id=thread_id, q_node=charge,
        outcome={"status": status, "exit_time": exit_time, "entangled": entangled,
                 "diag": {"newton_failures": nf}},
    )
    P = sweep_params(pt, N_t=N_t)
    obs = _trajectory_observables(P, traj)
    valid = pt.Fbar_l > 0.0
    return {
        "N": pt.N,
        "N_t": N_t,
        "sigma_w": pt.sigma_w,
        "ell": SWEEP_ELL,
        "x": pt.x,
        "Fbar_l": pt.Fbar_l,
        "q_nC": q_t * 1e9,
        "seed": pt.seed,
        "split": pt.split,
        "outcome": status,
        "exit_time": exit_time,
        "R_over_L_mean": obs["R_over_L_mean"],
        "R_over_L_std": obs["R_over_L_std"],
        "theta_L_mean": obs["theta_L_mean"],
        "dmin_min_um": obs["dmin_min_um"],
        "entangled": entangled,
        "newton_failures": nf,
        "wall_time_s": "",
        "hdf5_path": str(path.resolve()),
        "valid": valid,
        "i": pt.i,
    }


def is_snapshot_point(pt: SweepPoint) -> bool:
    """Fig. fig:snapshot: N=4, sigma_w=0.30, x=0, first 3 seeds (i=0,1,2)."""
    return (pt.N == 4 and abs(pt.sigma_w - 0.30) < 1e-12
            and pt.x == 0 and pt.i < 3)
