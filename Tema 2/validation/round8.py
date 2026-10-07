"""Round 8: last physics runs (mass scan, lambda map, etaB free, Test 3, …).

Resume by run id: each finished run writes JSON under
``paper_results/round8/<shard>/runs/<run_id>.json``. Rebuilding a CSV is a
merge of those files; a crash loses at most the in-flight workers.

Usage:
  python -m validation.round8 --shard pilot|A|B|C|D
  python -m validation.round8 --merge
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/netsim_mpl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_REF = str(_ROOT / "ref")
if _REF not in sys.path:
    sys.path.insert(0, _REF)

from netsim.config import (
    SimConfig, MaterialConfig, NetConfig, DroneConfig, NumericsConfig,
    ContactConfig, OutputConfig, KinematicConfig,
)
from netsim.discretize import discretize
from netsim.io_hdf5 import git_commit, write_diagnostic
from netsim.materials import get_material
from netsim.mmin import MminConfig, minimum_mass
from netsim.simulate import simulate, simulate_config, build_net
from netsim.topology import star
from netsim.integrator import PenetrationError

import riemann

from validation.junction_linear_2d import S_N, junction_linear_2d
from validation.offcentre import run_offcentre
from validation.export_paper import (
    export_params_used, export_conv, export_etaa, _header_dirty_flag,
)
from validation.test3_junction import run_junction_sim, _thread_segment_order

OUT = Path(os.environ.get("ROUND8_OUT", _ROOT / "paper_results" / "round8"))
_COMMIT = os.environ.get("GIT_COMMIT", "").strip() or git_commit()
_DATE = datetime.datetime.now().isoformat(timespec="seconds")
N_JOBS = max(1, int(os.environ.get("JOBS", os.cpu_count() or 1)))
PROGRESS_EVERY = 1

# tab_mmin star n_s=40, M=1 (kg).
MMIN_7C_KG = {
    ("S", 15): dict(A=8.46636e-3, Bloc=4.15809e-3, Bany=4.08741e-3),
    ("S", 20): dict(A=15.533e-3, Bloc=7.455e-3, Bany=7.39216e-3),
    ("D", 15): dict(A=10.1639e-3, Bloc=10.0554e-3, Bany=10.0554e-3),
    ("D", 20): dict(A=18.2856e-3, Bloc=17.6505e-3, Bany=17.6505e-3),
}
MSTAR_KG = {"S": 1.92657e-3, "D": 3.46236e-3}  # Ekin=200 J
MU_B = (0.90, 0.95, 0.98, 1.00, 1.02, 1.05, 1.10, 1.25, 1.50)
MU_A = (0.98, 1.00, 1.02)
A_OVER_R_SCAN = (0.25, 0.5)
LAMBDA_A = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
LAMBDA_M = (1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 10)
IMPACT3 = [(0.0, 0.0), (0.25, 0.0), (0.5, 0.0)]
RING_NS10 = {
    ("S", 20): dict(A=30.5843e-3, Bany=18.3862e-3),
    ("D", 20): dict(A=37.8695e-3, Bany=34.172e-3),
}


def _code_dirty() -> str:
    return os.environ.get("CODE_DIRTY", _header_dirty_flag())


def _header(config: str) -> str:
    return (f"# commit={_COMMIT} date={_DATE} config={config} "
            f"code_dirty={_code_dirty()}\n")


def _write_csv(path: Path, fieldnames, rows, config: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        fh.write(_header(config))
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _run_dir(shard: str) -> Path:
    d = OUT / shard / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _done_path(shard: str, run_id: str) -> Path:
    return _run_dir(shard) / f"{run_id}.json"


def _save_run(shard: str, run_id: str, payload: dict):
    p = _done_path(shard, run_id)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, default=str))
    tmp.replace(p)


def _load_run(shard: str, run_id: str):
    p = _done_path(shard, run_id)
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def _progress(shard: str, done: int, total: int, t0: float, msg: str = ""):
    elapsed = time.time() - t0
    rate = done / elapsed if elapsed > 0 and done else 0.0
    remain = (total - done) / rate if rate > 0 else float("nan")
    print(f"[{shard}] {done}/{total} {msg} elapsed={elapsed:.0f}s "
          f"remain~{remain:.0f}s rate={3600 * rate:.1f}/h", flush=True)


def _contact(mat: str, mode: str = "gripped") -> ContactConfig:
    return ContactConfig(
        mode=mode, k_c=1e7,
        k_c_mode=("relative" if mat == "D" else "absolute"),
        k_c_factor=8.0, penetration_guard="reduce_dt",
    )


def paper_star_cfg(mat, v0, a, s, *, lock_xy=False, n_s=40, t_end=0.25,
                   dt_out=5e-3, store_mesh=True, energy_groups=False,
                   mode="gripped", N=8, kind="star"):
    m = get_material(mat)
    ep = 0.1 * m.eps_b
    net_kw = dict(kind=kind, N=N, R=1.0, eps_p=ep, A_hat=1e-6)
    if kind == "star_with_rings":
        net_kw.update(radii=[0.5, 1.0], q_ratio=1.0, fix_radii=True)
    return SimConfig(
        material=MaterialConfig(name=mat),
        net=NetConfig(**net_kw),
        drone=DroneConfig(M=1.0, r_d=0.15, v0=v0, p=(a, 0.0),
                          lock_xy=bool(lock_xy)),
        numerics=NumericsConfig(
            n_s=n_s, C=0.5, t_end=t_end, dt_out=dt_out, use_numba=True,
            area_scale=s, store_mesh=store_mesh, energy_groups=energy_groups,
        ),
        contact=_contact(mat, mode),
        output=OutputConfig(hdf5=None, R_max=0.5, k_max=10),
    )


def _s_for_mass(cfg: SimConfig, m_net: float) -> float:
    mat = cfg.material.resolve()
    net = build_net(cfg)
    m1 = net.net_mass(mat.rho, 1.0)
    return float(m_net / m1) if m1 > 0 else 1.0


def _groups(traj, disc):
    n = int(traj.t.size)
    g1 = np.zeros(n)
    g2 = np.zeros(n)
    g3 = np.asarray(getattr(traj, "uel_other", np.zeros(n)), dtype=float)
    if g3.size != n:
        g3 = np.zeros(n)
    uel = np.asarray(getattr(traj, "uel_r0", np.zeros((0, 0))))
    r0 = np.nonzero(disc.seg_parent == 0)[0]
    failed = set()
    if traj.failures.size:
        failed = {int(s) for s in traj.failures[:, 0]}
    if uel.ndim == 2 and uel.shape[0] == n and uel.shape[1] == r0.size:
        for j, s in enumerate(r0):
            if int(s) in failed:
                g1 += uel[:, j]
            else:
                g2 += uel[:, j]
    return g1, g2, g3


def _n_failed_threads(res):
    traj = res.trajectory
    if not traj.failures.size:
        return 0
    return int(len(set(int(p) for p in traj.failures[:, 1])))


def _classify(res, R=1.0, R_max=0.5):
    if res.outcome == "timeout" and not res.arrested:
        return "timeout"
    if not res.arrested:
        return "perforated"
    n_thr = _n_failed_threads(res)
    if n_thr == 0:
        return "intact"
    traj = res.trajectory
    # First failure of each parent vs drone at that time (B_loc).
    first = {}
    order = np.argsort(traj.failures[:, 2]) if traj.failures.size else []
    for j in order:
        p = int(traj.failures[j, 1])
        if p in first:
            continue
        mid = traj.failure_midpoints[j, :2]
        dxy = traj.failure_drone_xy[j]
        dist = float(np.hypot(mid[0] - dxy[0], mid[1] - dxy[1]))
        first[p] = dist
    if first and max(first.values()) > R_max:
        return "nonlocal"
    return "local"


def _before_after(traj, Ekin):
    """Diagnostics around first failure; NaN if none."""
    out = dict(
        t_first_fail_ms=float("nan"), Kd_before=float("nan"),
        Uel_g1_before=float("nan"), Uel_g2_before=float("nan"),
        Uel_g3_before=float("nan"), Kd_after5ms=float("nan"),
        vxy_after5ms=float("nan"), vz_after5ms=float("nan"),
        Wfail_total=float("nan"), Knet_after5ms=float("nan"),
        first_fail_seg="", first_fail_dist_to_drone=float("nan"),
        Uel_total_before=float("nan"),
    )
    if not traj.failures.size:
        out["Wfail_total"] = float(traj.energy[-1, 3])
        return out
    t0 = float(traj.failures[0, 2])
    out["t_first_fail_ms"] = 1e3 * t0
    out["first_fail_seg"] = int(traj.failures[0, 0])
    mid = traj.failure_midpoints[0, :2]
    dxy = traj.failure_drone_xy[0]
    out["first_fail_dist_to_drone"] = float(
        np.hypot(mid[0] - dxy[0], mid[1] - dxy[1]))
    before = np.nonzero(traj.t < t0)[0]
    i0 = int(before[-1]) if before.size else 0
    out["Kd_before"] = float(traj.energy[i0, 0])
    out["Uel_total_before"] = float(traj.energy[i0, 2])
    after = np.nonzero(traj.t >= t0 + 0.005)[0]
    i1 = int(after[0]) if after.size else traj.t.size - 1
    out["Kd_after5ms"] = float(traj.energy[i1, 0])
    out["Knet_after5ms"] = float(traj.energy[i1, 1])
    out["Wfail_total"] = float(traj.energy[-1, 3])
    vd = traj.drone[i1, 3:6]
    out["vxy_after5ms"] = float(np.hypot(vd[0], vd[1]))
    out["vz_after5ms"] = float(vd[2])
    return out


def _fill_groups_before(row, traj, disc):
    if not traj.failures.size or traj.uel_r0.size == 0:
        return row
    t0 = float(traj.failures[0, 2])
    before = np.nonzero(traj.t < t0)[0]
    i0 = int(before[-1]) if before.size else 0
    g1, g2, g3 = _groups(traj, disc)
    row["Uel_g1_before"] = float(g1[i0])
    row["Uel_g2_before"] = float(g2[i0])
    row["Uel_g3_before"] = float(g3[i0])
    return row


# --------------------------------------------------------------------------- #
# Item 1
# --------------------------------------------------------------------------- #
def _mass_scan_jobs(lock_only=False):
    jobs = []
    mats_v0 = [("S", 15), ("S", 20), ("D", 15), ("D", 20)]
    if lock_only:
        mats_v0 = [("S", 20), ("D", 20)]
    locks = (1,) if lock_only else ((0, 1) if False else (0,))
    # Free: all v0; lock_xy: v0=20 only (handled by lock_only / extra pass).
    for mat, v0 in mats_v0:
        masses = MMIN_7C_KG[(mat, v0)]
        for a in A_OVER_R_SCAN:
            for mu in MU_B:
                jobs.append(dict(
                    material=mat, v0=v0, a=a, lock_xy=False,
                    mu_label=f"muB_{mu:.2f}", m_net=mu * masses["Bany"],
                ))
            for mu in MU_A:
                jobs.append(dict(
                    material=mat, v0=v0, a=a, lock_xy=False,
                    mu_label=f"muA_{mu:.2f}", m_net=mu * masses["A"],
                ))
    if not lock_only:
        for mat in ("S", "D"):
            masses = MMIN_7C_KG[(mat, 20)]
            for a in A_OVER_R_SCAN:
                for mu in MU_B:
                    jobs.append(dict(
                        material=mat, v0=20, a=a, lock_xy=True,
                        mu_label=f"muB_{mu:.2f}", m_net=mu * masses["Bany"],
                    ))
                for mu in MU_A:
                    jobs.append(dict(
                        material=mat, v0=20, a=a, lock_xy=True,
                        mu_label=f"muA_{mu:.2f}", m_net=mu * masses["A"],
                    ))
    return jobs


def _run_mass_scan_one(job):
    shard = job["shard"]
    run_id = job["run_id"]
    cached = _load_run(shard, run_id)
    if cached is not None:
        return cached
    mat, v0, a = job["material"], job["v0"], job["a"]
    cfg0 = paper_star_cfg(mat, v0, a, 1.0, lock_xy=job["lock_xy"])
    s = _s_for_mass(cfg0, job["m_net"])
    cfg = paper_star_cfg(
        mat, v0, a, s, lock_xy=job["lock_xy"], dt_out=1e-4,
        store_mesh=False, energy_groups=True, t_end=0.25,
    )
    disc = discretize(build_net(cfg), cfg.material.resolve(), cfg.numerics.n_s,
                      area_scale=s, r_d=cfg.drone.r_d)
    t0 = time.time()
    try:
        res = simulate_config(cfg, write=False)
    except PenetrationError as e:
        payload = dict(job, outcome="penetration_error", arrested=False,
                       error=str(e), cpu_s=time.time() - t0)
        _save_run(shard, run_id, payload)
        return payload
    traj = res.trajectory
    Ekin = 0.5 * cfg.drone.M * v0 ** 2
    row = dict(
        material=mat, v0=f"{v0:.6g}", a_over_R=f"{a:.6g}",
        lock_xy=int(bool(job["lock_xy"])), mu_label=job["mu_label"],
        m_net_g=f"{1e3 * res.m_net:.6g}",
        outcome=traj.outcome, arrested=int(bool(res.arrested)),
        n_failed_threads=_n_failed_threads(res),
        eta_at_arrest=f"{res.eta:.6g}",
        w_max_over_R=f"{res.w_max:.6g}",
        drone_x_arrest=f"{res.drone_x_arrest:.6g}",
        drone_y_arrest=f"{res.drone_y_arrest:.6g}",
        cpu_s=time.time() - t0, run_id=run_id,
    )
    row.update({k: (f"{v:.6g}" if isinstance(v, float) else v)
                for k, v in _before_after(traj, Ekin).items()})
    row = _fill_groups_before(row, traj, disc)
    g1, g2, g3 = _groups(traj, disc)
    h5_dir = OUT / shard / "hdf5"
    h5_dir.mkdir(parents=True, exist_ok=True)
    h5_path = h5_dir / f"{run_id}.h5"
    write_diagnostic(
        str(h5_path), cfg, traj, eta=res.eta, g1=g1, g2=g2, g3=g3, disc=disc,
        extra={"run_id": run_id, "mu_label": job["mu_label"],
               "m_net": res.m_net},
    )
    row["hdf5"] = str(h5_path.relative_to(OUT))
    _save_run(shard, run_id, row)
    return row


def _pool_jobs(shard, jobs, worker, desc):
    t0 = time.time()
    pending = []
    done_rows = []
    for i, job in enumerate(jobs):
        job = dict(job)
        job["shard"] = shard
        if "run_id" not in job:
            job["run_id"] = f"{desc}_{i:04d}"
        if _load_run(shard, job["run_id"]) is not None:
            done_rows.append(_load_run(shard, job["run_id"]))
        else:
            pending.append(job)
    total = len(jobs)
    done = len(done_rows)
    _progress(shard, done, total, t0, f"{desc} resume")
    if not pending:
        return done_rows
    n = min(N_JOBS, len(pending))
    if n <= 1:
        for job in pending:
            done_rows.append(worker(job))
            done += 1
            _progress(shard, done, total, t0, desc)
        return done_rows
    print(f"[{shard}] {len(pending)} new {desc} jobs × {n} workers", flush=True)
    with ProcessPoolExecutor(max_workers=n) as pool:
        futs = {pool.submit(worker, job): job for job in pending}
        for fut in as_completed(futs):
            done_rows.append(fut.result())
            done += 1
            if done % PROGRESS_EVERY == 0 or done == total:
                _progress(shard, done, total, t0, desc)
    return done_rows


def item1_mass_scan(shard="B"):
    jobs = _mass_scan_jobs()
    for i, j in enumerate(jobs):
        j["run_id"] = (
            f"mscan_{j['material']}_v{j['v0']}_a{j['a']}_"
            f"{'L' if j['lock_xy'] else 'F'}_{j['mu_label']}"
        )
    rows = _pool_jobs(shard, jobs, _run_mass_scan_one, "mass_scan")
    fields = [
        "material", "v0", "a_over_R", "lock_xy", "mu_label", "m_net_g",
        "outcome", "arrested", "n_failed_threads", "first_fail_seg",
        "first_fail_dist_to_drone", "t_first_fail_ms", "Kd_before",
        "Uel_g1_before", "Uel_g2_before", "Uel_g3_before", "Kd_after5ms",
        "vxy_after5ms", "vz_after5ms", "Wfail_total", "Knet_after5ms",
        "eta_at_arrest", "w_max_over_R", "drone_x_arrest", "drone_y_arrest",
    ]
    _write_csv(OUT / shard / "mass_scan.csv", fields, rows, "round8_item1")
    item1_mmin_lockxy(shard)
    return rows


def item1_mmin_lockxy(shard="B"):
    """Alg.2 with lock_xy at v0=20, a/R in {0,0.25,0.5}."""
    rows = []
    for mat in ("S", "D"):
        run_id = f"mmin_lockxy_{mat}_v20"
        cached = _load_run(shard, run_id)
        if cached is not None:
            rows.append(cached)
            continue
        cfg = paper_star_cfg(mat, 20.0, 0.0, 1.0, lock_xy=True, n_s=40,
                             t_end=0.25, dt_out=5e-3)
        cache = str(OUT / shard / "mmin_cache" / run_id)
        Path(cache).mkdir(parents=True, exist_ok=True)
        n_procs = min(3, max(1, N_JOBS))
        t0 = time.time()
        rec = {"material": mat, "v0": "20", "lock_xy": 1}
        for crit, key in (("A", "mA_lock_g"), ("B_any", "mBany_lock_g")):
            mc = MminConfig(
                criterion=crit, tol=0.01, impact_points=IMPACT3,
                n_procs=n_procs, cache_dir=cache + f"_{crit}",
                R_max=0.5,
            )
            r = minimum_mass(cfg, mc)
            rec[key] = f"{1e3 * r.m_min:.6g}"
            rec[f"worst_{crit}"] = f"{r.worst_point[0]:.6g},{r.worst_point[1]:.6g}"
        free = MMIN_7C_KG[(mat, 20)]
        rec["mA_free_g"] = f"{1e3 * free['A']:.6g}"
        rec["mBany_free_g"] = f"{1e3 * free['Bany']:.6g}"
        rec["ratio_Bany_lock_over_free"] = (
            f"{float(rec['mBany_lock_g']) / (1e3 * free['Bany']):.6g}")
        rec["cpu_s"] = time.time() - t0
        rec["run_id"] = run_id
        _save_run(shard, run_id, rec)
        rows.append(rec)
        print(f"[{shard}] lockxy mmin {mat} {rec}", flush=True)
    _write_csv(OUT / shard / "mmin_lockxy.csv",
               ["material", "v0", "lock_xy", "mA_lock_g", "mBany_lock_g",
                "mA_free_g", "mBany_free_g", "ratio_Bany_lock_over_free",
                "worst_A", "worst_B_any", "cpu_s"],
               rows, "round8_item1_lockxy")
    return rows


# --------------------------------------------------------------------------- #
# Item 2
# --------------------------------------------------------------------------- #
def _run_lambda_one(job):
    shard, run_id = job["shard"], job["run_id"]
    cached = _load_run(shard, run_id)
    if cached is not None:
        return cached
    mat, a, ratio = job["material"], job["a"], job["m_over_mstar"]
    m_net = ratio * MSTAR_KG[mat]
    cfg0 = paper_star_cfg(mat, 20.0, a, 1.0)
    s = _s_for_mass(cfg0, m_net)
    cfg = paper_star_cfg(mat, 20.0, a, s, dt_out=5e-3, t_end=0.25)
    t0 = time.time()
    res = simulate_config(cfg, write=False)
    Ekin = 0.5 * 1.0 * 20.0 ** 2
    m = get_material(mat)
    PiE = Ekin / (res.m_net * m.e_mat) if res.m_net > 0 else float("nan")
    ba = _before_after(res.trajectory, Ekin)
    row = dict(
        material=mat, a_over_R=f"{a:.6g}", m_over_mstar=f"{ratio:.6g}",
        m_net_g=f"{1e3 * res.m_net:.6g}",
        PiE_M_over_m=f"{PiE:.6g}",
        class_=_classify(res),
        n_failed_threads=_n_failed_threads(res),
        eta_at_arrest=f"{res.eta:.6g}",
        Kd_before=f"{ba['Kd_before']:.6g}",
        Uel_total_before=f"{ba['Uel_total_before']:.6g}",
        Kd_after5ms=f"{ba['Kd_after5ms']:.6g}",
        cpu_s=time.time() - t0, run_id=run_id,
    )
    _save_run(shard, run_id, row)
    return row


def item2_lambda_map(shard="C"):
    jobs = []
    for mat in ("S", "D"):
        for a in LAMBDA_A:
            for ratio in LAMBDA_M:
                jobs.append(dict(
                    material=mat, a=a, m_over_mstar=ratio,
                    run_id=f"lam_{mat}_a{a}_m{ratio}",
                ))
    rows = _pool_jobs(shard, jobs, _run_lambda_one, "lambda")
    for r in rows:
        r["class"] = r.get("class_") or r.get("class", "")
    fields = ["material", "a_over_R", "m_over_mstar", "m_net_g", "PiE_M_over_m",
              "class", "n_failed_threads", "eta_at_arrest",
              "Kd_before", "Uel_total_before", "Kd_after5ms"]
    _write_csv(OUT / shard / "lambda_map.csv", fields, rows, "round8_item2")
    _plot_lambda(rows, OUT / shard / "lambda_map.png")
    return rows


def _plot_lambda(rows, path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"lambda_map.png skipped: {e}", flush=True)
        return
    colours = {
        "intact": "#2ca02c", "local": "#1f77b4", "nonlocal": "#ff7f0e",
        "perforated": "#d62728", "timeout": "#7f7f7f",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    etaa_path = _ROOT / "paper_results" / "fig_etaa.csv"
    etaa = []
    if etaa_path.is_file():
        with open(etaa_path) as fh:
            next(fh)
            etaa = list(csv.DictReader(fh))
    for ax, mat in zip(axes, ("S", "D")):
        sub = [r for r in rows if r["material"] == mat]
        for r in sub:
            ax.scatter(float(r["a_over_R"]), float(r["m_over_mstar"]),
                       c=colours.get(r.get("class") or r.get("class_"), "k"),
                       s=28, edgecolors="none")
        mstar = MSTAR_KG[mat]
        e_mat = get_material(mat).e_mat
        # Overlay m*/eta(a) from quasi-static fig_etaa (drone held), ep=0.1.
        xs, yA, yB = [], [], []
        for e in etaa:
            if e["material"] != mat or abs(float(e["ep_frac"]) - 0.1) > 1e-9:
                continue
            a = float(e["a_over_R"])
            try:
                etaA = float(e["etaA_num"])
                etaB = float(e["etaB_num"]) if e["etaB_num"] else float("nan")
            except (TypeError, ValueError):
                continue
            xs.append(a)
            yA.append(1.0 / etaA if etaA else float("nan"))
            yB.append(1.0 / etaB if etaB == etaB and etaB else float("nan"))
        if xs:
            ax.plot(xs, yA, "k-", lw=0.8, label=r"$m^*/\eta_A$")
            ax.plot(xs, yB, "k--", lw=0.8, label=r"$m^*/\eta_B$")
        ax.set_title(mat)
        ax.set_xlabel(r"$a/R$")
        ax.set_xlim(0.05, 0.75)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(r"$m_{\mathrm{net}}/m^*$")
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=c, label=k, markersize=8)
               for k, c in colours.items()]
    axes[1].legend(handles=handles, loc="upper right", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Item 3 (also via export_etaa)
# --------------------------------------------------------------------------- #
def item3_etaB_free(shard="A"):
    print(f"[{shard}] fig_etaa with etaB_free …", flush=True)
    from validation import export_paper as ep
    ep.OUT_DIR = _ROOT / "paper_results"
    path = export_etaa()
    # Copy into the shard dir too.
    dest = OUT / shard / "fig_etaa.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(Path(path).read_text())
    print(f"[{shard}] wrote {path}", flush=True)
    return path


# --------------------------------------------------------------------------- #
# Item 4
# --------------------------------------------------------------------------- #
def _Et(mat, eps):
    return mat.E0 + 3.0 * mat.b * eps ** 2


def _nl_preds(material, N, eps0, ep):
    r = np.sqrt(_Et(material, eps0) / _Et(material, ep)) if ep > 0 else 1.0
    s = S_N(N)
    T0 = 2 * s / (r + s)
    dTmax = 2 / (r + s)
    Tfrac = 4 * r * s / (r + s) ** 2
    j2 = junction_linear_2d(material, N, ep, variant="with")
    s2 = s + j2["ZT_over_ZL"] * (N / 2.0)
    T0p = 2 * s2 / (r + s2)
    dTmaxp = 2 / (r + s2)
    Tfracp = 4 * r * s2 / (r + s2) ** 2
    return dict(r=r, SN=s, SN2d=s2, T0_1d=T0, dTmax_1d=dTmax, T_1d=Tfrac,
                T0_2d=T0p, dTmax_2d=dTmaxp, T_2d=Tfracp,
                ZT_over_ZL=j2["ZT_over_ZL"])


def run_junction_nl(name="S", N=8, eps0_frac=0.33, L=4.0, n_seg=400,
                    d_frac=1e-3):
    material = get_material(name)
    ref_mat = getattr(riemann, name)
    ep = 0.1 * material.eps_b
    eps0 = min(eps0_frac, 0.98) * material.eps_b
    e1 = eps0 + d_frac * material.eps_b
    v1 = ref_mat.Phi(e1, eps0)
    T_hub = material.sigma(eps0) - material.sigma(ep)
    A_hat = 1e-6
    force = (-T_hub * A_hat, 0.0, 0.0)  # away from thread 0 (+x)
    net = star(N, L, ep, material=material, A_hat=A_hat, eps_p_thread0=eps0)
    cfg = SimConfig(
        material=MaterialConfig(name=name),
        net=NetConfig(kind="star", N=N, R=L, eps_p=ep, A_hat=A_hat,
                      eps_p_thread0=eps0),
        drone=DroneConfig(M=1.0, r_d=1.0, v0=0.0),
        numerics=NumericsConfig(n_s=n_seg, C=0.4, t_end=0.0, dt_out=1e-6,
                                use_numba=True, store_mesh=True),
        contact=ContactConfig(mode="frictionless", k_c=1e7),
        output=OutputConfig(hdf5=None, R_max=1e9, k_max=10 ** 9),
        kinematic=KinematicConfig(
            enabled=True, node=1, mode="velocity",
            direction=(1.0, 0.0, 0.0), amplitude=v1,
            extra_force_node=0, extra_force=force,
        ),
    )
    c_inc = ref_mat.c(e1) if e1 > 0 else material.c_L0
    tarr = L / c_inc
    cfg.numerics.t_end = tarr + 1.6 * L / c_inc
    cfg.numerics.dt_out = cfg.numerics.t_end / 160
    res = simulate(net, material, cfg.drone, cfg.numerics, cfg.contact,
                   cfg.output, kinematic=cfg.kinematic)
    disc = discretize(net, material, n_seg, r_d=1.0, warn_ratio=1e9)
    order = _thread_segment_order(disc, N)
    from validation.test3_junction import WIN_LO, WIN_HI
    win = ((res.trajectory.t >= (tarr + WIN_LO * L / c_inc))
           & (res.trajectory.t <= (tarr + WIN_HI * L / c_inc)))
    frames = np.nonzero(win)[0]
    if frames.size == 0:
        frames = np.array([res.trajectory.t.size - 1])
    strains = np.zeros(N)
    cells = 5
    for k in range(N):
        s = order[k][cells]
        i, j = disc.seg_edges[s]
        vals = []
        for fr in frames:
            d = res.trajectory.x[fr, j] - res.trajectory.x[fr, i]
            vals.append(np.linalg.norm(d) / disc.seg_rest_length[s] - 1.0)
        strains[k] = float(np.mean(vals))
    Tinc = float(material.sigma(e1) - material.sigma(eps0))
    dT = np.array([(material.sigma(strains[k]) - material.sigma(
        eps0 if k == 0 else ep)) / Tinc for k in range(N)])
    # Transmitted energy: excess elastic+KE on threads 1..N-1 vs incident
    # excess on thread 0 in the pre-arrival window.
    pre = res.trajectory.t < 0.5 * tarr
    post = win
    E_thr = np.zeros(N)
    E_inc = 0.0
    # Approximate with tension-velocity at the measurement cell.
    pred = _nl_preds(material, N, eps0, ep)
    T_num = float("nan")
    if Tinc != 0:
        # Energy transmission from amplitude form using measured T0, r.
        r = pred["r"]
        T0 = float(dT[0])
        T_num = T0 * (2.0 - T0) * r if r == r else float("nan")
    return dict(
        N=N, material=name, eps0_frac=eps0_frac, n_seg=n_seg,
        T0_Tinc=float(dT[0]), dTmax_Tinc=float(dT[1:].max()),
        T_frac=T_num, **{f"pred_{k}": v for k, v in pred.items()},
        strains=strains.tolist(),
    )


def _run_jnl_one(job):
    shard, run_id = job["shard"], job["run_id"]
    cached = _load_run(shard, run_id)
    if cached is not None:
        return cached
    t0 = time.time()
    r = run_junction_nl(job["material"], N=job["N"], eps0_frac=job["eps0_frac"],
                        L=job.get("L", 4.0), n_seg=job["n_seg"])
    r["cpu_s"] = time.time() - t0
    r["run_id"] = run_id
    _save_run(shard, run_id, r)
    return r


def item4a_junction_nl(shard="A", n_seg=400):
    jobs = []
    for N in (4, 8, 16):
        for ef in (0.10, 0.33, 0.50, 0.67, 0.83, 0.98):
            jobs.append(dict(
                material="S", N=N, eps0_frac=ef, n_seg=n_seg, L=4.0,
                run_id=f"jnl_S_N{N}_e{ef}_n{n_seg}",
            ))
    rows = _pool_jobs(shard, jobs, _run_jnl_one, "jnl")
    fields = ["material", "N", "eps0_frac", "n_seg", "T0_Tinc", "dTmax_Tinc",
              "T_frac", "pred_r", "pred_SN", "pred_T0_1d", "pred_dTmax_1d",
              "pred_T_1d", "pred_SN2d", "pred_T0_2d", "pred_dTmax_2d",
              "pred_T_2d", "cpu_s"]
    out_rows = []
    for r in rows:
        out_rows.append({
            "material": r["material"], "N": r["N"],
            "eps0_frac": f"{r['eps0_frac']:.6g}", "n_seg": r["n_seg"],
            "T0_Tinc": f"{r['T0_Tinc']:.6g}",
            "dTmax_Tinc": f"{r['dTmax_Tinc']:.6g}",
            "T_frac": f"{r.get('T_frac', float('nan')):.6g}",
            "pred_r": f"{r.get('pred_r', r.get('r', float('nan'))):.6g}",
            "pred_SN": f"{r.get('pred_SN', r.get('SN', float('nan'))):.6g}",
            "pred_T0_1d": f"{r.get('pred_T0_1d', r.get('T0_1d', float('nan'))):.6g}",
            "pred_dTmax_1d": f"{r.get('pred_dTmax_1d', float('nan')):.6g}",
            "pred_T_1d": f"{r.get('pred_T_1d', float('nan')):.6g}",
            "pred_SN2d": f"{r.get('pred_SN2d', float('nan')):.6g}",
            "pred_T0_2d": f"{r.get('pred_T0_2d', float('nan')):.6g}",
            "pred_dTmax_2d": f"{r.get('pred_dTmax_2d', float('nan')):.6g}",
            "pred_T_2d": f"{r.get('pred_T_2d', float('nan')):.6g}",
            "cpu_s": f"{r.get('cpu_s', float('nan')):.4g}",
        })
    _write_csv(OUT / shard / "tab_junction_nl_num.csv", fields, out_rows,
               "round8_item4a")
    return rows


def _fa_break_or_slack(job):
    """Bisection on e1_frac for hub-break of thread 0, or neighbour slack."""
    shard, run_id = job["shard"], job["run_id"]
    cached = _load_run(shard, run_id)
    if cached is not None:
        return cached
    name, N, mode = job["material"], job["N"], job["mode"]
    n_seg = job.get("n_seg", 400)
    L = job.get("L", 4.0)
    lo, hi = 0.12, 1.05
    t0 = time.time()
    mat = get_material(name)

    def probe(e1f):
        r = run_junction_sim(name, N=N, ep_frac=0.1, e1_frac=e1f, L=L,
                             n_seg=n_seg, return_res=True)
        res = r["result"]
        disc = r["disc"]
        order = r["seg_order"]
        hub_seg = order[0][0]
        broke = False
        if res.trajectory.failures.size:
            segs = set(int(s) for s in res.trajectory.failures[:, 0])
            broke = hub_seg in segs
        strains = np.asarray(r["strains"])
        phi = 2 * np.pi * np.arange(N) / N
        j_near = 1 + int(np.argmin(np.abs(np.cos(phi[1:]))))
        slack = bool(strains[j_near] <= 1e-8)
        n_slack = int(np.sum(strains[1:] <= 1e-8))
        Tnear = float(mat.sigma(max(strains[j_near], 0.0)))
        return dict(broke=broke, slack=slack, n_slack=n_slack,
                    dTmax=r["dTmax_sim"], T0=r["T0_sim"],
                    dTopp=r["dTopp_sim"], Tnear=Tnear,
                    strains=strains.tolist())

    target = "broke" if mode == "break" else "slack"
    p_lo, p_hi = probe(lo), probe(hi)
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        p = probe(mid)
        if p[target]:
            hi, p_hi = mid, p
        else:
            lo, p_lo = mid, p
        if hi - lo < 0.005 * max(hi, 1e-6):
            break
    rec = dict(
        material=name, N=N, mode=mode, eps1_frac=hi, n_seg=n_seg,
        n_slack=p_hi["n_slack"], dTmax=p_hi["dTmax"], T0=p_hi["T0"],
        dTopp=p_hi["dTopp"], Tnear=p_hi.get("Tnear"),
        reached=bool(p_hi[target]), cpu_s=time.time() - t0, run_id=run_id,
    )
    if not rec["reached"]:
        rec["note"] = f"{mode} not reached at e1_frac<={hi}"
    _save_run(shard, run_id, rec)
    return rec


def _fa_amp_one(job):
    shard, run_id = job["shard"], job["run_id"]
    cached = _load_run(shard, run_id)
    if cached is not None:
        return cached
    t0 = time.time()
    r = run_junction_sim("S", N=job["N"], ep_frac=0.1, e1_frac=job["e1_frac"],
                         L=job.get("L", 4.0), n_seg=job.get("n_seg", 400))
    rec = dict(
        material="S", N=job["N"], e1_frac=job["e1_frac"],
        dT0_Tinc=r["T0_sim"], dTopp_Tinc=r["dTopp_sim"],
        dT0_1d=r["T0_ref"], dTopp_1d=r["dTopp_ref"],
        dT0_2d=r["T0_2d_with"], dTopp_2d=r["dTopp_2d_with"],
        cpu_s=time.time() - t0, run_id=run_id,
    )
    _save_run(shard, run_id, rec)
    return rec


def item4b_finite_amp(shard="A", n_seg=400):
    jobs = []
    for mat in ("S", "D"):
        for N in (5, 6, 8, 12, 16):
            for mode in ("break", "slack"):
                jobs.append(dict(
                    material=mat, N=N, mode=mode, n_seg=n_seg, L=4.0,
                    run_id=f"fa_{mode}_{mat}_N{N}_n{n_seg}",
                ))
    rows_b = _pool_jobs(shard, jobs, _fa_break_or_slack, "fa_bisect")
    jobs_a = []
    for N in (8, 16):
        for e1 in (0.2, 0.3, 0.4, 0.6, 0.8):
            jobs_a.append(dict(
                N=N, e1_frac=e1, n_seg=n_seg, L=4.0,
                run_id=f"fa_amp_S_N{N}_e{e1}_n{n_seg}",
            ))
    rows_a = _pool_jobs(shard, jobs_a, _fa_amp_one, "fa_amp")
    fields = ["kind", "material", "N", "mode", "eps1_frac", "e1_frac",
              "n_slack", "dTmax", "T0", "dTopp",
              "dT0_Tinc", "dTopp_Tinc", "dT0_1d", "dTopp_1d",
              "dT0_2d", "dTopp_2d", "n_seg"]
    out = []
    for r in rows_b:
        out.append({
            "kind": "bisect", "material": r["material"], "N": r["N"],
            "mode": r["mode"], "eps1_frac": f"{r['eps1_frac']:.6g}",
            "n_slack": r.get("n_slack", ""),
            "dTmax": f"{r.get('dTmax', float('nan')):.6g}",
            "T0": f"{r.get('T0', float('nan')):.6g}",
            "dTopp": f"{r.get('dTopp', float('nan')):.6g}",
            "n_seg": r.get("n_seg", n_seg),
        })
    for r in rows_a:
        out.append({
            "kind": "amp", "material": "S", "N": r["N"],
            "e1_frac": f"{r['e1_frac']:.6g}",
            "dT0_Tinc": f"{r['dT0_Tinc']:.6g}",
            "dTopp_Tinc": f"{r['dTopp_Tinc']:.6g}",
            "dT0_1d": f"{r['dT0_1d']:.6g}",
            "dTopp_1d": f"{r['dTopp_1d']:.6g}",
            "dT0_2d": f"{r['dT0_2d']:.6g}",
            "dTopp_2d": f"{r['dTopp_2d']:.6g}",
            "n_seg": n_seg,
        })
    _write_csv(OUT / shard / "tab_fa_num.csv", fields, out, "round8_item4b")
    return rows_b, rows_a


def item4_ns_check(shard="A"):
    """Double n_s on one 4a case; require < 0.5 % change."""
    run_id = "jnl_nscheck"
    cached = _load_run(shard, run_id)
    if cached is None:
        a = run_junction_nl("S", N=8, eps0_frac=0.33, L=4.0, n_seg=400)
        b = run_junction_nl("S", N=8, eps0_frac=0.33, L=4.0, n_seg=800)
        rel = abs(a["T0_Tinc"] - b["T0_Tinc"]) / max(abs(b["T0_Tinc"]), 1e-12)
        rec = dict(T0_400=a["T0_Tinc"], T0_800=b["T0_Tinc"], rel=rel,
                   ok=bool(rel <= 0.005), run_id=run_id)
        _save_run(shard, run_id, rec)
        cached = rec
    print(f"[{shard}] item4 n_s 400 vs 800 T0 rel={cached['rel']:.4%} "
          f"ok={cached['ok']}", flush=True)
    if not cached["ok"]:
        print(f"[{shard}] WARNING: n_s=400 not within 0.5%; using n_s=800 "
              "for item 4", flush=True)
    return cached


def item4(shard="A"):
    chk = item4_ns_check(shard)
    n_seg = 800 if not chk.get("ok", True) else 400
    item4a_junction_nl(shard, n_seg=n_seg)
    item4b_finite_amp(shard, n_seg=n_seg)


# --------------------------------------------------------------------------- #
# Item 5
# --------------------------------------------------------------------------- #
def item5_ring_ns20(shard="A"):
    rows = []
    for mat in ("S", "D"):
        run_id = f"ring_ns20_{mat}_v20"
        cached = _load_run(shard, run_id)
        if cached is not None:
            rows.append(cached)
            continue
        cfg = paper_star_cfg(mat, 20.0, 0.0, 1.0, n_s=20, t_end=0.25,
                             kind="star_with_rings")
        cache = str(OUT / shard / "mmin_cache" / run_id)
        Path(cache).mkdir(parents=True, exist_ok=True)
        rec = {"material": mat, "v0": "20", "n_s": 20, "net": "star+ring"}
        t0 = time.time()
        for crit, key in (("A", "mA_ns20_g"), ("B_any", "mBany_ns20_g")):
            mc = MminConfig(
                criterion=crit, tol=0.01, impact_points=IMPACT3,
                n_procs=min(3, max(1, N_JOBS)),
                cache_dir=cache + f"_{crit}", R_max=0.5,
            )
            r = minimum_mass(cfg, mc)
            rec[key] = f"{1e3 * r.m_min:.6g}"
        ref = RING_NS10[(mat, 20)]
        rec["mA_ns10_g"] = f"{1e3 * ref['A']:.6g}"
        rec["mBany_ns10_g"] = f"{1e3 * ref['Bany']:.6g}"
        rec["d_mA_pct"] = f"{100 * abs(float(rec['mA_ns20_g']) - 1e3 * ref['A']) / (1e3 * ref['A']):.4g}"
        rec["d_mB_pct"] = f"{100 * abs(float(rec['mBany_ns20_g']) - 1e3 * ref['Bany']) / (1e3 * ref['Bany']):.4g}"
        r10 = ref["A"] / ref["Bany"]
        r20 = float(rec["mA_ns20_g"]) / float(rec["mBany_ns20_g"])
        rec["d_ratio_pct"] = f"{100 * abs(r20 - r10) / r10:.4g}"
        rec["warn_mass"] = int(max(float(rec["d_mA_pct"]),
                                   float(rec["d_mB_pct"])) > 5)
        rec["warn_ratio"] = int(float(rec["d_ratio_pct"]) > 3)
        rec["cpu_s"] = time.time() - t0
        rec["run_id"] = run_id
        _save_run(shard, run_id, rec)
        rows.append(rec)
        print(f"[{shard}] ring ns20 {mat} {rec}", flush=True)
        if rec["warn_mass"] or rec["warn_ratio"]:
            print(f"[{shard}] WARNING star+ring {mat}: n_s=20 vs 10 "
                  f"d_mA={rec['d_mA_pct']}% d_mB={rec['d_mB_pct']}% "
                  f"d_ratio={rec['d_ratio_pct']}%", flush=True)
    _write_csv(OUT / shard / "ring_ns20.csv",
               ["net", "material", "v0", "n_s", "mA_ns20_g", "mBany_ns20_g",
                "mA_ns10_g", "mBany_ns10_g", "d_mA_pct", "d_mB_pct",
                "d_ratio_pct", "warn_mass", "warn_ratio", "cpu_s"],
               rows, "round8_item5")
    return rows


# --------------------------------------------------------------------------- #
# Item 6
# --------------------------------------------------------------------------- #
def item6_leftovers(shard="A"):
    from validation import export_paper as ep
    ep.OUT_DIR = _ROOT / "paper_results"
    p1 = export_params_used()
    p2 = export_conv()
    for p in (p1, p2):
        dest = OUT / shard / Path(p).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(Path(p).read_text())
        print(f"[{shard}] wrote {p}", flush=True)


# --------------------------------------------------------------------------- #
# Item 7
# --------------------------------------------------------------------------- #
def item7_frictionless(shard="D"):
    rows = []
    for mat in ("S", "D"):
        run_id = f"mmin_fric_{mat}_v20"
        cached = _load_run(shard, run_id)
        if cached is not None:
            rows.append(cached)
            continue
        cfg = paper_star_cfg(mat, 20.0, 0.0, 1.0, n_s=40, t_end=0.25,
                             mode="frictionless")
        cache = str(OUT / shard / "mmin_cache" / run_id)
        Path(cache).mkdir(parents=True, exist_ok=True)
        rec = {"material": mat, "v0": "20", "n_s": 40, "contact": "frictionless"}
        t0 = time.time()
        free = MMIN_7C_KG[(mat, 20)]
        rec["mA_gripped_g"] = f"{1e3 * free['A']:.6g}"
        rec["mBany_gripped_g"] = f"{1e3 * free['Bany']:.6g}"
        rec["mBloc_gripped_g"] = f"{1e3 * free['Bloc']:.6g}"
        for crit, key in (("A", "mA_fric_g"), ("B_any", "mBany_fric_g"),
                          ("B_loc", "mBloc_fric_g")):
            mc = MminConfig(
                criterion=crit, tol=0.01, impact_points=IMPACT3,
                n_procs=min(3, max(1, N_JOBS)),
                cache_dir=cache + f"_{crit}", R_max=0.5,
            )
            try:
                r = minimum_mass(cfg, mc)
                rec[key] = f"{1e3 * r.m_min:.6g}"
                rec[f"worst_{crit}"] = (
                    f"{r.worst_point[0]:.6g},{r.worst_point[1]:.6g}")
                rec[f"scan_{crit}"] = str(r.scan_pattern)
            except Exception as e:
                rec[key] = ""
                rec[f"error_{crit}"] = str(e)
                print(f"[{shard}] frictionless {mat} {crit} failed: {e}",
                      flush=True)
        rec["cpu_s"] = time.time() - t0
        rec["run_id"] = run_id
        _save_run(shard, run_id, rec)
        rows.append(rec)
        print(f"[{shard}] frictionless {mat} {rec}", flush=True)
    _write_csv(OUT / shard / "mmin_frictionless.csv",
               ["material", "v0", "n_s", "contact",
                "mA_fric_g", "mBany_fric_g", "mBloc_fric_g",
                "mA_gripped_g", "mBany_gripped_g", "mBloc_gripped_g",
                "worst_A", "worst_B_any", "worst_B_loc", "cpu_s"],
               rows, "round8_item7")
    return rows


# --------------------------------------------------------------------------- #
# Pilot
# --------------------------------------------------------------------------- #
def _blas_report():
    info = {k: os.environ.get(k, "<unset>")
            for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                      "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
    try:
        import threadpoolctl
        info["threadpool"] = threadpoolctl.threadpool_info()
    except Exception as e:
        info["threadpool_error"] = str(e)
    print("BLAS/threads:", json.dumps(info, default=str), flush=True)
    return info


def _one_pilot_run(mat="S"):
    masses = MMIN_7C_KG[(mat, 20)]
    cfg0 = paper_star_cfg(mat, 20.0, 0.5, 1.0)
    s = _s_for_mass(cfg0, masses["Bany"])
    cfg = paper_star_cfg(mat, 20.0, 0.5, s, n_s=40, t_end=0.25, dt_out=5e-3)
    t0 = time.time()
    res = simulate_config(cfg, write=False)
    cpu = time.time() - t0
    return dict(material=mat, cpu_s=cpu, outcome=res.outcome,
                arrested=res.arrested, m_net_g=1e3 * res.m_net,
                steps=res.trajectory.steps)


def _pilot_copy_work(_=None):
    """Picklable worker for the 8-vs-16 throughput test."""
    return _one_pilot_run("S")


def _timed_copies(n_copies, n_workers):
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        list(pool.map(_pilot_copy_work, range(n_copies)))
    wall = time.time() - t0
    rph = 3600.0 * n_copies / wall if wall > 0 else float("nan")
    return dict(n_copies=n_copies, n_workers=n_workers, wall_s=wall,
                runs_per_hour=rph)


def item_pilot(shard="pilot"):
    info = _blas_report()
    print("[pilot] single S then D at n_s=40, m=mBany, a=R/2, v0=20",
          flush=True)
    s = _one_pilot_run("S")
    d = _one_pilot_run("D")
    print(f"[pilot] S cpu={s['cpu_s']:.2f}s outcome={s['outcome']}", flush=True)
    print(f"[pilot] D cpu={d['cpu_s']:.2f}s outcome={d['outcome']}", flush=True)
    t8 = _timed_copies(8, 8)
    t16 = _timed_copies(16, 16)
    print(f"[pilot] 8 copies / 8 workers: {t8['runs_per_hour']:.1f} runs/h "
          f"wall={t8['wall_s']:.1f}s", flush=True)
    print(f"[pilot] 16 copies / 16 workers: {t16['runs_per_hour']:.1f} runs/h "
          f"wall={t16['wall_s']:.1f}s", flush=True)
    better = 16 if t16["runs_per_hour"] > t8["runs_per_hour"] else 8
    print(f"[pilot] using JOBS={better} on shards", flush=True)
    rec = dict(blas=info, S=s, D=d, t8=t8, t16=t16, better_jobs=better)
    # Extrapolate items.
    cpu = 0.5 * (s["cpu_s"] + d["cpu_s"])
    rph = t8["runs_per_hour"] if better == 8 else t16["runs_per_hour"]
    est = {
        "item1_144": 144 / rph,
        "item2_182": 182 / rph,
        "item5_mmin": "Alg2 ~ tens of S/D bisection evals",
        "item7_mmin": "Alg2 frictionless, similar to 7c half-table",
        "cpu_s_per_dyn": cpu,
        "runs_per_hour": rph,
        "jobs": better,
    }
    print("[pilot] wall-time extrapolation (hours):", json.dumps(est),
          flush=True)
    rec["estimate_h"] = est
    p = OUT / shard / "pilot.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, default=str, indent=2))
    print(f"[pilot] wrote {p}", flush=True)
    return rec


# --------------------------------------------------------------------------- #
# Merge
# --------------------------------------------------------------------------- #
def merge_shards():
    """Copy shard CSVs into paper_results/round8/ and check run ids."""
    dest = _ROOT / "paper_results" / "round8"
    dest.mkdir(parents=True, exist_ok=True)
    mapping = {
        "B": ["mass_scan.csv", "mmin_lockxy.csv"],
        "C": ["lambda_map.csv", "lambda_map.png"],
        "A": ["tab_junction_nl_num.csv", "tab_fa_num.csv", "ring_ns20.csv",
              "fig_etaa.csv", "tab_conv.csv", "params_used.csv"],
        "D": ["mmin_frictionless.csv"],
    }
    ids = []
    dups = []
    seen = set()
    for shard, files in mapping.items():
        sdir = dest / shard
        if not sdir.is_dir():
            print(f"merge: missing shard {shard}", flush=True)
            continue
        for name in files:
            src = sdir / name
            if src.is_file():
                (dest / name).write_bytes(src.read_bytes())
                print(f"merge: {src} -> {dest / name}", flush=True)
        rdir = sdir / "runs"
        if rdir.is_dir():
            for p in rdir.glob("*.json"):
                rid = p.stem
                if rid in seen:
                    dups.append(rid)
                seen.add(rid)
                ids.append(rid)
        h5 = sdir / "hdf5"
        if h5.is_dir():
            t = dest / "hdf5"
            t.mkdir(exist_ok=True)
            for p in h5.glob("*.h5"):
                target = t / p.name
                if not target.exists():
                    target.write_bytes(p.read_bytes())
    report = dict(n_run_ids=len(ids), n_unique=len(seen), duplicates=dups)
    (dest / "merge_report.json").write_text(json.dumps(report, indent=2))
    print("merge report", report, flush=True)
    if dups:
        raise SystemExit(f"duplicate run ids: {dups}")
    return report


def run_shard(shard: str):
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"=== round8 shard={shard} jobs={N_JOBS} commit={_COMMIT} "
          f"dirty={_code_dirty()} ===", flush=True)
    t0 = time.time()
    if shard == "pilot":
        item_pilot(shard)
    elif shard == "A":
        item6_leftovers(shard)
        item3_etaB_free(shard)
        item4(shard)
        item5_ring_ns20(shard)
    elif shard == "B":
        item1_mass_scan(shard)
    elif shard == "C":
        item2_lambda_map(shard)
    elif shard == "D":
        item7_frictionless(shard)
    elif shard == "merge":
        merge_shards()
    else:
        raise SystemExit(f"unknown shard {shard}")
    print(f"=== shard {shard} done in {time.time() - t0:.0f}s ===", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="pilot")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.merge:
        merge_shards()
        return
    run_shard(args.shard)


if __name__ == "__main__":
    main()
