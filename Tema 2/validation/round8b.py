"""Round 8b: fix 4a/4b, classify timeouts, extend Alg.2 past R/2.

Local (minutes): items 1–3. GCP: item 4 (12 tab_mmin rows, a/R in {0.6,0.7,0.8}).

Usage:
  python -m validation.round8b --local
  python -m validation.round8b --shard A|B|C|D
  python -m validation.round8b --merge
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import shutil
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

from netsim.discretize import discretize
from netsim.io_hdf5 import git_commit
from netsim.materials import get_material
from netsim.mmin import MminConfig, evaluate, minimum_mass
from netsim.simulate import simulate_config, build_net
from netsim.integrator import PenetrationError

import riemann

from validation.export_paper import (
    _MMIN_CACHE, _mmin_cfg, _mmin_n_procs, _MMIN_TOL,
)
from validation.round8 import (
    _classify, _code_dirty, _pool_jobs, _progress, _s_for_mass,
    _save_run, _load_run, energy_T_frac, paper_star_cfg, N_JOBS,
)
from validation.test3_junction import run_junction_sim

OUT = Path(os.environ.get("ROUND8B_OUT", _ROOT / "paper_results" / "round8b"))
R8 = Path(os.environ.get("ROUND8_OUT", _ROOT / "paper_results" / "round8"))
_COMMIT = os.environ.get("GIT_COMMIT", "").strip() or git_commit()
_DATE = datetime.datetime.now().isoformat(timespec="seconds")

PISTON_N = 24
PTS_OLD = [(0.0, 0.0), (0.25, 0.0), (0.5, 0.0)]
PTS_NEW = [(0.6, 0.0), (0.7, 0.0), (0.8, 0.0)]
PTS_EXT = PTS_OLD + PTS_NEW
TAB_MMIN_CASES = [
    ("star", "S", 1.0, 10.0, 40),
    ("star", "S", 1.0, 15.0, 40),
    ("star", "S", 1.0, 20.0, 40),
    ("star", "D", 1.0, 10.0, 40),
    ("star", "D", 1.0, 15.0, 40),
    ("star", "D", 1.0, 20.0, 40),
    ("star+ring", "S", 1.0, 10.0, 10),
    ("star+ring", "S", 1.0, 15.0, 10),
    ("star+ring", "S", 1.0, 20.0, 10),
    ("star+ring", "D", 1.0, 10.0, 10),
    ("star+ring", "D", 1.0, 15.0, 10),
    ("star+ring", "D", 1.0, 20.0, 10),
]
SHARD_CASES = {
    "A": TAB_MMIN_CASES[0:3],
    "B": TAB_MMIN_CASES[3:6],
    "C": TAB_MMIN_CASES[6:9],
    "D": TAB_MMIN_CASES[9:12],
}

# Import round8 helpers that close over OUT; redirect them to round8b.
import validation.round8 as _r8
_r8.OUT = OUT


def _header(config: str) -> str:
    return (f"# commit={_COMMIT} date={_DATE} config={config} "
            f"code_dirty={_code_dirty()}\n")


def _write(path: Path, fields, rows, config: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        fh.write(_header(config))
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


# --------------------------------------------------------------------------- #
# Item 1: 4b diagnosis + rerun
# --------------------------------------------------------------------------- #
def _strain_along_thread0(res, disc, order, n_sample=41):
    segs = order[0]
    idx = np.linspace(0, len(segs) - 1, n_sample, dtype=int)
    out = []
    x = res.trajectory.x[-1]
    for k in idx:
        s = segs[int(k)]
        i, j = disc.seg_edges[s]
        eps = float(np.linalg.norm(x[j] - x[i]) / disc.seg_rest_length[s] - 1.0)
        out.append((int(k), int(s), eps))
    return out


def diagnose_4b(shard="local"):
    """S, N=8, e1=0.6 eps_b: plateau, sign, first failure."""
    run_id = "diag2_fa_S_N8_e0.6"
    cached = _load_run(shard, run_id)
    if cached is not None:
        return cached
    name, N, e1f, L, n_seg = "S", 8, 0.6, 4.0, 400
    mat = get_material(name)
    ref_mat = getattr(riemann, name)
    ep = 0.1 * mat.eps_b
    e1 = e1f * mat.eps_b
    v1 = float(ref_mat.Phi(e1, ep))
    t0 = time.time()
    r = run_junction_sim(name, N=N, ep_frac=0.1, e1_frac=e1f, L=L,
                         n_seg=n_seg, return_res=True, piston_n_seg=PISTON_N,
                         clip_before_fail=True, use_numba=True)
    res, disc, order = r["result"], r["disc"], r["seg_order"]
    traj = res.trajectory
    c_inc = ref_mat.c(e1) if e1 > 0 else mat.c_L0
    tarr = L / c_inc
    times = np.linspace(0.15 * tarr, 0.95 * tarr, 5)
    snapshots = []
    segs = order[0]
    hub_seg, last_seg = segs[0], segs[-1]
    for tt in times:
        fr = int(np.argmin(np.abs(traj.t - tt)))
        xs = traj.x[fr]
        eps = []
        for s in segs:
            i, j = disc.seg_edges[s]
            eps.append(float(np.linalg.norm(xs[j] - xs[i])
                             / disc.seg_rest_length[s] - 1.0))
        eps = np.asarray(eps)
        # Plateau: last 40% of the thread (behind a right-going? left-going
        # from the far end: high indices).
        plat = float(np.mean(eps[int(0.6 * eps.size):]))
        snapshots.append(dict(
            t=float(traj.t[fr]), t_over_tarr=float(traj.t[fr] / tarr),
            plat=plat, plat_over_e1=plat / e1 if e1 else float("nan"),
            eps_hub=float(eps[0]), eps_drive=float(eps[-1]),
            max_eps=float(eps.max()),
        ))
    fails = []
    if traj.failures.size:
        for s, p, tf in traj.failures:
            fails.append(dict(seg=int(s), parent=int(p), t=float(tf),
                              t_over_tarr=float(tf / tarr),
                              is_hub=int(s) == int(hub_seg),
                              is_drive=int(s) == int(last_seg)))
    rec = dict(
        run_id=run_id, material=name, N=N, e1_frac=e1f, e1=e1, ep=ep,
        Phi=v1, v1_sign="+x (outward, same as riemann.Phi)",
        tarr=tarr, piston_n_seg=PISTON_N,
        T0=r["T0_sim"], T0_ref=r["T0_ref"], T0_2d=r["T0_2d_with"],
        snapshots=snapshots, failures=fails,
        n_fail=len(fails),
        plateau_ok=all(abs(s["plat_over_e1"] - 1.0) < 0.01 for s in snapshots),
        failed_before_hub=any(f["t"] < tarr and not f["is_hub"] for f in fails),
        cpu_s=time.time() - t0,
    )
    _save_run(shard, run_id, rec)
    fields = ["t", "t_over_tarr", "plat", "plat_over_e1", "eps_hub",
              "eps_drive", "max_eps"]
    _write(OUT / "diag_fa_snapshots.csv", fields, snapshots, "round8b_4b_diag")
    return rec


def _fa_probe(name, N, e1f, L, n_seg):
    r = run_junction_sim(name, N=N, ep_frac=0.1, e1_frac=e1f, L=L,
                         n_seg=n_seg, return_res=True, piston_n_seg=PISTON_N,
                         clip_before_fail=True, use_numba=True)
    res, order = r["result"], r["seg_order"]
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
    mat = get_material(name)
    Tnear = float(mat.sigma(max(strains[j_near], 0.0)))
    return dict(broke=broke, slack=slack, n_slack=n_slack,
                dTmax=r["dTmax_sim"], T0=r["T0_sim"],
                dTopp=r["dTopp_sim"], Tnear=Tnear,
                strains=strains.tolist())


def _fa_break_or_slack(job):
    shard, run_id = job["shard"], job["run_id"]
    cached = _load_run(shard, run_id)
    if cached is not None:
        return cached
    name, N, mode = job["material"], job["N"], job["mode"]
    n_seg = job.get("n_seg", 400)
    L = job.get("L", 4.0)
    lo, hi = 0.12, 0.98
    t0 = time.time()
    target = "broke" if mode == "break" else "slack"

    def probe(e1f):
        return _fa_probe(name, N, e1f, L, n_seg)

    p_lo, p_hi = probe(lo), probe(hi)
    if not p_hi[target]:
        grid = np.linspace(lo, hi, 18)
        last_no, first_yes = lo, None
        p_yes = None
        for e in grid[1:]:
            p = probe(float(e))
            if p[target]:
                first_yes, p_yes = float(e), p
                break
            last_no, p_lo = float(e), p
        if first_yes is None:
            rec = dict(
                material=name, N=N, mode=mode, eps1_frac=hi, n_seg=n_seg,
                n_slack=p_hi["n_slack"], dTmax=p_hi["dTmax"], T0=p_hi["T0"],
                dTopp=p_hi["dTopp"], Tnear=p_hi.get("Tnear"),
                reached=False, cpu_s=time.time() - t0, run_id=run_id,
                note=f"{mode} not reached at e1_frac<={hi} with piston={PISTON_N}",
            )
            _save_run(shard, run_id, rec)
            return rec
        lo, hi, p_hi = last_no, first_yes, p_yes
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        p = probe(mid)
        if p[target]:
            hi, p_hi = mid, p
        else:
            lo, p_lo = mid, p
        if hi - lo < 0.005 * max(hi, 1e-6):
            break
    # Amplitudes and ratios from the last intact probe; eps1_frac is the
    # first amplitude that meets the target (hub break or slack).
    meas = p_lo if p_lo is not None else p_hi
    rec = dict(
        material=name, N=N, mode=mode, eps1_frac=hi, n_seg=n_seg,
        n_slack=meas["n_slack"], dTmax=meas["dTmax"], T0=meas["T0"],
        dTopp=meas["dTopp"], Tnear=meas.get("Tnear"),
        reached=bool(p_hi[target]), cpu_s=time.time() - t0, run_id=run_id,
        piston_n_seg=PISTON_N,
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
    r = run_junction_sim(
        "S", N=job["N"], ep_frac=0.1, e1_frac=job["e1_frac"],
        L=job.get("L", 4.0), n_seg=job.get("n_seg", 400),
        piston_n_seg=PISTON_N, clip_before_fail=True, use_numba=True,
    )
    rec = dict(
        material="S", N=job["N"], e1_frac=job["e1_frac"],
        dT0_Tinc=r["T0_sim"], dTopp_Tinc=r["dTopp_sim"],
        dT0_1d=r["T0_ref"], dTopp_1d=r["dTopp_ref"],
        dT0_2d=r["T0_2d_with"], dTopp_2d=r["dTopp_2d_with"],
        cpu_s=time.time() - t0, run_id=run_id, piston_n_seg=PISTON_N,
    )
    _save_run(shard, run_id, rec)
    return rec


def item4b(shard="local", n_seg=400):
    diagnose_4b(shard)
    jobs = []
    for mat in ("S", "D"):
        for N in (5, 6, 8, 12, 16):
            for mode in ("break", "slack"):
                jobs.append(dict(
                    material=mat, N=N, mode=mode, n_seg=n_seg, L=4.0,
                    run_id=f"fa8b2_{mode}_{mat}_N{N}_n{n_seg}_p{PISTON_N}",
                ))
    rows_b = _pool_jobs(shard, jobs, _fa_break_or_slack, "fa_bisect")
    jobs_a = []
    for N in (8, 16):
        for e1 in (0.2, 0.3, 0.4, 0.6, 0.8):
            jobs_a.append(dict(
                N=N, e1_frac=e1, n_seg=n_seg, L=4.0,
                run_id=f"fa8b2_amp_S_N{N}_e{e1}_n{n_seg}_p{PISTON_N}",
            ))
    rows_a = _pool_jobs(shard, jobs_a, _fa_amp_one, "fa_amp")
    fields = ["kind", "material", "N", "mode", "eps1_frac", "e1_frac",
              "n_slack", "dTmax", "T0", "dTopp",
              "dT0_Tinc", "dTopp_Tinc", "dT0_1d", "dTopp_1d",
              "dT0_2d", "dTopp_2d", "n_seg", "reached"]
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
            "reached": int(bool(r.get("reached", False))),
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
    _write(OUT / "tab_fa_num.csv", fields, out, "round8b_item4b")
    dest = R8 / "tab_fa_num.csv"
    if OUT.joinpath("tab_fa_num.csv").is_file():
        shutil.copy2(OUT / "tab_fa_num.csv", dest)
        (R8 / "A").mkdir(parents=True, exist_ok=True)
        shutil.copy2(OUT / "tab_fa_num.csv", R8 / "A" / "tab_fa_num.csv")
    return rows_b, rows_a


# --------------------------------------------------------------------------- #
# Item 2: T_frac rewrite + large-ε0 diagnostics
# --------------------------------------------------------------------------- #
def _t_frac_from_cached(r):
    mat = get_material(r["material"])
    ep = 0.1 * mat.eps_b
    eps0 = min(float(r["eps0_frac"]), 0.98) * mat.eps_b
    strains = np.asarray(r["strains"], dtype=float)
    d_frac = float(r.get("d_frac", 1e-3))
    e1 = eps0 + d_frac * mat.eps_b
    Tinc = float(mat.sigma(e1) - mat.sigma(eps0))
    N = int(r["N"])
    dT = np.array([(mat.sigma(strains[k]) - mat.sigma(
        eps0 if k == 0 else ep)) / Tinc for k in range(N)])
    return float(dT[0]), float(dT[1:].max()), energy_T_frac(mat, eps0, ep, dT)


def item4a_rewrite_and_diag(shard="local"):
    rows = []
    src = R8 / "A" / "runs"
    for N in (4, 8, 16):
        for ef in (0.10, 0.33, 0.50, 0.67, 0.83, 0.98):
            p = src / f"jnl_S_N{N}_e{ef}_n400.json"
            if not p.is_file():
                continue
            r = json.loads(p.read_text())
            T0, dTmax, Tf = _t_frac_from_cached(r)
            r2 = dict(r)
            r2["T0_Tinc"] = T0
            r2["dTmax_Tinc"] = dTmax
            r2["T_frac"] = Tf
            r2["T_frac_old"] = r.get("T_frac")
            rows.append(r2)
    fields = ["material", "N", "eps0_frac", "n_seg", "T0_Tinc", "dTmax_Tinc",
              "T_frac", "T_frac_old", "pred_r", "pred_SN", "pred_T0_1d",
              "pred_dTmax_1d", "pred_T_1d", "pred_SN2d", "pred_T0_2d",
              "pred_dTmax_2d", "pred_T_2d", "cpu_s"]
    out_rows = []
    for r in rows:
        out_rows.append({
            "material": r["material"], "N": r["N"],
            "eps0_frac": f"{r['eps0_frac']:.6g}", "n_seg": r["n_seg"],
            "T0_Tinc": f"{r['T0_Tinc']:.6g}",
            "dTmax_Tinc": f"{r['dTmax_Tinc']:.6g}",
            "T_frac": f"{r['T_frac']:.6g}",
            "T_frac_old": f"{r.get('T_frac_old', float('nan')):.6g}",
            "pred_r": f"{r.get('pred_r', float('nan')):.6g}",
            "pred_SN": f"{r.get('pred_SN', float('nan')):.6g}",
            "pred_T0_1d": f"{r.get('pred_T0_1d', float('nan')):.6g}",
            "pred_dTmax_1d": f"{r.get('pred_dTmax_1d', float('nan')):.6g}",
            "pred_T_1d": f"{r.get('pred_T_1d', float('nan')):.6g}",
            "pred_SN2d": f"{r.get('pred_SN2d', float('nan')):.6g}",
            "pred_T0_2d": f"{r.get('pred_T0_2d', float('nan')):.6g}",
            "pred_dTmax_2d": f"{r.get('pred_dTmax_2d', float('nan')):.6g}",
            "pred_T_2d": f"{r.get('pred_T_2d', float('nan')):.6g}",
            "cpu_s": f"{r.get('cpu_s', float('nan')):.4g}",
        })
    _write(OUT / "tab_junction_nl_num.csv", fields, out_rows, "round8b_item4a")
    shutil.copy2(OUT / "tab_junction_nl_num.csv", R8 / "tab_junction_nl_num.csv")
    (R8 / "A").mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUT / "tab_junction_nl_num.csv",
                 R8 / "A" / "tab_junction_nl_num.csv")

    diag_jobs = []
    for ef in (0.83, 0.98):
        base = dict(material="S", N=16, eps0_frac=ef, L=4.0)
        diag_jobs.append(dict(
            **base, n_seg=400, d_frac=1e-4, win_hi_scale=1.0,
            run_id=f"jnl8b_S_N16_e{ef}_pulse0.1",
        ))
        diag_jobs.append(dict(
            **base, n_seg=800, d_frac=1e-3, win_hi_scale=1.0,
            run_id=f"jnl8b_S_N16_e{ef}_nseg800",
        ))
        diag_jobs.append(dict(
            **base, n_seg=400, d_frac=1e-3, win_hi_scale=0.5,
            run_id=f"jnl8b_S_N16_e{ef}_win0.5",
        ))
    drows = _pool_jobs(shard, diag_jobs, _r8._run_jnl_one, "jnl_diag")
    dfields = ["run_id", "eps0_frac", "n_seg", "d_frac", "win_hi_scale",
               "T0_Tinc", "dTmax_Tinc", "T_frac", "pred_T0_2d", "rel_err_T0_2d",
               "cpu_s"]
    dout = []
    baseline = {float(r["eps0_frac"]): r for r in rows if int(r["N"]) == 16}
    for r in drows:
        pred = r.get("pred_T0_2d", float("nan"))
        T0 = float(r["T0_Tinc"])
        rel = (T0 - pred) / pred if pred else float("nan")
        dout.append({
            "run_id": r["run_id"],
            "eps0_frac": f"{r['eps0_frac']:.6g}",
            "n_seg": r["n_seg"],
            "d_frac": f"{r.get('d_frac', 1e-3):.6g}",
            "win_hi_scale": f"{r.get('win_hi_scale', 1):.6g}",
            "T0_Tinc": f"{T0:.6g}",
            "dTmax_Tinc": f"{r['dTmax_Tinc']:.6g}",
            "T_frac": f"{r.get('T_frac', float('nan')):.6g}",
            "pred_T0_2d": f"{pred:.6g}",
            "rel_err_T0_2d": f"{rel:.6g}",
            "cpu_s": f"{r.get('cpu_s', float('nan')):.4g}",
        })
    for ef, br in baseline.items():
        if ef not in (0.83, 0.98):
            continue
        pred = float(br.get("pred_T0_2d", float("nan")))
        T0 = float(br["T0_Tinc"])
        rel = (T0 - pred) / pred if pred else float("nan")
        dout.append({
            "run_id": "baseline_n400",
            "eps0_frac": f"{ef:.6g}",
            "n_seg": 400, "d_frac": "0.001", "win_hi_scale": "1",
            "T0_Tinc": f"{T0:.6g}",
            "dTmax_Tinc": f"{br['dTmax_Tinc']:.6g}",
            "T_frac": f"{br['T_frac']:.6g}",
            "pred_T0_2d": f"{pred:.6g}",
            "rel_err_T0_2d": f"{rel:.6g}",
            "cpu_s": f"{br.get('cpu_s', float('nan')):.4g}",
        })
    _write(OUT / "tab_junction_nl_diag.csv", dfields, dout, "round8b_item4a_diag")
    return rows, drows


# --------------------------------------------------------------------------- #
# Item 3: timeout subclass
# --------------------------------------------------------------------------- #
def classify_timeout(res, disc=None, *, R=1.0, r_d=0.15):
    """Classify a finished run. t_end is numerics.t_end (0.25 s for scans)."""
    traj = res.trajectory
    t_end = float(traj.t[-1])
    xd = np.asarray(traj.drone[-1, :3], dtype=float)
    vd = np.asarray(traj.drone[-1, 3:6], dtype=float)
    vxy = float(np.hypot(vd[0], vd[1]))
    vz = float(vd[2])
    speed = float(np.linalg.norm(vd))
    r_xy = float(np.hypot(xd[0], xd[1]))
    E = np.asarray(traj.energy[-1], dtype=float)
    partition = dict(
        KE_dr=float(E[0]), KE_net=float(E[1]), U_el=float(E[2]),
        U_fail=float(E[3]), U_c=float(E[4]), dE_cap=float(E[5]),
    )
    attached = False
    if disc is not None and traj.x.ndim == 3 and traj.x.shape[1] > 0:
        nodes = traj.x[-1]
        d = np.linalg.norm(nodes - xd.reshape(1, 3), axis=1)
        near = np.nonzero(d <= 1.05 * r_d)[0]
        intact = traj.intact[-1]
        if near.size and intact.size:
            for s, (i, j) in enumerate(disc.seg_edges):
                if intact[s] and (i in near or j in near):
                    attached = True
                    break
    else:
        attached = (res.outcome != "perforated") and (r_xy <= R)
    outside = r_xy > R
    if (not attached) or outside:
        subclass = "escaped"
    elif speed < 1.0:
        subclass = "slow"
    else:
        subclass = "moving"
    return dict(
        t_end=t_end, vxy=vxy, vz=vz, speed=speed,
        drone_x=float(xd[0]), drone_y=float(xd[1]), drone_z=float(xd[2]),
        r_xy=r_xy, attached=int(attached), outside=int(outside),
        timeout_subclass=subclass, **partition,
    )


def _timeout_jobs_from_csv():
    jobs = []
    mass_p = R8 / "mass_scan.csv"
    if mass_p.is_file():
        with open(mass_p) as fh:
            rows = list(csv.DictReader(
                (ln for ln in fh if not ln.startswith("#"))))
        for r in rows:
            if r.get("outcome") != "timeout":
                continue
            lock = int(float(r["lock_xy"]))
            a = float(r["a_over_R"])
            v0 = float(r["v0"])
            mat = r["material"]
            mu = r["mu_label"]
            m_net = 1e-3 * float(r["m_net_g"])
            jobs.append(dict(
                source="mass_scan", material=mat, v0=v0, a=a,
                lock_xy=bool(lock), mu_label=mu, m_net=m_net,
                t_end=0.25, n_s=40, kind="star",
                run_id=f"to8b_mscan_{mat}_v{v0:g}_a{a:g}_"
                       f"{'L' if lock else 'F'}_{mu}",
            ))
    lam_p = R8 / "lambda_map.csv"
    if lam_p.is_file():
        with open(lam_p) as fh:
            rows = list(csv.DictReader(
                (ln for ln in fh if not ln.startswith("#"))))
        for r in rows:
            cls = r.get("class") or r.get("class_")
            if cls != "timeout":
                continue
            mat = r["material"]
            a = float(r["a_over_R"])
            ratio = float(r["m_over_mstar"])
            m_net = 1e-3 * float(r["m_net_g"])
            jobs.append(dict(
                source="lambda_map", material=mat, v0=20.0, a=a,
                lock_xy=False, mu_label=f"m{ratio:g}", m_net=m_net,
                t_end=0.25, n_s=40, kind="star",
                run_id=f"to8b_lam_{mat}_a{a:g}_m{ratio:g}",
            ))
    return jobs


def _run_timeout_one(job):
    shard, run_id = job["shard"], job["run_id"]
    cached = _load_run(shard, run_id)
    if cached is not None:
        return cached
    mat, v0, a = job["material"], job["v0"], job["a"]
    t_end = float(job.get("t_end", 0.25))
    cfg0 = paper_star_cfg(mat, v0, a, 1.0, lock_xy=job.get("lock_xy", False),
                          n_s=job.get("n_s", 40), kind=job.get("kind", "star"))
    s = _s_for_mass(cfg0, job["m_net"])
    cfg = paper_star_cfg(
        mat, v0, a, s, lock_xy=job.get("lock_xy", False),
        n_s=job.get("n_s", 40), kind=job.get("kind", "star"),
        dt_out=5e-3, store_mesh=True, energy_groups=True, t_end=t_end,
    )
    disc = discretize(build_net(cfg), cfg.material.resolve(), cfg.numerics.n_s,
                      area_scale=s, r_d=cfg.drone.r_d)
    t0 = time.time()
    try:
        res = simulate_config(cfg, write=False)
    except PenetrationError as e:
        payload = dict(job, outcome="penetration_error", error=str(e),
                       cpu_s=time.time() - t0, run_id=run_id)
        _save_run(shard, run_id, payload)
        return payload
    info = classify_timeout(res, disc, R=float(cfg.net.R), r_d=cfg.drone.r_d)
    row = dict(job)
    row.update(info)
    row["outcome"] = res.trajectory.outcome
    row["class"] = _classify(res)
    row["arrested"] = int(bool(res.arrested))
    row["cpu_s"] = time.time() - t0
    row["run_id"] = run_id
    row["t_end_setting"] = t_end
    _save_run(shard, run_id, row)
    return row


def item3_timeouts(shard="local"):
    jobs = _timeout_jobs_from_csv()
    rows = _pool_jobs(shard, jobs, _run_timeout_one, "timeout")
    slow = [r for r in rows if r.get("timeout_subclass") == "slow"]
    doubled = []
    if slow:
        djobs = []
        for r in slow:
            j = dict(r)
            j["t_end"] = 2.0 * float(r.get("t_end_setting", 0.25))
            j["run_id"] = r["run_id"] + "_t2"
            djobs.append(j)
        doubled = _pool_jobs(shard, djobs, _run_timeout_one, "timeout_t2")
    fields = [
        "source", "material", "v0", "a", "lock_xy", "mu_label", "m_net",
        "outcome", "class", "timeout_subclass", "t_end", "t_end_setting",
        "speed", "vxy", "vz", "drone_x", "drone_y", "drone_z", "r_xy",
        "attached", "outside", "KE_dr", "KE_net", "U_el", "U_fail", "U_c",
        "run_id",
    ]
    _write(OUT / "timeout_class.csv", fields, rows, "round8b_item3")
    if doubled:
        _write(OUT / "timeout_class_t2.csv", fields, doubled, "round8b_item3_t2")

    # Add subclass column to copies of the round-8 CSVs (do not change class).
    def _annotate(src, key_fn, out_name, class_col):
        if not src.is_file():
            return
        with open(src) as fh:
            raw = fh.read()
        lines = raw.splitlines()
        hdr_line = next(i for i, ln in enumerate(lines) if not ln.startswith("#"))
        reader = csv.DictReader(lines[hdr_line:])
        by_id = {}
        for r in rows:
            by_id[key_fn(r)] = r.get("timeout_subclass", "")
        new_rows = []
        for row in reader:
            k = None
            if class_col == "outcome":
                k = (row["material"], float(row["v0"]), float(row["a_over_R"]),
                     int(float(row["lock_xy"])), row["mu_label"])
            else:
                k = (row["material"], float(row["a_over_R"]),
                     float(row["m_over_mstar"]))
            sub = by_id.get(k, "")
            if (row.get(class_col) or row.get("class")) != "timeout":
                sub = ""
            nr = dict(row)
            nr["timeout_subclass"] = sub
            new_rows.append(nr)
        fields2 = list(reader.fieldnames) + ["timeout_subclass"]
        _write(OUT / out_name, fields2, new_rows, "round8b_item3_annot")
        shutil.copy2(OUT / out_name, R8 / out_name.replace("timeout_", ""))

    def mass_key(r):
        return (r["material"], float(r["v0"]), float(r["a"]),
                int(bool(r.get("lock_xy"))), r["mu_label"])

    def lam_key(r):
        lab = r["mu_label"]
        ratio = float(str(lab).lstrip("m"))
        return (r["material"], float(r["a"]), ratio)

    # Write annotated copies next to originals without renaming the class col.
    if (R8 / "mass_scan.csv").is_file():
        with open(R8 / "mass_scan.csv") as fh:
            lines = fh.read().splitlines()
        hdr_i = next(i for i, ln in enumerate(lines) if not ln.startswith("#"))
        reader = csv.DictReader(lines[hdr_i:])
        lookup = {mass_key(r): r.get("timeout_subclass", "")
                  for r in rows if r.get("source") == "mass_scan"}
        new_rows = []
        for row in reader:
            k = (row["material"], float(row["v0"]), float(row["a_over_R"]),
                 int(float(row["lock_xy"])), row["mu_label"])
            sub = lookup.get(k, "") if row.get("outcome") == "timeout" else ""
            nr = dict(row)
            nr["timeout_subclass"] = sub
            new_rows.append(nr)
        fields2 = list(reader.fieldnames) + ["timeout_subclass"]
        _write(OUT / "mass_scan.csv", fields2, new_rows, "round8b_item3_mass")
        shutil.copy2(OUT / "mass_scan.csv", R8 / "mass_scan.csv")
    if (R8 / "lambda_map.csv").is_file():
        with open(R8 / "lambda_map.csv") as fh:
            lines = fh.read().splitlines()
        hdr_i = next(i for i, ln in enumerate(lines) if not ln.startswith("#"))
        reader = csv.DictReader(lines[hdr_i:])
        lookup = {}
        for r in rows:
            if r.get("source") != "lambda_map":
                continue
            ratio = float(str(r["mu_label"]).lstrip("m"))
            lookup[(r["material"], float(r["a"]), ratio)] = r.get(
                "timeout_subclass", "")
        new_rows = []
        for row in reader:
            k = (row["material"], float(row["a_over_R"]),
                 float(row["m_over_mstar"]))
            cls = row.get("class") or row.get("class_")
            sub = lookup.get(k, "") if cls == "timeout" else ""
            nr = dict(row)
            nr["timeout_subclass"] = sub
            new_rows.append(nr)
        fields2 = list(reader.fieldnames) + ["timeout_subclass"]
        _write(OUT / "lambda_map.csv", fields2, new_rows, "round8b_item3_lambda")
        shutil.copy2(OUT / "lambda_map.csv", R8 / "lambda_map.csv")

    counts = {}
    for r in rows:
        k = r.get("timeout_subclass", "?")
        counts[k] = counts.get(k, 0) + 1
    changed = []
    for d in doubled:
        orig_id = d["run_id"].replace("_t2", "")
        orig = next((x for x in rows if x["run_id"] == orig_id), None)
        if orig and d.get("class") != orig.get("class"):
            changed.append(dict(orig_run=orig_id, old=orig.get("class"),
                                new=d.get("class"),
                                new_subclass=d.get("timeout_subclass")))
    summary = dict(n=len(rows), counts=counts, n_slow=len(slow),
                   n_doubled=len(doubled), class_changed=changed,
                   t_end_setting=0.25,
                   t_end_note="paper_star_cfg / mass_scan / lambda_map use t_end=0.25 s")
    (OUT / "timeout_summary.json").write_text(json.dumps(summary, indent=2))
    return rows, doubled, summary


# --------------------------------------------------------------------------- #
# Item 4: Alg.2 at a/R ∈ {0.6, 0.7, 0.8}
# --------------------------------------------------------------------------- #
def _pt_key(p):
    if isinstance(p, str) and "," in p:
        a, b = p.split(",")
        return (float(a), float(b))
    return (float(p[0]), float(p[1]))


def _first_fail_seg(info):
    segs = info.get("fail_segs") or []
    if len(segs) == 0:
        return ""
    return int(segs[0])


def _mmin_ext_one(job):
    shard, run_id = job["shard"], job["run_id"]
    cached = _load_run(shard, run_id)
    if cached is not None:
        return cached
    net_kind, mat, M, v0, n_s = (
        job["net"], job["material"], job["M"], job["v0"], job["n_s"])
    cfg = _mmin_cfg(mat, M, v0, net_kind=net_kind, n_s=int(n_s))
    net = build_net(cfg)
    material = cfg.material.resolve()
    m1 = net.net_mass(material.rho, 1.0)
    n_inner = _mmin_n_procs()
    cache = str(_MMIN_CACHE / f"{net_kind}_{mat}_M{M:g}_v{v0:g}_ns{int(n_s)}")
    Path(cache).mkdir(parents=True, exist_ok=True)
    pts = list(PTS_EXT)
    t0 = time.time()
    mmA = MminConfig(criterion="A", tol=_MMIN_TOL, impact_points=pts,
                     n_procs=n_inner, cache_dir=cache)
    mm_any = MminConfig(criterion="B_any", tol=_MMIN_TOL, impact_points=pts,
                        n_procs=n_inner, n_scan=24, cache_dir=cache)
    mm_locs = {
        frac: MminConfig(criterion="B_loc", tol=_MMIN_TOL, impact_points=pts,
                         n_procs=n_inner, n_scan=24, R_max=frac * 1.0,
                         cache_dir=cache)
        for frac in (0.25, 0.5, 0.75)
    }
    MminConfig.assert_same_tol(mmA, mm_any, *mm_locs.values())
    rA = minimum_mass(cfg, mmA)
    r_any = minimum_mass(cfg, mm_any, s_cap=rA.s_min)
    blocs = {frac: minimum_mass(cfg, mm, s_cap=rA.s_min)
             for frac, mm in mm_locs.items()}
    named = {
        "A": (mmA, rA),
        "Bany": (mm_any, r_any),
        "Bloc025": (mm_locs[0.25], blocs[0.25]),
        "Bloc050": (mm_locs[0.5], blocs[0.5]),
        "Bloc075": (mm_locs[0.75], blocs[0.75]),
    }
    point_rows = []
    nesting_ok_all = True
    for crit, (mm, rx) in named.items():
        per = rx.per_point
        for idx, p in enumerate(pts):
            s_pt = None
            for pk, sv in per.items():
                if abs(_pt_key(pk)[0] - p[0]) < 1e-12 and abs(
                        _pt_key(pk)[1] - p[1]) < 1e-12:
                    s_pt = float(sv)
                    break
            if s_pt is None:
                s_pt = float(rx.s_min)
            m_g = 1e3 * s_pt * m1
            s_lo = 0.99 * s_pt
            info_lo = evaluate(cfg, float(s_lo), p, mm, idx)
            outcome = info_lo.get("outcome", "")
            subclass = ""
            if outcome == "timeout":
                cfg2 = _mmin_cfg(mat, M, v0, net_kind=net_kind, n_s=int(n_s))
                cfg2.numerics.area_scale = float(s_lo)
                cfg2.drone.p = (p[0], p[1])
                cfg2.numerics.store_mesh = True
                try:
                    res = simulate_config(cfg2, write=False)
                    disc = discretize(build_net(cfg2), material,
                                      int(n_s), area_scale=s_lo,
                                      r_d=cfg2.drone.r_d)
                    subclass = classify_timeout(
                        res, disc, R=1.0, r_d=cfg2.drone.r_d)[
                            "timeout_subclass"]
                except Exception as e:
                    subclass = f"error:{e}"
            point_rows.append(dict(
                criterion=crit, a_over_R=p[0], m_min_g=m_g,
                outcome_below=outcome,
                first_fail_seg=_first_fail_seg(info_lo),
                timeout_subclass=subclass, s_min=s_pt,
            ))
        # per-point nesting checked after collecting this case's A/B
    by_pt = {}
    for pr in point_rows:
        by_pt.setdefault(pr["a_over_R"], {})[pr["criterion"]] = pr["m_min_g"]
    for a, masses in by_pt.items():
        mA = masses.get("A", float("nan"))
        mBloc = min(masses.get("Bloc025", mA),
                    masses.get("Bloc050", mA),
                    masses.get("Bloc075", mA))
        mBany = masses.get("Bany", float("nan"))
        ok = (mA + 1e-12 >= mBloc) and (mBloc + 1e-12 >= mBany)
        if not ok:
            nesting_ok_all = False
    rec = dict(
        net=net_kind, material=mat, M=M, v0=v0, n_s=n_s,
        mA_min_g=1e3 * rA.m_min,
        mBany_min_g=1e3 * r_any.m_min,
        mBloc025_min_g=1e3 * blocs[0.25].m_min,
        mBloc050_min_g=1e3 * blocs[0.5].m_min,
        mBloc075_min_g=1e3 * blocs[0.75].m_min,
        worst_A=list(rA.worst_point),
        worst_Bany=list(r_any.worst_point),
        nesting_ok=int(nesting_ok_all),
        points=point_rows,
        cpu_s=time.time() - t0, run_id=run_id,
    )
    _save_run(shard, run_id, rec)
    print(f"[{shard}] mmin_ext {net_kind} {mat} v0={v0} "
          f"mA={rec['mA_min_g']:.4g}g worstA={rec['worst_A']} "
          f"cpu={rec['cpu_s']:.0f}s", flush=True)
    return rec


def item4_mmin_ext(shard):
    cases = SHARD_CASES[shard]
    jobs = []
    for net, mat, M, v0, n_s in cases:
        jobs.append(dict(
            net=net, material=mat, M=M, v0=v0, n_s=n_s,
            run_id=f"mmin8b_{net}_{mat}_M{M:g}_v{v0:g}_ns{n_s}",
        ))
    # Sequential: each case already uses MMIN_N_PROCS inner workers.
    rows = []
    t0 = time.time()
    for i, job in enumerate(jobs):
        job = dict(job)
        job["shard"] = shard
        rows.append(_mmin_ext_one(job))
        _progress(shard, i + 1, len(jobs), t0, "mmin_ext")
    _write_ext_csv(rows, OUT / shard / "tab_mmin_ext.csv")
    return rows


def _load_7c_worst():
    p = _ROOT / "paper_results" / "tab_mmin.csv"
    out = {}
    if not p.is_file():
        return out
    with open(p) as fh:
        rows = list(csv.DictReader((ln for ln in fh if not ln.startswith("#"))))
    for r in rows:
        key = (r["net"], r["material"], int(float(r["n_s"])),
               float(r["M"]), float(r["v0"]))
        out[key] = r
    return out


def _write_ext_csv(recs, path):
    old = _load_7c_worst()
    fields = [
        "net", "material", "n_s", "M", "v0", "criterion", "a_over_R",
        "m_min_g", "outcome_below", "first_fail_seg", "timeout_subclass",
        "m7c_worst_g", "m_new_worst_g", "new_worst_a", "nesting_ok",
        "cpu_s",
    ]
    crit_7c = {
        "A": "mA_min_g", "Bany": "mBany_min_g",
        "Bloc025": "mBloc025_min_g", "Bloc050": "mBloc050_min_g",
        "Bloc075": "mBloc075_min_g",
    }
    rows = []
    for rec in recs:
        key = (rec["net"], rec["material"], int(rec["n_s"]),
               float(rec["M"]), float(rec["v0"]))
        o = old.get(key, {})
        by_crit = {}
        for pr in rec["points"]:
            by_crit.setdefault(pr["criterion"], []).append(pr)
        for crit, plist in by_crit.items():
            worst = max(plist, key=lambda x: float(x["m_min_g"]))
            m7c = o.get(crit_7c.get(crit, ""), "")
            for pr in plist:
                rows.append({
                    "net": rec["net"], "material": rec["material"],
                    "n_s": rec["n_s"], "M": f"{rec['M']:.6g}",
                    "v0": f"{rec['v0']:.6g}",
                    "criterion": crit,
                    "a_over_R": f"{pr['a_over_R']:.6g}",
                    "m_min_g": f"{pr['m_min_g']:.6g}",
                    "outcome_below": pr.get("outcome_below", ""),
                    "first_fail_seg": pr.get("first_fail_seg", ""),
                    "timeout_subclass": pr.get("timeout_subclass", ""),
                    "m7c_worst_g": m7c,
                    "m_new_worst_g": f"{worst['m_min_g']:.6g}",
                    "new_worst_a": f"{worst['a_over_R']:.6g}",
                    "nesting_ok": rec.get("nesting_ok", ""),
                    "cpu_s": f"{rec.get('cpu_s', float('nan')):.4g}",
                })
    _write(path, fields, rows, "round8b_mmin_ext")
    return rows


def merge_shards():
    dest = OUT
    dest.mkdir(parents=True, exist_ok=True)
    recs = []
    seen = set()
    dups = []
    for shard in ("A", "B", "C", "D"):
        sdir = dest / shard
        rdir = sdir / "runs"
        if not rdir.is_dir():
            print(f"merge: missing {shard}", flush=True)
            continue
        for p in rdir.glob("mmin8b_*.json"):
            rec = json.loads(p.read_text())
            rid = p.stem
            if rid in seen:
                dups.append(rid)
            seen.add(rid)
            recs.append(rec)
    _write_ext_csv(recs, dest / "tab_mmin_ext.csv")
    report = dict(n=len(recs), n_unique=len(seen), duplicates=dups)
    (dest / "merge_report.json").write_text(json.dumps(report, indent=2))
    print("merge", report, flush=True)
    if dups:
        raise SystemExit(f"duplicate run ids: {dups}")
    return report


def run_local(items=None):
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"=== round8b local jobs={N_JOBS} commit={_COMMIT} "
          f"dirty={_code_dirty()} ===", flush=True)
    want = set(items or ("4b", "4a", "timeout"))
    if "4b" in want:
        item4b("local")
    if "4a" in want:
        item4a_rewrite_and_diag("local")
    if "timeout" in want:
        item3_timeouts("local")
    print(f"=== round8b local done in {time.time() - t0:.0f}s ===", flush=True)


def run_shard(shard: str):
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"=== round8b shard={shard} jobs={N_JOBS} commit={_COMMIT} "
          f"dirty={_code_dirty()} ===", flush=True)
    t0 = time.time()
    if shard == "local":
        run_local()
    elif shard in SHARD_CASES:
        item4_mmin_ext(shard)
    elif shard == "merge":
        merge_shards()
    else:
        raise SystemExit(f"unknown shard {shard}")
    print(f"=== shard {shard} done in {time.time() - t0:.0f}s ===", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--shard", default="")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--items", default="",
                    help="comma list of local items: 4b,4a,timeout")
    args = ap.parse_args()
    if args.merge:
        merge_shards()
        return
    if args.local or args.shard == "local":
        items = [x.strip() for x in args.items.split(",") if x.strip()] or None
        run_local(items)
        return
    if not args.shard:
        raise SystemExit("need --local, --shard A|B|C|D, or --merge")
    run_shard(args.shard)


if __name__ == "__main__":
    main()
