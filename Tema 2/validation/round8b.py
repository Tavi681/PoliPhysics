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
from netsim.mmin import MminConfig, analytical_s0, evaluate, minimum_mass_point
from netsim.simulate import simulate_config, build_net
from netsim.integrator import PenetrationError

import riemann

from validation.export_paper import (
    _MMIN_CACHE, _mmin_cfg, _MMIN_TOL,
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
                         n_seg=n_seg, return_res=True, **FA_KW)
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


FA_KW = dict(piston_n_seg=PISTON_N, clip_before_fail=True, use_numba=True,
             shock_visc=0.6, fail_avg_n_seg=0, damping=0.0)


def _fa_tag():
    return "p%d_v%g_a%d" % (
        FA_KW.get("piston_n_seg", PISTON_N),
        FA_KW.get("shock_visc", 0.0),
        FA_KW.get("fail_avg_n_seg", 0),
    )


def _fa_probe(name, N, e1f, L, n_seg):
    r = run_junction_sim(name, N=N, ep_frac=0.1, e1_frac=e1f, L=L,
                         n_seg=n_seg, return_res=True, **FA_KW)
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
        L=job.get("L", 4.0), n_seg=job.get("n_seg", 400), **FA_KW,
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
                    run_id=f"fa8b3_{mode}_{mat}_N{N}_n{n_seg}_{_fa_tag()}",
                ))
    rows_b = _pool_jobs(shard, jobs, _fa_break_or_slack, "fa_bisect")
    jobs_a = []
    for N in (8, 16):
        for e1 in (0.2, 0.3, 0.4, 0.6, 0.8):
            jobs_a.append(dict(
                N=N, e1_frac=e1, n_seg=n_seg, L=4.0,
                    run_id=f"fa8b3_amp_S_N{N}_e{e1}_n{n_seg}_{_fa_tag()}",
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


_CRIT_SPEC = (
    ("A", "A", None),
    ("Bany", "B_any", None),
    ("Bloc025", "B_loc", 0.25),
    ("Bloc050", "B_loc", 0.5),
    ("Bloc075", "B_loc", 0.75),
)


def _mmin_point_work(job):
    """One (row, point, criterion) Alg.2 bisection. Cache is per (point_idx, s)."""
    net_kind, mat, M, v0, n_s = (
        job["net"], job["material"], job["M"], job["v0"], job["n_s"])
    cfg = _mmin_cfg(mat, M, v0, net_kind=net_kind, n_s=int(n_s))
    cache = str(_MMIN_CACHE / f"{net_kind}_{mat}_M{M:g}_v{v0:g}_ns{int(n_s)}")
    Path(cache).mkdir(parents=True, exist_ok=True)
    s0 = analytical_s0(cfg)
    p = (float(job["px"]), float(job["py"]))
    idx = int(job["point_idx"])
    mm = MminConfig(
        criterion=job["mmin_crit"], tol=_MMIN_TOL, impact_points=[p],
        n_procs=1, n_scan=24, cache_dir=cache,
        R_max=(None if job.get("R_max") is None else float(job["R_max"])),
    )
    t0 = time.time()
    rec = minimum_mass_point((cfg, p, mm, idx, s0))
    s_cap = job.get("s_cap")
    if s_cap is not None and rec["s_min"] > float(s_cap):
        rec["s_below"] = float(rec.get("s_below", 0.995 * s_cap))
        rec["s_lo"] = float(rec.get("s_lo", rec["s_below"]))
        rec["s_min"] = float(s_cap)
        rec["s_hi"] = float(s_cap)
        rec["info"] = evaluate(cfg, float(s_cap), p, mm, idx)
    rec["cpu_s"] = time.time() - t0
    rec["crit_key"] = job["crit_key"]
    rec["px"], rec["py"] = p
    rec["net"] = net_kind
    rec["material"] = mat
    rec["M"] = M
    rec["v0"] = v0
    rec["n_s"] = n_s
    rec["run_id"] = job["row_id"]
    rec["point_idx"] = idx
    print(f"  point {job['crit_key']} {mat} v0={v0} a={p[0]:.2f} "
          f"s={rec['s_min']:.4g} cpu={rec['cpu_s']:.0f}s", flush=True)
    return rec


def _pool_point_jobs(jobs, n_workers):
    if not jobs:
        return []
    n = min(max(1, n_workers), len(jobs))
    if n <= 1:
        return [_mmin_point_work(j) for j in jobs]
    out = []
    with ProcessPoolExecutor(max_workers=n) as pool:
        futs = {pool.submit(_mmin_point_work, j): j for j in jobs}
        for fut in as_completed(futs):
            out.append(fut.result())
    return out


def _assemble_mmin_row(shard, net_kind, mat, M, v0, n_s, by_crit_point, t0):
    cfg = _mmin_cfg(mat, M, v0, net_kind=net_kind, n_s=int(n_s))
    material = cfg.material.resolve()
    m1 = build_net(cfg).net_mass(material.rho, 1.0)
    pts = list(PTS_EXT)
    point_rows = []
    nesting_ok_all = True
    masses_pt = {}
    for crit_key, mmin_crit, rmax in _CRIT_SPEC:
        mm = MminConfig(
            criterion=mmin_crit, tol=_MMIN_TOL, impact_points=pts, n_procs=1,
            n_scan=24, cache_dir=str(
                _MMIN_CACHE / f"{net_kind}_{mat}_M{M:g}_v{v0:g}_ns{int(n_s)}"),
            R_max=(None if rmax is None else rmax),
        )
        for idx, p in enumerate(pts):
            rec = by_crit_point[(crit_key, idx)]
            s_pt = float(rec["s_min"])
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
                criterion=crit_key, a_over_R=p[0], m_min_g=m_g,
                outcome_below=outcome,
                first_fail_seg=_first_fail_seg(info_lo),
                timeout_subclass=subclass, s_min=s_pt,
            ))
            masses_pt.setdefault(p[0], {})[crit_key] = m_g
    for a, masses in masses_pt.items():
        mA = masses.get("A", float("nan"))
        mBloc = min(masses.get("Bloc025", mA),
                    masses.get("Bloc050", mA),
                    masses.get("Bloc075", mA))
        mBany = masses.get("Bany", float("nan"))
        ok = (mA + 1e-12 >= mBloc) and (mBloc + 1e-12 >= mBany)
        if not ok:
            nesting_ok_all = False

    def worst(key):
        block = [pr for pr in point_rows if pr["criterion"] == key]
        w = max(block, key=lambda x: float(x["m_min_g"]))
        return float(w["m_min_g"]), (w["a_over_R"], 0.0)

    mA, wA = worst("A")
    mBany, wB = worst("Bany")
    run_id = f"mmin8b_{net_kind}_{mat}_M{M:g}_v{v0:g}_ns{n_s}"
    rec = dict(
        net=net_kind, material=mat, M=M, v0=v0, n_s=n_s,
        mA_min_g=mA, mBany_min_g=mBany,
        mBloc025_min_g=worst("Bloc025")[0],
        mBloc050_min_g=worst("Bloc050")[0],
        mBloc075_min_g=worst("Bloc075")[0],
        worst_A=list(wA), worst_Bany=list(wB),
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
    """All row×point×criterion bisections, up to JOBS workers (cache per sim)."""
    cases = SHARD_CASES[shard]
    n_workers = min(16, max(1, N_JOBS))
    pts = list(PTS_EXT)
    pending_rows = []
    done_rows = []
    for net, mat, M, v0, n_s in cases:
        run_id = f"mmin8b_{net}_{mat}_M{M:g}_v{v0:g}_ns{n_s}"
        cached = _load_run(shard, run_id)
        if cached is not None:
            done_rows.append(cached)
        else:
            pending_rows.append((net, mat, M, v0, n_s, run_id))
    t0 = time.time()
    if not pending_rows:
        _write_ext_csv(done_rows, OUT / shard / "tab_mmin_ext.csv")
        return done_rows

    a_jobs = []
    for net, mat, M, v0, n_s, run_id in pending_rows:
        for idx, p in enumerate(pts):
            a_jobs.append(dict(
                net=net, material=mat, M=M, v0=v0, n_s=n_s, row_id=run_id,
                crit_key="A", mmin_crit="A", R_max=None, s_cap=None,
                px=p[0], py=p[1], point_idx=idx,
            ))
    print(f"[{shard}] {len(a_jobs)} A-point jobs × {n_workers} workers",
          flush=True)
    a_res = _pool_point_jobs(a_jobs, n_workers)
    s_cap_row = {}
    a_by = {}
    for r in a_res:
        key = (r["run_id"], r["point_idx"])
        a_by[key] = r
        s_cap_row[r["run_id"]] = max(s_cap_row.get(r["run_id"], 0.0),
                                     float(r["s_min"]))

    b_jobs = []
    for net, mat, M, v0, n_s, run_id in pending_rows:
        cap = s_cap_row[run_id]
        for crit_key, mmin_crit, rmax in _CRIT_SPEC:
            if crit_key == "A":
                continue
            for idx, p in enumerate(pts):
                b_jobs.append(dict(
                    net=net, material=mat, M=M, v0=v0, n_s=n_s, row_id=run_id,
                    crit_key=crit_key, mmin_crit=mmin_crit,
                    R_max=(None if rmax is None else rmax),
                    s_cap=cap, px=p[0], py=p[1], point_idx=idx,
                ))
    print(f"[{shard}] {len(b_jobs)} B-point jobs × {n_workers} workers "
          f"(s_cap from A)", flush=True)
    b_res = _pool_point_jobs(b_jobs, n_workers)

    grouped = {}
    for r in a_res + b_res:
        grouped.setdefault(r["run_id"], {})[(r["crit_key"], r["point_idx"])] = r
    for net, mat, M, v0, n_s, run_id in pending_rows:
        rec = _assemble_mmin_row(
            shard, net, mat, M, v0, n_s, grouped[run_id], t0)
        done_rows.append(rec)
        _progress(shard, len(done_rows), len(cases), t0, "mmin_ext")
    _write_ext_csv(done_rows, OUT / shard / "tab_mmin_ext.csv")
    return done_rows


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


def _hub_hist(name, N, e1f, *, shock_visc=0.0, fail_avg_n_seg=0, damping=0.0,
              L=4.0, n_seg=400):
    """Peak/plateau of hub segment 0 and neighbour-1 tension history."""
    mat = get_material(name)
    r = run_junction_sim(
        name, N=N, ep_frac=0.1, e1_frac=e1f, L=L, n_seg=n_seg,
        return_res=True, piston_n_seg=PISTON_N, clip_before_fail=True,
        use_numba=True, shock_visc=shock_visc, fail_avg_n_seg=fail_avg_n_seg,
        damping=damping)
    res, disc, order = r["result"], r["disc"], r["seg_order"]
    traj = res.trajectory
    hub = order[0][0]
    near = order[1][0]
    def _eps(fr, s):
        ii, jj = disc.seg_edges[s]
        return float(np.linalg.norm(traj.x[fr, jj] - traj.x[fr, ii])
                     / disc.seg_rest_length[s] - 1.0)

    hub3 = list(order[0][:3])
    near3 = list(order[1][:3])
    eps_h, eps_h3, T_n, T_n3 = [], [], [], []
    for fr in range(traj.t.size):
        eh = _eps(fr, hub)
        en = _eps(fr, near)
        eh3 = float(np.mean([_eps(fr, s) for s in hub3]))
        en3 = float(np.mean([_eps(fr, s) for s in near3]))
        eps_h.append(eh)
        eps_h3.append(eh3)
        T_n.append(float(mat.sigma(max(en, 0.0))))
        T_n3.append(float(mat.sigma(max(en3, 0.0))))
    eps_h = np.asarray(eps_h)
    eps_h3 = np.asarray(eps_h3)
    e1 = e1f * mat.eps_b
    peak = float(eps_h.max())
    ref_mat = getattr(riemann, name)
    tarr = L / (ref_mat.c(e1) if e1 > 0 else mat.c_L0)
    t_fail = float("inf")
    if traj.failures.size:
        for s, _p, tf in traj.failures:
            if int(s) == int(hub):
                t_fail = float(tf)
                break
    # Plateau: median of the late, non-exploded hub strain (after 0.5 t_end,
    # before hub failure). The 0.7 tarr window sits in the prestress if the
    # nonlinear shock is slower than c(e1).
    t_hi = min(t_fail, float(traj.t[-1]))
    mask = (traj.t >= 0.5 * t_hi) & (traj.t < t_hi)
    if not np.any(mask):
        mask = np.ones(traj.t.size, dtype=bool)
    body = eps_h[mask]
    plat = float(np.median(body))
    if np.isfinite(plat) and plat > 0 and peak > 2.0 * plat:
        body = body[body < 1.5 * np.median(body)]
        if body.size:
            plat = float(np.median(body))
    peak3 = float(eps_h3.max())
    plat3 = float(np.mean(eps_h3[mask])) if np.any(mask) else float("nan")
    slack_1 = bool(np.min([_eps(traj.t.size - 1, s) for s in near3[:1]]) <= 1e-8)
    slack_3 = bool(float(np.mean([_eps(traj.t.size - 1, s) for s in near3])) <= 1e-8)
    return dict(
        material=name, N=N, e1_frac=e1f, shock_visc=shock_visc,
        fail_avg_n_seg=fail_avg_n_seg, damping=damping,
        peak=peak, plateau=plat, peak_over_plat=peak / plat if plat else float("nan"),
        peak_over_eb=peak / mat.eps_b, plat_over_e1=plat / e1 if e1 else float("nan"),
        peak_avg3=peak3, plateau_avg3=plat3,
        peak_avg3_over_eb=peak3 / mat.eps_b,
        t_hub_fail=(t_fail if np.isfinite(t_fail) else None),
        T_near=T_n, T_near_avg3=T_n3, t=traj.t.tolist(), hub_eps=eps_h.tolist(),
        slack_1seg=slack_1, slack_3seg=slack_3,
        T0=r["T0_sim"], n_fail=int(traj.failures.shape[0]),
    )


def _over_row(rec):
    skip = {"T_near", "T_near_avg3", "t", "hub_eps"}
    return {k: rec[k] for k in rec if k not in skip}


def _over_series(rec):
    out = []
    for ti, eh, Tn in zip(rec.get("t") or [], rec.get("hub_eps") or [],
                          rec.get("T_near") or []):
        out.append(dict(
            material=rec["material"], e1_frac=rec["e1_frac"],
            shock_visc=rec["shock_visc"], fail_avg_n_seg=rec["fail_avg_n_seg"],
            t=ti, hub_eps=eh, T_near=Tn,
        ))
    return out


def item4b_overshoot(shard="local"):
    rows, series = [], []
    visc_grid = (0.0, 0.3, 0.6, 1.0)
    for name in ("S", "D"):
        for e1f in (0.4, 0.5):
            jobs = [dict(shock_visc=0.0, fail_avg_n_seg=0, tag="base"),
                    dict(shock_visc=0.0, fail_avg_n_seg=3, tag="avg3")]
            jobs.extend(dict(shock_visc=v, fail_avg_n_seg=0, tag=f"visc{v}")
                        for v in visc_grid[1:])
            for j in jobs:
                rec = _hub_hist(name, 8, e1f, shock_visc=j["shock_visc"],
                                fail_avg_n_seg=j["fail_avg_n_seg"])
                _save_run(shard, f"over_{j['tag']}_{name}_e{e1f}", rec)
                rows.append(_over_row(rec))
                series.extend(_over_series(rec))
    fields = ["material", "N", "e1_frac", "shock_visc", "fail_avg_n_seg",
              "damping", "peak", "plateau", "peak_over_plat", "peak_over_eb",
              "plat_over_e1", "peak_avg3", "plateau_avg3", "peak_avg3_over_eb",
              "slack_1seg", "slack_3seg", "t_hub_fail", "T0", "n_fail"]
    _write(OUT / "tab_fa_overshoot.csv", fields, rows, "round8b_4b_overshoot")
    _write(OUT / "tab_fa_overshoot_series.csv",
           ["material", "e1_frac", "shock_visc", "fail_avg_n_seg",
            "t", "hub_eps", "T_near"],
           series, "round8b_4b_overshoot_series")
    return rows


def item_ring_ns(shard="local"):
    """Criterion A at a/R in {0.7,0.8} for star+ring, n_s=20 (and S 0.8 n_s=40)."""
    jobs = []
    for mat in ("S", "D"):
        for ns in (20,):
            for a in (0.7, 0.8):
                jobs.append(dict(
                    net="star+ring", material=mat, M=1.0, v0=10.0, n_s=ns,
                    row_id=f"ringns_{mat}_ns{ns}", crit_key="A", mmin_crit="A",
                    R_max=None, s_cap=None, px=a, py=0.0,
                    point_idx=PTS_EXT.index((a, 0.0)),
                ))
    jobs.append(dict(
        net="star+ring", material="S", M=1.0, v0=10.0, n_s=40,
        row_id="ringns_S_ns40", crit_key="A", mmin_crit="A",
        R_max=None, s_cap=None, px=0.8, py=0.0, point_idx=5,
    ))
    n_workers = min(8, max(1, N_JOBS))
    print(f"[ring-ns] {len(jobs)} A-only jobs × {n_workers}", flush=True)
    res = _pool_point_jobs(jobs, n_workers)
    out = []
    for r in res:
        cfg = _mmin_cfg(r["material"], r["M"], r["v0"], net_kind=r["net"],
                        n_s=int(r["n_s"]))
        m1 = build_net(cfg).net_mass(cfg.material.resolve().rho, 1.0)
        out.append(dict(
            net=r["net"], material=r["material"], v0=r["v0"], n_s=r["n_s"],
            a_over_R=r["px"], mA_g=1e3 * float(r["s_min"]) * m1,
            s_min=r["s_min"], cpu_s=r["cpu_s"],
        ))
    _write(OUT / "tab_mmin_ring_ns.csv",
           ["net", "material", "v0", "n_s", "a_over_R", "mA_g", "s_min", "cpu_s"],
           out, "round8b_ring_ns")
    return out


def item3_moving_t2(shard="local"):
    """Rerun the 18 moving timeouts with t_end doubled (original t_end=0.25 s)."""
    src = OUT / "timeout_class.csv"
    if not src.is_file():
        src = R8 / "timeout_class.csv"
    jobs = []
    with open(src) as fh:
        rows = list(csv.DictReader((ln for ln in fh if not ln.startswith("#"))))
    for r in rows:
        if r.get("timeout_subclass") != "moving":
            continue
        lock = str(r.get("lock_xy", "False")).lower() in ("1", "true")
        jobs.append(dict(
            source=r["source"], material=r["material"],
            v0=float(r["v0"]), a=float(r["a"]), lock_xy=lock,
            mu_label=r["mu_label"], m_net=float(r["m_net"]),
            t_end=0.50, n_s=40, kind="star",
            run_id=r["run_id"] + "_t2",
            orig_t_end=0.25, orig_class=r.get("class"),
        ))
    print(f"[timeout-t2] {len(jobs)} moving cases, t_end 0.25 -> 0.50", flush=True)
    got = _pool_jobs(shard, jobs, _run_timeout_one, "timeout_moving_t2")
    by_id = {d["run_id"]: d for d in got}
    changed = []
    for j in jobs:
        d = by_id.get(j["run_id"], {})
        if d.get("class") != j.get("orig_class"):
            changed.append(dict(
                run_id=j["run_id"], old=j.get("orig_class"),
                new=d.get("class"), subclass=d.get("timeout_subclass"),
                material=j["material"], v0=j["v0"], a=j["a"],
                mu_label=j["mu_label"], lock_xy=j["lock_xy"],
            ))
    fields = ["source", "material", "v0", "a", "lock_xy", "mu_label",
              "outcome", "class", "timeout_subclass", "t_end", "t_end_setting",
              "speed", "run_id"]
    _write(OUT / "timeout_moving_t2.csv", fields, got, "round8b_item3_moving_t2")
    summary = dict(n=len(got), orig_t_end=0.25, new_t_end=0.50,
                   n_changed=len(changed), changed=changed)
    (OUT / "timeout_moving_t2_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    return got, summary


def run_local(items=None):
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"=== round8b local jobs={N_JOBS} commit={_COMMIT} "
          f"dirty={_code_dirty()} ===", flush=True)
    want = set(items or ("4b", "4a", "timeout"))
    if "overshoot" in want:
        item4b_overshoot("local")
    if "4b" in want:
        item4b("local")
    if "4a" in want:
        item4a_rewrite_and_diag("local")
    if "timeout" in want:
        item3_timeouts("local")
    if "moving" in want:
        item3_moving_t2("local")
    if "ringns" in want:
        item_ring_ns("local")
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
                    help="comma list of local items: 4b,4a,timeout,overshoot,moving,ringns")
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
