"""Round 8c: checks only. No new physics, no new bisections, no retuning.

GCP: items 2 and 4 as a global (row, point, s) job list, cost-sorted, 4 VMs
× 16 workers. Item 3 uses existing caches/CSVs only. Item 6 applies the
fixed paper-status rules.

Usage:
  python -m validation.round8c --plan
  python -m validation.round8c --shard A|B|C|D
  python -m validation.round8c --merge
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
import sys
import time
from collections import defaultdict
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
from netsim.simulate import simulate_config, build_net
from netsim.integrator import PenetrationError

from validation.export_paper import _mmin_cfg
from validation.round8b import (
    TAB_MMIN_CASES, classify_timeout, _first_fail_seg,
)

OUT = Path(os.environ.get("ROUND8C_OUT", _ROOT / "paper_results" / "round8c"))
R8 = _ROOT / "paper_results" / "round8"
R8B = _ROOT / "paper_results" / "round8b"
_CACHE = _ROOT / ".cache" / "mmin"
_COMMIT = os.environ.get("GIT_COMMIT", "").strip() or git_commit()
_DATE = datetime.datetime.now().isoformat(timespec="seconds")
N_JOBS = max(1, int(os.environ.get("JOBS", "16")))
N_SHARDS = 4
SHARD_NAMES = ("A", "B", "C", "D")
PROGRESS_S = 30 * 60


def _code_dirty() -> str:
    return os.environ.get("CODE_DIRTY", "true")


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


def _run_dir(shard: str) -> Path:
    d = OUT / shard / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_run(shard: str, run_id: str, payload: dict):
    p = _run_dir(shard) / f"{run_id}.json"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, default=str))
    tmp.replace(p)


def _load_run(shard: str, run_id: str):
    p = _run_dir(shard) / f"{run_id}.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def expected_cpu(net, mat, n_s, t_end):
    """Wall-clock CPU seconds for one continued run (order-of-magnitude)."""
    te = float(t_end) / 0.25
    if net == "star" and mat == "D" and int(n_s) >= 40:
        return 2300.0 * te
    if net == "star" and mat == "S" and int(n_s) >= 40:
        return 400.0 * te
    if net == "star+ring" and mat == "D":
        return 900.0 * te
    return 220.0 * te


def _mstar_g(mat, M, v0):
    e_mat = float(get_material(mat).e_mat)
    ekin = 0.5 * float(M) * float(v0) ** 2
    return 1e3 * ekin / e_mat if e_mat > 0 else float("nan")


def _load_mmin_ext_recs():
    recs = []
    seen = set()
    for shard in SHARD_NAMES:
        rdir = R8B / shard / "runs"
        if not rdir.is_dir():
            continue
        for p in sorted(rdir.glob("mmin8b_*.json")):
            rec = json.loads(p.read_text())
            rid = rec.get("run_id") or p.stem
            if rid in seen:
                continue
            seen.add(rid)
            recs.append(rec)
    if not recs:
        merged = R8B / "tab_mmin_ext.csv"
        if merged.is_file():
            raise SystemExit("need round8b shard jsons under paper_results/round8b/{A,B,C,D}/runs")
    return recs


def build_jobs():
    """Unique (row, point, s) jobs for items 2 and 4, plus criteria tags."""
    recs = _load_mmin_ext_recs()
    pa_key = {}
    slow_key = {}
    for rec in recs:
        net, mat = rec["net"], rec["material"]
        M, v0, n_s = rec["M"], rec["v0"], rec["n_s"]
        for pr in rec.get("points") or []:
            a = float(pr["a_over_R"])
            s = float(pr["s_min"])
            crit = pr["criterion"]
            k = (net, mat, float(M), float(v0), int(n_s), a, round(s, 8))
            pa_key.setdefault(k, {
                "kind": "post_arrest",
                "net": net, "material": mat, "M": M, "v0": v0, "n_s": n_s,
                "a": a, "s": s, "m_g": float(pr["m_min_g"]),
                "criteria": [], "t_end": 0.25,
                "continue_after_arrest": True,
            })
            pa_key[k]["criteria"].append(crit)
            if pr.get("timeout_subclass") == "slow":
                s_lo = 0.99 * s
                ks = (net, mat, float(M), float(v0), int(n_s), a, round(s_lo, 8))
                slow_key.setdefault(ks, {
                    "kind": "slow_timeout",
                    "net": net, "material": mat, "M": M, "v0": v0, "n_s": n_s,
                    "a": a, "s": s_lo, "m_g": 0.99 * float(pr["m_min_g"]),
                    "criteria": [], "t_end": 0.5,
                    "continue_after_arrest": False,
                    "s_hi": s, "m_hi_g": float(pr["m_min_g"]),
                    "orig_outcome": pr.get("outcome_below", "timeout"),
                })
                slow_key[ks]["criteria"].append(crit)

    jobs = []
    for d in list(pa_key.values()) + list(slow_key.values()):
        d["criteria"] = sorted(set(d["criteria"]))
        d["expected_cpu"] = expected_cpu(d["net"], d["material"], d["n_s"],
                                         d["t_end"])
        tag = "pa" if d["kind"] == "post_arrest" else "to"
        d["run_id"] = (
            f"{tag}_{d['net']}_{d['material']}_v{d['v0']:g}_ns{int(d['n_s'])}"
            f"_a{d['a']:.2f}_s{d['s']:.6f}".replace("+", "p")
        )
        jobs.append(d)
    jobs.sort(key=lambda j: (-float(j["expected_cpu"]),
                             0 if j["net"] == "star" and j["material"] == "D"
                             and int(j["n_s"]) >= 40 else 1,
                             j["run_id"]))
    return jobs


def split_shards(jobs, n=N_SHARDS):
    bins = [[] for _ in range(n)]
    load = [0.0] * n
    for j in jobs:
        i = int(np.argmin(load))
        bins[i].append(j)
        load[i] += float(j["expected_cpu"])
    return bins, load


def plan_jobs():
    jobs = build_jobs()
    bins, load = split_shards(jobs)
    workers = N_JOBS
    plan = {
        "n_jobs": len(jobs),
        "n_post_arrest": sum(1 for j in jobs if j["kind"] == "post_arrest"),
        "n_slow_timeout": sum(1 for j in jobs if j["kind"] == "slow_timeout"),
        "workers_per_vm": workers,
        "n_vms": N_SHARDS,
        "cpu_s_total": float(sum(j["expected_cpu"] for j in jobs)),
        "per_shard": [],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plan").mkdir(exist_ok=True)
    for name, chunk, cpu in zip(SHARD_NAMES, bins, load):
        wall = cpu / max(workers, 1)
        rec = dict(shard=name, n_jobs=len(chunk), cpu_s=cpu,
                   wall_s_proj=wall, n_star_D40=sum(
                       1 for j in chunk if j["net"] == "star"
                       and j["material"] == "D" and int(j["n_s"]) >= 40))
        plan["per_shard"].append(rec)
        (OUT / "plan" / f"shard_{name}.json").write_text(
            json.dumps(chunk, indent=2, default=str))
        print(f"  shard {name}: {len(chunk)} jobs  "
              f"CPU~{cpu/3600:.2f} h  wall~{wall/3600:.2f} h "
              f"(16 workers)  starD40={rec['n_star_D40']}", flush=True)
    plan["wall_s_proj"] = max(load) / max(workers, 1)
    (OUT / "plan" / "plan.json").write_text(json.dumps(plan, indent=2))
    print(f"jobs={plan['n_jobs']} post_arrest={plan['n_post_arrest']} "
          f"slow={plan['n_slow_timeout']}  "
          f"proj wall={plan['wall_s_proj']/3600:.2f} h", flush=True)
    return jobs, bins, plan


def _cfg_for(job):
    cfg = _mmin_cfg(job["material"], job["M"], job["v0"],
                    net_kind=job["net"], n_s=int(job["n_s"]))
    s = float(job["s"])
    cfg.numerics.area_scale = s
    cfg.numerics.t_end = float(job["t_end"])
    cfg.numerics.dt_out = 5e-3
    cfg.numerics.store_mesh = True
    cfg.numerics.continue_after_arrest = bool(job.get("continue_after_arrest"))
    cfg.drone.p = (float(job["a"]), 0.0)
    cfg.output.hdf5 = None
    return cfg, s


def _arrest_frame(traj):
    t_arr = float(getattr(traj, "t_arrest", float("nan")))
    if np.isfinite(t_arr):
        return int(np.argmin(np.abs(traj.t - t_arr))), t_arr
    if traj.arrested:
        return int(traj.t.size - 1), float(traj.t[-1])
    return int(traj.t.size - 1), float(traj.t[-1])


def _ke_split(energy_row, vd):
    kd = float(energy_row[0])
    v2 = float(np.dot(vd, vd))
    if v2 <= 0 or kd <= 0:
        return 0.0, 0.0, kd
    kz = kd * (float(vd[2]) ** 2 / v2)
    return kd - kz, kz, kd


def _contacts_at(traj, disc, fr, r_d):
    if traj.x.ndim != 3 or traj.x.shape[1] == 0:
        return []
    xd = np.asarray(traj.drone[fr, :3], dtype=float)
    x = traj.x[fr]
    xa = x[disc.anchored]
    h = float(np.max(disc.seg_rest_length)) if disc.seg_rest_length.size else 0.0
    out = []
    d = np.linalg.norm(x - xd.reshape(1, 3), axis=1)
    for i in np.nonzero(d < r_d)[0]:
        if disc.anchored[int(i)]:
            da = 0.0
        elif xa.size:
            da = float(np.min(np.linalg.norm(x[int(i)] - xa, axis=1)))
        else:
            da = float("nan")
        out.append(dict(node=int(i), dist_drone=float(d[int(i)]),
                        dist_anchor=da, near_frame=int(da <= h + 1e-12)))
    return out


def _failed_by(traj, t_cut):
    if not traj.failures.size:
        return [], 0
    segs, parents = [], []
    for s, p, tf in traj.failures:
        if float(tf) <= t_cut + 1e-12:
            segs.append(int(s))
            parents.append(int(p))
    return sorted(set(parents)), len(segs)


def _summarise_run(job, res, disc, cpu_s):
    traj = res.trajectory
    i_arr, t_arr = _arrest_frame(traj)
    E = np.asarray(traj.energy, dtype=float)
    drone = np.asarray(traj.drone, dtype=float)
    vd_a = drone[i_arr, 3:6]
    xd_a = drone[i_arr, :3]
    kxy, kz, kd = _ke_split(E[i_arr], vd_a)
    ekin = 0.5 * float(job["M"]) * float(job["v0"]) ** 2
    u_el = float(E[i_arr, 2])
    u_fail = float(E[i_arr, 3])
    u_c = float(E[i_arr, 4])
    u_cap = float(E[i_arr, 5])
    k_net = float(E[i_arr, 1])
    absorbed = (u_el + u_fail) - float(traj.U_prestress)
    m_net = float(res.m_net)
    e_mat = float(res.material.e_mat)
    eta = absorbed / (m_net * e_mat) if (m_net * e_mat) > 0 else 0.0
    thr_a, nseg_a = _failed_by(traj, t_arr)
    thr_e, nseg_e = _failed_by(traj, float(traj.t[-1]))
    fail_after = [int(p) for s, p, tf in (traj.failures if traj.failures.size
                                          else np.zeros((0, 3)))
                  if float(tf) > t_arr + 1e-12]
    perf_after = (str(traj.outcome) == "perforated"
                  and np.isfinite(getattr(traj, "t_perforate", float("nan")))
                  and float(traj.t_perforate) > t_arr + 1e-12)
    if traj.arrested and i_arr < drone.shape[0] - 1:
        dxy = drone[i_arr:, :2] - xd_a[:2]
        max_dxy = float(np.max(np.hypot(dxy[:, 0], dxy[:, 1])))
    else:
        max_dxy = 0.0
    t_fc = float(getattr(traj, "t_frame_contact", float("nan")))
    frame_before = bool(np.isfinite(t_fc) and t_fc <= t_arr + 1e-12)
    contacts = _contacts_at(traj, disc, i_arr, float(disc.r_d
                          if hasattr(disc, "r_d") else 0.15))
    r_d = 0.15
    info = classify_timeout(res, disc, R=1.0, r_d=r_d)
    mstar = _mstar_g(job["material"], job["M"], job["v0"])
    return dict(
        job, outcome=str(traj.outcome), arrested=int(bool(res.arrested)),
        t_end=float(traj.t[-1]), t_arrest=t_arr,
        Kd_xy=kxy, Kd_z=kz, Kd=kd, Knet=k_net, Uel=u_el, Ufail=u_fail,
        Ucontact=u_c, Ucapture=u_cap, eta=eta,
        Kd_over_Ekin=kd / ekin if ekin else float("nan"),
        m_net_g=1e3 * m_net, mstar_g=mstar,
        below_mstar=int(1e3 * m_net < mstar - 1e-9),
        drone_x=float(xd_a[0]), drone_y=float(xd_a[1]), drone_z=float(xd_a[2]),
        vx=float(vd_a[0]), vy=float(vd_a[1]), vz=float(vd_a[2]),
        n_failed_threads_arrest=len(thr_a),
        n_failed_threads_end=len(thr_e),
        n_failed_seg_arrest=nseg_a, n_failed_seg_end=nseg_e,
        failed_threads_arrest=thr_a, failed_threads_end=list(sorted(set(
            int(p) for _s, p, _tf in (traj.failures if traj.failures.size
                                      else np.zeros((0, 3)))))),
        fail_after_arrest=int(bool(fail_after)),
        fail_after_parents=sorted(set(fail_after)),
        perforate_after_arrest=int(perf_after),
        t_perforate=float(getattr(traj, "t_perforate", float("nan"))),
        max_dxy_after_arrest=max_dxy,
        frame_contact=int(np.isfinite(t_fc)),
        t_frame_contact=t_fc,
        frame_contact_before_arrest=int(frame_before),
        contacts_at_arrest=contacts,
        timeout_subclass=info.get("timeout_subclass", ""),
        cpu_s=cpu_s,
        criteria=",".join(job.get("criteria") or []),
    )


def run_one(job):
    shard, run_id = job["shard"], job["run_id"]
    cached = _load_run(shard, run_id)
    if cached is not None:
        return cached
    cfg, s = _cfg_for(job)
    material = cfg.material.resolve()
    net = build_net(cfg)
    disc = discretize(net, material, int(job["n_s"]), area_scale=s,
                      r_d=cfg.drone.r_d)
    t0 = time.time()
    try:
        res = simulate_config(cfg, write=False)
    except PenetrationError as e:
        payload = dict(job, outcome="penetration_error", error=str(e),
                       cpu_s=time.time() - t0)
        _save_run(shard, run_id, payload)
        return payload
    row = _summarise_run(job, res, disc, time.time() - t0)
    _save_run(shard, run_id, row)
    print(f"  {run_id} out={row['outcome']} t_arr={row['t_arrest']:.4g} "
          f"cpu={row['cpu_s']:.0f}s", flush=True)
    return row


def _progress_loop(t0, total, get_done, get_cpu_left, stop):
    last = 0.0
    while not stop[0]:
        time.sleep(5)
        now = time.time()
        if now - last < PROGRESS_S and get_done() < total:
            continue
        last = now
        done = get_done()
        elapsed = now - t0
        left = get_cpu_left()
        wall_left = left / max(N_JOBS, 1)
        print(f"[progress] {done}/{total} elapsed={elapsed/3600:.2f}h "
              f"remain~{wall_left/3600:.2f}h cpu_left~{left/3600:.2f}h",
              flush=True)


def run_shard(shard: str):
    jobs, bins, plan = plan_jobs()
    idx = SHARD_NAMES.index(shard)
    chunk = bins[idx]
    for j in chunk:
        j["shard"] = shard
    print(f"=== round8c shard={shard} jobs={len(chunk)} workers={N_JOBS} "
          f"commit={_COMMIT} dirty={_code_dirty()} ===", flush=True)
    print(f"projected shard wall {plan['per_shard'][idx]['wall_s_proj']/3600:.2f} h",
          flush=True)
    t0 = time.time()
    pending, done_rows = [], []
    cpu_left = 0.0
    for j in chunk:
        cached = _load_run(shard, j["run_id"])
        if cached is not None:
            done_rows.append(cached)
        else:
            pending.append(j)
            cpu_left += float(j["expected_cpu"])
    state = dict(done=len(done_rows), cpu_left=cpu_left)
    print(f"[{shard}] resume {state['done']}/{len(chunk)} "
          f"pending={len(pending)}", flush=True)
    stop = [False]

    def get_done():
        return state["done"]

    def get_cpu_left():
        return state["cpu_left"]

    import threading
    th = threading.Thread(target=_progress_loop,
                          args=(t0, len(chunk), get_done, get_cpu_left, stop),
                          daemon=True)
    th.start()
    n = min(N_JOBS, max(1, len(pending)))
    if not pending:
        stop[0] = True
        return done_rows
    if n <= 1:
        for j in pending:
            done_rows.append(run_one(j))
            state["done"] += 1
            state["cpu_left"] = max(0.0, state["cpu_left"] - float(
                j["expected_cpu"]))
    else:
        with ProcessPoolExecutor(max_workers=n) as pool:
            futs = {pool.submit(run_one, j): j for j in pending}
            for fut in as_completed(futs):
                j = futs[fut]
                done_rows.append(fut.result())
                state["done"] += 1
                state["cpu_left"] = max(0.0, state["cpu_left"] - float(
                    j["expected_cpu"]))
                if state["done"] % 5 == 0 or state["done"] == len(chunk):
                    elapsed = time.time() - t0
                    wall_left = state["cpu_left"] / max(n, 1)
                    print(f"[{shard}] {state['done']}/{len(chunk)} "
                          f"elapsed={elapsed:.0f}s remain~{wall_left:.0f}s",
                          flush=True)
    stop[0] = True
    print(f"=== shard {shard} done in {time.time()-t0:.0f}s ===", flush=True)
    return done_rows


def _read_csv(path: Path):
    if not path.is_file():
        return []
    with open(path) as fh:
        return list(csv.DictReader((ln for ln in fh if not ln.startswith("#"))))


def _cache_flags():
    """Item 3 from 7c/8b caches: only m_net and arrested; no Kd stored."""
    note = ("mmin cache stores arrested/outcome/n_failures/R_d/energy_error/"
            "fail_segs only — not Kd, eta, or the energy partition. "
            "eta and Kd/Ekin are taken from mass_scan.csv, lambda_map.csv, "
            "and the item-2 threshold runs.")
    flags = []
    if not _CACHE.is_dir():
        return flags, note, dict(n_h5=0, has_Kd=False)
    import h5py
    n = 0
    sample_keys = []
    for d in sorted(_CACHE.iterdir()):
        if not d.is_dir() or not d.name.startswith("star"):
            continue
        # star_S_M1_v20_ns40  or tagged _A / _B_any
        parts = d.name.split("_")
        try:
            # kind may be star or star+ring encoded as star+ring
            kind = "star+ring" if "ring" in d.name else "star"
            # find material token
            mat = "S" if "_S_" in f"_{d.name}_" else ("D" if "_D_" in f"_{d.name}_" else "")
            if mat == "":
                continue
            # M and v0
            M = v0 = n_s = None
            for tok in parts:
                if tok.startswith("M") and tok[1:2].isdigit():
                    M = float(tok[1:])
                elif tok.startswith("v") and tok[1:2].isdigit():
                    v0 = float(tok[1:])
                elif tok.startswith("ns") and tok[2:].isdigit():
                    n_s = int(tok[2:])
            if None in (M, v0, n_s):
                continue
        except Exception:
            continue
        m1 = None
        mstar = _mstar_g(mat, M, v0)
        for p in d.glob("mmin_p*.h5"):
            n += 1
            try:
                with h5py.File(p, "r") as h:
                    a = h["eval"].attrs
                    if not sample_keys:
                        sample_keys = list(a.keys())
                    arrested = bool(a["arrested"])
                    outcome = str(a["outcome"]) if "outcome" in a else ""
                stem = p.stem  # mmin_p0_s0.123456
                s = float(stem.split("_s")[-1])
            except Exception:
                continue
            if m1 is None:
                cfg = _mmin_cfg(mat, M, v0, net_kind=kind, n_s=n_s)
                m1 = build_net(cfg).net_mass(cfg.material.resolve().rho, 1.0)
            m_g = 1e3 * s * m1
            if arrested and m_g < mstar:
                flags.append(dict(
                    source="cache", net=kind, material=mat, M=M, v0=v0,
                    n_s=n_s, s=s, m_net_g=m_g, mstar_g=mstar,
                    eta="", Kd_over_Ekin="", outcome=outcome,
                    reason="m_net<mstar", cache_file=str(p.relative_to(_CACHE)),
                ))
    meta = dict(n_h5=n, has_Kd=False, sample_attrs=sample_keys)
    return flags, note, meta


def item3_energy_flags(post_rows):
    flags, note, meta = _cache_flags()
    for src, path, mcol, ecol, kcol, arrcol in (
        ("mass_scan", R8 / "mass_scan.csv", "m_net_g", "eta_at_arrest",
         "Kd_before", "arrested"),
        ("lambda_map", R8 / "lambda_map.csv", "m_net_g", "eta_at_arrest",
         "Kd_before", None),
        ("mass_scan", R8B / "mass_scan.csv", "m_net_g", "eta_at_arrest",
         "Kd_before", "arrested"),
        ("lambda_map", R8B / "lambda_map.csv", "m_net_g", "eta_at_arrest",
         "Kd_before", None),
    ):
        for r in _read_csv(path):
            try:
                m_g = float(r.get(mcol) or "nan")
                mat = r["material"]
                v0 = float(r.get("v0") or 20.0)
            except (KeyError, ValueError):
                continue
            mstar = _mstar_g(mat, 1.0, v0)
            eta = r.get(ecol)
            try:
                eta_f = float(eta) if eta not in ("", None, "nan") else float("nan")
            except ValueError:
                eta_f = float("nan")
            kd = r.get(kcol)
            try:
                kd_f = float(kd) if kd not in ("", None, "nan") else float("nan")
            except ValueError:
                kd_f = float("nan")
            ekin = 0.5 * v0 ** 2
            kd_frac = kd_f / ekin if np.isfinite(kd_f) and ekin else float("nan")
            arrested = True
            if arrcol and r.get(arrcol) not in ("1", "true", "True"):
                if (r.get("class") or r.get("class_") or r.get("outcome")) not in (
                        "arrested", "intact"):
                    arrested = False
            reasons = []
            if arrested and m_g < mstar:
                reasons.append("m_net<mstar")
            if np.isfinite(eta_f) and eta_f > 1.0:
                reasons.append("eta>1")
            if np.isfinite(kd_frac) and kd_frac > 0.5:
                reasons.append("Kd/Ekin>0.5")
            if not reasons:
                continue
            flags.append(dict(
                source=src, net="star", material=mat, v0=v0,
                a_over_R=r.get("a_over_R", ""),
                m_net_g=m_g, mstar_g=mstar, eta=eta_f,
                Kd_over_Ekin=kd_frac, reason=";".join(reasons),
            ))
    for r in post_rows:
        if r.get("kind") != "post_arrest":
            continue
        reasons = []
        if r.get("arrested") and r.get("below_mstar"):
            reasons.append("m_net<mstar")
        eta = r.get("eta")
        try:
            if eta not in ("", None) and float(eta) > 1.0:
                reasons.append("eta>1")
        except (TypeError, ValueError):
            pass
        try:
            if float(r.get("Kd_over_Ekin") or 0) > 0.5:
                reasons.append("Kd/Ekin>0.5")
        except (TypeError, ValueError):
            pass
        if not reasons:
            continue
        flags.append(dict(
            source="post_arrest", net=r.get("net"), material=r.get("material"),
            v0=r.get("v0"), a_over_R=r.get("a"),
            m_net_g=r.get("m_net_g"), mstar_g=r.get("mstar_g"),
            eta=r.get("eta"), Kd_over_Ekin=r.get("Kd_over_Ekin"),
            reason=";".join(reasons), criterion=r.get("criteria"),
        ))
    fields = ["source", "net", "material", "v0", "a_over_R", "criterion",
              "m_net_g", "mstar_g", "eta", "Kd_over_Ekin", "reason"]
    _write(OUT / "energy_flags.csv", fields, flags, "round8c_item3")
    counts = defaultdict(int)
    for f in flags:
        counts[(f.get("net"), f.get("material"), f.get("a_over_R", ""))] += 1
    (OUT / "energy_flags_meta.json").write_text(json.dumps(
        dict(note=note, cache=meta, n_flags=len(flags),
             counts={str(k): v for k, v in counts.items()}), indent=2))
    return flags, note, meta


def _expand_post(rows):
    """One CSV line per criterion sharing a threshold run."""
    out = []
    for r in rows:
        if r.get("kind") != "post_arrest":
            continue
        crits = [c for c in str(r.get("criteria") or "").split(",") if c]
        if not crits:
            crits = [""]
        for c in crits:
            d = dict(r)
            d["criterion"] = c
            out.append(d)
    return out


def item6_paper_status(post_rows, slow_rows, flags):
    lines = []

    def add(**kw):
        lines.append(dict(
            table_or_figure=kw.get("table_or_figure", ""),
            net=kw.get("net", ""), material=kw.get("material", ""),
            v0=kw.get("v0", ""), criterion=kw.get("criterion", ""),
            a_over_R=kw.get("a_over_R", ""), value=kw.get("value", ""),
            status=kw.get("status", "keep"), reason=kw.get("reason", ""),
        ))

    post_by = {}
    for r in post_rows:
        if r.get("kind") != "post_arrest":
            continue
        for c in str(r.get("criteria") or "").split(","):
            if not c:
                continue
            post_by[(r["net"], r["material"], float(r["v0"]), c,
                     float(r["a"]))] = r

    slow_by = {}
    for r in slow_rows:
        for c in str(r.get("criteria") or "").split(","):
            if not c:
                continue
            slow_by[(r.get("net"), r.get("material"), float(r.get("v0", 0)),
                     c, float(r.get("a", 0)))] = r

    # tab_mmin_ext
    for rec in _load_mmin_ext_recs():
        for pr in rec["points"]:
            key = (rec["net"], rec["material"], float(rec["v0"]),
                   pr["criterion"], float(pr["a_over_R"]))
            run = post_by.get(key, {})
            status, reason = "keep", ""
            mstar = _mstar_g(rec["material"], rec["M"], rec["v0"])
            m = float(pr["m_min_g"])
            eta = run.get("eta")
            try:
                eta_f = float(eta) if eta not in ("", None) else float("nan")
            except (TypeError, ValueError):
                eta_f = float("nan")
            if m < mstar or (np.isfinite(eta_f) and eta_f > 1.0):
                status, reason = "exclude", "energy bound (rule 1)"
            else:
                a = float(pr["a_over_R"])
                fc = int(run.get("frame_contact_before_arrest") or 0)
                if a >= 0.6 - 1e-12 and fc:
                    if pr["criterion"] == "A":
                        status, reason = "keep_with_note", "frame contact (rule 2, A)"
                    elif pr["criterion"].startswith("B"):
                        status, reason = "exclude", "frame contact (rule 2, B)"
                elif int(run.get("fail_after_arrest") or 0) or int(
                        run.get("perforate_after_arrest") or 0):
                    tfa = ""
                    parents = run.get("fail_after_parents") or []
                    tfa = f"fails after first rebound at t = {run.get('t_arrest')}"
                    status, reason = "keep_with_note", tfa
                sl = slow_by.get(key)
                if sl is not None and status != "exclude":
                    new_out = sl.get("outcome")
                    if new_out == "arrested" and sl.get("orig_outcome") == "timeout":
                        corr = sl.get("m_g")
                        status, reason = "keep_with_note", (
                            f"slow timeout at 0.5 s arrested; "
                            f"corrected m_min={corr} g from 1% bracket")
                    elif new_out == "timeout":
                        if status == "keep":
                            reason = "slow timeout still timeout at 0.5 s"
            add(table_or_figure="tab_mmin_ext", net=rec["net"],
                material=rec["material"], v0=rec["v0"],
                criterion=pr["criterion"], a_over_R=pr["a_over_R"],
                value=pr["m_min_g"], status=status, reason=reason)

    # tab_fa_num (8b, no new runs)
    fa = _read_csv(R8B / "tab_fa_num.csv") or _read_csv(R8 / "tab_fa_num.csv")
    for r in fa:
        kind = r.get("kind", "")
        mode = r.get("mode", "")
        mat = r.get("material", "")
        try:
            N = int(float(r.get("N") or 0))
        except ValueError:
            N = 0
        try:
            e1 = float(r.get("e1_frac") or "nan")
        except ValueError:
            e1 = float("nan")
        keep = False
        if kind == "amp" and np.isfinite(e1) and e1 <= 0.3 + 1e-12:
            keep = True
        if kind == "bisect" and mode == "break" and mat == "S" and N >= 8:
            keep = True
        status = "keep" if keep else "exclude"
        reason = ("rule 5 keep" if keep
                  else "rule 5 exclude (D break / slack / N<8 / amp e1>=0.4)")
        val = r.get("eps1_frac") or r.get("dT0_Tinc") or r.get("e1_frac")
        add(table_or_figure="tab_fa_num", net="star", material=mat, v0="",
            criterion=f"{kind}:{mode}:N{N}", a_over_R="", value=val,
            status=status, reason=reason)

    # localization map: rules 1 and 2
    for r in _read_csv(R8B / "lambda_map.csv") or _read_csv(R8 / "lambda_map.csv"):
        mat = r["material"]
        a = float(r["a_over_R"])
        m_g = float(r["m_net_g"])
        mstar = _mstar_g(mat, 1.0, 20.0)
        eta = r.get("eta_at_arrest")
        try:
            eta_f = float(eta) if eta not in ("", None, "nan") else float("nan")
        except ValueError:
            eta_f = float("nan")
        cls = r.get("class") or r.get("class_")
        status, reason = "keep", "rule 6"
        if cls in ("intact", "arrested") or r.get("arrested") in ("1", "true"):
            if m_g < mstar or (np.isfinite(eta_f) and eta_f > 1.0):
                status, reason = "exclude", "energy bound (rule 1)"
        keyA = ("star", mat, 20.0, "A", a)
        keyB = ("star", mat, 20.0, "Bany", a)
        run = post_by.get(keyB) or post_by.get(keyA)
        if status != "exclude" and a >= 0.6 - 1e-12 and run and int(
                run.get("frame_contact_before_arrest") or 0):
            status, reason = "keep_with_note", "frame contact on nearby threshold (rule 2)"
        add(table_or_figure="lambda_map", net="star", material=mat, v0=20,
            criterion=cls, a_over_R=a, value=r.get("m_over_mstar"),
            status=status, reason=reason)

    keep_tables = [
        ("mass_scan", R8 / "mass_scan.csv", "mass_scan"),
        ("mmin_lockxy", R8 / "mmin_lockxy.csv", "mmin_lockxy"),
        ("fig_etaa", R8 / "fig_etaa.csv", "fig_etaa"),
        ("tab_junction_nl_num", R8 / "tab_junction_nl_num.csv", "tab_junction_nl"),
        ("tab_mmin_ring_ns", R8B / "tab_mmin_ring_ns.csv", "tab_mmin_ring_ns"),
        ("mmin_frictionless", R8 / "mmin_frictionless.csv", "mmin_frictionless"),
        ("tab_conv", R8 / "tab_conv.csv", "tab_conv"),
        ("params_used", R8 / "params_used.csv", "params_used"),
        ("tab_mmin", _ROOT / "paper_results" / "tab_mmin.csv", "tab_mmin"),
    ]
    for table, path, fig in keep_tables:
        if not path.is_file():
            continue
        rows = _read_csv(path)
        if not rows:
            add(table_or_figure=fig, status="keep", reason="rule 6 (empty)")
            continue
        for r in rows:
            add(table_or_figure=fig,
                net=r.get("net", "star"),
                material=r.get("material", ""),
                v0=r.get("v0", ""),
                criterion=r.get("criterion", r.get("mu_label", "")),
                a_over_R=r.get("a_over_R", r.get("a", "")),
                value=r.get("m_min_g") or r.get("m_net_g") or r.get("eta") or "",
                status="keep", reason="rule 6")

    fields = ["table_or_figure", "net", "material", "v0", "criterion",
              "a_over_R", "value", "status", "reason"]
    _write(OUT / "paper_status.csv", fields, lines, "round8c_item6")
    counts = defaultdict(lambda: defaultdict(int))
    for ln in lines:
        counts[ln["table_or_figure"]][ln["status"]] += 1
    (OUT / "paper_status_counts.json").write_text(json.dumps(counts, indent=2))
    return lines, counts


def item1_dump(post_rows):
    """Star D, v0=10, a=0.8R, m=0.436 g (Bany = Bloc075)."""
    hits = [r for r in post_rows
            if r.get("net") == "star" and r.get("material") == "D"
            and float(r.get("v0", 0)) == 10.0
            and abs(float(r.get("a", -1)) - 0.8) < 1e-9
            and float(r.get("m_g") or r.get("m_net_g") or 0) < 0.5]
    (OUT / "item1_impossible.json").write_text(
        json.dumps(hits, indent=2, default=str))
    return hits


def merge_shards():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for shard in SHARD_NAMES:
        rdir = OUT / shard / "runs"
        if not rdir.is_dir():
            print(f"merge: missing {shard}", flush=True)
            continue
        for p in sorted(rdir.glob("*.json")):
            rows.append(json.loads(p.read_text()))
    post = [r for r in rows if r.get("kind") == "post_arrest"]
    slow = [r for r in rows if r.get("kind") == "slow_timeout"]
    post_lines = _expand_post(post)
    fields_pa = [
        "net", "material", "v0", "n_s", "criterion", "a_over_R", "s", "m_g",
        "outcome", "arrested", "t_arrest", "t_end",
        "Kd_xy", "Kd_z", "Kd", "Knet", "Uel", "Ufail", "Ucontact", "Ucapture",
        "eta", "Kd_over_Ekin", "n_failed_threads_arrest", "n_failed_threads_end",
        "fail_after_arrest", "fail_after_parents", "perforate_after_arrest",
        "t_perforate", "max_dxy_after_arrest",
        "frame_contact", "t_frame_contact", "frame_contact_before_arrest",
        "timeout_subclass", "run_id", "cpu_s",
    ]
    for r in post_lines:
        r["a_over_R"] = r.get("a")
    _write(OUT / "post_arrest.csv", fields_pa, post_lines, "round8c_item2")
    fields_to = fields_pa + ["s_hi", "m_hi_g", "orig_outcome"]
    for r in slow:
        r["a_over_R"] = r.get("a")
        r["criterion"] = r.get("criteria")
    _write(OUT / "timeout_slow_t05.csv", fields_to, slow, "round8c_item4")

    # per-criterion summary of post-arrest status change
    n_by = defaultdict(lambda: dict(n=0, fail_after=0, perf_after=0))
    for r in post_lines:
        c = r.get("criterion") or "?"
        n_by[c]["n"] += 1
        if int(r.get("fail_after_arrest") or 0):
            n_by[c]["fail_after"] += 1
        if int(r.get("perforate_after_arrest") or 0):
            n_by[c]["perf_after"] += 1
    (OUT / "post_arrest_summary.json").write_text(json.dumps(n_by, indent=2))

    flags, note, meta = item3_energy_flags(rows)
    hits = item1_dump(post)
    lines, counts = item6_paper_status(rows, slow, flags)
    report = dict(
        n_post=len(post), n_slow=len(slow), n_flags=len(flags),
        cache_note=note, cache_meta=meta, item1_n=len(hits),
        paper_status_counts=counts, post_arrest_by_crit=n_by,
        commit=_COMMIT,
    )
    (OUT / "merge_report.json").write_text(json.dumps(
        report, indent=2, default=str))
    print("merge", {k: report[k] for k in
                    ("n_post", "n_slow", "n_flags", "item1_n")}, flush=True)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--shard", default="")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.plan:
        plan_jobs()
        return
    if args.merge:
        merge_shards()
        return
    if args.shard in SHARD_NAMES:
        run_shard(args.shard)
        return
    raise SystemExit("need --plan, --shard A|B|C|D, or --merge")


if __name__ == "__main__":
    main()
