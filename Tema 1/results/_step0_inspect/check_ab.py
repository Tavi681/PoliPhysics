#!/usr/bin/env python3
"""Checks A (frame spacing) and B (a_x vs a_v) for Tema 1 ML."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from google.cloud import storage

BUCKET = "authorship-verification-poliphysics"
PREFIX = "tema1-stageb"
DT0 = 0.01
TOL = 1e-6
ROOT = Path(__file__).resolve().parents[2]
INSPECT = Path(__file__).resolve().parent


def gcs_blob(name: str, shard: int) -> str:
    return f"{PREFIX}/shard-{int(shard)}/ml_samples/{name}"


def download_h5(client: storage.Client, name: str, shard: int, dest: Path) -> Path:
    blob = client.bucket(BUCKET).blob(gcs_blob(name, shard))
    dest.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(dest))
    return dest


def analyze_t(t32: np.ndarray) -> dict:
    t = np.asarray(t32, dtype=np.float64)
    if len(t) < 2:
        return {
            "n_frames": len(t),
            "n_intervals": 0,
            "n_bad": 0,
            "bad_idx": [],
            "dt": np.array([], dtype=np.float64),
            "float32_only": True,
        }
    dt = np.diff(t)
    bad = np.where(np.abs(dt - DT0) > TOL)[0]
    # Is jitter only float32 representation?
    t_round = np.round(t / DT0) * DT0
    dt_round = np.diff(t_round)
    bad_round = np.where(np.abs(dt_round - DT0) > TOL)[0]
    # Compare t vs rounded grid in float64 after casting stored float32
    max_grid_err = float(np.max(np.abs(t - t_round))) if len(t) else 0.0
    return {
        "n_frames": int(len(t)),
        "n_intervals": int(len(dt)),
        "n_bad": int(len(bad)),
        "bad_idx": bad.tolist(),
        "dt": dt,
        "n_bad_after_round": int(len(bad_round)),
        "max_grid_err": max_grid_err,
        "float32_only": bool(len(bad_round) == 0 and len(bad) > 0),
        "t0": float(t[0]),
        "tend": float(t[-1]),
    }


def masked_windows(n_frames: int, bad_idx: list[int], C: int = 5) -> tuple[int, int]:
    """Training windows: history C velocities need frames [t-C .. t+1] inclusive
    for a_t from positions (needs t-1,t,t+1) and history velocities from C steps.
    Window index t in [C, n_frames-2] (inclusive) for position-based a.
    All dt_i in [t-C, t] (C intervals of history) AND dt at t-1 and t (for a)
    must be good — i.e. intervals with indices in [t-C, t] inclusive.
    """
    if n_frames < C + 2:
        return 0, 0
    bad_set = set(bad_idx)
    total = 0
    ok = 0
    for t in range(C, n_frames - 1):
        total += 1
        # intervals used: (t-C-1?); velocities v_{t-C+1}..v_t need positions
        # For history of C velocities at times t-C+1..t: intervals [t-C .. t-1]
        # For a_t from x: intervals [t-1, t] i.e. indices t-1 and t? 
        # a uses x_{t-1},x_t,x_{t+1} so intervals (t-1)->t and t->(t+1): indices t-1, t
        # Combined interval indices: [t-C, t] inclusive
        need = range(t - C, t + 1)
        if any(i in bad_set for i in need):
            continue
        ok += 1
    return total, ok


def check_a_one(path: Path, meta: dict, C: int = 5) -> dict:
    with h5py.File(path, "r") as f:
        t = f["t"][:]
    info = analyze_t(t)
    total, ok = masked_windows(info["n_frames"], info["bad_idx"], C=C)
    bad_details = []
    if info["n_bad"]:
        t64 = np.asarray(t, dtype=np.float64)
        for i in info["bad_idx"][:20]:
            bad_details.append({
                "interval_i": int(i),
                "t_i": float(t64[i]),
                "t_ip1": float(t64[i + 1]),
                "dt": float(t64[i + 1] - t64[i]),
                "is_last": bool(i == info["n_intervals"] - 1),
                "is_first": bool(i == 0),
            })
    return {
        **meta,
        "n_frames": info["n_frames"],
        "n_intervals": info["n_intervals"],
        "n_bad": info["n_bad"],
        "n_bad_after_round": info["n_bad_after_round"],
        "max_grid_err": info["max_grid_err"],
        "float32_only_jitter": info["float32_only"],
        "windows_total": total,
        "windows_ok": ok,
        "windows_masked": total - ok,
        "bad_details": bad_details,
        "dt_min": float(info["dt"].min()) if info["n_intervals"] else None,
        "dt_max": float(info["dt"].max()) if info["n_intervals"] else None,
        "dt_median": float(np.median(info["dt"])) if info["n_intervals"] else None,
    }


def node_types_arr(f) -> np.ndarray:
    raw = f["topology/node_type"][:]
    out = []
    for x in raw:
        if isinstance(x, bytes):
            out.append(x.decode())
        else:
            out.append(str(x))
    return np.asarray(out)


def check_b_one(path: Path, meta: dict) -> dict:
    with h5py.File(path, "r") as f:
        x = np.asarray(f["x"][:], dtype=np.float64)
        v = np.asarray(f["v"][:], dtype=np.float64)
        t = np.asarray(f["t"][:], dtype=np.float64)
        ntype = node_types_arr(f)
    dt = DT0
    # only regular interior frames: use indices 1..T-2 where both neighboring dts ok
    dti = np.diff(t)
    good_interval = np.abs(dti - dt) <= TOL
    # frame t usable for a if intervals t-1 and t are good
    T = len(t)
    usable = np.zeros(T, dtype=bool)
    for i in range(1, T - 1):
        if good_interval[i - 1] and good_interval[i]:
            usable[i] = True
    idx = np.where(usable)[0]
    if len(idx) < 10:
        # fall back to all interior with fixed dt
        idx = np.arange(1, T - 1)

    a_x = (x[idx + 1] - 2.0 * x[idx] + x[idx - 1]) / (dt ** 2)
    a_v = (v[idx + 1] - v[idx]) / dt
    diff = a_x - a_v

    # spider horizontal distance from release (frame 0 spider xy)
    spider0 = x[0, 0, :2]
    r_h = np.linalg.norm(x[idx, 0, :2] - spider0, axis=-1)
    t_u = t[idx]

    results = {"meta": meta, "n_usable": int(len(idx)), "by_type": {}}
    for nt in ["spider", "interior", "tip"]:
        mask = ntype == nt
        if not mask.any():
            continue
        ax = a_x[:, mask, :].reshape(-1)
        av = a_v[:, mask, :].reshape(-1)
        d = diff[:, mask, :].reshape(-1)
        std_av = float(np.std(av))
        rms_d = float(np.sqrt(np.mean(d ** 2)))
        results["by_type"][nt] = {
            "std_a_v": std_av,
            "rms_ax_minus_av": rms_d,
            "ratio_pct": 100.0 * rms_d / std_av if std_av > 0 else float("nan"),
            "n": int(ax.size),
        }

    # ratio vs time and r_h (global, all nodes)
    # bin by time quintiles and r_h quintiles
    def binned_ratio(values, bins=5):
        qs = np.linspace(0, 100, bins + 1)
        edges = np.percentile(values, qs)
        edges = np.unique(edges)
        out = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (values >= lo) & (values <= hi)
            if m.sum() < 5:
                continue
            d = diff[m].reshape(-1)
            av = a_v[m].reshape(-1)
            std = float(np.std(av))
            rms = float(np.sqrt(np.mean(d ** 2)))
            out.append({
                "lo": float(lo), "hi": float(hi),
                "std_av": std, "rms": rms,
                "ratio_pct": 100.0 * rms / std if std > 0 else float("nan"),
                "n": int(m.sum()),
            })
        return out

    results["ratio_vs_time"] = binned_ratio(t_u)
    results["ratio_vs_rh"] = binned_ratio(r_h)

    # drift: integrate from (x_t, v_t) with a_v for K steps: 
    # v_{k+1}=v_k+dt*a_k with a_k=(v_{k+1}-v_k)/dt is tautological for v;
    # position: x_{t+1}?= x_t + dt * v_{t+1}
    drifts = {}
    for K in [1, 10, 100]:
        errs = []
        # sample starting frames
        starts = idx[idx + K < T]
        starts = starts[:: max(1, len(starts) // 200)]
        for s in starts:
            # one-step identity check chain
            xx = x[s].copy()
            vv = v[s].copy()
            ok = True
            for k in range(K):
                i = s + k
                if i + 1 >= T:
                    ok = False
                    break
                # a from stored v
                aa = (v[i + 1] - v[i]) / dt
                vv = vv + dt * aa  # = v[i+1] if start matched
                xx = xx + dt * vv
            if not ok:
                continue
            # compare to stored x[s+K]
            errs.append(np.sqrt(np.mean((xx - x[s + K]) ** 2)))
        drifts[K] = {
            "rms_mean": float(np.mean(errs)) if errs else float("nan"),
            "rms_median": float(np.median(errs)) if errs else float("nan"),
            "rms_max": float(np.max(errs)) if errs else float("nan"),
            "n": len(errs),
        }
    # also one-step position drift: ||x_t + dt*v_{t+1} - x_{t+1}||
    one = x[idx] + dt * v[idx + 1] - x[idx + 1]
    results["one_step_x_drift_rms"] = float(np.sqrt(np.mean(one ** 2)))
    results["drifts"] = drifts
    return results


def _read_t_gcs(fs, name: str, shard: int) -> np.ndarray:
    path = f"{BUCKET}/{PREFIX}/shard-{int(shard)}/ml_samples/{name}"
    with fs.open(path, "rb") as fobj:
        with h5py.File(fobj, "r") as f:
            return f["t"][:]


def run_a(workers: int = 16, limit: int | None = None, C: int = 5) -> dict:
    import gcsfs

    runs = pd.read_csv(INSPECT / "valid_runs.csv")
    if limit:
        runs = runs.sample(limit, random_state=0)
    fs = gcsfs.GCSFileSystem()
    rows = []
    t0 = time.time()
    cache = INSPECT / "t_cache"
    cache.mkdir(exist_ok=True)

    def job(row):
        name = row["name"]
        shard = int(row["shard"])
        npy = cache / f"{name}.t.npy"
        meta = {
            "name": name, "N": int(row["N"]), "outcome": row["outcome"],
            "sigma_w": float(row["sigma_w"]), "x": int(row["x"]),
            "n_frames_csv": int(row["n_frames"]),
        }
        try:
            if npy.exists():
                t = np.load(npy)
            else:
                try:
                    t = _read_t_gcs(fs, name, shard)
                except Exception:
                    t = _read_t_gcs(fs, name, 1 - shard)
                    meta["shard_corrected"] = 1 - shard
                np.save(npy, t)
            return check_a_one_from_t(t, meta, C=C)
        except Exception as e:
            return {**meta, "error": str(e)}

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(job, r._asdict() if hasattr(r, "_asdict") else r.to_dict()): i
                for i, r in runs.iterrows()}
        for fut in as_completed(futs):
            rows.append(fut.result())
            done += 1
            if done % 100 == 0 or done == len(futs):
                print(f"A progress {done}/{len(futs)} elapsed={time.time()-t0:.0f}s", flush=True)

    # aggregate
    ok_rows = [r for r in rows if "error" not in r]
    err_rows = [r for r in rows if "error" in r]
    all_dt_bad = sum(r["n_bad"] for r in ok_rows)
    all_intervals = sum(r["n_intervals"] for r in ok_rows)
    # collect bad occurrence stats
    bad_last = 0
    bad_first = 0
    bad_other = 0
    by_outcome = {}
    by_N = {}
    for r in ok_rows:
        for d in r.get("bad_details", []):
            if d["is_last"]:
                bad_last += 1
            elif d["is_first"]:
                bad_first += 1
            else:
                bad_other += 1
        # only count from details truncated — recount from n_bad + is_last heuristic
    # Better recount: for each run, classify bad indices
    class_counts = {"last": 0, "first": 0, "other": 0}
    times_bad = []
    for r in ok_rows:
        # reload from cache for full classification if n_bad
        if r["n_bad"] == 0:
            continue
        npy = cache / f"{r['name']}.t.npy"
        t = np.load(npy).astype(np.float64)
        dt = np.diff(t)
        bad = np.where(np.abs(dt - DT0) > TOL)[0]
        for i in bad:
            times_bad.append(float(t[i]))
            if i == len(dt) - 1:
                class_counts["last"] += 1
            elif i == 0:
                class_counts["first"] += 1
            else:
                class_counts["other"] += 1
        key = r["outcome"]
        by_outcome.setdefault(key, {"runs_with_bad": 0, "n_bad": 0})
        by_outcome[key]["runs_with_bad"] += 1
        by_outcome[key]["n_bad"] += r["n_bad"]
        by_N.setdefault(r["N"], {"runs": 0, "windows_total": 0, "windows_ok": 0, "n_bad": 0, "intervals": 0})
        by_N[r["N"]]["runs"] += 1
        by_N[r["N"]]["windows_total"] += r["windows_total"]
        by_N[r["N"]]["windows_ok"] += r["windows_ok"]
        by_N[r["N"]]["n_bad"] += r["n_bad"]
        by_N[r["N"]]["intervals"] += r["n_intervals"]

    # fill by_N for runs with zero bad too
    for r in ok_rows:
        by_N.setdefault(r["N"], {"runs": 0, "windows_total": 0, "windows_ok": 0, "n_bad": 0, "intervals": 0})
        if r["n_bad"] == 0:
            by_N[r["N"]]["runs"] += 1
            by_N[r["N"]]["windows_total"] += r["windows_total"]
            by_N[r["N"]]["windows_ok"] += r["windows_ok"]
            by_N[r["N"]]["intervals"] += r["n_intervals"]

    # Fix double-counting: rebuild by_N cleanly
    by_N = {}
    for r in ok_rows:
        d = by_N.setdefault(r["N"], {"runs": 0, "windows_total": 0, "windows_ok": 0, "n_bad": 0, "intervals": 0})
        d["runs"] += 1
        d["windows_total"] += r["windows_total"]
        d["windows_ok"] += r["windows_ok"]
        d["n_bad"] += r["n_bad"]
        d["intervals"] += r["n_intervals"]

    for N, d in by_N.items():
        d["masked_frac"] = 1.0 - d["windows_ok"] / d["windows_total"] if d["windows_total"] else 0.0
        d["bad_interval_frac"] = d["n_bad"] / d["intervals"] if d["intervals"] else 0.0

    float32_only_runs = sum(1 for r in ok_rows if r.get("float32_only_jitter"))
    runs_with_bad = sum(1 for r in ok_rows if r["n_bad"] > 0)
    summary = {
        "n_runs_ok": len(ok_rows),
        "n_runs_err": len(err_rows),
        "errors": err_rows[:20],
        "total_intervals": all_intervals,
        "total_bad_intervals": all_dt_bad,
        "bad_interval_frac": all_dt_bad / all_intervals if all_intervals else 0.0,
        "runs_with_bad": runs_with_bad,
        "float32_only_jitter_runs": float32_only_runs,
        "bad_class_counts": class_counts,
        "by_outcome": by_outcome,
        "by_N": {str(k): v for k, v in sorted(by_N.items())},
        "bad_time_percentiles": {
            "p0": float(np.min(times_bad)) if times_bad else None,
            "p50": float(np.median(times_bad)) if times_bad else None,
            "p100": float(np.max(times_bad)) if times_bad else None,
        },
        "elapsed_s": time.time() - t0,
    }
    out = INSPECT / "check_a_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame([{k: v for k, v in r.items() if k != "bad_details"} for r in ok_rows]).to_csv(
        INSPECT / "check_a_per_run.csv", index=False
    )
    print(json.dumps(summary, indent=2))
    return summary


def check_a_one_from_t(t, meta, C=5):
    info = analyze_t(t)
    total, ok = masked_windows(info["n_frames"], info["bad_idx"], C=C)
    bad_details = []
    t64 = np.asarray(t, dtype=np.float64)
    for i in info["bad_idx"][:5]:
        bad_details.append({
            "interval_i": int(i),
            "t_i": float(t64[i]),
            "t_ip1": float(t64[i + 1]),
            "dt": float(t64[i + 1] - t64[i]),
            "is_last": bool(i == info["n_intervals"] - 1),
            "is_first": bool(i == 0),
        })
    return {
        **meta,
        "n_frames": info["n_frames"],
        "n_intervals": info["n_intervals"],
        "n_bad": info["n_bad"],
        "n_bad_after_round": info["n_bad_after_round"],
        "max_grid_err": info["max_grid_err"],
        "float32_only_jitter": info["float32_only"] and info["n_bad_after_round"] == 0,
        "windows_total": total,
        "windows_ok": ok,
        "windows_masked": total - ok,
        "bad_details": bad_details,
        "dt_min": float(info["dt"].min()) if info["n_intervals"] else None,
        "dt_max": float(info["dt"].max()) if info["n_intervals"] else None,
        "dt_median": float(np.median(info["dt"])) if info["n_intervals"] else None,
    }


def run_b(workers: int = 8) -> dict:
    picks = pd.read_csv(INSPECT / "b_runs.csv")
    picks["name"] = picks["hdf5_path"].map(lambda p: Path(p).name)
    valid = pd.read_csv(INSPECT / "valid_runs.csv")
    picks = picks.merge(valid[["name", "shard"]], on="name", how="left")
    client = storage.Client()
    dest_dir = INSPECT / "b_h5"
    dest_dir.mkdir(exist_ok=True)
    results = []

    def job(row):
        name = row["name"]
        shard = int(row["shard"])
        dest = dest_dir / name
        if not dest.exists():
            try:
                download_h5(client, name, shard, dest)
            except Exception:
                download_h5(client, name, 1 - shard, dest)
        meta = {
            "name": name, "N": int(row["N"]), "outcome": row["outcome"],
            "n_frames": int(row["n_frames"]),
        }
        return check_b_one(dest, meta)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(job, r) for _, r in picks.iterrows()]
        for fut in as_completed(futs):
            results.append(fut.result())
            print(f"B done {results[-1]['meta']['name']}", flush=True)

    # aggregate decision
    decision_rows = []
    for r in results:
        for nt, d in r["by_type"].items():
            decision_rows.append({
                "name": r["meta"]["name"], "N": r["meta"]["N"], "node_type": nt,
                **d,
                "one_step_x_drift_rms": r["one_step_x_drift_rms"],
                "drift_1": r["drifts"][1]["rms_mean"],
                "drift_10": r["drifts"][10]["rms_mean"],
                "drift_100": r["drifts"][100]["rms_mean"],
            })
    df = pd.DataFrame(decision_rows)
    # rule: if any (N, node_type) group has mean ratio > 1%
    grp = df.groupby(["N", "node_type"])["ratio_pct"].mean().reset_index()
    trigger = grp[grp["ratio_pct"] > 1.0]
    use_velocity_target = len(trigger) > 0
    summary = {
        "use_velocity_target": use_velocity_target,
        "group_mean_ratio_pct": grp.to_dict(orient="records"),
        "trigger_groups": trigger.to_dict(orient="records"),
        "per_run": decision_rows,
        "drift_summary": {
            "one_step_x_drift_rms_mean": float(df["one_step_x_drift_rms"].mean()),
            "drift_1_mean": float(df["drift_1"].mean()),
            "drift_10_mean": float(df["drift_10"].mean()),
            "drift_100_mean": float(df["drift_100"].mean()),
        },
    }
    with open(INSPECT / "check_b_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    df.to_csv(INSPECT / "check_b_per_run.csv", index=False)
    print(json.dumps({k: summary[k] for k in ["use_velocity_target", "group_mean_ratio_pct", "trigger_groups", "drift_summary"]}, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", action="store_true")
    ap.add_argument("--b", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--C", type=int, default=5)
    args = ap.parse_args()
    if args.a:
        run_a(workers=args.workers, limit=args.limit, C=args.C)
    if args.b:
        run_b(workers=min(args.workers, 8))


if __name__ == "__main__":
    main()
