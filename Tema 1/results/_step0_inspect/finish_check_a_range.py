#!/usr/bin/env python3
"""Check A: read ONLY /t from GCS via gcsfs+h5py byte-range reads (no full HDF5 download)."""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

DT = 0.01
C = 5
BUCKET = "authorship-verification-poliphysics"
PREFIX = "tema1-stageb"
INSPECT = Path(__file__).resolve().parent
CACHE = INSPECT / "t_cache"
CACHE.mkdir(exist_ok=True)

# Per-process byte counters (aggregated in parent via return values).
_fs = None
_bytes_transferred = 0
_n_reads = 0


def _init_worker():
    """Each process gets its own GCSFileSystem (gcsfs is not safely shared)."""
    global _fs, _bytes_transferred, _n_reads
    import gcsfs

    _bytes_transferred = 0
    _n_reads = 0
    _fs = gcsfs.GCSFileSystem()


def _track_file(raw):
    global _bytes_transferred, _n_reads
    orig = raw.read

    def tracked(size=-1):
        global _bytes_transferred, _n_reads
        data = orig(size)
        _bytes_transferred += len(data)
        _n_reads += 1
        return data

    raw.read = tracked
    return raw


def read_t(name: str, shard: int) -> np.ndarray:
    path = f"{BUCKET}/{PREFIX}/shard-{int(shard)}/ml_samples/{name}"
    # cache_type='none' → true range GETs; default AdaptiveReadahead is 5 MiB and
    # would pull nearly the whole file for typical Stage B HDF5s.
    with _fs.open(path, "rb", cache_type="none") as raw:
        _track_file(raw)
        with h5py.File(raw, "r") as f:
            return f["t"][:]


def job(row: dict) -> dict:
    """Fetch /t for one run; returns bytes/read counts for aggregation."""
    global _bytes_transferred, _n_reads
    name = row["name"]
    shard = int(row["shard"])
    b0, r0 = _bytes_transferred, _n_reads
    try:
        try:
            t = read_t(name, shard)
        except Exception:
            t = read_t(name, 1 - shard)
        np.save(CACHE / f"{name}.t.npy", t)
        return {
            "name": name,
            "ok": True,
            "bytes": _bytes_transferred - b0,
            "reads": _n_reads - r0,
            "error": None,
        }
    except Exception as e:
        return {
            "name": name,
            "ok": False,
            "bytes": _bytes_transferred - b0,
            "reads": _n_reads - r0,
            "error": f"{name}: {e}",
        }


def analyze(name: str, N: int, outcome: str, t: np.ndarray) -> dict:
    t64 = np.asarray(t, dtype=np.float64)
    n_int = max(len(t64) - 1, 0)
    if n_int == 0:
        return {
            "name": name, "N": N, "outcome": outcome,
            "n_frames": len(t64), "n_intervals": 0,
            "n_jump": 0, "windows_total": 0, "windows_ok": 0,
        }
    k = np.round(t64 / DT).astype(np.int64)
    jumps = np.where(np.diff(k) != 1)[0]
    T = len(t64)
    bad = set(jumps.tolist())
    tot = ok = 0
    for ti in range(C, T - 1):
        tot += 1
        if any(i in bad for i in range(ti - C, ti + 1)):
            continue
        ok += 1
    return {
        "name": name, "N": int(N), "outcome": str(outcome),
        "n_frames": int(T), "n_intervals": int(n_int),
        "n_jump": int(len(jumps)),
        "windows_total": tot, "windows_ok": ok,
    }


def main():
    runs = pd.read_csv(INSPECT / "valid_runs.csv")
    missing = [
        {
            "name": r["name"],
            "shard": int(r["shard"]),
        }
        for _, r in runs.iterrows()
        if not (CACHE / f"{r['name']}.t.npy").exists()
    ]
    n_workers = min(32, max(8, (os.cpu_count() or 8)))
    print(
        f"total={len(runs)} cached={len(runs)-len(missing)} missing={len(missing)} "
        f"mode=gcsfs+h5py cache_type=none ProcessPool workers={n_workers}",
        flush=True,
    )

    t0 = time.time()
    done = 0
    n_ok = 0
    n_err = 0
    total_bytes = 0
    total_reads = 0
    errors: list[str] = []

    if missing:
        # spawn (not fork): clean interpreter + own gcsfs per worker
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=n_workers, mp_context=ctx, initializer=_init_worker
        ) as ex:
            futs = [ex.submit(job, r) for r in missing]
            for fut in as_completed(futs):
                res = fut.result()
                done += 1
                total_bytes += int(res["bytes"])
                total_reads += int(res["reads"])
                if res["ok"]:
                    n_ok += 1
                else:
                    n_err += 1
                    if len(errors) < 20:
                        errors.append(res["error"])
                if done % 100 == 0 or done == len(futs):
                    print(
                        f"fetch {done}/{len(futs)} ok={n_ok} err={n_err} "
                        f"bytes={total_bytes} ({total_bytes/1e6:.2f} MB) "
                        f"elapsed={time.time()-t0:.0f}s",
                        flush=True,
                    )

    # Aggregate all cached
    rows = []
    for _, r in runs.iterrows():
        npy = CACHE / f"{r['name']}.t.npy"
        if not npy.exists():
            continue
        t = np.load(npy)
        rows.append(analyze(r["name"], int(r["N"]), r["outcome"], t))
    df = pd.DataFrame(rows)
    df.to_csv(INSPECT / "check_a_final_per_run.csv", index=False)

    by_N = {}
    for N, g in df.groupby("N"):
        by_N[str(int(N))] = {
            "runs": int(len(g)),
            "n_jumps": int(g["n_jump"].sum()),
            "runs_with_jump": int((g["n_jump"] > 0).sum()),
            "windows_total": int(g["windows_total"].sum()),
            "windows_ok": int(g["windows_ok"].sum()),
            "masked_window_frac": float(
                1.0 - g["windows_ok"].sum() / g["windows_total"].sum()
                if g["windows_total"].sum() else 0.0
            ),
        }

    summary = {
        "n_runs": int(len(df)),
        "n_runs_expected": int(len(runs)),
        "total_jumps": int(df["n_jump"].sum()) if len(df) else 0,
        "runs_with_jump": int((df["n_jump"] > 0).sum()) if len(df) else 0,
        "masked_window_frac_overall": float(
            1.0 - df["windows_ok"].sum() / df["windows_total"].sum()
            if len(df) and df["windows_total"].sum() else 0.0
        ),
        "by_N": by_N,
        "mask_mode": "frame_index_jump (round(t/0.01) not consecutive)",
        "dt_nominal": DT,
        "C": C,
        "gcs_bytes_transferred_this_run": int(total_bytes),
        "gcs_read_ops_this_run": int(total_reads),
        "gcs_files_fetched_this_run": int(n_ok),
        "gcs_files_errors": int(n_err),
        "errors_sample": errors[:20],
        "method": "gcsfs.open(cache_type='none') + h5py File → only /t dataset (ProcessPool)",
        "elapsed_s": time.time() - t0,
    }
    (INSPECT / "check_a_final.json").write_text(json.dumps(summary, indent=2))
    out_ml = INSPECT.parent / "ml" / "preprocess_report.json"
    out_ml.parent.mkdir(parents=True, exist_ok=True)
    report = json.loads(out_ml.read_text()) if out_ml.exists() else {}
    report.update({
        "check_A": summary,
        "target": "velocity",
        "dt": DT,
        "store_v": True,
        "processed_size_estimate_GB": 52,
        "raw_size_GB": 59,
    })
    out_ml.write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
