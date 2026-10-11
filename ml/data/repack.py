"""Re-pack processed shards: materialize arrays + add u_next (no HDF5 re-read).

Reads an existing processed data_version (local or GCS), writes a new one with:
  - contiguous exact-sized ndarray copies (drops oversized parent storages)
  - u_next[t] = KS.u(x_abs_t, t+dt) float32 via ballooning.fields.KinematicSimulation
  - updated splits including test_unseenN_interp (N=3); train/val/norms unchanged
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ml.data.frames import DT
from ml.data.preprocess import (
    GCS_BUCKET,
    GCS_PROCESSED_PREFIX,
    _records_from_shards,
    _upload_processed,
    materialize_record,
)
from ml.data.splits import (
    assert_split_coverage,
    splits_hash,
    write_split_manifest,
)
from ml.utils.git_info import git_provenance, repo_root


def _ensure_tema1_path() -> None:
    tema1 = str(repo_root() / "Tema 1")
    if tema1 not in sys.path:
        sys.path.insert(0, tema1)


def make_ks(globals_: dict[str, Any]):
    """Simulator KinematicSimulation (same kwargs as evaluate / Stage B sweep)."""
    _ensure_tema1_path()
    from ballooning.fields import KinematicSimulation
    return KinematicSimulation(
        sigma=float(globals_["sigma_w"]),
        ell=float(globals_["ell"]),
        U_h=float(globals_["U_h"]),
        N_k=200,
        seed=int(globals_["seed"]),
        lambda_=0.5,
        renormalize=True,
    )


def compute_u_next(rec: dict[str, Any]) -> np.ndarray:
    """u_next[t, node] at positions x_t (abs) and time t+dt, float32."""
    g = rec["globals"]
    dt = float(g.get("dt", DT))
    flow = make_ks(g)
    t = np.asarray(rec["t"], dtype=np.float64)
    x_rel = np.asarray(rec["x_rel"], dtype=np.float64)
    spider = np.asarray(rec["spider_abs"], dtype=np.float64)
    T, n, _ = x_rel.shape
    out = np.empty((T, n, 3), dtype=np.float32)
    for i in range(T):
        x_abs = spider[i] + x_rel[i]
        out[i] = np.asarray(flow.u(x_abs, float(t[i] + dt)), dtype=np.float32)
    return out


def _process_one_run(rec: dict[str, Any]) -> dict[str, Any]:
    out = materialize_record(rec)
    out["u_next"] = compute_u_next(out)
    return out


def max_abs_field_diff(a: dict[str, Any], b: dict[str, Any], keys: tuple[str, ...]) -> dict[str, float]:
    diffs = {}
    for k in keys:
        if k not in a or k not in b:
            continue
        diffs[k] = float(np.max(np.abs(
            np.asarray(a[k]).astype(np.float64) - np.asarray(b[k]).astype(np.float64)
        )))
    return diffs


COMPARE_KEYS = (
    "t", "x_rel", "spider_abs", "v", "u",
    "charge", "thread_id", "node_type", "edges", "window_ok",
)


def _process_shard_job(args: tuple[str, str, str]) -> dict[str, Any]:
    """Worker: load one local src shard → materialize+u_next → consolidated out shard."""
    from ml.data.shard_format import load_shard, pack_records_consolidated
    local_sh, out_shard, _unused = args
    pack = load_shard(local_sh)
    max_diffs = {k: 0.0 for k in COMPARE_KEYS}
    new_pack = []
    done: list[str] = []
    for rec in pack:
        new_rec = _process_one_run(rec)
        for k, v in max_abs_field_diff(rec, new_rec, COMPARE_KEYS).items():
            max_diffs[k] = max(max_diffs[k], v)
        new_pack.append(new_rec)
        done.append(new_rec["run_id"])
    torch.save(pack_records_consolidated(new_pack), out_shard)
    return {
        "done": done,
        "max_diffs": max_diffs,
        "src_shard": local_sh,
        "out_shard": out_shard,
        "n_runs": len(new_pack),
    }


def patch_splits_with_n3(
    old_splits: dict[str, list[str]],
    records: list[dict],
    run_table: pd.DataFrame | None = None,
) -> dict[str, list[str]]:
    """Add test_unseenN_interp = all N=3 runs; keep every other split identical."""
    from ml.data.preprocess import build_run_table
    df = run_table if run_table is not None else build_run_table()
    # Prefer n_frames from processed records
    id2frames = {r["run_id"]: int(r["n_frames"]) for r in records if "n_frames" in r}
    if id2frames:
        df = df.copy()
        df["n_frames"] = df["run_id"].map(lambda x: id2frames.get(x, 0)).astype(int)

    assigned = {rid: sp for sp, ids in old_splits.items() for rid in ids}
    n3 = df.loc[df["N"] == 3, "run_id"].tolist()
    if len(n3) != 150:
        # still allow if subset; but full pack expects 150
        pass
    for rid in n3:
        if rid in assigned and assigned[rid] != "test_unseenN_interp":
            raise RuntimeError(f"{rid} already in {assigned[rid]}")
    new = {k: list(v) for k, v in old_splits.items()}
    # Preserve other splits exactly (sorted lists as before)
    for k in list(new.keys()):
        if k != "test_unseenN_interp":
            new[k] = sorted(new[k])
    new["test_unseenN_interp"] = sorted(set(n3))
    expected = set(df["run_id"])
    assert_split_coverage(new, expected)
    # Ensure train/val/other tests unchanged
    for k in old_splits:
        if k == "test_unseenN_interp":
            continue
        if sorted(old_splits[k]) != sorted(new[k]):
            raise RuntimeError(f"split {k} changed unexpectedly")
    return new


def repack(
    *,
    src_version: str = "b0a7c5b503af5d26",
    src_dir: Path | None = None,
    out_dir: Path,
    workers: int = 16,
    resume: bool = True,
    upload_gcs: bool = False,
    patch_src_splits: bool = True,
) -> dict[str, Any]:
    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done_shards_path = out_dir / "repack_done_shards.json"
    done_shards: set[str] = set()
    if resume and done_shards_path.exists():
        done_shards = set(json.loads(done_shards_path.read_text()))
    # Also treat existing output shards as done
    for p in out_dir.glob("shard_*.pt"):
        done_shards.add(p.name)

    import gcsfs
    fs = gcsfs.GCSFileSystem()
    gcs_prefix = f"{GCS_BUCKET}/{GCS_PROCESSED_PREFIX}/{src_version}"

    meta_dir = out_dir / "_src_meta"
    meta_dir.mkdir(exist_ok=True)
    # Always refresh small meta files
    for name in ("manifest.json", "norm_stats.json", "DATA_VERSION",
                 "splits/splits.json", "splits/tab_mlsplit.csv", "checksums.json"):
        src = f"{gcs_prefix}/{name}"
        if fs.exists(src):
            dest = meta_dir / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            fs.get(src, str(dest))

    man = json.loads((meta_dir / "manifest.json").read_text())
    old_splits = json.loads((meta_dir / "splits" / "splits.json").read_text())["splits"]
    records_meta = man["records"]
    shard_names = list(man["shards"])

    # Build new splits (N=3) using index + record frame counts
    from ml.data.preprocess import build_run_table
    df = build_run_table()
    id2frames = {r["run_id"]: int(r["n_frames"]) for r in records_meta}
    df = df.copy()
    df["n_frames"] = df["run_id"].map(lambda x: id2frames.get(x, 0)).astype(int)
    new_splits = patch_splits_with_n3(old_splits, records_meta, df)
    splits_dir = out_dir / "splits"
    write_split_manifest(new_splits, df, splits_dir)

    if patch_src_splits:
        # Update b0a7 in place (splits + checksums only; norms untouched)
        _upload_splits_only(fs, src_version, splits_dir, meta_dir)

    # Copy norms unchanged
    shutil.copy2(meta_dir / "norm_stats.json", out_dir / "norm_stats.json")

    all_ids = [r["run_id"] for r in records_meta]
    todo_shards = [s for s in shard_names if s not in done_shards]
    print(
        f"repack: src={src_version} shards={len(shard_names)} "
        f"done={len(done_shards)} todo={len(todo_shards)}",
        flush=True,
    )

    # Stream shards: download → rewrite in place name → drop cache (low disk).
    shard_cache = out_dir / "_shard_cache"
    shard_cache.mkdir(exist_ok=True)

    max_diffs: dict[str, float] = {k: 0.0 for k in COMPARE_KEYS}
    wave = max(1, min(workers, 4))
    for i in tqdm(range(0, len(todo_shards), wave), desc="shard_waves"):
        wave_names = todo_shards[i : i + wave]
        jobs = []
        for sh_name in wave_names:
            local_sh = shard_cache / sh_name
            if not local_sh.exists():
                fs.get(f"{gcs_prefix}/{sh_name}", str(local_sh))
            out_sh = str(out_dir / sh_name)
            jobs.append((str(local_sh), out_sh, ""))
        with ProcessPoolExecutor(max_workers=len(jobs)) as ex:
            futs = [ex.submit(_process_shard_job, j) for j in jobs]
            for fut in as_completed(futs):
                res = fut.result()
                for k, v in res["max_diffs"].items():
                    max_diffs[k] = max(max_diffs[k], v)
                done_shards.add(Path(res["out_shard"]).name)
                Path(res["src_shard"]).unlink(missing_ok=True)
        done_shards_path.write_text(json.dumps(sorted(done_shards)))

    if any(v != 0.0 for v in max_diffs.values()):
        raise RuntimeError(f"repack field mismatch vs source shards: {max_diffs}")

    shard_paths, new_records = _records_from_shards(out_dir)
    if len(new_records) != len(all_ids):
        raise RuntimeError(f"expected {len(all_ids)} records, got {len(new_records)}")
    if set(shard_paths) != set(shard_names):
        raise RuntimeError(
            f"shard set mismatch: missing={set(shard_names)-set(shard_paths)} "
            f"extra={set(shard_paths)-set(shard_names)}"
        )

    # Verify every new record has u_next
    from ml.data.shard_format import load_shard
    for sh in shard_paths:
        for rec in load_shard(out_dir / sh):
            if "u_next" not in rec:
                raise RuntimeError(f"missing u_next in {rec['run_id']}")
            if rec["u_next"].shape != rec["u"].shape:
                raise RuntimeError(f"u_next shape mismatch {rec['run_id']}")

    # Leak test on 3 runs per N (local shards)
    leak = run_leak_tests(out_dir, new_records, per_n=3, seed=42)

    prov = git_provenance()
    norm = json.loads((out_dir / "norm_stats.json").read_text())
    payload = json.dumps({
        "run_ids": sorted(all_ids),
        "target": "velocity",
        "dt": DT,
        "norm_a_mean": norm["a"]["mean"],
        "norm_a_std": norm["a"]["std"],
        "git": prov,
        "n_train_norm": man.get("n_train_for_norm", 4720),
        "u_next": True,
        "shard_format": "consolidated_v1",
        "repack_from": src_version,
        "splits_hash": splits_hash(new_splits),
    }, sort_keys=True).encode()
    data_version = hashlib.sha256(payload).hexdigest()[:16]

    manifest = {
        "data_version": data_version,
        "n_runs": len(all_ids),
        "shards": shard_paths,
        "target": "velocity",
        "dt": DT,
        "store_v": True,
        "store_u_next": True,
        "shard_format": "consolidated_v1",
        "repack_from": src_version,
        "git": prov,
        "n_train_for_norm": man.get("n_train_for_norm", 4720),
        "splits_hash": splits_hash(new_splits),
        "wall_time_s": time.time() - t0,
        "repack_max_abs_vs_src": max_diffs,
        "leak_test": leak,
        "records": new_records,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out_dir / "DATA_VERSION").write_text(data_version + "\n")
    (out_dir / "packed_ids.json").write_text(json.dumps(sorted(all_ids)))

    if upload_gcs:
        _upload_processed(out_dir, data_version)

    return manifest


def _upload_splits_only(fs, src_version: str, splits_dir: Path, meta_dir: Path) -> None:
    """Patch splits on existing data_version; refresh checksums for changed keys."""
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    prefix = f"{GCS_PROCESSED_PREFIX}/{src_version}"

    for name in ("splits.json", "tab_mlsplit.csv"):
        local = splits_dir / name
        bucket.blob(f"{prefix}/splits/{name}").upload_from_filename(str(local))

    check_path = meta_dir / "checksums.json"
    checks = json.loads(check_path.read_text()) if check_path.exists() else {
        "data_version": src_version, "sha256": {}, "n_files": 0, "total_bytes": 0,
    }
    sha = dict(checks.get("sha256", {}))
    for name in ("splits.json", "tab_mlsplit.csv"):
        data = (splits_dir / name).read_bytes()
        sha[f"splits/{name}"] = hashlib.sha256(data).hexdigest()

    man = json.loads((meta_dir / "manifest.json").read_text())
    man["splits_hash"] = json.loads((splits_dir / "splits.json").read_text())["splits_hash"]
    man["splits_note"] = "test_unseenN_interp (N=3) added; train/val/norms unchanged"
    man_bytes = json.dumps(man, indent=2).encode()
    (meta_dir / "manifest.json").write_bytes(man_bytes)
    bucket.blob(f"{prefix}/manifest.json").upload_from_string(
        man_bytes, content_type="application/json"
    )
    sha["manifest.json"] = hashlib.sha256(man_bytes).hexdigest()

    # Sum sizes of every hashed object currently on GCS
    total = 0
    for rel in sha:
        path = f"{GCS_BUCKET}/{prefix}/{rel}"
        if fs.exists(path):
            total += int(fs.info(path)["size"])

    out = {
        "data_version": src_version,
        "n_files": len(sha),
        "total_bytes": total,
        "sha256": sha,
        "note": "splits patched with test_unseenN_interp; shard hashes unchanged",
    }
    raw = json.dumps(out, indent=2).encode()
    check_path.write_bytes(raw)
    bucket.blob(f"{prefix}/checksums.json").upload_from_string(
        raw, content_type="application/json"
    )
    print(f"patched splits on gs://{GCS_BUCKET}/{prefix}/", flush=True)


def run_leak_tests(
    data_dir: Path,
    records: list[dict],
    *,
    per_n: int = 3,
    seed: int = 42,
) -> dict[str, Any]:
    """u_next == KS(x_t, t+dt) to f32 roundoff; differs from u[t+1]."""
    import random
    from collections import defaultdict
    by_n: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        by_n[int(r["N"])].append(r)
    rng = random.Random(seed)
    samples = []
    for N, recs in sorted(by_n.items()):
        samples.extend(recs if len(recs) <= per_n else rng.sample(recs, per_n))

    from ml.data.shard_format import load_shard
    shard_cache: dict[str, list] = {}
    results = []
    for meta in samples:
        sh = meta["shard"]
        if sh not in shard_cache:
            shard_cache[sh] = load_shard(Path(data_dir) / sh)
        rec = next(r for r in shard_cache[sh] if r["run_id"] == meta["run_id"])
        g = rec["globals"]
        dt = float(g.get("dt", DT))
        flow = make_ks(g)
        t = np.asarray(rec["t"], dtype=np.float64)
        x_rel = np.asarray(rec["x_rel"], dtype=np.float64)
        spider = np.asarray(rec["spider_abs"], dtype=np.float64)
        u_next = np.asarray(rec["u_next"], dtype=np.float32)
        u = np.asarray(rec["u"], dtype=np.float32)
        # check a few frames + mid
        T = len(t)
        frames = sorted(set([0, max(0, T // 2), max(0, T - 2), max(0, T - 1)]))
        max_eq = 0.0
        max_ne = 0.0
        for i in frames:
            x_abs = spider[i] + x_rel[i]
            direct = np.asarray(flow.u(x_abs, float(t[i] + dt)), dtype=np.float32)
            max_eq = max(max_eq, float(np.max(np.abs(direct - u_next[i]))))
            if i + 1 < T:
                max_ne = max(max_ne, float(np.max(np.abs(u_next[i] - u[i + 1]))))
        results.append({
            "run_id": meta["run_id"],
            "N": int(meta["N"]),
            "max_abs_vs_direct_ks": max_eq,
            "max_abs_vs_u_tp1": max_ne,
            "eq_ok": max_eq == 0.0,
            "differs_from_u_tp1": max_ne > 0.0,
        })
    ok = all(r["eq_ok"] and r["differs_from_u_tp1"] for r in results)
    return {"ok": ok, "n": len(results), "per_run": results}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-version", default="b0a7c5b503af5d26")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--upload-gcs", action="store_true")
    ap.add_argument("--skip-patch-src-splits", action="store_true")
    args = ap.parse_args()
    man = repack(
        src_version=args.src_version,
        out_dir=args.out,
        workers=args.workers,
        resume=not args.no_resume,
        upload_gcs=args.upload_gcs,
        patch_src_splits=not args.skip_patch_src_splits,
    )
    print(json.dumps({
        "data_version": man["data_version"],
        "n_runs": man["n_runs"],
        "store_u_next": man["store_u_next"],
        "wall_time_s": man["wall_time_s"],
        "leak_ok": man["leak_test"]["ok"],
        "repack_max_abs_vs_src": man["repack_max_abs_vs_src"],
    }, indent=2))


if __name__ == "__main__":
    main()
