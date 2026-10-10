"""Preprocess Stage B HDF5 → per-run packs + shards + TRAIN-only norms.

Streams x/v/u_air from GCS via gcsfs+h5py (cache_type=none); no full raw copy.
Per-run outputs under out/runs/<run_id>.pt support resume after interruption.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ml.data.frames import window_ok_mask, DT
from ml.data.normalize import running_mean_var, save_norm_stats
from ml.utils.git_info import git_provenance, repo_root

NODE_TYPE_MAP = {"spider": 0, "interior": 1, "tip": 2}

TEMA1 = repo_root() / "Tema 1"
DEFAULT_INDEX = TEMA1 / "results" / "ml_index.csv"
DEFAULT_SWEEP = TEMA1 / "results" / "sweep.csv"
GCS_BUCKET = "authorship-verification-poliphysics"
GCS_RAW_PREFIX = "tema1-stageb"
GCS_PROCESSED_PREFIX = "tema1-ml/processed"

# Module-level GCS filesystem for ProcessPool workers
_FS = None


def _basename(p: str) -> str:
    return Path(str(p)).name


def _run_id(name: str) -> str:
    return name[:-3] if name.endswith(".h5") else name


def _shard_for(name: str) -> int:
    import re
    m = re.search(r"_i(\d+)\.h5$", name)
    return int(m.group(1)) % 2 if m else 0


def build_run_table(
    index_csv: Path = DEFAULT_INDEX,
    sweep_csv: Path = DEFAULT_SWEEP,
) -> pd.DataFrame:
    idx = pd.read_csv(index_csv)
    sw = pd.read_csv(sweep_csv)
    idx["name"] = idx["hdf5_path"].map(_basename)
    sw["name"] = sw["hdf5_path"].map(_basename)
    cols = [
        "name", "N", "N_t", "sigma_w", "ell", "x", "Fbar_l", "q_nC", "seed",
        "split", "outcome", "exit_time", "wall_time_s", "valid",
    ]
    m = idx.merge(sw[cols], on="name", suffixes=("", "_sw"), how="left")
    m["valid"] = m["valid"].astype(bool)
    m = m.loc[m["valid"]].copy()
    m["run_id"] = m["name"].map(_run_id)
    m["x_grid"] = m["x"].astype(int)
    m["q"] = m["q_nC"].astype(float) * 1e-9
    m["split_source"] = m["split"]
    m["n_nodes"] = 1 + m["N"].astype(int) * m["N_t"].astype(int)
    m["shard"] = m["name"].map(_shard_for)
    if "n_frames" not in m.columns:
        m["n_frames"] = 0
    return m.reset_index(drop=True)


def stratified_subset(df: pd.DataFrame, frac: float, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for _, g in df.groupby(["N", "sigma_w", "x_grid"], sort=False):
        n = len(g)
        k = max(1, int(round(frac * n))) if frac > 0 else 0
        k = min(k, n)
        parts.append(g.sample(n=k, random_state=int(rng.integers(0, 2**31 - 1))))
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0]


def _decode_node_type(raw) -> np.ndarray:
    out = []
    for x in raw:
        s = x.decode() if isinstance(x, bytes) else str(x)
        out.append(NODE_TYPE_MAP[s])
    return np.asarray(out, dtype=np.int8)


def process_one_h5_file(f: h5py.File, meta: dict[str, Any]) -> dict[str, Any]:
    t = np.asarray(f["t"][:], dtype=np.float64)
    x_abs = np.asarray(f["x"][:], dtype=np.float64)
    v = np.asarray(f["v"][:], dtype=np.float32)
    u = np.asarray(f["u_air"][:], dtype=np.float32)
    charge = np.asarray(f["charge"][:], dtype=np.float32)
    thread_id = np.asarray(f["topology/thread_id"][:], dtype=np.int32)
    node_type = _decode_node_type(f["topology/node_type"][:])
    edges = np.asarray(f["topology/edges"][:], dtype=np.int64)
    params = dict(f["params"].attrs)
    turb = dict(f["turb"].attrs)
    source_commit = str(params.get("git_commit", ""))
    source_dirty = bool(params.get("git_dirty", False))

    spider = x_abs[:, 0:1, :]
    x_rel = (x_abs - spider).astype(np.float32)
    spider_abs = spider[:, 0, :].astype(np.float64)

    ok = window_ok_mask(t, C=5, mode="index")

    return {
        "run_id": meta["run_id"],
        "t": t.astype(np.float32),
        "x_rel": x_rel,
        "spider_abs": spider_abs,
        "v": v,
        "u": u,
        "charge": charge,
        "thread_id": thread_id,
        "node_type": node_type,
        "edges": edges,
        "window_ok": ok,
        "globals": {
            "N": int(meta["N"]),
            "N_t": int(meta["N_t"]),
            "sigma_w": float(meta["sigma_w"]),
            "ell": float(meta.get("ell", turb.get("ell", 1.0))),
            "U_h": float(params.get("U_h", turb.get("U_h", 1.0))),
            "x_grid": int(meta["x_grid"]),
            "Fbar_l": float(meta["Fbar_l"]),
            "q": float(meta["q"]),
            "m": float(params.get("m", 1e-6)),
            "E": float(params.get("E_constant", 7410.0)),
            "L": float(params.get("L", 0.5)),
            "z0": float(params.get("z0", 0.5)),
            "h": float(params.get("h", 2.0)),
            "seed": int(meta["seed"]),
            "split_source": str(meta["split_source"]),
            "outcome": str(meta["outcome"]),
            "exit_time": float(meta["exit_time"]) if pd.notna(meta["exit_time"]) else float("nan"),
            "wall_time_s": float(meta["wall_time_s"]) if pd.notna(meta.get("wall_time_s")) else float("nan"),
            "source_commit": source_commit,
            "source_dirty": source_dirty,
            "dt": DT,
            "n_frames": int(len(t)),
            "n_nodes": int(x_rel.shape[1]),
            "hdf5_name": str(meta["name"]),
            "hdf5_shard": int(meta["shard"]),
        },
    }


def process_one_h5(path: Path, meta: dict[str, Any]) -> dict[str, Any]:
    with h5py.File(path, "r") as f:
        return process_one_h5_file(f, meta)


def _init_worker():
    global _FS
    import gcsfs
    _FS = gcsfs.GCSFileSystem()


def process_one_gcs(meta: dict[str, Any]) -> dict[str, Any]:
    """Stream /x,/v,/u_air,/t from GCS (range reads; no full-file cache)."""
    name = meta["name"]
    shard = int(meta["shard"])
    last_err = None
    for s in (shard, 1 - shard):
        path = f"{GCS_BUCKET}/{GCS_RAW_PREFIX}/shard-{s}/ml_samples/{name}"
        try:
            with _FS.open(path, "rb", cache_type="none") as raw:
                with h5py.File(raw, "r") as f:
                    rec = process_one_h5_file(f, {**meta, "shard": s})
                    rec["globals"]["hdf5_shard"] = int(s)
                    return rec
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"{name}: {last_err}")


def _run_path(out_dir: Path, run_id: str) -> Path:
    return out_dir / "runs" / f"{run_id}.pt"


def _records_from_shards(out_dir: Path) -> tuple[list[str], list[dict]]:
    shard_paths = sorted(p.name for p in out_dir.glob("shard_*.pt"))
    records_meta: list[dict] = []
    for sh in shard_paths:
        for rec in torch.load(out_dir / sh, weights_only=False):
            g = rec["globals"]
            records_meta.append({
                "run_id": rec["run_id"],
                "shard": sh,
                **{k: g[k] for k in (
                    "N", "N_t", "sigma_w", "x_grid", "n_frames", "n_nodes",
                    "outcome", "seed", "split_source", "wall_time_s",
                    "hdf5_name", "hdf5_shard",
                ) if k in g},
            })
    return shard_paths, records_meta


def pack_pending_runs(
    out_dir: Path,
    shard_size: int = 50,
) -> tuple[list[str], list[dict]]:
    """Pack any remaining runs/*.pt into new shards; delete each after pack."""
    out_dir = Path(out_dir)
    existing = sorted(out_dir.glob("shard_*.pt"))
    shard_idx = 0
    if existing:
        # next index after highest shard_NNNN
        shard_idx = max(int(p.stem.split("_")[1]) for p in existing) + 1
    pending = sorted((out_dir / "runs").glob("*.pt"))
    buf: list[dict] = []
    buf_paths: list[Path] = []

    def flush():
        nonlocal shard_idx, buf, buf_paths
        if not buf:
            return
        name = f"shard_{shard_idx:04d}.pt"
        torch.save(buf, out_dir / name)
        for p in buf_paths:
            p.unlink(missing_ok=True)
        shard_idx += 1
        buf = []
        buf_paths = []

    for p in tqdm(pending, desc="pack_shards"):
        buf.append(torch.load(p, weights_only=False))
        buf_paths.append(p)
        if len(buf) >= shard_size:
            flush()
    flush()
    return _records_from_shards(out_dir)


def compute_train_norms(out_dir: Path, train_ids: set[str], shard_paths: list[str]) -> dict:
    upd_v, fin_v = running_mean_var()
    upd_u, fin_u = running_mean_var()
    upd_a, fin_a = running_mean_var()
    upd_x, fin_x = running_mean_var()
    for sh in tqdm(shard_paths, desc="norm"):
        pack = torch.load(out_dir / sh, weights_only=False)
        for rec in pack:
            if rec["run_id"] not in train_ids:
                continue
            v, u, x, ok = rec["v"], rec["u"], rec["x_rel"], rec["window_ok"]
            T = v.shape[0]
            for ti in range(T - 1):
                if ti < 5 or not ok[ti]:
                    continue
                a = (v[ti + 1] - v[ti]) / DT
                upd_a(a)
                upd_v(v[ti])
                upd_u(u[ti])
                upd_x(x[ti])
    return {
        "v": fin_v(),
        "u": fin_u(),
        "x_rel": fin_x(),
        "a": fin_a(),
        "target": "velocity",
        "dt": DT,
    }


def preprocess(
    out_dir: Path,
    *,
    subset_frac: float | None = None,
    subset_seed: int = 42,
    runs: pd.DataFrame | None = None,
    local_ml_dir: Path | None = None,
    workers: int = 16,
    shard_size: int = 50,
    split_ids: dict[str, list[str]] | None = None,
    upload_gcs: bool = False,
    resume: bool = True,
    stream_gcs: bool = True,
    splits_dir: Path | None = None,
) -> dict[str, Any]:
    t0 = time.time()
    out_dir = Path(out_dir)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    df = runs if runs is not None else build_run_table()
    if subset_frac is not None and subset_frac < 1.0:
        df = stratified_subset(df, subset_frac, seed=subset_seed)

    prov = git_provenance()
    rows = [r.to_dict() for _, r in df.iterrows()]

    # Resume: skip runs already in per-run files OR already packed into shards
    packed_ids: set[str] = set()
    if resume:
        for sh in sorted(out_dir.glob("shard_*.pt")):
            for rec in torch.load(sh, weights_only=False):
                packed_ids.add(rec["run_id"])
    todo = []
    skipped = 0
    for r in rows:
        rid = r["run_id"]
        if resume and (_run_path(out_dir, rid).exists() or rid in packed_ids):
            skipped += 1
        else:
            todo.append(r)
    print(f"preprocess: total={len(rows)} done={skipped} todo={len(todo)} "
          f"packed={len(packed_ids)} stream_gcs={stream_gcs} workers={workers}", flush=True)

    errors: list[str] = []
    if todo:
        if stream_gcs and local_ml_dir is None:
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=workers, mp_context=ctx, initializer=_init_worker
            ) as ex:
                futs = {ex.submit(process_one_gcs, r): r["run_id"] for r in todo}
                for fut in tqdm(as_completed(futs), total=len(futs), desc="fetch"):
                    rid = futs[fut]
                    try:
                        rec = fut.result()
                        torch.save(rec, _run_path(out_dir, rid))
                    except Exception as e:
                        errors.append(f"{rid}: {e}")
        else:
            # local files (or download-to-temp fallback)
            from concurrent.futures import ThreadPoolExecutor

            def job(r):
                if local_ml_dir is not None:
                    p = Path(local_ml_dir) / r["name"]
                    if p.exists():
                        return process_one_h5(p, r)
                # fallback: stream via gcsfs in-thread
                _init_worker()
                return process_one_gcs(r)

            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(job, r): r["run_id"] for r in todo}
                for fut in tqdm(as_completed(futs), total=len(futs), desc="fetch"):
                    rid = futs[fut]
                    try:
                        rec = fut.result()
                        torch.save(rec, _run_path(out_dir, rid))
                    except Exception as e:
                        errors.append(f"{rid}: {e}")

    if errors:
        (out_dir / "preprocess_errors.json").write_text(json.dumps(errors[:100], indent=2))
        raise RuntimeError(f"{len(errors)} preprocess failures; see preprocess_errors.json")

    # Ensure every requested run is available as per-run file or packed shard
    have = set(packed_ids)
    for r in rows:
        if _run_path(out_dir, r["run_id"]).exists():
            have.add(r["run_id"])
    missing = [r["run_id"] for r in rows if r["run_id"] not in have]
    if missing:
        raise RuntimeError(f"missing {len(missing)} runs after fetch e.g. {missing[:5]}")

    shard_paths, records_meta = pack_pending_runs(out_dir, shard_size=shard_size)
    done_ids = [r["run_id"] for r in rows]
    have_final = {m["run_id"] for m in records_meta}
    if have_final != set(done_ids):
        raise RuntimeError(
            f"shard coverage mismatch: have={len(have_final)} want={len(done_ids)}"
        )

    train_ids = set(split_ids.get("train", [])) if split_ids else set(done_ids)
    norm = compute_train_norms(out_dir, train_ids, shard_paths)
    save_norm_stats(norm, out_dir / "norm_stats.json")

    payload = json.dumps({
        "run_ids": sorted(done_ids),
        "target": "velocity",
        "dt": DT,
        "norm_a_mean": norm["a"]["mean"].tolist(),
        "norm_a_std": norm["a"]["std"].tolist(),
        "git": prov,
        "n_train_norm": len(train_ids),
    }, sort_keys=True).encode()
    data_version = hashlib.sha256(payload).hexdigest()[:16]

    manifest = {
        "data_version": data_version,
        "n_runs": int(len(done_ids)),
        "shards": shard_paths,
        "target": "velocity",
        "dt": DT,
        "store_v": True,
        "git": prov,
        "subset_frac": subset_frac,
        "stream_gcs": bool(stream_gcs and local_ml_dir is None),
        "resume": resume,
        "n_train_for_norm": len(train_ids),
        "wall_time_s": time.time() - t0,
        "records": records_meta,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    with open(out_dir / "DATA_VERSION", "w") as f:
        f.write(data_version + "\n")

    if splits_dir is not None and Path(splits_dir).exists():
        dest = out_dir / "splits"
        src = Path(splits_dir).resolve()
        if src != dest.resolve():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)

    if upload_gcs:
        _upload_processed(out_dir, data_version)

    return manifest


def _upload_processed(local_dir: Path, data_version: str) -> dict[str, Any]:
    """Upload shards/manifest/norms/splits (not per-run cache) + write checksums."""
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    prefix = f"{GCS_PROCESSED_PREFIX}/{data_version}"
    # Prefer packed artifacts; skip bulky runs/ cache on GCS (rebuildable)
    include = []
    for path in sorted(Path(local_dir).rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(local_dir).as_posix()
        if rel.startswith("runs/"):
            continue
        include.append(path)

    checksums = {}
    total_bytes = 0
    for path in tqdm(include, desc="upload"):
        rel = path.relative_to(local_dir).as_posix()
        data = path.read_bytes()
        total_bytes += len(data)
        checksums[rel] = hashlib.sha256(data).hexdigest()
        blob = bucket.blob(f"{prefix}/{rel}")
        blob.upload_from_filename(str(path))
    check_path = Path(local_dir) / "checksums.json"
    check_path.write_text(json.dumps({
        "data_version": data_version,
        "n_files": len(checksums),
        "total_bytes": total_bytes,
        "sha256": checksums,
    }, indent=2))
    bucket.blob(f"{prefix}/checksums.json").upload_from_filename(str(check_path))
    print(f"uploaded gs://{GCS_BUCKET}/{prefix}/ ({len(checksums)} files, {total_bytes/1e9:.3f} GB)")
    return {"prefix": prefix, "n_files": len(checksums), "total_bytes": total_bytes}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--subset", type=float, default=None)
    ap.add_argument("--subset-seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--shard-size", type=int, default=50)
    ap.add_argument("--local-ml-dir", type=Path, default=None)
    ap.add_argument("--splits", type=Path, default=None)
    ap.add_argument("--upload-gcs", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--no-stream", action="store_true", help="Disable gcsfs stream (use local only)")
    args = ap.parse_args()
    split_ids = None
    if args.splits:
        with open(args.splits) as f:
            split_ids = json.load(f).get("splits")
    splits_dir = Path(args.splits).parent if args.splits else None
    man = preprocess(
        args.out,
        subset_frac=args.subset,
        subset_seed=args.subset_seed,
        local_ml_dir=args.local_ml_dir,
        workers=args.workers,
        shard_size=args.shard_size,
        split_ids=split_ids,
        upload_gcs=args.upload_gcs,
        resume=not args.no_resume,
        stream_gcs=not args.no_stream,
        splits_dir=splits_dir,
    )
    print(json.dumps({k: man[k] for k in (
        "data_version", "n_runs", "target", "dt", "wall_time_s", "n_train_for_norm"
    )}, indent=2))


if __name__ == "__main__":
    main()
