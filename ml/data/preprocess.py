"""Preprocess Stage B HDF5 → sharded torch packs + manifest.

Velocity-based target (check B): store x_rel (float32), spider_abs (float64),
v (float32), u (float32). Nominal dt = 0.01 s.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    # prefer idx valid
    m["valid"] = m["valid"].astype(bool)
    m = m.loc[m["valid"]].copy()
    m["run_id"] = m["name"].map(_run_id)
    m["x_grid"] = m["x"].astype(int)
    m["q"] = m["q_nC"].astype(float) * 1e-9  # Coulombs
    m["split_source"] = m["split"]
    m["n_nodes"] = 1 + m["N"].astype(int) * m["N_t"].astype(int)
    m["shard"] = m["name"].map(_shard_for)
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


def process_one_h5(path: Path, meta: dict[str, Any]) -> dict[str, Any]:
    with h5py.File(path, "r") as f:
        t = np.asarray(f["t"][:], dtype=np.float64)
        x_abs = np.asarray(f["x"][:], dtype=np.float64)  # (T,n,3)
        v = np.asarray(f["v"][:], dtype=np.float32)
        u = np.asarray(f["u_air"][:], dtype=np.float32)
        charge = np.asarray(f["charge"][:], dtype=np.float32)
        thread_id = np.asarray(f["topology/thread_id"][:], dtype=np.int32)
        node_type = _decode_node_type(f["topology/node_type"][:])
        edges = np.asarray(f["topology/edges"][:], dtype=np.int64)
        params = dict(f["params"].attrs)
        outcome = dict(f["outcome"].attrs)
        turb = dict(f["turb"].attrs)
        source_commit = str(params.get("git_commit", ""))
        source_dirty = bool(params.get("git_dirty", False))

    spider = x_abs[:, 0:1, :]  # (T,1,3)
    x_rel = (x_abs - spider).astype(np.float32)
    spider_abs = spider[:, 0, :].astype(np.float64)  # (T,3)

    ok = window_ok_mask(t, C=5, mode="index")  # C default; dataset may use other C

    rec = {
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
        },
    }
    return rec


def open_raw_h5(name: str, shard: int, local_ml_dir: Path | None = None):
    """Yield a local path to an HDF5 file (download if needed)."""
    if local_ml_dir is not None:
        p = Path(local_ml_dir) / name
        if p.exists():
            return p, None
    # download to temp
    from google.cloud import storage
    client = storage.Client()
    tmp = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    tmp.close()
    dest = Path(tmp.name)
    blob = client.bucket(GCS_BUCKET).blob(f"{GCS_RAW_PREFIX}/shard-{shard}/ml_samples/{name}")
    try:
        blob.download_to_filename(str(dest))
    except Exception:
        blob = client.bucket(GCS_BUCKET).blob(f"{GCS_RAW_PREFIX}/shard-{1 - shard}/ml_samples/{name}")
        blob.download_to_filename(str(dest))
    return dest, dest  # path, to_delete


def content_hash_bytes(parts: list[bytes]) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.hexdigest()[:16]


def preprocess(
    out_dir: Path,
    *,
    subset_frac: float | None = None,
    subset_seed: int = 42,
    runs: pd.DataFrame | None = None,
    local_ml_dir: Path | None = None,
    workers: int = 4,
    shard_size: int = 50,
    split_ids: dict[str, list[str]] | None = None,
    upload_gcs: bool = False,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = runs if runs is not None else build_run_table()
    if subset_frac is not None and subset_frac < 1.0:
        df = stratified_subset(df, subset_frac, seed=subset_seed)

    prov = git_provenance()
    records_meta = []
    shard_paths = []
    buf = []
    shard_idx = 0

    def flush():
        nonlocal shard_idx, buf
        if not buf:
            return
        path = out_dir / f"shard_{shard_idx:04d}.pt"
        torch.save(buf, path)
        shard_paths.append(str(path.name))
        for r in buf:
            records_meta.append({
                "run_id": r["run_id"],
                "shard": path.name,
                **{k: r["globals"][k] for k in (
                    "N", "N_t", "sigma_w", "x_grid", "n_frames", "n_nodes",
                    "outcome", "seed", "split_source", "wall_time_s",
                )},
            })
        shard_idx += 1
        buf = []

    def job(row):
        path, doomed = open_raw_h5(row["name"], int(row["shard"]), local_ml_dir)
        try:
            return process_one_h5(path, row.to_dict())
        finally:
            if doomed is not None and Path(doomed).exists():
                Path(doomed).unlink(missing_ok=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(job, r): r.run_id for _, r in df.iterrows()}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="preprocess"):
            rec = fut.result()
            buf.append(rec)
            if len(buf) >= shard_size:
                flush()
    flush()

    # normalization on TRAIN split only
    train_ids = set(split_ids.get("train", [])) if split_ids else set(df["run_id"])
    upd_v, fin_v = running_mean_var()
    upd_u, fin_u = running_mean_var()
    upd_a, fin_a = running_mean_var()
    upd_x, fin_x = running_mean_var()

    for sh in shard_paths:
        pack = torch.load(out_dir / sh, weights_only=False)
        for rec in pack:
            if rec["run_id"] not in train_ids:
                continue
            v = rec["v"]
            u = rec["u"]
            x = rec["x_rel"]
            ok = rec["window_ok"]
            # a_t at usable t: (v[t+1]-v[t])/dt
            T = v.shape[0]
            for ti in range(T - 1):
                if ti < 5 or not ok[ti]:
                    continue
                a = (v[ti + 1] - v[ti]) / DT
                upd_a(a)
                upd_v(v[ti])
                upd_u(u[ti])
                upd_x(x[ti])

    norm = {
        "v": fin_v(),
        "u": fin_u(),
        "x_rel": fin_x(),
        "a": fin_a(),
        "target": "velocity",
        "dt": DT,
    }
    save_norm_stats(norm, out_dir / "norm_stats.json")

    # data_version = content hash of run ids + target + dt + norm means
    payload = json.dumps({
        "run_ids": sorted(df["run_id"].tolist()),
        "target": "velocity",
        "dt": DT,
        "norm_a_mean": norm["a"]["mean"].tolist(),
        "norm_a_std": norm["a"]["std"].tolist(),
        "git": prov,
    }, sort_keys=True).encode()
    data_version = hashlib.sha256(payload).hexdigest()[:16]

    manifest = {
        "data_version": data_version,
        "n_runs": int(len(df)),
        "shards": shard_paths,
        "target": "velocity",
        "dt": DT,
        "store_v": True,
        "git": prov,
        "subset_frac": subset_frac,
        "records": records_meta,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    # rename / copy into versioned subdir marker
    with open(out_dir / "DATA_VERSION", "w") as f:
        f.write(data_version + "\n")

    if upload_gcs:
        _upload_processed(out_dir, data_version)

    return manifest


def _upload_processed(local_dir: Path, data_version: str) -> None:
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    prefix = f"{GCS_PROCESSED_PREFIX}/{data_version}"
    for path in Path(local_dir).rglob("*"):
        if path.is_file():
            rel = path.relative_to(local_dir).as_posix()
            blob = bucket.blob(f"{prefix}/{rel}")
            blob.upload_from_filename(str(path))
            print(f"uploaded gs://{GCS_BUCKET}/{prefix}/{rel}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--subset", type=float, default=None, help="Stratified fraction of runs")
    ap.add_argument("--subset-seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--shard-size", type=int, default=50)
    ap.add_argument("--local-ml-dir", type=Path, default=None)
    ap.add_argument("--splits", type=Path, default=None, help="splits.json for train-only norms")
    ap.add_argument("--upload-gcs", action="store_true")
    args = ap.parse_args()
    split_ids = None
    if args.splits:
        with open(args.splits) as f:
            split_ids = json.load(f).get("splits")
    man = preprocess(
        args.out,
        subset_frac=args.subset,
        subset_seed=args.subset_seed,
        local_ml_dir=args.local_ml_dir,
        workers=args.workers,
        shard_size=args.shard_size,
        split_ids=split_ids,
        upload_gcs=args.upload_gcs,
    )
    print(json.dumps({k: man[k] for k in ("data_version", "n_runs", "target", "dt")}, indent=2))


if __name__ == "__main__":
    main()
