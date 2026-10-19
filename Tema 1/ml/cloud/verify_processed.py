"""Verify processed shards on GCS against original Stage B HDF5.

Checks: file count, total size, per-shard sha256, and for 3 random runs per N
reloads x/v/u from processed vs original HDF5 (max abs difference).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch

from ml.data.preprocess import GCS_BUCKET, GCS_PROCESSED_PREFIX, GCS_RAW_PREFIX, DT


def _gcs():
    import gcsfs
    return gcsfs.GCSFileSystem()


def verify(
    data_version: str,
    *,
    seed: int = 42,
    per_n: int = 3,
    local_dir: Path | None = None,
) -> dict:
    fs = _gcs()
    prefix = f"{GCS_BUCKET}/{GCS_PROCESSED_PREFIX}/{data_version}"
    files = [p for p in fs.ls(prefix) if not p.endswith("/")]
    sizes = {p: fs.info(p)["size"] for p in files}
    total = int(sum(sizes.values()))

    # checksums.json if present
    check_blob = f"{prefix}/checksums.json"
    checksum_ok = None
    shard_checks = {}
    if fs.exists(check_blob):
        with fs.open(check_blob, "rb") as f:
            checks = json.load(f)
        for rel, expect in checks.get("sha256", {}).items():
            if not rel.startswith("shard_"):
                continue
            path = f"{prefix}/{rel}"
            h = hashlib.sha256()
            with fs.open(path, "rb") as f:
                while True:
                    b = f.read(8 << 20)
                    if not b:
                        break
                    h.update(b)
            got = h.hexdigest()
            shard_checks[rel] = {"expected": expect, "got": got, "ok": got == expect}
        checksum_ok = all(v["ok"] for v in shard_checks.values()) if shard_checks else None

    # load manifest
    with fs.open(f"{prefix}/manifest.json", "rb") as f:
        man = json.load(f)
    by_n: dict[int, list[dict]] = defaultdict(list)
    for rec in man["records"]:
        by_n[int(rec["N"])].append(rec)

    rng = random.Random(seed)
    samples = []
    for N, recs in sorted(by_n.items()):
        pick = recs if len(recs) <= per_n else rng.sample(recs, per_n)
        samples.extend(pick)

    # index run_id -> record in shard
    diffs = []
    for meta in samples:
        rid = meta["run_id"]
        sh = meta["shard"]
        from ml.data.shard_format import load_shard
        import tempfile
        with fs.open(f"{prefix}/{sh}", "rb") as f:
            raw = f.read()
        with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
            tmp.write(raw)
            tmp.flush()
            pack = load_shard(tmp.name)
        rec = next(r for r in pack if r["run_id"] == rid)
        name = rec["globals"].get("hdf5_name", rid + ".h5")
        shard = int(rec["globals"].get("hdf5_shard", meta.get("hdf5_shard", 0)))
        # stream original
        last_err = None
        x_abs = v = u = None
        for s in (shard, 1 - shard):
            raw_path = f"{GCS_BUCKET}/{GCS_RAW_PREFIX}/shard-{s}/ml_samples/{name}"
            try:
                with fs.open(raw_path, "rb", cache_type="none") as raw:
                    with h5py.File(raw, "r") as hf:
                        x_abs = np.asarray(hf["x"][:], dtype=np.float64)
                        v = np.asarray(hf["v"][:], dtype=np.float32)
                        u = np.asarray(hf["u_air"][:], dtype=np.float32)
                break
            except Exception as e:
                last_err = e
        if x_abs is None:
            raise RuntimeError(f"raw open failed {name}: {last_err}")
        spider = x_abs[:, 0:1, :]
        x_rel = (x_abs - spider).astype(np.float32)
        d = {
            "run_id": rid,
            "N": int(meta["N"]),
            "max_abs_x_rel": float(np.max(np.abs(rec["x_rel"] - x_rel))),
            "max_abs_v": float(np.max(np.abs(rec["v"] - v))),
            "max_abs_u": float(np.max(np.abs(rec["u"] - u))),
            "dt": DT,
        }
        diffs.append(d)

    report = {
        "data_version": data_version,
        "gcs_prefix": f"gs://{prefix}",
        "n_files": len(files),
        "total_bytes": total,
        "total_GB": total / 1e9,
        "checksum_shards_ok": checksum_ok,
        "n_shard_checksums": len(shard_checks),
        "n_compare": len(diffs),
        "max_abs_x_rel": max(d["max_abs_x_rel"] for d in diffs),
        "max_abs_v": max(d["max_abs_v"] for d in diffs),
        "max_abs_u": max(d["max_abs_u"] for d in diffs),
        "per_run": diffs,
        "ok": (
            (checksum_ok is not False)
            and max(d["max_abs_x_rel"] for d in diffs) == 0.0
            and max(d["max_abs_v"] for d in diffs) == 0.0
            and max(d["max_abs_u"] for d in diffs) == 0.0
        ),
    }
    if local_dir:
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        Path(local_dir, "verify_report.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-version", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-n", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    rep = verify(args.data_version, seed=args.seed, per_n=args.per_n, local_dir=args.out)
    print(json.dumps({k: rep[k] for k in (
        "data_version", "n_files", "total_GB", "checksum_shards_ok",
        "max_abs_x_rel", "max_abs_v", "max_abs_u", "ok",
    )}, indent=2))
    if not rep["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
