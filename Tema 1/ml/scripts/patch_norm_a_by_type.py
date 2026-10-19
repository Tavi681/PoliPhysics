"""Add a_by_type to norm_stats; delete legacy noise_stats; refresh checksums.

  PYTHONPATH=. python -m ml.scripts.patch_norm_a_by_type \\
    --data-dir /data/processed/5c60fa669a719404 \\
    --upload-gcs gs://authorship-verification-poliphysics/tema1-ml/processed/5c60fa669a719404/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from ml.data.frames import DT
from ml.data.normalize import load_norm_stats, running_mean_var, save_norm_stats
from ml.data.shard_format import load_shard
from ml.data.splits import load_splits
from ml.train import _resolve
from ml.utils.config import load_yaml


TYPE_NAMES = ("spider", "interior", "tip")


def compute_a_by_type(data_dir: Path, train_ids: set[str]) -> dict:
    type_upds = [running_mean_var() for _ in range(3)]
    shards = sorted(data_dir.glob("shard_*.pt"))
    if not shards:
        raise FileNotFoundError(f"no shards in {data_dir}")
    for sh in shards:
        pack = load_shard(sh)
        for rec in pack:
            if rec["run_id"] not in train_ids:
                continue
            v = np.asarray(rec["v"], dtype=np.float32)
            nt = np.asarray(rec["node_type"], dtype=np.int64)
            ok = np.asarray(rec["window_ok"], dtype=bool)
            T = v.shape[0]
            if T < 7:
                continue
            t_idx = np.arange(5, T - 1)
            mask = ok[t_idx]
            if not np.any(mask):
                continue
            t_use = t_idx[mask]
            a = (v[t_use + 1] - v[t_use]) / np.float32(DT)
            for typ in range(3):
                m = nt == typ
                if not np.any(m):
                    continue
                type_upds[typ][0](a[:, m, :].reshape(-1, 3))
    return {TYPE_NAMES[t]: type_upds[t][1]() for t in range(3)}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def refresh_checksums(data_dir: Path) -> dict:
    """Update checksums.json for norm_stats; drop noise_stats keys."""
    ck_path = data_dir / "checksums.json"
    if ck_path.exists():
        obj = json.loads(ck_path.read_text())
    else:
        obj = {"sha256": {}}
    # Support both flat and nested {"sha256": {...}} layouts.
    if "sha256" in obj and isinstance(obj["sha256"], dict):
        table = obj["sha256"]
        nested = True
    else:
        table = obj
        nested = False
    for dead in ("noise_stats.json", "noise_stats.sha256"):
        table.pop(dead, None)
    digest = _sha256_file(data_dir / "norm_stats.json")
    table["norm_stats.json"] = digest
    if nested:
        obj["sha256"] = table
        obj["n_files"] = len(table)
        payload = obj
    else:
        payload = table
    ck_path.write_text(json.dumps(payload, indent=2))
    return {"norm_stats.json": digest, "removed": ["noise_stats.json", "noise_stats.sha256"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--splits", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=None,
                    help="optional YAML to resolve data_dir/splits")
    ap.add_argument("--upload-gcs", type=str, default=None)
    ap.add_argument("--keep-global-a", action="store_true", default=True,
                    help="keep existing global a mean/std (default)")
    args = ap.parse_args()

    if args.config is not None:
        cfg = load_yaml(_resolve(args.config) if not args.config.is_absolute() else args.config)
        dcfg = cfg["data"]
        data_dir = Path(dcfg["data_dir"])
        if not data_dir.is_absolute():
            data_dir = _resolve(data_dir)
        splits_path = Path(dcfg["splits"])
        if not splits_path.is_absolute():
            splits_path = _resolve(splits_path)
    else:
        data_dir = Path(args.data_dir) if Path(args.data_dir).is_absolute() else _resolve(args.data_dir)
        splits_path = args.splits
        if splits_path is None:
            splits_path = data_dir / "splits" / "splits.json"
        elif not Path(splits_path).is_absolute():
            splits_path = _resolve(splits_path)

    splits = load_splits(splits_path)
    train_ids = set(splits["train"])
    print(f"Computing a_by_type on {len(train_ids)} train runs in {data_dir}", flush=True)
    a_by_type = compute_a_by_type(data_dir, train_ids)

    norm_path = data_dir / "norm_stats.json"
    norm = load_norm_stats(norm_path)
    norm["a_by_type"] = a_by_type
    save_norm_stats(norm, norm_path)
    print("a_by_type:", json.dumps(
        {k: {"std": v["std"].tolist(), "count": v["count"]} for k, v in a_by_type.items()},
        indent=2,
    ), flush=True)

    # Delete legacy noise_stats so nothing can use it again.
    removed = []
    for name in ("noise_stats.json", "noise_stats.sha256"):
        p = data_dir / name
        if p.exists():
            p.unlink()
            removed.append(name)
            print(f"deleted {p}", flush=True)

    ck = refresh_checksums(data_dir)
    print(f"checksums updated: {ck}", flush=True)

    if args.upload_gcs:
        dest = args.upload_gcs.rstrip("/") + "/"
        cmds = [
            ["gcloud", "storage", "cp", str(norm_path), dest + "norm_stats.json"],
            ["gcloud", "storage", "cp", str(data_dir / "checksums.json"), dest + "checksums.json"],
        ]
        for name in removed:
            cmds.append(["gcloud", "storage", "rm", dest + name])
        for cmd in cmds:
            print(" ".join(cmd), flush=True)
            subprocess.run(cmd, check=False)
        print(f"uploaded patched norms to {dest}", flush=True)


if __name__ == "__main__":
    main()
