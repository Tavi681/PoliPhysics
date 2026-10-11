"""Compute train-only std(dv) once; write noise_stats.json + sha256 next to norm_stats.

  PYTHONPATH=. python -m ml.scripts.compute_noise_stats \\
    --data-dir /data/processed/5c60fa669a719404 \\
    --upload-gcs gs://authorship-verification-poliphysics/tema1-ml/processed/5c60fa669a719404/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from ml.data.dataset import BallooningRunStore
from ml.data.noise_stats import compute_dv_std_by_type, save_noise_stats
from ml.data.splits import load_splits


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--splits", type=Path, default=None)
    ap.add_argument("--max-frames-per-run", type=int, default=80)
    ap.add_argument("--upload-gcs", type=str, default=None)
    args = ap.parse_args()
    root = args.data_dir
    splits_path = args.splits or (root / "splits" / "splits.json")
    store = BallooningRunStore(root, cache_shards=8, mmap=True)
    splits = load_splits(splits_path)
    train = list(splits["train"])
    stats = compute_dv_std_by_type(
        store, train, max_frames_per_run=args.max_frames_per_run
    )
    stats["n_train_runs"] = len(train)
    stats["source"] = "full_train_split"
    out = root / "noise_stats.json"
    save_noise_stats(stats, out)
    digest = _sha256(out)
    (root / "noise_stats.sha256").write_text(f"{digest}  noise_stats.json\n")
    # link from checksums.json if present
    ck = root / "checksums.json"
    if ck.exists():
        obj = json.loads(ck.read_text())
        if isinstance(obj, dict):
            obj["noise_stats.json"] = digest
            ck.write_text(json.dumps(obj, indent=2))
    print(json.dumps({"path": str(out), "sha256": digest, "stats": stats}, indent=2))
    if args.upload_gcs:
        dest = args.upload_gcs.rstrip("/") + "/"
        subprocess.check_call(
            ["gcloud", "storage", "cp", str(out), str(root / "noise_stats.sha256"), dest]
        )
        print("uploaded to", dest)


if __name__ == "__main__":
    main()
