"""Extract a run-id subset from a GCS (or local) consolidated processed pack.

Example (local_subset from 5c60fa with stored u_next):
  PYTHONPATH=. python -m ml.data.extract_subset \\
    --src-uri gs://authorship-verification-poliphysics/tema1-ml/processed/5c60fa669a719404 \\
    --run-ids-from \"Tema 1/results/ml/processed/local_subset/manifest.json\" \\
    --out \"Tema 1/results/ml/processed/local_subset\" \\
    --splits \"Tema 1/results/ml/splits/splits.json\"
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch

from ml.data.shard_format import load_shard, pack_records_consolidated
from ml.utils.git_info import git_provenance, repo_root


def _gcs_cp(src: str, dst: Path, *, timeout_s: float = 60.0) -> None:
    """Copy one GCS object; fall back to gcloud storage if gsutil stalls."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    gsutil = [
        "gsutil",
        "-o",
        "GSUtil:parallel_thread_count=1",
        "cp",
        src,
        str(dst),
    ]
    try:
        subprocess.check_call(gsutil, stdout=subprocess.DEVNULL, timeout=timeout_s)
        return
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        if dst.exists():
            dst.unlink(missing_ok=True)
    subprocess.check_call(
        ["gcloud", "storage", "cp", src, str(dst)],
        stdout=subprocess.DEVNULL,
        timeout=max(timeout_s, 120.0),
    )


def _gsutil_cp(src: str, dst: Path) -> None:
    _gcs_cp(src, dst)


def _gcs_cp_many(uris: list[str], dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    if not uris:
        return
    # gcloud storage avoids gsutil -m hangs on macOS; one invocation for all shards.
    subprocess.check_call(
        ["gcloud", "storage", "cp", *uris, str(dst_dir) + "/"],
    )


def _load_json_uri(uri: str, cache: Path) -> dict:
    if uri.startswith("gs://"):
        _gsutil_cp(uri, cache)
        return json.loads(cache.read_text())
    return json.loads(Path(uri).read_text())


def extract_subset(
    *,
    src_uri: str,
    run_ids: list[str],
    out_dir: Path,
    splits_path: Path | None = None,
    shard_size: int = 50,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    want = set(run_ids)
    tmp = Path(tempfile.mkdtemp(prefix="extract_subset_"))
    try:
        man = _load_json_uri(f"{src_uri.rstrip('/')}/manifest.json", tmp / "manifest.json")
        _load_json_uri(f"{src_uri.rstrip('/')}/norm_stats.json", tmp / "norm_stats.json")

        rid_to_shard: dict[str, str] = {}
        for r in man.get("records", []):
            if r["run_id"] in want:
                rid_to_shard[r["run_id"]] = r["shard"]
        missing = want - set(rid_to_shard)
        if missing:
            raise RuntimeError(f"run_ids not in source manifest: {sorted(missing)[:10]}")

        shards_needed = sorted(set(rid_to_shard.values()))
        shard_dir = tmp / "shards"
        if src_uri.startswith("gs://"):
            uris = [f"{src_uri.rstrip('/')}/{sh}" for sh in shards_needed]
            print(f"Downloading {len(uris)} shards via gcloud storage cp ...", flush=True)
            _gcs_cp_many(uris, shard_dir)
        else:
            shard_dir.mkdir(parents=True, exist_ok=True)
            for sh in shards_needed:
                shutil.copy2(Path(src_uri) / sh, shard_dir / sh)

        collected: list[dict] = []
        for sh in shards_needed:
            local = shard_dir / sh
            for rec in load_shard(local):
                if rec["run_id"] in want:
                    if man.get("store_u_next") and "u_next" not in rec:
                        raise RuntimeError(f"missing u_next for {rec['run_id']}")
                    collected.append(rec)
            local.unlink(missing_ok=True)

        collected.sort(key=lambda r: r["run_id"])
        if len(collected) != len(want):
            raise RuntimeError(f"expected {len(want)} runs, got {len(collected)}")

        for p in out_dir.glob("shard_*.pt"):
            p.unlink()

        shard_names: list[str] = []
        rid_shard: dict[str, str] = {}
        for i in range(0, len(collected), shard_size):
            chunk = collected[i : i + shard_size]
            name = f"shard_{i // shard_size:04d}.pt"
            torch.save(pack_records_consolidated(chunk), out_dir / name)
            shard_names.append(name)
            for r in chunk:
                rid_shard[r["run_id"]] = name

        shutil.copy2(tmp / "norm_stats.json", out_dir / "norm_stats.json")
        src_ver = man.get("data_version") or Path(src_uri.rstrip("/")).name
        (out_dir / "DATA_VERSION").write_text(f"{src_ver}_subset\n")

        records = []
        for r in collected:
            g = r["globals"]
            records.append({
                "run_id": r["run_id"],
                "shard": rid_shard[r["run_id"]],
                "N": int(g["N"]),
                "N_t": int(g["N_t"]),
                "sigma_w": float(g["sigma_w"]),
                "x_grid": int(g.get("x_grid", 0)),
                "n_frames": int(r["t"].shape[0]),
                "n_nodes": int(r["x_rel"].shape[1]),
                "outcome": g.get("outcome"),
                "seed": g.get("seed"),
                "split_source": g.get("split_source"),
                "wall_time_s": g.get("wall_time_s"),
            })

        out_man = {
            "data_version": f"{src_ver}_subset",
            "extracted_from": src_ver,
            "n_runs": len(collected),
            "shards": shard_names,
            "target": man.get("target", "velocity"),
            "dt": man.get("dt", 0.01),
            "store_v": True,
            "store_u_next": bool(man.get("store_u_next", True)),
            "shard_format": "consolidated_v1",
            "git": git_provenance(),
            "records": records,
        }
        (out_dir / "manifest.json").write_text(json.dumps(out_man, indent=2))
        return {"manifest": out_man, "n_shards": len(shard_names)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-uri", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--run-ids-from", type=Path, required=True)
    ap.add_argument("--splits", type=Path, default=None)
    ap.add_argument("--shard-size", type=int, default=50)
    args = ap.parse_args()
    man = json.loads(Path(args.run_ids_from).read_text())
    ids = [r["run_id"] for r in man["records"]]
    splits = args.splits
    if splits is None:
        cand = repo_root() / "Tema 1/results/ml/splits/splits.json"
        splits = cand if cand.exists() else None
    out = extract_subset(
        src_uri=args.src_uri,
        run_ids=ids,
        out_dir=args.out,
        splits_path=splits,
        shard_size=args.shard_size,
    )
    print(json.dumps({"n_runs": out["manifest"]["n_runs"], "n_shards": out["n_shards"]}, indent=2))


if __name__ == "__main__":
    main()
