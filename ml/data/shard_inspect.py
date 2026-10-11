"""Inspect torch.save overhead for processed shards (nbytes vs storage)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _walk(obj: Any, path: str, rows: list[dict]) -> None:
    if isinstance(obj, torch.Tensor):
        t = obj
        nbytes = int(t.nbytes)
        try:
            storage_nbytes = int(t.untyped_storage().nbytes())
        except Exception:
            storage_nbytes = int(t.storage().nbytes()) if t.storage() is not None else nbytes
        is_view = storage_nbytes > nbytes or (
            t.storage_offset() != 0 if hasattr(t, "storage_offset") else False
        )
        rows.append({
            "path": path,
            "kind": "torch.Tensor",
            "dtype": str(t.dtype),
            "shape": list(t.shape),
            "nbytes": nbytes,
            "storage_nbytes": storage_nbytes,
            "is_view_or_slice": bool(is_view),
            "storage_offset": int(t.storage_offset()) if hasattr(t, "storage_offset") else 0,
        })
        return
    if isinstance(obj, np.ndarray):
        base = obj.base
        base_nbytes = int(np.asarray(base).nbytes) if base is not None else int(obj.nbytes)
        rows.append({
            "path": path,
            "kind": "numpy.ndarray",
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
            "nbytes": int(obj.nbytes),
            "storage_nbytes": base_nbytes,
            "is_view_or_slice": bool(base is not None and base_nbytes > obj.nbytes),
            "storage_offset": 0,
        })
        return
    if isinstance(obj, (list, tuple)):
        rows.append({
            "path": path,
            "kind": type(obj).__name__,
            "dtype": f"len={len(obj)}",
            "shape": [],
            "nbytes": 0,
            "storage_nbytes": 0,
            "is_view_or_slice": False,
            "storage_offset": 0,
        })
        for i, x in enumerate(obj):
            _walk(x, f"{path}[{i}]", rows)
        return
    if isinstance(obj, dict):
        rows.append({
            "path": path,
            "kind": "dict",
            "dtype": f"keys={sorted(obj.keys())}",
            "shape": [],
            "nbytes": 0,
            "storage_nbytes": 0,
            "is_view_or_slice": False,
            "storage_offset": 0,
        })
        for k, v in obj.items():
            _walk(v, f"{path}.{k}" if path else str(k), rows)
        return
    rows.append({
        "path": path,
        "kind": type(obj).__name__,
        "dtype": repr(obj)[:80],
        "shape": [],
        "nbytes": 0,
        "storage_nbytes": 0,
        "is_view_or_slice": False,
        "storage_offset": 0,
    })


def inspect_shard(path: Path) -> dict:
    path = Path(path)
    file_size = path.stat().st_size
    pack = torch.load(path, weights_only=False)
    rows: list[dict] = []
    _walk(pack, "root", rows)
    arr_rows = [r for r in rows if r["nbytes"] > 0]

    # Second pass for accurate storage dedupe
    seen_ptrs: set[int] = set()
    sum_nbytes = 0
    sum_storage_dedup = 0
    sum_storage_raw = 0

    def acc(obj: Any) -> None:
        nonlocal sum_nbytes, sum_storage_dedup, sum_storage_raw
        if isinstance(obj, torch.Tensor):
            sum_nbytes += int(obj.nbytes)
            try:
                sn = int(obj.untyped_storage().nbytes())
                ptr = int(obj.untyped_storage().data_ptr())
            except Exception:
                sn = int(obj.nbytes)
                ptr = id(obj)
            sum_storage_raw += sn
            if ptr not in seen_ptrs:
                seen_ptrs.add(ptr)
                sum_storage_dedup += sn
        elif isinstance(obj, np.ndarray):
            sum_nbytes += int(obj.nbytes)
            base = obj.base if obj.base is not None else obj
            sn = int(np.asarray(base).nbytes)
            ptr = id(base)
            sum_storage_raw += sn
            if ptr not in seen_ptrs:
                seen_ptrs.add(ptr)
                sum_storage_dedup += sn
        elif isinstance(obj, (list, tuple)):
            for x in obj:
                acc(x)
        elif isinstance(obj, dict):
            for v in obj.values():
                acc(v)

    acc(pack)
    kinds = sorted({r["kind"] for r in rows})
    n_views = sum(1 for r in arr_rows if r["is_view_or_slice"])
    if isinstance(pack, list):
        n_runs = len(pack)
        fmt = "legacy_list"
    elif isinstance(pack, dict):
        n_runs = len(pack.get("run_ids", []))
        fmt = pack.get("format", "dict")
    else:
        n_runs = None
        fmt = type(pack).__name__
    return {
        "path": str(path),
        "file_size": file_size,
        "format": fmt,
        "n_runs": n_runs,
        "sum_tensor_nbytes": sum_nbytes,
        "sum_storage_nbytes_raw": sum_storage_raw,
        "sum_storage_nbytes_dedup": sum_storage_dedup,
        "overhead_file_vs_nbytes": file_size - sum_nbytes,
        "overhead_file_vs_storage_dedup": file_size - sum_storage_dedup,
        "n_array_objects": len(arr_rows),
        "n_views_or_slices": n_views,
        "kinds_present": kinds,
        "dtypes_present": sorted({r["dtype"] for r in arr_rows}),
        "has_python_list_payload": any(
            r["kind"] == "list" and str(r.get("dtype", "")).startswith("len=") for r in rows
        ),
        "objects": arr_rows[:200],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("shards", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    reports = [inspect_shard(p) for p in args.shards]
    text = json.dumps(reports, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
