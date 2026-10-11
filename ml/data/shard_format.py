"""Consolidated shard format: one contiguous tensor per field per shard.

On disk (torch.save dict):
  format: "consolidated_v1"
  run_ids: list[str]
  globals: list[dict]
  n_frames: int32 [R]
  n_nodes: int32 [R]
  frame_ptr: int64 [R+1]  # into t / spider_abs / window_ok
  node_ptr: int64 [R+1]   # into x_rel / v / u / u_next  (frame-major: T*n)
  static_ptr: int64 [R+1] # into charge / thread_id / node_type
  edge_ptr: int64 [R+1]   # into edges (E, 2)
  t, spider_abs, window_ok, x_rel, v, u, [u_next], charge, thread_id, node_type, edges

Loading expands to the legacy per-run dict API used by BallooningRunStore.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ml.data.preprocess import materialize_array

FORMAT = "consolidated_v1"

TIME_KEYS = ("t", "spider_abs", "window_ok")
NODE_KEYS = ("x_rel", "v", "u", "u_next")  # u_next optional
STATIC_KEYS = ("charge", "thread_id", "node_type")


def _as_np(x: Any) -> np.ndarray:
    return materialize_array(x) if not isinstance(x, np.ndarray) else np.ascontiguousarray(x).copy()


def pack_records_consolidated(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build one consolidated shard dict from per-run records (must include arrays)."""
    run_ids = [r["run_id"] for r in records]
    globals_ = [dict(r["globals"]) for r in records]
    n_frames = np.asarray([int(r["t"].shape[0]) for r in records], dtype=np.int32)
    n_nodes = np.asarray([int(r["x_rel"].shape[1]) for r in records], dtype=np.int32)
    frame_ptr = np.zeros(len(records) + 1, dtype=np.int64)
    node_ptr = np.zeros(len(records) + 1, dtype=np.int64)
    static_ptr = np.zeros(len(records) + 1, dtype=np.int64)
    edge_ptr = np.zeros(len(records) + 1, dtype=np.int64)
    frame_ptr[1:] = np.cumsum(n_frames)
    node_ptr[1:] = np.cumsum(n_frames.astype(np.int64) * n_nodes.astype(np.int64))
    static_ptr[1:] = np.cumsum(n_nodes)
    edge_counts = np.asarray([int(np.asarray(r["edges"]).shape[0]) for r in records], dtype=np.int64)
    edge_ptr[1:] = np.cumsum(edge_counts)

    out: dict[str, Any] = {
        "format": FORMAT,
        "run_ids": run_ids,
        "globals": globals_,
        "n_frames": n_frames,
        "n_nodes": n_nodes,
        "frame_ptr": frame_ptr,
        "node_ptr": node_ptr,
        "static_ptr": static_ptr,
        "edge_ptr": edge_ptr,
    }

    # time-like
    out["t"] = np.concatenate([_as_np(r["t"]).astype(np.float32, copy=False) for r in records], axis=0)
    out["spider_abs"] = np.concatenate(
        [_as_np(r["spider_abs"]).astype(np.float64, copy=False) for r in records], axis=0
    )
    out["window_ok"] = np.concatenate(
        [_as_np(r["window_ok"]).astype(bool, copy=False) for r in records], axis=0
    )

    # (T,n,3) → flat (sum T*n, 3)
    for k in ("x_rel", "v", "u"):
        out[k] = np.concatenate(
            [_as_np(r[k]).astype(np.float32, copy=False).reshape(-1, 3) for r in records],
            axis=0,
        )
    if any("u_next" in r for r in records):
        out["u_next"] = np.concatenate(
            [_as_np(r["u_next"]).astype(np.float32, copy=False).reshape(-1, 3) for r in records],
            axis=0,
        )

    out["charge"] = np.concatenate(
        [_as_np(r["charge"]).astype(np.float32, copy=False).reshape(-1) for r in records], axis=0
    )
    out["thread_id"] = np.concatenate(
        [_as_np(r["thread_id"]).astype(np.int32, copy=False).reshape(-1) for r in records], axis=0
    )
    out["node_type"] = np.concatenate(
        [_as_np(r["node_type"]).astype(np.int8, copy=False).reshape(-1) for r in records], axis=0
    )
    out["edges"] = np.concatenate(
        [_as_np(r["edges"]).astype(np.int64, copy=False).reshape(-1, 2) for r in records],
        axis=0,
    ) if int(edge_ptr[-1]) > 0 else np.zeros((0, 2), dtype=np.int64)

    # Store as torch tensors so zip serialization shares storages cleanly
    for k, v in list(out.items()):
        if isinstance(v, np.ndarray):
            out[k] = torch.from_numpy(np.ascontiguousarray(v))
    return out


def expand_consolidated(blob: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand consolidated shard to list of per-run dicts (numpy arrays)."""
    if blob.get("format") != FORMAT:
        raise ValueError(f"not consolidated: {blob.get('format')}")
    run_ids = blob["run_ids"]
    globals_ = blob["globals"]
    n_frames = np.asarray(blob["n_frames"])
    n_nodes = np.asarray(blob["n_nodes"])
    frame_ptr = np.asarray(blob["frame_ptr"])
    node_ptr = np.asarray(blob["node_ptr"])
    static_ptr = np.asarray(blob["static_ptr"])
    edge_ptr = np.asarray(blob["edge_ptr"])

    def arr(key: str) -> np.ndarray:
        v = blob[key]
        return v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v)

    t = arr("t")
    spider = arr("spider_abs")
    wok = arr("window_ok")
    x_rel = arr("x_rel")
    v = arr("v")
    u = arr("u")
    u_next = arr("u_next") if "u_next" in blob else None
    charge = arr("charge")
    thread_id = arr("thread_id")
    node_type = arr("node_type")
    edges = arr("edges")

    out = []
    for i, rid in enumerate(run_ids):
        T = int(n_frames[i])
        n = int(n_nodes[i])
        f0, f1 = int(frame_ptr[i]), int(frame_ptr[i + 1])
        n0, n1 = int(node_ptr[i]), int(node_ptr[i + 1])
        s0, s1 = int(static_ptr[i]), int(static_ptr[i + 1])
        e0, e1 = int(edge_ptr[i]), int(edge_ptr[i + 1])
        rec: dict[str, Any] = {
            "run_id": rid,
            "globals": dict(globals_[i]),
            "t": np.ascontiguousarray(t[f0:f1]),
            "spider_abs": np.ascontiguousarray(spider[f0:f1]),
            "window_ok": np.ascontiguousarray(wok[f0:f1]),
            "x_rel": np.ascontiguousarray(x_rel[n0:n1].reshape(T, n, 3)),
            "v": np.ascontiguousarray(v[n0:n1].reshape(T, n, 3)),
            "u": np.ascontiguousarray(u[n0:n1].reshape(T, n, 3)),
            "charge": np.ascontiguousarray(charge[s0:s1]),
            "thread_id": np.ascontiguousarray(thread_id[s0:s1]),
            "node_type": np.ascontiguousarray(node_type[s0:s1]),
            "edges": np.ascontiguousarray(edges[e0:e1]),
        }
        if u_next is not None:
            rec["u_next"] = np.ascontiguousarray(u_next[n0:n1].reshape(T, n, 3))
        out.append(rec)
    return out


def load_shard(path) -> list[dict[str, Any]]:
    """Load shard as list of run dicts (supports legacy list and consolidated_v1)."""
    obj = torch.load(path, weights_only=False)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and obj.get("format") == FORMAT:
        return expand_consolidated(obj)
    raise TypeError(f"unknown shard type {type(obj)} at {path}")
