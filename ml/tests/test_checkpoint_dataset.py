import json
from pathlib import Path

import numpy as np
import torch

from ml.data.dataset import BallooningDataset, BallooningRunStore
from ml.models import build_model


def _write_synthetic_store(tmp: Path, N: int = 1, T: int = 40):
    n = 1 + N * 50
    # minimal topology: spider + chain
    edges = np.array([[i, i + 1] for i in range(n - 1)], dtype=np.int64)
    thread_id = np.full(n, 0, dtype=np.int32)
    thread_id[0] = -1
    node_type = np.ones(n, dtype=np.int8)
    node_type[0] = 0
    node_type[-1] = 2
    x_rel = np.zeros((T, n, 3), dtype=np.float32)
    for i in range(1, n):
        x_rel[:, i, 2] = -0.01 * i
    spider = np.zeros((T, 3), dtype=np.float64)
    spider[:, 2] = 0.5 - 0.001 * np.arange(T)
    v = np.zeros((T, n, 3), dtype=np.float32)
    v[:, :, 2] = -0.1
    u = np.zeros((T, n, 3), dtype=np.float32)
    u[:, :, 0] = 1.0
    ok = np.ones(T, dtype=bool)
    ok[:5] = False
    ok[-1] = False
    rec = {
        "run_id": f"synth_N{N}",
        "t": (np.arange(T) * 0.01).astype(np.float32),
        "x_rel": x_rel,
        "spider_abs": spider,
        "v": v,
        "u": u,
        "charge": np.zeros(n, dtype=np.float32),
        "thread_id": thread_id,
        "node_type": node_type,
        "edges": edges,
        "window_ok": ok,
        "globals": {
            "N": N, "N_t": 50, "sigma_w": 0.15, "ell": 1.0, "U_h": 1.0,
            "x_grid": 0, "Fbar_l": 1.0, "q": 1e-9, "m": 1e-6, "E": 7410.0,
            "L": 0.5, "z0": 0.5, "h": 2.0, "seed": 0, "split_source": "main",
            "outcome": "fall", "exit_time": 0.3, "wall_time_s": 1.0,
            "source_commit": "x", "source_dirty": False, "dt": 0.01,
            "n_frames": T, "n_nodes": n,
        },
    }
    from ml.data.shard_format import pack_records_consolidated
    torch.save(pack_records_consolidated([rec]), tmp / "shard_0000.pt")
    norm = {
        "v": {"mean": [0, 0, 0], "std": [1, 1, 1], "count": 1},
        "u": {"mean": [0, 0, 0], "std": [1, 1, 1], "count": 1},
        "x_rel": {"mean": [0, 0, 0], "std": [1, 1, 1], "count": 1},
        "a": {"mean": [0, 0, 0], "std": [1, 1, 1], "count": 1},
        "dt": 0.01,
        "target": "velocity",
    }
    (tmp / "norm_stats.json").write_text(json.dumps(norm))
    rid = f"synth_N{N}"
    man = {
        "data_version": "test",
        "shards": ["shard_0000.pt"],
        "n_runs": 1,
        "records": [{"run_id": rid, "shard": "shard_0000.pt"}],
    }
    (tmp / "manifest.json").write_text(json.dumps(man))
    return rid


def test_dataset_shapes_N1_N8(tmp_path):
    for N in (1, 8):
        d = tmp_path / f"N{N}"
        d.mkdir()
        rid = _write_synthetic_store(d, N=N)
        store = BallooningRunStore(d)
        ds = BallooningDataset(store, [rid], C=5, edge_sets=["material"])
        assert len(ds) > 0
        data = ds[0]
        n = 1 + N * 50
        assert data.pos.shape == (n, 3)
        assert data.y.shape == (n, 3)
        assert data.edge_index_material.shape[0] == 2
        assert data.edge_index_material.shape[1] == 2 * (n - 1)


def test_checkpoint_resume(tmp_path):
    model = build_model("stub_gnn", node_dim=22, hidden=8)
    ckpt = tmp_path / "last.pt"
    torch.save({"model": model.state_dict(), "step": 10}, ckpt)
    model2 = build_model("stub_gnn", node_dim=22, hidden=8)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model2.load_state_dict(state["model"])
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)
