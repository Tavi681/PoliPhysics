import numpy as np
import pandas as pd
import torch

from ml.data.graph import all_tips_edges, fully_connected, material_edges, radius_cross_thread
from ml.data.normalize import denormalize, normalize
from ml.data.splits import assert_no_leakage, build_splits
from ml.models import MODEL_REGISTRY, build_model


def _fake_index(n_per=20):
    rows = []
    i = 0
    for N in [1, 2, 4, 6, 8]:
        for sw in [0.15, 0.30, 0.22]:
            if sw == 0.22 and N not in (2, 4):
                continue
            if N in (6, 8) and sw == 0.22:
                continue
            src = "gen_sigma" if sw == 0.22 else ("gen_N" if N in (3, 6) and sw == 0.30 and N != 8 else "main")
            if N == 6:
                src = "gen_N"
            for x in [-1, 0, 1]:
                for _ in range(n_per // 3 + 1):
                    rows.append({
                        "run_id": f"N{N}_sw{sw}_x{x:+d}_i{i:04d}",
                        "N": N, "sigma_w": sw, "x_grid": x, "q": 1e-9, "q_nC": 1.0,
                        "n_frames": 100, "valid": True, "split_source": src if N != 8 else "main",
                    })
                    i += 1
    return pd.DataFrame(rows)


def test_no_split_leakage():
    cfg = {
        "seed": 42, "val_frac": 0.1, "test_interp_frac": 0.1,
        "train_val_pool": {"N": [1, 2, 4], "sigma_w": [0.15, 0.30], "sources": ["main"]},
        "test_unseenN": {"N": [6, 8], "sources": ["main", "gen_N"]},
        "test_unseenSigma": {"sigma_w": [0.22]},
    }
    splits = build_splits(_fake_index(), cfg)
    assert_no_leakage(splits)
    assert len(splits["train"]) > 0
    assert len(splits["test_unseenN"]) > 0


def test_material_edges_bidirectional_and_count():
    edges = np.array([[0, 1], [1, 2], [0, 3]], dtype=np.int64)
    e = material_edges(edges)
    assert e.shape[0] == 2
    assert e.shape[1] == 6  # 3 * 2
    # spider links present
    pairs = set(map(tuple, e.T.tolist()))
    assert (0, 1) in pairs and (1, 0) in pairs


def test_radius_and_tips():
    pos = torch.tensor([[0.0, 0, 0], [0.05, 0, 0], [1.0, 0, 0]], dtype=torch.float32)
    tid = torch.tensor([-1, 0, 1])
    e = radius_cross_thread(pos, tid, radius=0.1)
    assert e.shape[1] >= 2
    nt = torch.tensor([0, 2, 2])
    et = all_tips_edges(nt)
    assert et.shape[1] == 2
    assert fully_connected(3).shape[1] == 6


def test_normalize_roundtrip():
    mean = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    x = np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]], dtype=np.float32)
    z = normalize(x, mean, std)
    xr = denormalize(z, mean, std)
    np.testing.assert_allclose(xr, x, rtol=1e-6, atol=1e-6)


def test_model_registry():
    assert "stub_gnn" in MODEL_REGISTRY
    assert "constant_velocity" in MODEL_REGISTRY
    assert "follow_air" in MODEL_REGISTRY
    m = build_model("stub_gnn", node_dim=22, hidden=16)
    assert m.required_edges() == ["material"]
    assert m.count_parameters() > 0
