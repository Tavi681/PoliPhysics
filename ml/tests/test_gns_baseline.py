"""GNS / MeshGraphNet: shapes, invariance, edges, u_next leak test."""
from __future__ import annotations

import numpy as np
import torch

from ml.data.air import compute_u_next_run, make_ks, u_at
from ml.data.dataset import BallooningDataset, BallooningRunStore
from ml.data.edge_features import build_edge_sets_gns, interaction_edges
from ml.models import MODEL_REGISTRY, build_model


def _synth_rec(N: int = 1, T: int = 40, N_t: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = 1 + N * N_t
    edges = []
    thread_id = np.full(n, -1, dtype=np.int32)
    node_type = np.ones(n, dtype=np.int8)
    node_type[0] = 0
    idx = 1
    for th in range(N):
        prev = 0
        for k in range(N_t):
            edges.append([prev, idx])
            thread_id[idx] = th
            node_type[idx] = 2 if k == N_t - 1 else 1
            prev = idx
            idx += 1
    edges = np.asarray(edges, dtype=np.int64)
    x_rel = np.zeros((T, n, 3), dtype=np.float32)
    for i in range(1, n):
        x_rel[:, i, 2] = -0.01 * ((i - 1) % N_t + 1)
        x_rel[:, i, 0] = 0.02 * ((i - 1) // N_t)
    spider = np.zeros((T, 3), dtype=np.float64)
    spider[:, 2] = 0.5 - 0.001 * np.arange(T)
    spider[:, 0] = 0.01 * np.arange(T)
    v = np.zeros((T, n, 3), dtype=np.float32)
    v[:, :, 2] = -0.1
    u = np.zeros((T, n, 3), dtype=np.float32)
    u[:, :, 0] = 1.0
    ok = np.ones(T, dtype=bool)
    ok[:5] = False
    ok[-1] = False
    charge = np.full(n, 1e-9, dtype=np.float32)
    charge[0] = 0.0
    g = {
        "N": N, "N_t": N_t, "sigma_w": 0.15, "ell": 1.0, "U_h": 1.0,
        "x_grid": 0, "Fbar_l": 1.0, "q": 1e-9, "m": 1e-6, "E": 7410.0,
        "L": 0.5, "z0": 0.5, "h": 2.0, "seed": 7, "split_source": "main",
        "outcome": "fall", "exit_time": 0.3, "wall_time_s": 1.0,
        "dt": 0.01, "n_frames": T, "n_nodes": n,
    }
    return {
        "run_id": f"synth_N{N}",
        "t": (np.arange(T) * 0.01).astype(np.float32),
        "x_rel": x_rel,
        "spider_abs": spider,
        "v": v,
        "u": u,
        "charge": charge,
        "thread_id": thread_id,
        "node_type": node_type,
        "edges": edges,
        "window_ok": ok,
        "globals": g,
    }


def _write_store(tmp, recs, with_unext=True):
    from ml.data.shard_format import pack_records_consolidated
    import json
    packed = []
    for r in recs:
        rr = dict(r)
        if with_unext and "u_next" not in rr:
            # placeholder (same shape as u); real KS checked in test_u_next_leak_three_runs
            rr["u_next"] = (rr["u"] + 0.01).astype(np.float32)
        packed.append(rr)
    torch.save(pack_records_consolidated(packed), tmp / "shard_0000.pt")
    norm = {
        "v": {"mean": [0, 0, 0], "std": [1, 1, 1], "count": 1},
        "u": {"mean": [0, 0, 0], "std": [1, 1, 1], "count": 1},
        "x_rel": {"mean": [0, 0, 0], "std": [1, 1, 1], "count": 1},
        "a": {"mean": [0, 0, 0], "std": [1, 1, 1], "count": 1},
        "dt": 0.01, "target": "velocity",
    }
    (tmp / "norm_stats.json").write_text(json.dumps(norm))
    man = {
        "data_version": "test", "shards": ["shard_0000.pt"], "n_runs": len(packed),
        "store_u_next": True,
        "records": [
            {"run_id": r["run_id"], "shard": "shard_0000.pt", "N": r["globals"]["N"],
             "n_frames": r["globals"]["n_frames"], "n_nodes": r["globals"]["n_nodes"]}
            for r in packed
        ],
    }
    (tmp / "manifest.json").write_text(json.dumps(man))
    return [r["run_id"] for r in packed]


def test_registry():
    assert "gns_baseline" in MODEL_REGISTRY
    assert "gns_local" in MODEL_REGISTRY


def test_forward_shapes_N1_4_8(tmp_path):
    for N in (1, 4, 8):
        d = tmp_path / f"N{N}"
        d.mkdir()
        ids = _write_store(d, [_synth_rec(N=N, N_t=8, T=30)])
        store = BallooningRunStore(d)
        ds = BallooningDataset(
            store, ids, C=5,
            node_features=["v_hist", "u", "u_next", "charge"],
            globals_fields=["qE", "m", "N"],
            edge_sets=["material", "interaction"],
            radius=0.05,
        )
        assert len(ds) > 0
        sample = ds[0]
        m = build_model(
            "gns_baseline",
            node_dim=int(sample.x.shape[-1]),
            edge_sets=["material", "interaction"],
            edge_dims={"material": 5, "interaction": 7},
            latent=32, n_mp_steps=2,
        )
        batch = sample
        out = m(batch)
        n = sample.num_nodes
        assert out.shape == (n, 3)


def test_translation_invariance(tmp_path):
    d = tmp_path / "trans"
    d.mkdir()
    ids = _write_store(d, [_synth_rec(N=2, N_t=8, T=30)])
    store = BallooningRunStore(d)
    ds = BallooningDataset(
        store, ids, C=5,
        node_features=["v_hist", "u", "u_next", "charge"],
        globals_fields=["qE", "m", "N"],
        edge_sets=["material", "interaction"],
        radius=0.05,
    )
    sample = ds[0]
    m = build_model(
        "gns_baseline", node_dim=int(sample.x.shape[-1]),
        edge_sets=["material", "interaction"],
        edge_dims={"material": 5, "interaction": 7},
        latent=32, n_mp_steps=2,
    )
    m.eval()
    with torch.no_grad():
        y0 = m(sample)
        # shift absolute frame via spider; relative pos unchanged → same edge feats on pos
        sample2 = sample.clone()
        sample2.spider_abs = sample.spider_abs + torch.tensor([3.0, -2.0, 0.0])
        # node features / relative pos unchanged
        y1 = m(sample2)
        assert torch.allclose(y0, y1, atol=1e-5, rtol=1e-5)


def test_thread_permutation(tmp_path):
    d = tmp_path / "perm"
    d.mkdir()
    ids = _write_store(d, [_synth_rec(N=2, N_t=6, T=30)])
    store = BallooningRunStore(d)
    ds = BallooningDataset(
        store, ids, C=5,
        node_features=["v_hist", "u", "charge"],
        globals_fields=["qE", "m", "N"],
        edge_sets=["material"],
        radius=0.05,
    )
    sample = ds[0]
    m = build_model(
        "gns_local", node_dim=int(sample.x.shape[-1]),
        edge_sets=["material"], edge_dims={"material": 5},
        latent=32, n_mp_steps=2,
    )
    m.eval()
    n = sample.num_nodes
    N_t = 6
    perm = torch.arange(n)
    a = list(range(1, 1 + N_t))
    b = list(range(1 + N_t, 1 + 2 * N_t))
    for i, j in zip(a, b):
        perm[i], perm[j] = int(j), int(i)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(n)

    from ml.data.edge_features import material_edge_attr
    from torch_geometric.data import Data

    def gather(data, p):
        ei = inv[data.edge_index_material]
        pos = data.pos[p]
        return Data(
            x=data.x[p], pos=pos, node_type=data.node_type[p],
            thread_id=data.thread_id[p], charge=data.charge[p],
            edge_index=ei, edge_index_material=ei,
            edge_attr_material=material_edge_attr(pos, ei, l0=0.5 / N_t),
            num_nodes=n,
        )

    with torch.no_grad():
        y0 = m(sample)
        y1 = m(gather(sample, perm))
        assert torch.allclose(y1, y0[perm], atol=1e-4, rtol=1e-4)


def test_edge_counts_N1_3_4_6_8():
    rows = []
    for N in (1, 3, 4, 6, 8):
        rec = _synth_rec(N=N, N_t=50, T=10)
        pos = torch.from_numpy(rec["x_rel"][0])
        pos = pos + torch.from_numpy(rec["spider_abs"][0].astype(np.float32))
        tid = torch.from_numpy(rec["thread_id"].astype(np.int64))
        nt = torch.from_numpy(rec["node_type"].astype(np.int64))
        ed = build_edge_sets_gns(
            ["material", "interaction"],
            edges_material=rec["edges"], pos=pos, thread_id=tid, node_type=nt,
            radius=0.05,
        )
        n_mat = int(ed["material"].shape[1])
        n_int = int(ed["interaction"].shape[1])
        rows.append((N, n_mat, n_int))
        assert n_mat == N * 50 * 2
        ei = ed["interaction"]
        if ei.numel():
            src, dst = ei[0], ei[1]
            # no spider, no same-thread pairs
            assert (tid[src] >= 0).all() and (tid[dst] >= 0).all()
            assert (tid[src] != tid[dst]).all()
            assert (nt[src] != 0).all() and (nt[dst] != 0).all()
        if N == 1:
            assert n_int == 0
    print("edge_counts", rows)


def test_interaction_N1_is_zero_no_spider_same_thread():
    rec = _synth_rec(N=1, N_t=50, T=10)
    pos = torch.from_numpy(rec["x_rel"][0]) + torch.from_numpy(
        rec["spider_abs"][0].astype(np.float32)
    )
    tid = torch.from_numpy(rec["thread_id"].astype(np.int64))
    nt = torch.from_numpy(rec["node_type"].astype(np.int64))
    ei, flags = interaction_edges(pos, tid, nt, radius=0.05)
    assert ei.shape[1] == 0
    assert flags.numel() == 0


def test_u_next_leak_three_runs():
    """u_next equals KS(x_t, t+dt); differs from u_air[t+1]."""
    for seed, N in enumerate((1, 2, 4)):
        rec = _synth_rec(N=N, N_t=4, T=8, seed=seed)
        flow = make_ks(rec["globals"])
        T = rec["t"].shape[0]
        u_air = np.zeros_like(rec["u"])
        for i in range(T):
            x_abs = rec["spider_abs"][i] + rec["x_rel"][i].astype(np.float64)
            u_air[i] = u_at(flow, x_abs, float(rec["t"][i]))
        rec["u"] = u_air
        # only compute u_next for the frames we check (faster)
        for i in (0, T // 2, max(0, T - 2)):
            x_abs = rec["spider_abs"][i] + rec["x_rel"][i].astype(np.float64)
            direct = u_at(flow, x_abs, float(rec["t"][i]) + 0.01)
            # via helper path used by preprocessing
            from ml.data.air import u_next_frame
            got = u_next_frame({**rec, "u_next": None} if False else rec, i, flow=flow)
            # force on-the-fly (no stored field)
            rec2 = {k: v for k, v in rec.items() if k != "u_next"}
            got = u_next_frame(rec2, i, flow=flow)
            assert np.max(np.abs(direct - got)) == 0.0
            if i + 1 < T:
                assert np.max(np.abs(got - u_air[i + 1])) > 0.0
