"""FollowAir as massless tracers for every node, spider included."""
from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from ml.data.integrator import DT, step_np
from ml.models.baselines import FollowAir


def _thread_geometry(*, z0: float = 5.0, L: float = 0.5, n_seg: int = 4):
    """Spider at (0,0,z0); one thread of n_seg segments along −z (relative)."""
    n = n_seg + 1
    x_rel = np.zeros((n, 3), dtype=np.float32)
    for i in range(1, n):
        x_rel[i, 2] = -float(i) * (L / n_seg)
    spider = np.array([0.0, 0.0, z0], dtype=np.float64)
    edges = np.array([[i, i + 1] for i in range(n_seg)], dtype=np.int64)
    node_type = np.zeros(n, dtype=np.int64)
    node_type[1:-1] = 1
    node_type[-1] = 2
    l0 = L / n_seg
    return x_rel, spider, edges, node_type, l0


def _rollout_constant_u(
    u_const: np.ndarray,
    *,
    z0: float = 5.0,
    h: float = 10.0,
    L: float = 0.5,
    t_end: float = 60.0,
) -> dict:
    """Integrate FollowAir under spatially uniform air ``u_const`` (paper path)."""
    x_rel, spider, edges, node_type, l0 = _thread_geometry(z0=z0, L=L)
    v = np.zeros_like(x_rel)
    model = FollowAir(a_mean=torch.zeros(3), a_std=torch.ones(3), dt=DT)
    model.eval()
    u = np.broadcast_to(np.asarray(u_const, dtype=np.float32).reshape(1, 3), x_rel.shape).copy()
    status = "timeout"
    exit_time = float("nan")
    t = 0.0
    edge_lens0 = None
    x_abs0 = spider[None, :] + x_rel.astype(np.float64)
    edge_lens0 = np.linalg.norm(
        x_abs0[edges[:, 1]] - x_abs0[edges[:, 0]], axis=-1
    )
    max_edge_drift = 0.0
    while t < t_end - 1e-12:
        data = Data(
            x=torch.zeros(x_rel.shape[0], 1),
            v_t=torch.from_numpy(v),
            u_t=torch.from_numpy(u),
            node_type=torch.from_numpy(node_type),
            num_nodes=x_rel.shape[0],
        )
        a_n = model(data).detach().cpu().numpy().astype(np.float32)
        a = a_n  # a_mean=0, a_std=1
        x_rel, v = step_np(x_rel, v, a, dt=DT)
        spider = spider + DT * v[0].astype(np.float64)
        x_rel = (x_rel - x_rel[0:1]).astype(np.float32)
        t = t + DT
        x_abs = spider[None, :] + x_rel.astype(np.float64)
        lens = np.linalg.norm(x_abs[edges[:, 1]] - x_abs[edges[:, 0]], axis=-1)
        max_edge_drift = max(max_edge_drift, float(np.max(np.abs(lens - edge_lens0) / edge_lens0)))
        if float(spider[2]) >= h:
            status = "rise"
            exit_time = t
            break
        if float(spider[2]) <= 0.0:
            status = "fall"
            exit_time = t
            break
    return {
        "status": status,
        "exit_time": exit_time,
        "spider": spider.copy(),
        "x_rel": x_rel.copy(),
        "v": v.copy(),
        "max_edge_drift": max_edge_drift,
        "t": t,
    }


def test_follow_air_forward_sets_all_nodes_to_u():
    n = 5
    u = torch.tensor([[0.1, -0.2, 0.3]] * n, dtype=torch.float32)
    v = torch.zeros(n, 3)
    model = FollowAir(a_mean=torch.zeros(3), a_std=torch.ones(3), dt=DT)
    data = Data(
        x=torch.zeros(n, 1),
        v_t=v,
        u_t=u,
        node_type=torch.tensor([0, 1, 1, 1, 2]),
        num_nodes=n,
    )
    a = model(data)
    v_next = v + DT * a
    torch.testing.assert_close(v_next, u)


def test_uniform_updraft_rise_and_rigid_edges():
    w = 1.0
    z0, h = 5.0, 10.0
    out = _rollout_constant_u(np.array([0.0, 0.0, w]), z0=z0, h=h, t_end=60.0)
    assert out["status"] == "rise"
    t_expected = (h - z0) / w
    # One Euler step of size DT: exit when z first crosses h.
    assert abs(out["exit_time"] - t_expected) <= DT + 1e-9
    assert abs(float(out["spider"][2]) - h) <= w * DT + 1e-6
    assert out["max_edge_drift"] < 1e-5
    # Relative shape unchanged under uniform advection.
    x_rel0 = _thread_geometry(z0=z0)[0]
    np.testing.assert_allclose(out["x_rel"], x_rel0, atol=1e-5)


def test_uniform_horizontal_wind_preserves_relative_shape():
    out = _rollout_constant_u(np.array([2.0, -1.0, 0.0]), z0=5.0, h=10.0, t_end=1.0)
    assert out["status"] == "timeout"
    x_rel0 = _thread_geometry(z0=5.0)[0]
    np.testing.assert_allclose(out["x_rel"], x_rel0, atol=1e-5)
    assert out["max_edge_drift"] < 1e-5
    # Spider has drifted horizontally with the wind; height unchanged.
    assert abs(float(out["spider"][2]) - 5.0) < 1e-5
    assert abs(float(out["spider"][0]) - 2.0 * out["t"]) < 1e-4


def test_zero_air_timeout_nothing_moves():
    x_rel0, spider0, *_ = _thread_geometry(z0=5.0)
    out = _rollout_constant_u(np.zeros(3), z0=5.0, h=10.0, t_end=2.0)
    assert out["status"] == "timeout"
    np.testing.assert_allclose(out["x_rel"], x_rel0, atol=1e-6)
    np.testing.assert_allclose(out["spider"], spider0, atol=1e-6)
    np.testing.assert_allclose(out["v"], 0.0, atol=1e-6)
