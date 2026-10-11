"""Noise scale vs normalizer: global and per-type added variance checks."""
from __future__ import annotations

import numpy as np

from ml.data.noise_stats import a_std_by_type_matrix, resolve_noise


def test_resolve_noise_global_matches_a_std_dt():
    a_std = np.array([0.6, 0.5, 0.4], dtype=np.float32)
    dt = 0.01
    scalar, per_type, comp = resolve_noise(
        noise_k=1.0, noise_std=None, noise_mode="global", a_std=a_std, dt=dt
    )
    assert per_type is None
    assert comp is not None
    np.testing.assert_allclose(comp, a_std * dt, rtol=1e-6)


def test_resolve_noise_per_type_matrix():
    a_std = np.array([0.6, 0.5, 0.4], dtype=np.float32)
    a_type = np.array(
        [
            [0.1, 0.08, 0.07],  # spider smaller
            [0.6, 0.5, 0.4],
            [0.7, 0.55, 0.42],
        ],
        dtype=np.float32,
    )
    dt = 0.01
    scalar, mat, comp = resolve_noise(
        noise_k=1.0,
        noise_std=None,
        noise_mode="per_type",
        a_std=a_std,
        a_std_by_type=a_type,
        dt=dt,
    )
    assert comp is None
    assert mat is not None
    np.testing.assert_allclose(mat, a_type * dt, rtol=1e-6)


def test_noisy_target_adds_unit_normalized_variance_global():
    """Synthetic GNS target: a = (v_tp1 - v_noisy_t)/dt; global noise → +1 var."""
    rng = np.random.default_rng(0)
    C = 5
    k = 1.0
    dt = 0.01
    a_std = np.array([0.6, 0.5, 0.4], dtype=np.float64)
    a_mean = np.zeros(3)
    vel_std = k * a_std * dt
    n_trials = 20000
    n_nodes = 8

    a_clean = rng.normal(0.0, 1.0, size=(n_trials, n_nodes, 3)) * a_std
    v_t = rng.normal(0.0, 0.2, size=(n_trials, n_nodes, 3))
    v_tp1 = v_t + a_clean * dt

    eps = rng.normal(0.0, 1.0, size=(n_trials, C, n_nodes, 3))
    eps = eps * (vel_std.reshape(1, 1, 1, 3) / np.sqrt(C))
    noise_rw = np.cumsum(eps, axis=1)
    v_noisy_t = v_t + noise_rw[:, -1]
    a_noisy = (v_tp1 - v_noisy_t) / dt

    y_clean = (a_clean - a_mean) / a_std
    y_noisy = (a_noisy - a_mean) / a_std
    added = y_noisy.var(axis=(0, 1)) - y_clean.var(axis=(0, 1))
    assert np.all(np.abs(added - 1.0) < 0.08), added


def test_per_type_added_normalized_variance():
    """For each type/component, added var ≈ (a_std_type[c] / a_std[c])^2 at k=1."""
    rng = np.random.default_rng(1)
    C = 5
    k = 1.0
    dt = 0.01
    a_std = np.array([0.6, 0.5, 0.4], dtype=np.float64)
    a_mean = np.zeros(3)
    a_type = np.array(
        [
            [0.08, 0.06, 0.05],  # spider ~8x smaller
            [0.60, 0.50, 0.40],
            [0.70, 0.55, 0.45],
        ],
        dtype=np.float64,
    )
    vel_type = k * a_type * dt  # (3,3)
    n_trials = 8000
    # one node per type
    n_nodes = 3
    node_type = np.array([0, 1, 2])

    a_clean = rng.normal(0.0, 1.0, size=(n_trials, n_nodes, 3)) * a_std.reshape(1, 1, 3)
    v_t = rng.normal(0.0, 0.2, size=(n_trials, n_nodes, 3))
    v_tp1 = v_t + a_clean * dt

    scale = vel_type[node_type]  # (3,3)
    eps = rng.normal(0.0, 1.0, size=(n_trials, C, n_nodes, 3))
    eps = eps * (scale.reshape(1, 1, n_nodes, 3) / np.sqrt(C))
    noise_rw = np.cumsum(eps, axis=1)
    a_noisy = (v_tp1 - (v_t + noise_rw[:, -1])) / dt

    y_clean = (a_clean - a_mean) / a_std
    y_noisy = (a_noisy - a_mean) / a_std
    expected = (a_type / a_std.reshape(1, 3)) ** 2
    for typ in range(3):
        added = y_noisy[:, typ, :].var(axis=0) - y_clean[:, typ, :].var(axis=0)
        # sampling error tolerance
        assert np.all(np.abs(added - expected[typ]) < 0.12), (typ, added, expected[typ])


def test_a_std_by_type_matrix_loader():
    norm = {
        "a": {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0], "count": 1},
        "a_by_type": {
            "spider": {"mean": [0.0, 0.0, 0.0], "std": [0.1, 0.2, 0.3], "count": 10},
            "interior": {"mean": [0.0, 0.0, 0.0], "std": [0.4, 0.5, 0.6], "count": 20},
            "tip": {"mean": [0.0, 0.0, 0.0], "std": [0.7, 0.8, 0.9], "count": 30},
        },
    }
    mat = a_std_by_type_matrix(norm)
    assert mat is not None
    np.testing.assert_allclose(mat[0], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(mat[2], [0.7, 0.8, 0.9])
