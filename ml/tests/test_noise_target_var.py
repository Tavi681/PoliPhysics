"""With k=1 and vel noise = k*a_std*dt, normalized target gains ~1 variance per component."""
from __future__ import annotations

import numpy as np

from ml.data.noise_stats import resolve_noise


def test_resolve_noise_matches_a_std_dt():
    a_std = np.array([0.6, 0.5, 0.4], dtype=np.float32)
    dt = 0.01
    scalar, per_type, comp = resolve_noise(
        noise_k=1.0, noise_std=None, noise_stats=None, a_std=a_std, dt=dt
    )
    assert per_type is None
    assert comp is not None
    np.testing.assert_allclose(comp, a_std * dt, rtol=1e-6)


def test_noisy_target_adds_unit_normalized_variance():
    """Synthetic GNS target adjustment: a = (v_tp1 - v_noisy_t)/dt."""
    rng = np.random.default_rng(0)
    C = 5
    k = 1.0
    dt = 0.01
    a_std = np.array([0.6, 0.5, 0.4], dtype=np.float64)
    a_mean = np.zeros(3)
    vel_std = k * a_std * dt  # per component at latest velocity
    n_trials = 20000
    n_nodes = 8

    # clean a ~ N(0, a_std) so normalized clean has unit variance
    a_clean = rng.normal(0.0, 1.0, size=(n_trials, n_nodes, 3)) * a_std
    v_t = rng.normal(0.0, 0.2, size=(n_trials, n_nodes, 3))
    v_tp1 = v_t + a_clean * dt

    # RW over C steps with std at end = vel_std
    eps = rng.normal(0.0, 1.0, size=(n_trials, C, n_nodes, 3))
    eps = eps * (vel_std.reshape(1, 1, 1, 3) / np.sqrt(C))
    noise_rw = np.cumsum(eps, axis=1)
    v_noisy_t = v_t + noise_rw[:, -1]
    a_noisy = (v_tp1 - v_noisy_t) / dt

    y_clean = (a_clean - a_mean) / a_std
    y_noisy = (a_noisy - a_mean) / a_std
    # Added variance ≈ var(y_noisy) - var(y_clean); each component should be ~1
    added = y_noisy.var(axis=(0, 1)) - y_clean.var(axis=(0, 1))
    print(f"added_normalized_var_per_component={added.tolist()}", flush=True)
    assert np.all(np.abs(added - 1.0) < 0.08), added
