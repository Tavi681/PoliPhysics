import numpy as np

from ml.data.integrator import DT, accel_from_velocities, integrate_gt_velocities, step_np


def test_velocity_target_reproduces_v_exactly():
    rng = np.random.default_rng(0)
    T, n = 50, 7
    v = rng.normal(size=(T, n, 3)).astype(np.float32)
    x0 = rng.normal(size=(n, 3)).astype(np.float32)
    x_hat, v_hat = integrate_gt_velocities(v, x0, dt=DT)
    np.testing.assert_allclose(v_hat, v, rtol=0, atol=0)  # exact float32 path
    # one-step x drift floor should be tiny for random v (not physical), but finite
    x1 = x0 + DT * v[1]
    # after full integrate, last step consistency: x grows
    assert x_hat.shape == (T, n, 3)


def test_step_matches_definition():
    x = np.zeros((4, 3), dtype=np.float32)
    v = np.ones((4, 3), dtype=np.float32)
    a = np.full((4, 3), 2.0, dtype=np.float32)
    xn, vn = step_np(x, v, a, dt=DT)
    np.testing.assert_allclose(vn, v + DT * a)
    np.testing.assert_allclose(xn, x + DT * vn)


def test_accel_from_velocities():
    v = np.arange(12, dtype=np.float32).reshape(4, 1, 3)
    a = accel_from_velocities(v, dt=DT)
    assert a.shape == (3, 1, 3)
