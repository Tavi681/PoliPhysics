import numpy as np

from ml.data.integrator import DT, accel_from_velocities, integrate_gt_velocities, step_np


def test_velocity_target_reproduces_v_exactly():
    """Velocity-based target: stored v reproduced exactly; x within ~1.2e-7 m floor.

    On physical Stage B trajectories, check B measured one-step
    ||x_t + dt*v_{t+1} - x_{t+1}||_rms ≈ 1.2e-7 m (rollout metric floor).
    """
    rng = np.random.default_rng(0)
    T, n = 50, 7
    v = rng.normal(size=(T, n, 3)).astype(np.float32)
    x0 = rng.normal(size=(n, 3)).astype(np.float32)
    x_hat, v_hat = integrate_gt_velocities(v, x0, dt=DT)
    np.testing.assert_allclose(v_hat, v, rtol=0, atol=0)  # exact float32 v
    # one-step Euler identity vs an independently built x from the same rule
    x_ref = [x0.copy()]
    x = x0.copy()
    for t in range(T - 1):
        x = x + np.float32(DT) * v[t + 1]
        x_ref.append(x.copy())
    x_ref = np.stack(x_ref)
    np.testing.assert_allclose(x_hat, x_ref, rtol=0, atol=0)
    assert x_hat.shape == (T, n, 3)
    # Note: the ~1.2e-7 m RMS floor is measured on physical Stage B (x,v) in
    # check B (float32 identity vs stored x); not on O(1) synthetic noise.



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
