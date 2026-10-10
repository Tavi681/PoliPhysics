"""Unit tests for duration-normalized speed-up and Q*."""
import math

from ml.metrics.speedup import (
    mean_speedup,
    q_star,
    scale_surr_to_full_length,
    sim_seconds,
    speedup_per_run,
    wall_per_sim_second,
)


def test_sim_seconds_exit_and_timeout():
    assert sim_seconds(12.5, t_end=60.0) == 12.5
    assert sim_seconds(-1.0, t_end=60.0) == 60.0
    assert sim_seconds(float("nan"), t_end=60.0) == 60.0
    assert sim_seconds(None, t_end=60.0) == 60.0


def test_speedup_same_duration_synthetic():
    # Sim: 100 s wall for 10 s simulated → 10 wall/sim-s
    # Surr: 2 s wall for 10 s simulated → 0.2 wall/sim-s → speedup 50
    s = speedup_per_run(100.0, 10.0, 2.0, 10.0)
    assert abs(s - 50.0) < 1e-12


def test_speedup_different_durations_synthetic():
    # Sim: 200 s wall / 20 s sim = 10 /s
    # Surr smoke: 1 s wall / 5 s sim = 0.2 /s → speedup still 50
    s = speedup_per_run(200.0, 20.0, 1.0, 5.0)
    assert abs(s - 50.0) < 1e-12


def test_mean_speedup_skips_nan_sim_wall():
    m = mean_speedup(
        [100.0, float("nan"), 50.0],
        [10.0, 10.0, 5.0],
        [2.0, 2.0, 1.0],
        [10.0, 10.0, 5.0],
    )
    # first: 50, third: 50 → mean 50
    assert abs(m - 50.0) < 1e-12


def test_q_star_and_scale():
    # Full-length: t_sim=100, t_surr=2 → denom 98; C=980 → Q*=10
    assert abs(q_star(800.0, 180.0, 100.0, 2.0) - 10.0) < 1e-12
    assert math.isnan(q_star(100.0, 100.0, 10.0, 10.0))
    # smoke 5 s at 0.2 wall/sim-s → 12 wall-s for 60 s full
    assert abs(scale_surr_to_full_length(1.0, 5.0, 60.0) - 12.0) < 1e-12
    assert abs(wall_per_sim_second(10.0, 2.0) - 5.0) < 1e-12
