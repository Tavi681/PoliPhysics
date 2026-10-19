"""Unit tests for shared paper rollout metrics."""
from __future__ import annotations

import math

import numpy as np
import pytest

from ml.metrics.rollout_metrics import (
    ERROR_CAP_ABS_MSE,
    ERROR_CAP_REL_MSE,
    PHYS_EDGE_TOL,
    aggregate_P_grid,
    aggregate_horizon_errors,
    aggregate_ttf,
    cap_mse,
    default_error_time_grid,
    error_time_grid,
    is_physical_at_horizon,
    is_physical_series,
    parse_x_from_run_id,
    segment_rel_deviations,
    time_to_failure,
)


def test_error_time_grid_shape():
    g = error_time_grid(60.0)
    assert g[0] == pytest.approx(0.1)
    assert g[99] == pytest.approx(10.0)
    assert g[100] == pytest.approx(11.0)
    assert g[-1] == pytest.approx(60.0)
    assert len(g) == 100 + (60 - 10)  # 0.1..10 + 11..60
    assert default_error_time_grid(60.0) == g


def test_segment_rel_deviations():
    # Two nodes distance 1.1, rest length 1.0 → rel = 0.1
    X = np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]], dtype=np.float64)
    edges = np.array([[0, 1]], dtype=np.int64)
    dev = segment_rel_deviations(X, edges, L_seg=1.0)
    assert dev.shape == (1,)
    assert float(dev[0]) == pytest.approx(0.1)


def test_ttf_stretching_rollout():
    """Edge grows past 1.1 ℓ0 at a known time → TTF matches."""
    L_seg = 1.0
    edges = np.array([[0, 1]], dtype=np.int64)
    # t = 0,1,2,3,4 with length = 1.0 + 0.05*t → at t=2 length=1.10 → rel=0.10
    ts = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    lengths = 1.0 + 0.05 * ts
    max_edge_rel = np.abs(lengths - L_seg) / L_seg
    finite = np.ones_like(ts, dtype=bool)
    # Also verify via segment_rel_deviations at failure frame
    X_fail = np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]])
    assert float(segment_rel_deviations(X_fail, edges, L_seg)[0]) == pytest.approx(0.1)
    assert float(max_edge_rel[2]) == pytest.approx(PHYS_EDGE_TOL)
    ttf = time_to_failure(ts, max_edge_rel, finite)
    assert ttf == pytest.approx(2.0)
    assert is_physical_at_horizon(ttf, 1.0)
    assert not is_physical_at_horizon(ttf, 2.0)
    assert not is_physical_at_horizon(ttf, 3.0)


def test_ttf_nonfinite_rollout():
    """NaN at frame k → TTF at that time."""
    ts = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    max_edge_rel = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    finite = np.array([True, True, False, False, False])  # NaN at frame k=2
    ttf = time_to_failure(ts, max_edge_rel, finite)
    assert ttf == pytest.approx(0.2)
    phys = is_physical_series(max_edge_rel, finite)
    assert phys.tolist() == [True, True, False, False, False]


def test_ttf_good_rollout_never_fails():
    """Never fails → TTF = +inf; physical True; uncapped errors."""
    ts = np.linspace(0.0, 5.0, 51)
    max_edge_rel = np.full_like(ts, 0.01)
    finite = np.ones_like(ts, dtype=bool)
    ttf = time_to_failure(ts, max_edge_rel, finite)
    assert math.isinf(ttf) and ttf > 0
    assert is_physical_at_horizon(ttf, 10.0)
    assert bool(is_physical_series(max_edge_rel, finite).all())

    # Uncapped small MSE
    v, capped = cap_mse(0.5, cap=ERROR_CAP_REL_MSE)
    assert v == pytest.approx(0.5)
    assert capped is False
    v2, capped2 = cap_mse(float("nan"), cap=ERROR_CAP_REL_MSE)
    assert v2 == pytest.approx(ERROR_CAP_REL_MSE)
    assert capped2 is True
    v3, capped3 = cap_mse(1e6, cap=ERROR_CAP_ABS_MSE)
    assert v3 == pytest.approx(ERROR_CAP_ABS_MSE)
    assert capped3 is True

    runs = [
        {
            "ttf_s": ttf,
            "horizon_errors": {
                "5s": {"rel_mse": 0.25, "spider_abs_mse": 0.1},
            },
        }
    ]
    agg = aggregate_horizon_errors(runs, [5.0])
    assert agg["5s"]["rel_mse_mean"] == pytest.approx(0.25)
    assert agg["5s"]["n_capped_rel"] == 0
    assert agg["5s"]["physical_fraction"] == pytest.approx(1.0)
    ttf_stats = aggregate_ttf(runs)
    assert math.isinf(ttf_stats["median"])
    assert ttf_stats["n_never_fail"] == 1


def test_parse_x_from_run_id():
    assert parse_x_from_run_id("N1_sw0.15_x+4_i0099") == 4
    assert parse_x_from_run_id("N8_sw0.30_x-2_i0001") == -2
    assert parse_x_from_run_id("N3_sw0.22_x+0_i0014") == 0


def test_p_grid_groups_by_x_not_charge():
    """Charge q ~ 1e-9 C rounds to 0 and must not pool distinct x values."""
    runs = []
    for x, n_surr_rise in ((4, 2), (-4, 0)):
        for i in range(2):
            runs.append({
                "run_id": f"N1_sw0.15_x{x:+d}_i{i:04d}",
                "x_grid": x,
                "N": 1,
                "q": 1e-9,
                "sigma_w": 0.15,
                "status": "rise" if i < n_surr_rise else "fall",
                "outcome_sim": "rise" if i == 0 else "timeout",
            })
    rows = aggregate_P_grid(runs)
    assert [r["x"] for r in rows] == [-4, 4]
    by_x = {r["x"]: r for r in rows}
    assert by_x[4]["n_runs"] == 2
    assert by_x[4]["P_surr"] == pytest.approx(1.0)
    assert by_x[4]["P_sim"] == pytest.approx(0.5)
    assert by_x[-4]["P_surr"] == pytest.approx(0.0)
    assert by_x[-4]["P_sim"] == pytest.approx(0.5)
    assert "q" not in by_x[4]
    bad = dict(runs[0], x_grid=0)
    with pytest.raises(ValueError, match="disagrees"):
        aggregate_P_grid([bad])
