"""Validation Test 3 (pytest): N-thread junction vs transverse-corrected theory.

The collinear case N=2 matches the 1D theory (and ref/riemann.junction) exactly.
For N>=3 the full 2D simulator radiates transverse waves into the off-axis
threads; this is captured quantitatively by the transverse-corrected 2D theory
(validation/junction_linear_2d.py), which the simulator matches within a few
percent. The 1D-numerical discrepancy scales as Z_T/Z_L ~ sqrt(eps_p).
"""

import pytest

from validation.test3_junction import run_junction_sim


def test_junction_N2_matches_theory():
    r = run_junction_sim("S", N=2, ep_frac=0.1, L=2.0, n_seg=300)
    # N=2 is collinear: no transverse correction, 1D == 2D == numerical.
    assert r["T0_sim"] == pytest.approx(1.0, abs=0.03)
    assert r["dTopp_sim"] == pytest.approx(1.0, abs=0.05)
    assert r["T0_2d_with"] == pytest.approx(1.0, abs=1e-6)
    assert r["T0_ref"] == pytest.approx(r["T0_lin"], abs=0.02)


@pytest.mark.parametrize("N", [3, 4, 6, 8, 16])
def test_junction_matches_2d_theory(N):
    r = run_junction_sim("S", N=N, ep_frac=0.1, L=2.0, n_seg=300)
    # 2D theory matches numerics within a few percent...
    assert r["T0_sim"] == pytest.approx(r["T0_2d_with"], abs=0.04)
    # ...and is at least as good as the 1D theory (strictly better for N>=4).
    err_2d = abs(r["T0_sim"] - r["T0_2d_with"])
    err_1d = abs(r["T0_sim"] - r["T0_lin"])
    assert err_2d <= err_1d + 1e-9


def test_junction_trend_and_scaling():
    # T0 increases monotonically with N.
    vals = [run_junction_sim("S", N=N, ep_frac=0.1, L=2.0, n_seg=300)["T0_sim"]
            for N in (2, 4, 8, 16)]
    assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))
    # D scaling: larger eps_p -> larger 1D discrepancy, tracked by 2D theory.
    r_lo = run_junction_sim("D", N=8, ep_frac=0.02, L=2.0, n_seg=300)
    r_hi = run_junction_sim("D", N=8, ep_frac=0.3, L=2.0, n_seg=300)
    assert abs(r_hi["T0_sim"] - r_hi["T0_lin"]) > abs(r_lo["T0_sim"] - r_lo["T0_lin"])
    assert abs(r_hi["T0_sim"] - r_hi["T0_2d_with"]) < 0.02
