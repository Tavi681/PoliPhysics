"""Validation Test 4 (pytest, light): convergence in n_s on the reference case.

Checks that w_max/R for the gripped reference case converges as n_s is refined.
The full table (both contact modes, CPU times) is produced by
validation/test4_convergence.py.
"""

import pytest

from validation.test4_convergence import run_case


def test_wmax_converges_gripped():
    r20 = run_case(20, "gripped")
    r40 = run_case(40, "gripped")
    r80 = run_case(80, "gripped")
    d_40 = abs(r40["w_max_over_R"] - r80["w_max_over_R"]) / r80["w_max_over_R"]
    d_20 = abs(r20["w_max_over_R"] - r80["w_max_over_R"]) / r80["w_max_over_R"]
    # Refinement should reduce the difference from the finest grid.
    assert d_40 <= d_20 + 1e-6
    assert d_40 < 0.05


def test_run_case_reports_full_fields():
    # Test 4 must report more than w_max: eta, R_d, failure positions, cascade.
    r = run_case(40, "gripped")
    for key in ("eta", "R_d", "cascade", "fail_radii", "n_failures"):
        assert key in r
    # Failure bookkeeping must be self-consistent.
    assert len(r["fail_radii"]) == r["n_failures"]


def test_eta_converges_gripped():
    # Convergence of w_max alone is not enough; the absorbed-energy ratio eta
    # must also settle as the grid is refined.
    e20 = run_case(20, "gripped")["eta"]
    e40 = run_case(40, "gripped")["eta"]
    e80 = run_case(80, "gripped")["eta"]
    d_40 = abs(e40 - e80) / abs(e80) if e80 else abs(e40 - e80)
    d_20 = abs(e20 - e80) / abs(e80) if e80 else abs(e20 - e80)
    assert d_40 <= d_20 + 1e-6
    assert d_40 < 0.05
