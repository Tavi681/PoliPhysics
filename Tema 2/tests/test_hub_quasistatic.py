"""Validation Test 2 (pytest): quasi-static hub impact on the star."""

import pytest

from validation.test2_hub import run_hub, analytic
from netsim.materials import get_material


CASES = [
    ("S", 0.0, 0.831, 1.000),
    ("S", 0.1, 0.770, 0.993),
    ("S", 0.3, 0.650, 0.937),
    ("D", 0.0, 0.255, 1.000),
    ("D", 0.1, 0.241, 0.990),
    ("D", 0.3, 0.212, 0.910),
]


@pytest.mark.parametrize("mat,frac,wb_ref,eta_ref", CASES)
def test_hub_wb_and_eta(mat, frac, wb_ref, eta_ref):
    speed = 2.0 if mat == "S" else 0.5
    r = run_hub(mat, eps_p_frac=frac, speed=speed)
    assert r["wb_over_R_num"] == pytest.approx(wb_ref, abs=5e-3)
    assert r["eta_num"] == pytest.approx(eta_ref, abs=5e-3)


def test_hub_rate_independence():
    r1 = run_hub("S", eps_p_frac=0.1, speed=1.0, n_s=4)
    r2 = run_hub("S", eps_p_frac=0.1, speed=2.0, n_s=4)
    assert r1["wb_over_R_num"] == pytest.approx(r2["wb_over_R_num"], abs=1e-2)
