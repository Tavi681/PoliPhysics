"""Validation Test 6 (pytest): off-centre quasi-static (gripped) vs ref/offc.py.

Compares eta_A at first failure and the first-failing segment (inner vs outer)
with the reference solution. A small subset of ``a/R`` is used to keep the test
fast; the full table is produced by ``validation/offcentre.py``.
"""

import pytest

from validation.offcentre import run_offcentre

import offc  # ref/offc.py (on sys.path via conftest)


# (a/R, n_s, expected first-failure kind)
CASES = [
    (0.5, 2, "out"),
    (0.1, 10, "in"),
]


@pytest.mark.parametrize("a,n_s,kind", CASES)
def test_offcentre_gripped(a, n_s, kind):
    sim = run_offcentre("S", a_over_R=a, n_s=n_s, speed=0.3, damping=400.0)
    ref_fails = offc.run("S", 8, a, 0.0)
    ref_kind, _, ref_eta = ref_fails[0]
    assert sim["first_kind"] == kind == ref_kind
    assert sim["eta_ff"] == pytest.approx(ref_eta, abs=5e-3)
