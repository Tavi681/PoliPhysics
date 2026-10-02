"""Validation Test 1 (pytest): single thread Smith solution, multi-resolution.

The longitudinal-front and kink errors are reported separately and must shrink
as the thread is refined to 200/400/800 segments over the resolved length. For
the linear material D the longitudinal front is a clean step and converges; for
the stiffening material S the longitudinal wave is a dispersive shock, so only
the kink (transverse) front converges cleanly while the plateau strain stays
accurate -- we check the appropriate quantity per material (physics, not
tolerances, dictates which).
"""

import pytest

from validation.test1_smith import run_smith


@pytest.mark.parametrize("v0", [100.0, 200.0, 300.0])
def test_smith_strain(v0):
    r = run_smith(v0, material_name="S", n_resolved=400)
    assert r["eps_num"] == pytest.approx(r["eps_ana"], rel=0.03)


def test_smith_front_and_kink():
    r = run_smith(300.0, material_name="S", n_resolved=400)
    assert r["front_num"] == pytest.approx(r["front_ana"], rel=0.08)
    assert r["kink_num"] == pytest.approx(r["kink_ana"], rel=0.08)


def _errors(material, v0, resolutions=(200, 400, 800)):
    front, kink, eps = [], [], []
    for n in resolutions:
        r = run_smith(v0, material_name=material, n_resolved=n)
        front.append(r["front_relerr"])
        kink.append(r["kink_relerr"])
        eps.append(r["eps_relerr"])
    return front, kink, eps


def test_smith_kink_converges_both_materials():
    # The transverse (kink) front is a clean step for both materials and must
    # improve from the coarsest to the finest grid.
    for material, v0 in (("S", 200.0), ("D", 300.0)):
        front, kink, eps = _errors(material, v0)
        assert kink[-1] < kink[0], (material, kink)
        assert kink[-1] < 0.01, (material, kink)
        assert eps[-1] < 0.01, (material, eps)


def test_smith_longitudinal_front_converges_linear():
    # For the linear material D the longitudinal front is a true step and must
    # converge monotonically with resolution.
    front, kink, eps = _errors("D", 500.0)
    assert front[1] < front[0] and front[2] < front[1], front
    assert front[-1] < 0.01, front
