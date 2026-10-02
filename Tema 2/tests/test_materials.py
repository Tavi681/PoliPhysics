"""Validation 1: material-law unit checks (targets from the objective).

For S: sigma(eps_b)=1.20 GPa, e_mat=103.8 kJ/kg, c_L0=1240 m/s,
c_L(eps_b)=1754 m/s, c_T(eps_b)=843 m/s.
For D: sigma_b=3.52 GPa, e_mat=57.8 kJ/kg, c_L0=10622 m/s.
"""

import math

import pytest

from netsim.materials import get_material


def test_material_S():
    m = get_material("S")
    assert m.sigma_b == pytest.approx(1.20e9, rel=5e-3)
    assert m.e_mat == pytest.approx(103.8e3, rel=5e-3)
    assert m.c_L0 == pytest.approx(1240.0, rel=5e-3)
    assert float(m.c_L(m.eps_b)) == pytest.approx(1754.0, rel=5e-3)
    assert float(m.c_T(m.eps_b)) == pytest.approx(843.0, rel=5e-3)
    # tangent speed correction cited in the objective
    assert m.c_tan_max == pytest.approx(2480.0, rel=5e-3)


def test_material_D():
    m = get_material("D")
    assert m.sigma_b == pytest.approx(3.52e9, rel=5e-3)
    assert m.e_mat == pytest.approx(57.8e3, rel=5e-3)
    assert m.c_L0 == pytest.approx(10622.0, rel=5e-3)


def test_inverse_law_roundtrip():
    m = get_material("S")
    A = 1e-6
    for eps in (0.01, 0.1, 0.25):
        T = m.tension(eps, A)
        eps_back = m.strain_from_tension(float(T), A)
        assert eps_back == pytest.approx(eps, rel=1e-8)


def test_no_compression():
    m = get_material("S")
    assert float(m.sigma(-0.1)) == 0.0
    assert float(m.Phi(-0.1)) == 0.0
