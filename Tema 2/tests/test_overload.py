"""Validation 5a (pytest): dynamic overload of an N-star hub."""

import pytest

from netsim.materials import get_material
from netsim.overload import run_overload, remove_thread, ThreadRemoval


def test_remove_thread_api():
    r = remove_thread(2, 0.5, t_f=0.1)
    assert isinstance(r, ThreadRemoval)
    assert (r.parent, r.t_b, r.t_f) == (2, 0.5, 0.1)


def test_static_strain_matches_target():
    # The force is derived analytically; the settled strain must equal eps0.
    m = get_material("D")
    r = run_overload(8, m, eps0=0.01, m_hub=0.01, n_s=2)
    assert r.eps0_sim == pytest.approx(0.01, rel=1e-3)


def test_removal_increases_strain_and_daf_gt_one():
    m = get_material("D")
    r = run_overload(8, m, eps0=0.01, m_hub=0.01, n_s=2, t_f_over_Tn=0.0)
    # Removing a radial raises the static strain in the survivors ...
    assert r.eps_s > r.eps0_sim
    # ... and the instantaneous removal overshoots it dynamically.
    assert r.eps_m > r.eps_s
    assert r.daf > 1.0


def test_daf_decreases_with_release_time():
    m = get_material("D")
    daf0 = run_overload(8, m, eps0=0.01, m_hub=0.01, n_s=2,
                        t_f_over_Tn=0.0).daf
    daf_slow = run_overload(8, m, eps0=0.01, m_hub=0.01, n_s=2,
                            t_f_over_Tn=4.0, dyn_periods=16.0).daf
    assert daf_slow < daf0
    assert daf_slow == pytest.approx(1.0, abs=0.08)


def test_stiffening_reduces_daf():
    # The cubic-dominated material S overshoots less than the linear material D.
    d = run_overload(8, get_material("D"), eps0=0.01, m_hub=0.01, n_s=2).daf
    s = run_overload(8, get_material("S"), eps0=0.24, m_hub=0.01, n_s=2).daf
    assert s < d


def test_n_eff_regimes():
    # Effective exponent n_eff = d ln F / d ln w. For the linear material in the
    # small-deflection geometric regime n_eff ~ 3; cubic stiffening raises it.
    d = run_overload(8, get_material("D"), eps0=0.01, m_hub=0.01, n_s=2).n_eff
    s = run_overload(8, get_material("S"), eps0=0.24, m_hub=0.01, n_s=2).n_eff
    assert d > 2.0
    assert s > d
