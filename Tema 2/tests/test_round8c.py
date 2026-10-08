"""Round 8c: production defaults match a6c4705; continue-after-arrest is opt-in."""

from pathlib import Path

import numpy as np
import pytest

from netsim.config import (
    SimConfig, MaterialConfig, NetConfig, DroneConfig, NumericsConfig,
    ContactConfig, OutputConfig,
)
from netsim.simulate import simulate_config


_FIXTURE = Path(__file__).resolve().parent / "data" / "a6c4705_starS.npz"


def _prod_cfg(*, continue_after_arrest=False, n_s=6, t_end=0.03, area_scale=2.0):
    return SimConfig(
        material=MaterialConfig(name="S"),
        net=NetConfig(kind="star", N=8, R=1.0, eps_p=0.0, A_hat=1e-6),
        drone=DroneConfig(M=1.0, r_d=0.15, v0=15.0, p=(0.0, 0.0)),
        numerics=NumericsConfig(n_s=n_s, C=0.5, t_end=t_end, dt_out=5e-3,
                                use_numba=False, area_scale=area_scale,
                                shock_visc=0.0, continue_after_arrest=continue_after_arrest),
        contact=ContactConfig(mode="gripped", k_c=1e7,
                              penetration_guard="reduce_dt"),
        output=OutputConfig(hdf5=None, R_max=0.5, k_max=10),
    )


def test_production_defaults_off():
    n = NumericsConfig()
    assert n.shock_visc == 0.0
    assert n.continue_after_arrest is False


def test_continue_after_arrest_default_matches_explicit_false():
    a = simulate_config(_prod_cfg(), write=False)
    b = simulate_config(_prod_cfg(continue_after_arrest=False), write=False)
    np.testing.assert_array_equal(a.trajectory.energy, b.trajectory.energy)
    np.testing.assert_array_equal(a.trajectory.t, b.trajectory.t)
    assert a.outcome == b.outcome


@pytest.mark.skipif(not _FIXTURE.is_file(), reason="a6c4705 fixture missing")
def test_production_run_bit_identical_to_a6c4705():
    data = np.load(_FIXTURE)
    res = simulate_config(_prod_cfg(), write=False)
    np.testing.assert_array_equal(res.trajectory.energy, data["energy"])
    np.testing.assert_array_equal(res.trajectory.drone, data["drone"])
    np.testing.assert_array_equal(res.trajectory.t, data["t"])
    assert res.outcome == str(data["outcome"][0])
