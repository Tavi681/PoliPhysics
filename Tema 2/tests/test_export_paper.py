"""Item 6 (pytest): paper-export helpers (format only, no heavy runs)."""

import csv

import pytest

import validation.export_paper as ep


def test_write_header_and_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(ep, "OUT_DIR", tmp_path)
    rows = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    path = ep._write("demo.csv", ["a", "b"], rows, config="unit")
    text = path.read_text().splitlines()
    # First line is the required comment header.
    assert text[0].startswith("# commit=")
    assert "date=" in text[0] and "config=unit" in text[0]
    # Then the column header, then the data.
    assert text[1] == "a,b"
    assert text[2] == "1,2"


def test_select_light_frames_caps_and_keeps_ends():
    from types import SimpleNamespace
    import numpy as np
    from netsim.io_hdf5 import select_light_frames

    n = 200
    t = np.linspace(0.0, 1.0, n)
    energy = np.zeros((n, 6))
    energy[20, 4] = 1.0
    drone = np.zeros((n, 6))
    drone[:40, 5] = -1.0
    drone[40:, 5] = 0.1
    failures = np.array([[0, 0, t[80]]])
    traj = SimpleNamespace(t=t, energy=energy, drone=drone, failures=failures)
    idx = select_light_frames(traj, max_frames=50)
    assert idx[0] == 0 and idx[-1] == n - 1
    assert len(idx) <= 50
    assert 20 in idx and 40 in idx and 80 in idx


def test_header_uses_env_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(ep, "OUT_DIR", tmp_path)
    monkeypatch.setattr(ep, "_COMMIT", "deadbeef")
    monkeypatch.setattr(ep, "_CODE_DIRTY", False)
    path = ep._write("prov.csv", ["a"], [{"a": 1}], config="unit")
    hdr = path.read_text().splitlines()[0]
    assert "commit=deadbeef" in hdr
    assert "code_dirty=false" in hdr


def test_params_used_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(ep, "OUT_DIR", tmp_path)
    path = ep.export_params_used()
    with open(path) as fh:
        assert fh.readline().startswith("# commit=")
        reader = csv.DictReader(fh)
        assert reader.fieldnames == ["symbol", "value", "unit", "comment"]
        syms = [r["symbol"] for r in reader]
    # Must include the material parameters and the working values.
    for key in ("E0_S", "eps_b_D", "r_d", "k_c", "R_max", "k_max"):
        assert key in syms


def test_scale_mmin_row_m_prop_M():
    src = {"net": "star", "material": "S", "n_s": "40", "M": "1", "v0": "10",
           "Ekin": "50", "m_lower_g": "1", "mA_min_g": "8", "mBany_min_g": "4",
           "mBloc025_min_g": "8", "mBloc050_min_g": "8", "mBloc075_min_g": "5",
           "s_lo_A": "0.8", "s_hi_A": "0.81", "m_source": "computed"}
    row = ep._scale_mmin_row(src, 2.0, "scaled_m_prop_M from M=1 v0=10")
    assert row["M"] == "2"
    assert float(row["mA_min_g"]) == pytest.approx(16.0)
    assert float(row["Ekin"]) == pytest.approx(100.0)
    assert float(row["s_lo_A"]) == pytest.approx(1.6)
    assert row["m_source"].startswith("scaled_m_prop_M")
