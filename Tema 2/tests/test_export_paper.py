"""Item 6 (pytest): paper-export helpers (format only, no heavy runs)."""

import csv

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
