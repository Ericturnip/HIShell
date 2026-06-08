from __future__ import annotations

from pathlib import Path

from src.pv.shell_catalog import load_bagetakos_table7


ROOT = Path(__file__).resolve().parents[1]


def test_checked_in_bagetakos_catalog_loads():
    catalog = load_bagetakos_table7(ROOT / "J_AJ_141_23_table7.dat.txt")

    assert len(catalog) == 1046
    assert sorted(catalog["shell_type"].dropna().astype(int).unique().tolist()) == [1, 2, 3]
    assert {"Name", "shell_id", "ra_deg", "dec_deg", "vel_center_kms", "d_pc"}.issubset(catalog.columns)
