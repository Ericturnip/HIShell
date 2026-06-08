from pathlib import Path

from hishells_pv.catalog.shell_catalog import load_bagetakos_table7

TABLE7 = Path(__file__).resolve().parents[1] / "catalogs" / "J_AJ_141_23_table7.dat.txt"


def test_load_bagetakos_table7():
    df = load_bagetakos_table7(TABLE7)
    assert len(df) == 1046
    for col in ("Name", "shell_type", "ra_deg", "dec_deg", "d_pc"):
        assert col in df.columns
    # Three shell types are present in the Bagetakos catalog.
    assert set(int(t) for t in df["shell_type"].dropna().unique()) <= {1, 2, 3}


def test_keep_types_filter():
    df = load_bagetakos_table7(TABLE7, keep_types=[2, 3])
    assert set(int(t) for t in df["shell_type"].dropna().unique()) <= {2, 3}
