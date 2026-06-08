from hishells_pv.catalog import load_bagetakos_table7


def test_load_bagetakos_table7():
    catalog = load_bagetakos_table7("J_AJ_141_23_table7.dat.txt")

    assert len(catalog) == 1046
    assert sorted(catalog["shell_type"].dropna().astype(int).unique()) == [1, 2, 3]

