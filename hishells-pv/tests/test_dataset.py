from hishells_pv.data.dataset import PVPatchDataset


def test_dataset_shapes(tiny_dataset):
    ds = PVPatchDataset(tiny_dataset, "train", seed=1)
    assert len(ds) == 8
    x, y = ds[0]
    assert x.shape == (1, 32, 64)
    assert y.shape == (1, 32, 64)
    # Labels are binary.
    assert float(y.max()) <= 1.0 and float(y.min()) >= 0.0
