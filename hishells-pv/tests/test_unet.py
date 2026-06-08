import torch

from hishells_pv.models.unet import UNetPV


def test_unet_forward_shape():
    model = UNetPV(base_filters=8, depth=3, dropout=0.0).eval()
    x = torch.randn(2, 1, 96, 256)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1, 96, 256)
    assert torch.isfinite(out).all()


def test_unet_from_config():
    model = UNetPV.from_config({"base_filters": 8, "depth": 2, "dilation_rate": 1, "dropout": 0.1})
    assert model.config["base_filters"] == 8
    assert model.config["depth"] == 2
