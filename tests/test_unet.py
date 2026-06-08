import torch

from hishells_pv.models import UNetPV


def test_unet_pv_forward_shape():
    model = UNetPV(base_filters=4, depth=2, dilation_rate=1, dropout=0.0)
    x = torch.randn(2, 1, 32, 64)

    y = model(x)

    assert y.shape == (2, 1, 32, 64)

