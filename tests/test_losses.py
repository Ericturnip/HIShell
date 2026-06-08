import torch

from hishells_pv.models import BCETverskyLoss


def test_bce_tversky_loss_is_finite_and_differentiable():
    logits = torch.randn(2, 1, 16, 16, requires_grad=True)
    target = torch.zeros_like(logits)
    target[:, :, 4:8, 4:8] = 1.0
    loss_fn = BCETverskyLoss(alpha=0.3, beta=0.7, bce_weight=0.5, tversky_weight=0.5)

    loss = loss_fn(logits, target)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None

