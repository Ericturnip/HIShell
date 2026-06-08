import torch

from hishells_pv.models.losses import BCETverskyLoss, tversky_loss_from_logits


def test_tversky_loss_range():
    logits = torch.zeros((2, 1, 8, 16))
    y = torch.zeros((2, 1, 8, 16))
    loss = tversky_loss_from_logits(logits, y)
    assert torch.isfinite(loss)
    assert 0.0 <= float(loss) <= 1.0


def test_bce_tversky_perfect_vs_wrong():
    y = torch.zeros((2, 1, 8, 16))
    y[:, :, 2:5, 3:7] = 1.0
    loss_fn = BCETverskyLoss()
    big_neg = torch.full_like(y, -10.0)
    big_pos = torch.full_like(y, 10.0)
    # Confidently-correct logits should yield a lower loss than confidently-wrong ones.
    correct = loss_fn(torch.where(y > 0.5, big_pos, big_neg), y)
    wrong = loss_fn(torch.where(y > 0.5, big_neg, big_pos), y)
    assert float(correct) < float(wrong)
