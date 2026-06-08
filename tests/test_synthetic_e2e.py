import numpy as np
import torch
from torch.utils.data import DataLoader

from hishells_pv.data import PVPatchDataset
from hishells_pv.models import BCETverskyLoss, UNetPV, evaluate


def test_synthetic_end_to_end_training_step(tmp_path):
    for sub in ("pv", "labels", "splits"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    pv = np.random.default_rng(42).normal(size=(32, 64)).astype(np.float32)
    label = np.zeros((32, 64), dtype=np.uint8)
    label[8:16, 24:40] = 1
    np.save(tmp_path / "pv" / "synthetic.npy", pv)
    np.save(tmp_path / "labels" / "synthetic.npy", label)
    (tmp_path / "splits" / "train_manifest.txt").write_text("synthetic.npy\n")

    cfg = {
        "output_root": str(tmp_path),
        "standardized_pv": {"enabled": True},
        "train": {
            "norm_method": "zscore_galaxy_only",
            "patch_vel": 32,
            "patch_pos": 64,
            "strict_fixed_shape": True,
            "samples_per_pv": 1,
        },
    }
    loader = DataLoader(PVPatchDataset(cfg, "train"), batch_size=1)
    model = UNetPV(base_filters=4, depth=2, dilation_rate=1, dropout=0.0)
    loss_fn = BCETverskyLoss(alpha=0.3, beta=0.7, bce_weight=0.5, tversky_weight=0.5)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x, y = next(iter(loader))
    loss = loss_fn(model(x), y)
    loss.backward()
    opt.step()
    metrics = evaluate(model, loader, loss_fn, torch.device("cpu"), thresholds=[0.075], max_steps=1)

    assert torch.isfinite(loss)
    assert "patch_recall_0p075" in metrics

