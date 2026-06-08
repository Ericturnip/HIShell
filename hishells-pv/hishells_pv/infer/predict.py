"""Inference: load a trained checkpoint and write per-PV probability maps.

A checkpoint is a dict ``{"model_state_dict", "cfg", "history"}`` as written by
:func:`hishells_pv.train.trainer.train`. The model architecture is rebuilt from
the embedded ``cfg['model']`` section so no separate config is required to load.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from hishells_pv.data.dataset import normalize
from hishells_pv.models.unet import UNetPV
from hishells_pv.train.trainer import resolve_device
from hishells_pv.utils.io import load_yaml


def load_checkpoint(model_path: str | Path, device: torch.device | None = None) -> tuple[UNetPV, dict[str, Any]]:
    """Rebuild a :class:`UNetPV` from a checkpoint and return ``(model, cfg)``."""
    device = device or resolve_device("auto")
    ckpt = torch.load(Path(model_path), map_location=device, weights_only=False)
    if "model_state_dict" not in ckpt:
        raise ValueError(f"{model_path} is not a hishells-pv checkpoint (missing model_state_dict)")
    cfg = ckpt.get("cfg", {})
    model = UNetPV.from_config(cfg.get("model", {}))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, cfg


@torch.no_grad()
def predict_pv(
    model: UNetPV,
    pv: np.ndarray,
    device: torch.device,
    *,
    norm_method: str | None = "zscore_galaxy_only",
) -> np.ndarray:
    """Return a probability map in [0, 1] for a single ``(V, S)`` PV array."""
    pv_norm = normalize(np.asarray(pv, dtype=np.float32), norm_method)
    x = torch.from_numpy(pv_norm).unsqueeze(0).unsqueeze(0).to(device)
    prob = torch.sigmoid(model(x))[0, 0].detach().cpu().numpy()
    return np.clip(prob.astype(np.float32), 0.0, 1.0)


def predict_run(
    config: str | None,
    model_path: str | Path,
    *,
    out_dir: str | Path | None = None,
    device_name: str = "auto",
) -> Path:
    """Predict probability maps for every PV in ``output_root/pv`` and save to ``pred/``.

    ``output_root`` is taken from ``config`` when provided, else from the
    checkpoint's embedded config.
    """
    device = resolve_device(device_name)
    model, ckpt_cfg = load_checkpoint(model_path, device)

    cfg = load_yaml(config) if config else ckpt_cfg
    if "output_root" not in cfg:
        raise ValueError("No output_root found in config or checkpoint; pass --config.")
    root = Path(cfg["output_root"])
    pv_dir = root / "pv"
    if not pv_dir.exists():
        raise FileNotFoundError(f"PV directory not found: {pv_dir}")
    norm_method = cfg.get("train", {}).get("norm_method", "zscore_galaxy_only")

    out = Path(out_dir) if out_dir else (root / "pred")
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in pv_dir.glob("*.npy") if not p.name.endswith("_posxy.npy"))
    print(f"[predict] model={Path(model_path).name} device={device} pv_files={len(files)}")
    for i, pv_path in enumerate(files, start=1):
        pv = np.load(pv_path)
        prob = predict_pv(model, pv, device, norm_method=norm_method)
        np.save(out / pv_path.name, prob)
        if i % 50 == 0:
            print(f"[predict] {i}/{len(files)}")
    print(f"[predict] wrote {len(files)} probability maps -> {out.resolve()}")
    return out
