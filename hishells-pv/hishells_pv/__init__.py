"""HIShells-PV: PyTorch HI-shell detection from position-velocity (PV) cuts.

Public API
----------
- Catalog loading: :func:`load_bagetakos_table7`, :func:`catalog_to_pixel_shells`
- Model: :class:`UNetPV`
- Losses: :class:`BCETverskyLoss`
- Data: :class:`PVPatchDataset`
- Training: :func:`train`
- Inference: :func:`predict_run`, :func:`calibrate_threshold`, :func:`postprocess_run`

The heavy ML symbols (torch) are imported lazily so that the lightweight
catalog/data-generation half of the library can be used without importing
torch.
"""
from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

__version__ = "0.2.0"

# Lightweight (no-torch) re-exports.
from hishells_pv.catalog.shell_catalog import (  # noqa: E402
    catalog_to_pixel_shells,
    load_bagetakos_table7,
)

_LAZY = {
    "UNetPV": "hishells_pv.models.unet",
    "BCETverskyLoss": "hishells_pv.models.losses",
    "PVPatchDataset": "hishells_pv.data.dataset",
    "train": "hishells_pv.train.trainer",
    "predict_run": "hishells_pv.infer.predict",
    "calibrate_threshold": "hishells_pv.infer.calibrate",
    "postprocess_run": "hishells_pv.infer.postprocess",
}

__all__ = [
    "__version__",
    "load_bagetakos_table7",
    "catalog_to_pixel_shells",
    *_LAZY.keys(),
]

if TYPE_CHECKING:  # pragma: no cover - import-time hints only
    from hishells_pv.data.dataset import PVPatchDataset
    from hishells_pv.infer.calibrate import calibrate_threshold
    from hishells_pv.infer.postprocess import postprocess_run
    from hishells_pv.infer.predict import predict_run
    from hishells_pv.models.losses import BCETverskyLoss
    from hishells_pv.models.unet import UNetPV
    from hishells_pv.train.trainer import train


def __getattr__(name: str):
    """Lazily import torch-backed symbols on first access (PEP 562)."""
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module 'hishells_pv' has no attribute {name!r}")
    module = import_module(module_path)
    return getattr(module, name)
