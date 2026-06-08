"""TensorFlow/Keras inference wrapper for saved PV models."""

from __future__ import annotations


def run_inference(*args, **kwargs):
    """Run Keras inference, importing TensorFlow only when this path is used."""
    from hishells_pv.infer.infer_pv import main

    return main(*args, **kwargs)

__all__ = ["run_inference"]
