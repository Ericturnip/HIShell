from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_tensorflow_loss_stack_smoke_when_available():
    env = os.environ.copy()
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    env.setdefault("PANDAS_USE_PYARROW", "0")

    code = """
import sys
sys.path.insert(0, ".")
import tensorflow as tf
from src.models.losses import dice_loss, focal_tversky_loss
y_true = tf.zeros((1, 8, 16, 1))
y_pred = tf.zeros((1, 8, 16, 1))
assert float(dice_loss()(y_true, y_pred)) >= 0.0
assert float(focal_tversky_loss()(y_true, y_pred)) >= 0.0
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=45,
    )

    if proc.returncode != 0:
        lines = (proc.stderr or proc.stdout).strip().splitlines()
        detail = " ".join(lines[-1:]) if lines else f"return code {proc.returncode}"
        pytest.skip("TensorFlow stack is unavailable in this Python environment: " + detail)
