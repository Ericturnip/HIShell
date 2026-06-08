from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_published_model_metadata_matches_checkpoint():
    metadata = json.loads((ROOT / "artifacts" / "model_metadata.json").read_text())
    model_path = ROOT / metadata["model"]["file"]

    assert model_path.exists()
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == metadata["model"]["sha256"]
    assert metadata["model"]["input_shape"] == [96, 256, 1]
    assert metadata["model"]["recommended_probability_threshold"] == 0.075


def test_metadata_contains_headline_metrics_only():
    metadata = json.loads((ROOT / "artifacts" / "model_metadata.json").read_text())

    assert not (ROOT / "artifacts" / "evaluation").exists()
    assert set(metadata["metrics_at_threshold_0p075"]) == {"val", "test", "stress"}
    assert metadata["metrics_at_threshold_0p075"]["test"]["patch_recall"] > 0.99
    assert metadata["ddo53_physical_grid_shell_recall"]["shell_level_detection_recall"] == 1.0
