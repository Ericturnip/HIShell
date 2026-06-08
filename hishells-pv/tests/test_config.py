"""Tests for the pure-Python config override/merge helpers.

The full ``resolve_config`` performs FITS header inference, which requires a
real cube on disk (intentionally excluded from the repo), so we exercise the
deterministic override/merge logic directly.
"""
from hishells_pv.utils import config as cfgmod


def test_cli_overrides_type_inference():
    out = cfgmod._cli_overrides(["optim.lr=0.0005", "model.base_filters=48", "train.flag=true"])
    assert out["optim"]["lr"] == 0.0005
    assert out["model"]["base_filters"] == 48
    assert out["train"]["flag"] is True


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("PV_model.base_filters", "48")
    out = cfgmod._env_overrides()
    assert out["model"]["base_filters"] == 48


def test_deep_update_merges_nested():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    merged = cfgmod._deep_update(base, {"a": {"y": 20, "z": 30}})
    assert merged["a"] == {"x": 1, "y": 20, "z": 30}
    assert merged["b"] == 3


def test_hash_is_stable():
    obj = {"a": 1, "b": [1, 2, 3]}
    assert cfgmod._hash_cfg(obj) == cfgmod._hash_cfg(dict(obj))
