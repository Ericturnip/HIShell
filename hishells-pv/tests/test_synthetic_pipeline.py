"""End-to-end smoke test: train -> infer -> calibrate -> postprocess on synthetic data."""
from pathlib import Path

import yaml

from hishells_pv.infer.calibrate import calibrate_threshold
from hishells_pv.infer.postprocess import postprocess_run
from hishells_pv.infer.predict import predict_run
from hishells_pv.train.trainer import train


def test_train_infer_postprocess(tiny_dataset, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(tiny_dataset, sort_keys=False))
    runs_root = tmp_path / "runs"

    run_dir = train(
        str(config_path),
        run_name="smoke",
        runs_root=str(runs_root),
        device_name="cpu",
        smoke=True,
    )
    best = Path(run_dir) / "best_model.pt"
    final = Path(run_dir) / "final_model.pt"
    assert final.exists()
    model_path = best if best.exists() else final

    pred_dir = predict_run(str(config_path), model_path, device_name="cpu")
    pred_files = list(Path(pred_dir).glob("*.npy"))
    assert pred_files, "inference produced no probability maps"

    report = calibrate_threshold(str(config_path), model_path, split="val", device_name="cpu")
    assert 0.0 <= report["best_threshold"] <= 1.0

    out_json = postprocess_run(str(config_path), model_path, split="test", threshold=0.5, device_name="cpu")
    assert Path(out_json).exists()
