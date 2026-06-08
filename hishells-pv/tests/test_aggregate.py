"""End-to-end smoke test for the aggregate stage: train -> aggregate."""
import json
import shutil
from pathlib import Path

import yaml

from hishells_pv.infer.aggregate import aggregate
from hishells_pv.train.trainer import train


def _train_checkpoint(train_config: dict, tmp_path: Path) -> Path:
    cfg_path = tmp_path / "train_config.yaml"
    cfg_path.write_text(yaml.safe_dump(train_config, sort_keys=False))
    runs_root = tmp_path / "runs"
    run_dir = Path(train(str(cfg_path), run_name="smoke", runs_root=str(runs_root), device_name="cpu", smoke=True))
    best = run_dir / "best_model.pt"
    if not best.exists():
        shutil.copy(run_dir / "final_model.pt", best)
    return run_dir


def test_train_then_aggregate(tiny_agg_dataset, tmp_path):
    run_dir = _train_checkpoint(tiny_agg_dataset["train_config"], tmp_path)

    agg_cfg_path = tmp_path / "agg_config.yaml"
    agg_cfg_path.write_text(yaml.safe_dump(tiny_agg_dataset["agg_config"], sort_keys=False))
    label = tiny_agg_dataset["label"]

    aggregate(
        cfg_path=str(agg_cfg_path),
        run_dir=str(run_dir),
        split=label,
        thresh=0.4,
        device_name="cpu",
        write_regions=True,
    )

    out_dir = run_dir / f"aggregate_{label}"
    det_json = out_dir / f"detections_{label}.json"
    vote_fits = out_dir / f"vote_map_{label}.fits"
    assert det_json.exists(), "aggregate did not write detections JSON"
    assert vote_fits.exists(), "aggregate did not write the vote-map FITS"

    report = json.loads(det_json.read_text())
    assert report["split"] == label
    assert "detections" in report
    assert "params" in report


def test_aggregate_all_driver(tiny_agg_dataset, tmp_path):
    """aggregate_all should pick up a generate-emitted agg config and run it."""
    from hishells_pv.infer.aggregate_galaxies import aggregate_all

    run_dir = _train_checkpoint(tiny_agg_dataset["train_config"], tmp_path)

    # Lay out a standardized dataset whose test split points at the galaxy, with
    # the per-galaxy agg config already emitted (as `generate` would do).
    std_root = tmp_path / "std"
    (std_root / "splits").mkdir(parents=True, exist_ok=True)
    galaxy_id = tiny_agg_dataset["galaxy_id"]
    label = tiny_agg_dataset["label"]

    agg_src = Path(tiny_agg_dataset["agg_config"]["output_root"])
    dst = std_root / "agg" / galaxy_id
    shutil.copytree(agg_src, dst)
    # Rewrite the copied config's output_root to the new location.
    cfg = yaml.safe_load((dst / f"{galaxy_id}_agg_config.yaml").read_text()) if (dst / f"{galaxy_id}_agg_config.yaml").exists() else dict(tiny_agg_dataset["agg_config"])
    cfg["output_root"] = str(dst.resolve())
    (dst / f"{galaxy_id}_agg_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    # The dataset-level test manifest groups by `<galaxy>__...` filenames.
    names = [ln for ln in (dst / "splits" / f"{label}_manifest.txt").read_text().splitlines() if ln.strip()]
    (std_root / "splits" / "test_manifest.txt").write_text("".join(f"{n}\n" for n in names))

    produced = aggregate_all(
        output_root=std_root,
        run_dir=run_dir,
        splits=("test",),
        thresh=0.4,
        device_name="cpu",
        write_regions=False,
    )
    assert produced, "aggregate_all produced nothing"
    out_dir = run_dir / f"aggregate_{label}"
    assert (out_dir / f"detections_{label}.json").exists()
