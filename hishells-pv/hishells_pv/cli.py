"""Unified command-line interface for the HIShells-PV tool.

Usage::

    hishells-pv <command> [options]

Commands
--------
- ``generate``        Generate standardized PV cuts + labels (training data).
- ``generate-all``    Per-galaxy PV/label generation driver.
- ``validate``        Validate generated training data.
- ``combine``         Combine per-galaxy training data into one dataset.
- ``train``           Train the PyTorch U-Net.
- ``infer``           Write per-PV probability maps from a checkpoint.
- ``calibrate``       Calibrate a global threshold against labelled PVs.
- ``postprocess``     Extract ranked candidate components from probability maps.
- ``aggregate``       Aggregate PV predictions into sky-plane candidates.
- ``aggregate-all``   Aggregate every galaxy in a standardized dataset.
- ``resolve-config``  Resolve a config (FITS inference + defaults) and print it.

Data-generation commands forward their remaining arguments to the underlying
module; run e.g. ``hishells-pv generate --help`` for their full option set.
"""
from __future__ import annotations

import argparse
import sys
from typing import Sequence


def _add_train(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("train", help="Train the PyTorch U-Net")
    p.add_argument("--config", required=True)
    p.add_argument("--run", default=None, help="Run name (output dir under runs/)")
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--device", default="auto", help="auto, mps, cuda, or cpu")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--max-train-steps", type=int, default=None)
    p.add_argument("--max-val-steps", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="One short epoch for a wiring smoke test")


def _run_train(args: argparse.Namespace) -> None:
    from hishells_pv.train.trainer import train

    train(
        args.config,
        run_name=args.run,
        runs_root=args.runs_root,
        device_name=args.device,
        epochs_override=args.epochs,
        max_train_steps=args.max_train_steps,
        max_val_steps=args.max_val_steps,
        num_workers=args.num_workers,
        smoke=args.smoke,
    )


def _add_infer(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("infer", help="Write per-PV probability maps")
    p.add_argument("--model", required=True, help="Path to a .pt checkpoint")
    p.add_argument("--config", default=None, help="Optional config (else use checkpoint's)")
    p.add_argument("--out", default=None, help="Output dir (default: <output_root>/pred)")
    p.add_argument("--device", default="auto")


def _run_infer(args: argparse.Namespace) -> None:
    from hishells_pv.infer.predict import predict_run

    predict_run(args.config, args.model, out_dir=args.out, device_name=args.device)


def _add_calibrate(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("calibrate", help="Calibrate a global probability threshold")
    p.add_argument("--config", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--min-recall", type=float, default=0.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=None)


def _run_calibrate(args: argparse.Namespace) -> None:
    from hishells_pv.infer.calibrate import calibrate_threshold

    calibrate_threshold(
        args.config,
        args.model,
        split=args.split,
        min_recall=args.min_recall,
        device_name=args.device,
        out_dir=args.out,
    )


def _add_postprocess(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("postprocess", help="Extract ranked candidate components")
    p.add_argument("--model", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--split", default=None, help="Manifest split (default: all PVs)")
    p.add_argument("--threshold", type=float, default=None, help="Default: calib/threshold.txt or 0.075")
    p.add_argument("--min-area-pix", type=int, default=6)
    p.add_argument("--drop-edge", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=None)


def _run_postprocess(args: argparse.Namespace) -> None:
    from hishells_pv.infer.postprocess import postprocess_run

    postprocess_run(
        args.config,
        args.model,
        threshold=args.threshold,
        split=args.split,
        min_area_pix=args.min_area_pix,
        drop_edge=args.drop_edge,
        device_name=args.device,
        out_dir=args.out,
    )


def _add_aggregate(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("aggregate", help="Aggregate PV predictions into sky candidates")
    p.add_argument("--config", required=True)
    p.add_argument("--run-dir", required=True, help="Run dir containing best_model.pt")
    p.add_argument("--split", default="test", help="Manifest split label (e.g. test, stress, ngc_7793_test)")
    p.add_argument("--thresh", type=float, default=0.5)
    p.add_argument("--device", default="auto")
    p.add_argument("--output-root", default=None, help="Override cfg.output_root")
    p.add_argument("--cube-path", default=None, help="Override cfg.cube_path")
    p.add_argument("--no-regions", action="store_true")


def _run_aggregate(args: argparse.Namespace) -> None:
    from hishells_pv.infer.aggregate import aggregate

    aggregate(
        cfg_path=args.config,
        run_dir=args.run_dir,
        split=args.split,
        thresh=args.thresh,
        device_name=args.device,
        write_regions=(not args.no_regions),
        output_root=args.output_root,
        cube_path=args.cube_path,
    )


def _add_aggregate_all(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("aggregate-all", help="Aggregate every galaxy in a standardized dataset")
    p.add_argument("--output-root", required=True, help="Standardized dataset dir (has pv/ + splits/)")
    p.add_argument("--run-dir", required=True, help="Run dir containing best_model.pt")
    p.add_argument("--splits", nargs="*", default=["test", "stress"], help="Splits to aggregate")
    p.add_argument("--thresh", type=float, default=0.4)
    p.add_argument("--device", default="auto")
    p.add_argument("--configs-dir", default=None, help="Source per-galaxy configs (default: <output_root>/../configs)")
    p.add_argument("--no-regions", action="store_true")


def _run_aggregate_all(args: argparse.Namespace) -> None:
    from hishells_pv.infer.aggregate_galaxies import aggregate_all

    aggregate_all(
        output_root=args.output_root,
        run_dir=args.run_dir,
        splits=tuple(args.splits),
        thresh=args.thresh,
        device_name=args.device,
        write_regions=(not args.no_regions),
        configs_dir=args.configs_dir,
    )


def _add_resolve(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("resolve-config", help="Resolve and print a config")
    p.add_argument("--config", required=True)
    p.add_argument("--set", nargs="*", default=[], help="key=value overrides")


def _run_resolve(args: argparse.Namespace) -> None:
    from hishells_pv.qa.print_resolved_config import main as resolve_main

    resolve_main(args.config, args.set)


# Data-generation commands forward unparsed args to the module's main(argv).
_FORWARD = {
    "generate": "hishells_pv.datagen.prepare_standardized",
    "generate-all": "hishells_pv.datagen.generate_all",
    "validate": "hishells_pv.datagen.validate",
    "combine": "hishells_pv.datagen.prepare_combined",
}


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    parser = argparse.ArgumentParser(prog="hishells-pv", description="HIShells PV-shell detection tool")
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    for name, help_text in (
        ("generate", "Generate standardized PV cuts + labels"),
        ("generate-all", "Per-galaxy PV/label generation driver"),
        ("validate", "Validate generated training data"),
        ("combine", "Combine per-galaxy training data"),
    ):
        sub.add_parser(name, help=help_text, add_help=False)

    _add_train(sub)
    _add_infer(sub)
    _add_calibrate(sub)
    _add_postprocess(sub)
    _add_aggregate(sub)
    _add_aggregate_all(sub)
    _add_resolve(sub)

    if argv and argv[0] == "--version":
        from hishells_pv import __version__

        print(__version__)
        return 0

    if not argv:
        parser.print_help()
        return 1

    command = argv[0]
    if command in _FORWARD:
        from importlib import import_module

        module = import_module(_FORWARD[command])
        module.main(argv[1:])
        return 0

    args = parser.parse_args(argv)
    dispatch = {
        "train": _run_train,
        "infer": _run_infer,
        "calibrate": _run_calibrate,
        "postprocess": _run_postprocess,
        "aggregate": _run_aggregate,
        "aggregate-all": _run_aggregate_all,
        "resolve-config": _run_resolve,
    }
    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        return 1
    handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
