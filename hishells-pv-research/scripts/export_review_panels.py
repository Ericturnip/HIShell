#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
import yaml

THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.diagnostic_utils import load_json, patch_prediction, predict_full_pv, read_manifest, source_paths


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def write_png_rgb(path: Path, image: np.ndarray) -> None:
    image = np.asarray(np.clip(image, 0, 255), dtype=np.uint8)
    h, w, _ = image.shape
    raw = b"".join(b"\x00" + image[y].tobytes() for y in range(h))
    payload = b"\x89PNG\r\n\x1a\n"
    payload += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    payload += _png_chunk(b"IDAT", zlib.compress(raw, level=6))
    payload += _png_chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def stretch_image(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.nanpercentile(x, [1.0, 99.5]) if x.size else (0.0, 1.0)
    if hi <= lo:
        hi = lo + 1.0
    y = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    y = np.arcsinh(4.0 * y) / np.arcsinh(4.0)
    return y


def gray_rgb(x: np.ndarray) -> np.ndarray:
    g = (255.0 * stretch_image(x)).astype(np.uint8)
    return np.repeat(g[..., None], 3, axis=2)


def mask_rgb(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = np.zeros(mask.shape + (3,), dtype=np.uint8)
    out[mask > 0] = color
    return out


def prob_rgb(prob: np.ndarray) -> np.ndarray:
    p = np.clip(prob.astype(np.float32), 0.0, 1.0)
    r = (255.0 * p).astype(np.uint8)
    g = (255.0 * np.sqrt(p) * 0.75).astype(np.uint8)
    b = (255.0 * (1.0 - p) * 0.15).astype(np.uint8)
    return np.stack([r, g, b], axis=2)


def overlay_rgb(pv: np.ndarray, mask: np.ndarray) -> np.ndarray:
    base = gray_rgb(pv).astype(np.float32)
    red = np.zeros_like(base)
    red[..., 0] = 255
    alpha = 0.45 * (mask > 0)[..., None]
    return ((1 - alpha) * base + alpha * red).astype(np.uint8)


def nearest_resize(img: np.ndarray, height: int, width: int) -> np.ndarray:
    ys = np.linspace(0, img.shape[0] - 1, height).astype(int)
    xs = np.linspace(0, img.shape[1] - 1, width).astype(int)
    return img[ys][:, xs]


def panel_image(pv: np.ndarray, label: np.ndarray, prob: np.ndarray, *, scale_h: int = 192, scale_w: int = 256) -> np.ndarray:
    m005 = prob >= 0.05
    m0075 = prob >= 0.075
    tiles = [
        gray_rgb(pv),
        mask_rgb(label > 0, (80, 180, 255)),
        prob_rgb(prob),
        mask_rgb(m005, (255, 80, 70)),
        mask_rgb(m0075, (255, 160, 60)),
        overlay_rgb(pv, m0075),
    ]
    tiles = [nearest_resize(tile, scale_h, scale_w) for tile in tiles]
    gap = np.full((scale_h, 4, 3), 30, dtype=np.uint8)
    row = tiles[0]
    for tile in tiles[1:]:
        row = np.concatenate([row, gap, tile], axis=1)
    return row


def _category_for_record(label: np.ndarray, prob: np.ndarray) -> str:
    true_patch = bool(label.any())
    p005 = patch_prediction(prob, 0.05)
    p0075 = patch_prediction(prob, 0.075)
    if true_patch and p0075:
        return "true_positive_0p075"
    if true_patch and p005:
        return "detected_0p05_missed_0p075"
    if true_patch:
        return "false_negative_0p075"
    if p0075:
        return "false_positive_0p075"
    if p005:
        return "false_positive_0p05"
    return "true_negative"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="training_data/combined_train_overnight.yaml")
    ap.add_argument("--model", default="runs/pv_unet_overnight_20260520_015732/best_model.keras")
    ap.add_argument("--run-dir", default="runs/pv_unet_overnight_20260520_015732")
    ap.add_argument("--split", default="test")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-per-category", type=int, default=12)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    combined_root = Path(cfg["output_root"])
    data_root = Path(args.data_root) if args.data_root else combined_root.parent
    patch_vel = int(cfg["train"]["patch_vel"])
    patch_pos = int(cfg["train"]["patch_pos"])

    from tensorflow import keras

    model = keras.models.load_model(args.model, compile=False)
    out_dir = Path(args.run_dir) / "review_panels"
    index_path = Path(args.run_dir) / "review_panels_index.csv"
    counts: dict[str, int] = {}
    rows: list[dict] = []

    for name in read_manifest(combined_root, args.split):
        paths = source_paths(combined_root, data_root, name)
        if not paths["pv"].exists() or not paths["label"].exists():
            continue
        meta = load_json(paths["meta"])
        label_meta = load_json(paths["label_json"])
        pv = np.load(paths["pv"])
        label = np.load(paths["label"])
        prob = predict_full_pv(model, pv, patch_vel=patch_vel, patch_pos=patch_pos, batch_size=args.batch_size)
        category = _category_for_record(label, prob)

        if meta.get("type") == "catalog_shell" and abs(float(meta.get("offset_fraction", 0.0))) > 0:
            if category.startswith("false_negative"):
                category = "offset_cut_failures"
        if category == "true_negative":
            continue
        if counts.get(category, 0) >= args.max_per_category:
            continue
        counts[category] = counts.get(category, 0) + 1

        stem = Path(name).stem.replace("/", "_")
        filename = f"{category}_{counts[category]:03d}_{stem}.png"
        write_png_rgb(out_dir / filename, panel_image(pv, label, prob))

        mask = prob >= 0.075
        vals = prob[mask]
        rows.append(
            {
                "filename": filename,
                "category": category,
                "galaxy": name.split("__", 1)[0] if "__" in name else meta.get("galaxy"),
                "patch_id": stem,
                "shell_id": meta.get("target_shell_id"),
                "threshold": 0.075,
                "max_probability": float(np.nanmax(prob)),
                "mean_predicted_probability_inside_mask": float(np.nanmean(vals)) if vals.size else 0.0,
                "predicted_mask_area": int(mask.sum()),
                "notes": f"type={meta.get('type')}; orientation={meta.get('orientation')}; offset={meta.get('offset_fraction')}; n_objects={label_meta.get('n_objects')}",
            }
        )
        if all(v >= args.max_per_category for v in counts.values()) and len(counts) >= 5:
            break

    index_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "filename",
        "category",
        "galaxy",
        "patch_id",
        "shell_id",
        "threshold",
        "max_probability",
        "mean_predicted_probability_inside_mask",
        "predicted_mask_area",
        "notes",
    ]
    with index_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[review] wrote {len(rows)} panels to {out_dir.resolve()}")
    print(f"[review] wrote {index_path.resolve()}")


if __name__ == "__main__":
    main()
