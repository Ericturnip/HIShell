from __future__ import annotations

import csv
import json
from collections import deque
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def read_manifest(root: Path, split: str) -> list[str]:
    path = root / "splits" / f"{split}_manifest.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def split_combined_name(name: str) -> tuple[str, str]:
    if "__" not in name:
        return "", name
    galaxy, rest = name.split("__", 1)
    return galaxy, rest


def source_paths(combined_root: Path, data_root: Path, name: str) -> dict[str, Path]:
    galaxy, rest = split_combined_name(name)
    if galaxy:
        base = data_root / galaxy
        return {
            "pv": base / "pv" / rest,
            "label": base / "labels" / rest,
            "label_json": base / "labels" / f"{Path(rest).stem}.json",
            "meta": base / "pv" / f"{Path(rest).stem}.json",
            "posxy": base / "pv" / f"{Path(rest).stem}_posxy.npy",
        }
    return {
        "pv": combined_root / "pv" / name,
        "label": combined_root / "labels" / name,
        "label_json": combined_root / "labels" / f"{Path(name).stem}.json",
        "meta": combined_root / "pv" / f"{Path(name).stem}.json",
        "posxy": combined_root / "pv" / f"{Path(name).stem}_posxy.npy",
    }


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def zscore_finite(pv: np.ndarray) -> np.ndarray:
    finite = pv[np.isfinite(pv)]
    if finite.size == 0:
        return np.nan_to_num(pv, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    mu = float(np.mean(finite))
    sigma = float(np.std(finite) + 1e-6)
    return np.nan_to_num((pv - mu) / sigma, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def pad_to_patch(
    pv: np.ndarray,
    label: np.ndarray | None,
    patch_vel: int,
    patch_pos: int,
) -> tuple[np.ndarray, np.ndarray | None, tuple[int, int, int, int]]:
    v, s = pv.shape
    dv = max(0, int(patch_vel) - v)
    ds = max(0, int(patch_pos) - s)
    pad = (dv // 2, dv - dv // 2, ds // 2, ds - ds // 2)
    if dv or ds:
        pv = np.pad(pv, ((pad[0], pad[1]), (pad[2], pad[3])), mode="edge")
        if label is not None:
            label = np.pad(label, ((pad[0], pad[1]), (pad[2], pad[3])), mode="constant", constant_values=0)
    return pv, label, pad


def crop_from_pad(arr: np.ndarray, original_shape: tuple[int, int], pad: tuple[int, int, int, int]) -> np.ndarray:
    v, s = original_shape
    return arr[pad[0] : pad[0] + v, pad[2] : pad[2] + s]


def predict_full_pv(
    model: Any,
    pv: np.ndarray,
    *,
    patch_vel: int,
    patch_pos: int,
    batch_size: int = 16,
    normalize: bool = True,
    stride_vel: int | None = None,
    stride_pos: int | None = None,
) -> np.ndarray:
    """Predict a full PV probability map by padding/tiling and averaging overlaps."""
    original_shape = tuple(pv.shape)
    x = zscore_finite(pv) if normalize else np.nan_to_num(pv, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    x, _, pad = pad_to_patch(x, None, patch_vel, patch_pos)
    vdim, sdim = x.shape

    stride_vel = int(stride_vel or max(1, patch_vel // 2))
    stride_pos = int(stride_pos or max(1, patch_pos // 2))
    ys = list(range(0, max(1, vdim - patch_vel + 1), stride_vel))
    xs = list(range(0, max(1, sdim - patch_pos + 1), stride_pos))
    if ys[-1] != vdim - patch_vel:
        ys.append(vdim - patch_vel)
    if xs[-1] != sdim - patch_pos:
        xs.append(sdim - patch_pos)

    out = np.zeros((vdim, sdim), dtype=np.float32)
    count = np.zeros((vdim, sdim), dtype=np.float32)
    buf: list[np.ndarray] = []
    coords: list[tuple[int, int]] = []
    for y0 in ys:
        for x0 in xs:
            buf.append(x[y0 : y0 + patch_vel, x0 : x0 + patch_pos][..., None])
            coords.append((y0, x0))
            if len(buf) >= batch_size:
                probs = model.predict(np.stack(buf, axis=0), verbose=0)[..., 0]
                for (yy, xx), prob in zip(coords, probs):
                    out[yy : yy + patch_vel, xx : xx + patch_pos] += prob
                    count[yy : yy + patch_vel, xx : xx + patch_pos] += 1.0
                buf.clear()
                coords.clear()
    if buf:
        probs = model.predict(np.stack(buf, axis=0), verbose=0)[..., 0]
        for (yy, xx), prob in zip(coords, probs):
            out[yy : yy + patch_vel, xx : xx + patch_pos] += prob
            count[yy : yy + patch_vel, xx : xx + patch_pos] += 1.0

    count[count == 0] = 1.0
    prob = np.clip(out / count, 0.0, 1.0)
    return crop_from_pad(prob, original_shape, pad)


def pixel_metrics(prob: np.ndarray, label: np.ndarray, threshold: float) -> dict[str, float | int]:
    y = label > 0.5
    p = prob >= float(threshold)
    tp = int(np.logical_and(p, y).sum())
    fp = int(np.logical_and(p, ~y).sum())
    fn = int(np.logical_and(~p, y).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def patch_prediction(prob: np.ndarray, threshold: float) -> bool:
    return bool(np.nanmax(prob) >= float(threshold)) if prob.size else False


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    mask = np.asarray(mask, dtype=bool)
    labels = np.zeros(mask.shape, dtype=np.int32)
    h, w = mask.shape
    current = 0
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or labels[y, x]:
                continue
            current += 1
            q: deque[tuple[int, int]] = deque([(y, x)])
            labels[y, x] = current
            while q:
                yy, xx = q.popleft()
                for ny, nx in ((yy - 1, xx), (yy + 1, xx), (yy, xx - 1), (yy, xx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not labels[ny, nx]:
                        labels[ny, nx] = current
                        q.append((ny, nx))
    return labels, current


def component_features(
    prob: np.ndarray,
    mask: np.ndarray,
    *,
    threshold: float,
    meta: dict[str, Any] | None = None,
    min_area: int = 1,
) -> list[dict[str, Any]]:
    labels, n = connected_components(mask)
    meta = meta or {}
    vel_kms = meta.get("vel_kms") or []
    if not vel_kms:
        vel_kms = meta.get("velocity_axis_kms") or []
    pos_axis = meta.get("pos_axis_pix") or []
    if len(vel_kms) >= 3 and isinstance(vel_kms[0], (int, float)) and isinstance(vel_kms[2], (int, float)):
        if len(vel_kms) == 3 and abs(float(vel_kms[2])) > 0:
            vel_step = abs(float(vel_kms[2]))
        else:
            vel_step = abs(float(np.nanmedian(np.diff(np.asarray(vel_kms, dtype=float)))))
    else:
        vel_step = None
    pos_step = abs(float(meta.get("pos_step_pix", 1.0)))
    beam_pix = meta.get("beam_fwhm_pix")
    pixel_window_size = meta.get("pixel_window_size")
    spatial_window_kpc = meta.get("spatial_window_kpc")
    rows: list[dict[str, Any]] = []
    for comp_id in range(1, n + 1):
        yy, xx = np.where(labels == comp_id)
        area = int(yy.size)
        if area < int(min_area):
            continue
        vals = prob[yy, xx]
        y0, y1 = int(yy.min()), int(yy.max())
        x0, x1 = int(xx.min()), int(xx.max())
        height = y1 - y0 + 1
        width = x1 - x0 + 1
        edge_touching = y0 == 0 or x0 == 0 or y1 == prob.shape[0] - 1 or x1 == prob.shape[1] - 1
        rows.append(
            {
                "component_id": int(comp_id),
                "threshold": float(threshold),
                "area_pix": area,
                "max_probability": float(np.nanmax(vals)),
                "mean_probability": float(np.nanmean(vals)),
                "integrated_probability_mass": float(np.nansum(vals)),
                "bbox_v0": y0,
                "bbox_pos0": x0,
                "bbox_v1": y1,
                "bbox_pos1": x1,
                "centroid_v": float(np.mean(yy)),
                "centroid_pos": float(np.mean(xx)),
                "aspect_ratio_pos_over_vel": float(width / max(1, height)),
                "major_extent_pix": int(max(width, height)),
                "minor_extent_pix": int(min(width, height)),
                "distance_from_image_edge_pix": int(min(y0, x0, prob.shape[0] - 1 - y1, prob.shape[1] - 1 - x1)),
                "edge_touching": bool(edge_touching),
                "velocity_extent_kms": None if vel_step is None else float(height * vel_step),
                "spatial_extent_pix": int(width),
                "spatial_extent_native": float(width * pos_step),
                "spatial_extent_kpc": (
                    None
                    if not pixel_window_size or not spatial_window_kpc
                    else float(width * float(spatial_window_kpc) / max(float(pixel_window_size), 1e-9))
                ),
                "area_in_beam_units": (
                    None
                    if not beam_pix
                    else float(area / max(float(beam_pix) ** 2, 1e-9))
                ),
                "has_pos_axis_metadata": bool(pos_axis),
            }
        )
    largest = max((r["area_pix"] for r in rows), default=0)
    for row in rows:
        row["component_count_in_patch"] = len(rows)
        row["largest_component_area_pix"] = largest
    return rows


def write_csv(rows: Iterable[dict[str, Any]], path: Path) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
