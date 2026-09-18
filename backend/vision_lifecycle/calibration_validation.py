from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any


def _dataset_item_ids(snapshot: dict[str, Any] | None) -> set[str]:
    if not isinstance(snapshot, dict):
        return set()
    return {
        str(item.get("id"))
        for item in snapshot.get("images", [])
        if isinstance(item, dict) and item.get("id") is not None
    }


def validate_calibration_definition(sampling: dict[str, Any], preprocessing: dict[str, Any], snapshot: dict[str, Any] | None) -> dict[str, Any]:
    """Validate a portable calibration contract without reading source files."""
    errors: list[dict[str, Any]] = []
    sampling = sampling if isinstance(sampling, dict) else {}
    preprocessing = preprocessing if isinstance(preprocessing, dict) else {}
    item_values = sampling.get("items", sampling.get("image_ids", []))
    items = [str(value) for value in item_values] if isinstance(item_values, list) else []
    if item_values and not isinstance(item_values, list):
        errors.append({"field": "sampling.items", "reason": "must be a list of Dataset image IDs"})
    if len(items) != len(set(items)):
        errors.append({"field": "sampling.items", "reason": "duplicate calibration item IDs"})
    known_ids = _dataset_item_ids(snapshot)
    unknown_items = sorted(set(items) - known_ids) if known_ids and items else []
    if unknown_items:
        errors.append({"field": "sampling.items", "reason": "items are not present in DatasetVersion snapshot", "items": unknown_items[:50]})
    count = sampling.get("count", len(items) if items else None)
    if count is not None and (not isinstance(count, int) or isinstance(count, bool) or count <= 0):
        errors.append({"field": "sampling.count", "reason": "must be a positive integer"})
    if isinstance(count, int) and known_ids and count > len(known_ids):
        errors.append({"field": "sampling.count", "reason": "requested count exceeds DatasetVersion image count", "available": len(known_ids)})
    strategy = sampling.get("strategy", "explicit" if items else "unspecified")
    allowed_strategies = {"explicit", "random", "stratified", "sequential", "unknown", "unspecified"}
    if strategy not in allowed_strategies:
        errors.append({"field": "sampling.strategy", "reason": f"unsupported strategy: {strategy}"})
    if strategy in {"random", "stratified"} and "seed" not in sampling:
        errors.append({"field": "sampling.seed", "reason": "seed is required for reproducible random/stratified sampling"})
    if not preprocessing:
        errors.append({"field": "preprocessing", "reason": "preprocessing declaration is missing"})
    statistics = {
        "dataset_image_count": len(known_ids) if known_ids else None,
        "requested_count": count,
        "explicit_item_count": len(items),
        "strategy": strategy,
        "seed": sampling.get("seed"),
        "preprocessing_keys": sorted(str(key) for key in preprocessing),
    }
    return {"status": "failed" if errors else "passed", "errors": errors, "statistics": statistics}


def inspect_calibration_statistics(
    *,
    sampling: dict[str, Any],
    preprocessing: dict[str, Any],
    dataset_annotation_path: str | None,
    dataset_manifest_path: str | None,
    snapshot: dict[str, Any] | None,
    max_items: int = 256,
) -> dict[str, Any]:
    """Inspect a bounded sample of source images for calibration evidence.

    This deliberately reports source availability and decoded image statistics;
    it does not apply preprocessing or pretend that those values are tensor
    calibration encodings. The latter must be imported from the quantizer.
    """
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    images = [item for item in snapshot.get("images", []) if isinstance(item, dict)]
    requested = sampling.get("items", sampling.get("image_ids", []))
    requested_ids = {str(value) for value in requested} if isinstance(requested, list) else set()
    if requested_ids:
        selected = [item for item in images if str(item.get("id")) in requested_ids]
    else:
        count = sampling.get("count")
        limit = count if isinstance(count, int) and count > 0 else len(images)
        selected = images[: min(limit, max_items)]
    selected = selected[:max_items]
    base_path = Path(dataset_annotation_path or dataset_manifest_path).expanduser().parent if (dataset_annotation_path or dataset_manifest_path) else None
    missing: list[str] = []
    decode_errors: list[str] = []
    resolutions: Counter[str] = Counter()
    channels: Counter[str] = Counter()
    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    pixel_count = 0
    total_bytes = 0
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return {
            "status": "unavailable",
            "reason": "Pillow and numpy are required for image statistics",
            "requested_count": len(selected),
            "preprocessing_declared": bool(preprocessing),
        }
    for image in selected:
        image_id = str(image.get("id", image.get("file_name", "unknown")))
        file_name = image.get("file_name") or image.get("path")
        if not file_name:
            missing.append(image_id)
            continue
        path = Path(str(file_name)).expanduser()
        if not path.is_absolute() and base_path:
            path = base_path / path
        if not path.is_file():
            missing.append(image_id)
            continue
        try:
            with Image.open(path) as opened:
                rgb = opened.convert("RGB")
                array = np.asarray(rgb, dtype=np.float64)
            height, width = array.shape[:2]
            resolutions[f"{width}x{height}"] += 1
            channels[str(array.shape[2] if array.ndim == 3 else 1)] += 1
            pixel_sum += float(array.sum())
            pixel_sq_sum += float(np.square(array).sum())
            pixel_count += int(array.size)
            total_bytes += path.stat().st_size
        except (OSError, ValueError, RuntimeError):
            decode_errors.append(image_id)
    mean = pixel_sum / pixel_count if pixel_count else None
    variance = max(0.0, pixel_sq_sum / pixel_count - mean * mean) if mean is not None else None
    status = "complete" if selected and not missing and not decode_errors else "partial" if selected else "unavailable"
    return {
        "status": status,
        "requested_count": len(selected),
        "inspected_count": len(selected) - len(missing) - len(decode_errors),
        "missing_image_ids": missing[:50],
        "decode_error_image_ids": decode_errors[:50],
        "resolutions": dict(resolutions),
        "channels": dict(channels),
        "pixel_mean_0_255": round(mean, 6) if mean is not None else None,
        "pixel_std_0_255": round(variance ** 0.5, 6) if variance is not None else None,
        "total_bytes": total_bytes,
        "preprocessing_declared": bool(preprocessing),
        "preprocessing_keys": sorted(str(key) for key in preprocessing),
        "source": "decoded-image-sample",
    }
