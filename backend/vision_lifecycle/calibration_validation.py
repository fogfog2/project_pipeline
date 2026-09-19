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

    Reports decoded source statistics and bounded statistics after the declared
    preprocessing contract. These are observational calibration diagnostics,
    not quantizer encoding values.
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
    tensor_sum = 0.0
    tensor_sq_sum = 0.0
    tensor_count = 0
    preprocessing_errors: list[str] = []
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
                prepared = rgb
                requested_size = preprocessing.get("input_size", preprocessing.get("resize")) if isinstance(preprocessing, dict) else None
                if isinstance(requested_size, dict):
                    requested_size = [requested_size.get("width"), requested_size.get("height")]
                if requested_size is not None:
                    if not isinstance(requested_size, (list, tuple)) or len(requested_size) != 2 or any(not isinstance(value, int) or value <= 0 for value in requested_size):
                        preprocessing_errors.append(f"{image_id}: input_size/resize must be [width, height]")
                    else:
                        prepared = prepared.resize((int(requested_size[0]), int(requested_size[1])))
                tensor_array = np.asarray(prepared, dtype=np.float64)
                if str(preprocessing.get("color", "rgb")).lower() == "bgr":
                    tensor_array = tensor_array[..., ::-1]
                elif str(preprocessing.get("color", "rgb")).lower() not in {"rgb", "bgr"}:
                    preprocessing_errors.append(f"{image_id}: color must be rgb or bgr")
                normalization = preprocessing.get("normalization", preprocessing)
                mean_values = normalization.get("mean") if isinstance(normalization, dict) else None
                std_values = normalization.get("std") if isinstance(normalization, dict) else None
                if mean_values is not None or std_values is not None:
                    if not isinstance(mean_values, (list, tuple)) or not isinstance(std_values, (list, tuple)) or len(mean_values) != 3 or len(std_values) != 3 or any(float(value) == 0 for value in std_values):
                        preprocessing_errors.append(f"{image_id}: mean/std must contain three non-zero values")
                    else:
                        tensor_array = (tensor_array - np.asarray(mean_values, dtype=np.float64)) / np.asarray(std_values, dtype=np.float64)
                scale = preprocessing.get("scale")
                if scale is not None:
                    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
                        preprocessing_errors.append(f"{image_id}: scale must be numeric")
                    else:
                        tensor_array *= float(scale)
                tensor_sum += float(tensor_array.sum())
                tensor_sq_sum += float(np.square(tensor_array).sum())
                tensor_count += int(tensor_array.size)
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
    tensor_mean = tensor_sum / tensor_count if tensor_count else None
    tensor_variance = max(0.0, tensor_sq_sum / tensor_count - tensor_mean * tensor_mean) if tensor_mean is not None else None
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
        "tensor_mean": round(tensor_mean, 6) if tensor_mean is not None else None,
        "tensor_std": round(tensor_variance ** 0.5, 6) if tensor_variance is not None else None,
        "tensor_value_count": tensor_count,
        "preprocessing_applied": not preprocessing_errors,
        "preprocessing_errors": preprocessing_errors[:50],
        "total_bytes": total_bytes,
        "preprocessing_declared": bool(preprocessing),
        "preprocessing_keys": sorted(str(key) for key in preprocessing),
        "source": "decoded-image-sample",
    }
