from __future__ import annotations

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
