from __future__ import annotations

import math
from typing import Any


def validate_board_measurement(metrics: dict[str, Any], measurement: dict[str, Any]) -> dict[str, Any]:
    """Validate a portable board measurement contract without probing hardware."""
    errors: list[dict[str, Any]] = []
    metrics = metrics if isinstance(metrics, dict) else {}
    measurement = measurement if isinstance(measurement, dict) else {}
    for name, value in metrics.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            errors.append({"field": f"metrics.{name}", "reason": "metric must be a finite number"})
    for field, minimum, integer in (("batch_size", 1, True), ("warmup_runs", 0, True), ("iterations", 1, True)):
        if field not in measurement:
            continue
        value = measurement[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            errors.append({"field": f"measurement.{field}", "reason": f"must be an integer >= {minimum}"})
    source = measurement.get("source")
    if source is not None and source not in {"external", "imported", "mock", "fixture"}:
        errors.append({"field": "measurement.source", "reason": "must be external, imported, mock, or fixture"})
    scope = measurement.get("scope")
    if scope is not None and scope not in {"single_image", "batch", "full_dataset", "unknown"}:
        errors.append({"field": "measurement.scope", "reason": "unsupported measurement scope"})
    units = measurement.get("units")
    if units is not None and not isinstance(units, dict):
        errors.append({"field": "measurement.units", "reason": "must be an object mapping metric names to units"})
    required = ("source", "scope", "batch_size", "iterations")
    missing = [field for field in required if field not in measurement]
    return {
        "status": "failed" if errors else "passed" if metrics and not missing else "incomplete",
        "errors": errors,
        "missing_fields": missing,
        "metric_count": len(metrics),
        "source": source or "unknown",
        "scope": scope or "unknown",
    }
