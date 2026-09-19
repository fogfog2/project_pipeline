from __future__ import annotations

from typing import Any


def validate_encoding_document(document: Any) -> dict[str, Any]:
    """Validate the portable subset of common quantization encoding JSON files."""
    if not isinstance(document, dict):
        return {"status": "failed", "schema": "unknown", "errors": ["encoding document must be a JSON object"]}
    if "scale" in document:
        if isinstance(document["scale"], bool) or not isinstance(document["scale"], (int, float)) or float(document["scale"]) <= 0:
            return {"status": "failed", "schema": "legacy-scale", "errors": ["scale must be a positive number"]}
        return {"status": "passed", "schema": "legacy-scale", "tensor_count": 1, "entry_count": 1}
    encodings = document.get("encodings", document.get("tensor_encodings"))
    if isinstance(encodings, dict):
        entries = []
        for name, value in encodings.items():
            if not isinstance(value, dict):
                return {"status": "failed", "schema": "tensor-map", "errors": [f"encoding for {name} must be an object"]}
            entries.append({"name": str(name), **value})
        encodings = entries
    if not isinstance(encodings, list) or not encodings:
        return {"status": "failed", "schema": "unknown", "errors": ["expected a non-empty encodings list or tensor_encodings map"]}
    errors: list[str] = []
    for index, item in enumerate(encodings):
        if not isinstance(item, dict):
            errors.append(f"encodings[{index}] must be an object")
            continue
        if not item.get("name"):
            errors.append(f"encodings[{index}].name is required")
        if "scale" in item and (isinstance(item["scale"], bool) or not isinstance(item["scale"], (int, float)) or float(item["scale"]) <= 0):
            errors.append(f"encodings[{index}].scale must be positive")
        if "min" in item and "max" in item:
            if not all(isinstance(item[key], (int, float)) and not isinstance(item[key], bool) for key in ("min", "max")):
                errors.append(f"encodings[{index}] min/max must be numeric")
            elif float(item["min"]) > float(item["max"]):
                errors.append(f"encodings[{index}] min cannot exceed max")
    return {"status": "failed" if errors else "passed", "schema": "tensor-encodings-v1", "tensor_count": len(encodings), "entry_count": len(encodings), "errors": errors}
