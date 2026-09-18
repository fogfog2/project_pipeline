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


def validate_evaluation_definition(definition: dict[str, Any], snapshot: dict[str, Any] | None) -> dict[str, Any]:
    """Validate a reproducible evaluation scope without reading model outputs."""
    definition = definition if isinstance(definition, dict) else {}
    errors: list[dict[str, Any]] = []
    if "items" in definition and "image_ids" in definition:
        errors.append({"field": "definition", "reason": "use either items or image_ids, not both"})
    raw = definition.get("items", definition.get("image_ids"))
    known_ids = _dataset_item_ids(snapshot)
    if raw is None:
        return {
            "status": "passed" if known_ids else "incomplete",
            "errors": errors,
            "scope": "dataset",
            "selected_item_count": len(known_ids) if known_ids else None,
            "known_item_count": len(known_ids) if known_ids else None,
        }
    if not isinstance(raw, list):
        errors.append({"field": "definition.items", "reason": "must be a list"})
        raw = []
    values: list[str] = []
    for index, item in enumerate(raw):
        value = item
        if isinstance(item, dict):
            value = item.get("image_id", item.get("item_id", item.get("id")))
        if value is None or isinstance(value, bool) or not isinstance(value, (str, int, float)):
            errors.append({"index": index, "reason": "each evaluation item needs a scalar image/item ID"})
            continue
        values.append(str(value))
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        errors.append({"field": "definition.items", "reason": "duplicate evaluation item IDs", "items": duplicates[:100]})
    selected_ids = set(values)
    unknown_items = sorted(selected_ids - known_ids) if known_ids and selected_ids else []
    if unknown_items:
        errors.append({"field": "definition.items", "reason": "items are not present in DatasetVersion snapshot", "items": unknown_items[:100]})
    if not selected_ids:
        errors.append({"field": "definition.items", "reason": "evaluation set must contain at least one item"})
    return {
        "status": "passed" if not errors and known_ids else "incomplete" if not errors else "failed",
        "errors": errors,
        "scope": "explicit_items",
        "selected_item_count": len(selected_ids),
        "known_item_count": len(known_ids) if known_ids else None,
        "unknown_items": unknown_items,
        "duplicate_items": duplicates,
    }
