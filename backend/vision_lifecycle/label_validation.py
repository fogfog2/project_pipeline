"""Validation helpers for versioned label schema contracts."""

from __future__ import annotations

from typing import Any


def validate_label_schema(classes: list[dict[str, Any]], mapping: dict[str, Any]) -> list[str]:
    """Return human-readable contract errors; an empty list means valid.

    Empty classes are allowed for a draft/unknown label schema. Once classes are
    supplied, IDs and names must be unique and mapping targets must resolve to
    one of those IDs. This avoids silently creating an untraceable label map.
    """
    errors: list[str] = []
    ids: list[Any] = []
    names: list[str] = []
    for index, item in enumerate(classes):
        if not isinstance(item, dict):
            errors.append(f"classes[{index}] must be an object")
            continue
        if "id" not in item:
            errors.append(f"classes[{index}] is missing id")
        else:
            class_id = item["id"]
            if isinstance(class_id, bool) or not isinstance(class_id, (str, int, float)) or (isinstance(class_id, str) and not class_id.strip()):
                errors.append(f"classes[{index}].id must be a non-empty string or number")
            elif class_id in ids:
                errors.append(f"duplicate class id: {class_id}")
            else:
                ids.append(class_id)
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            errors.append(f"classes[{index}].name must be a non-empty string")
        elif name.strip() in names:
            errors.append(f"duplicate class name: {name.strip()}")
        else:
            names.append(name.strip())

    if not isinstance(mapping, dict):
        errors.append("mapping must be an object")
    else:
        for source, target in mapping.items():
            if not isinstance(source, str) or not source.strip():
                errors.append("mapping keys must be non-empty strings")
            # A null target is useful while a draft mapping is being completed.
            if target is not None and target not in ids:
                errors.append(f"mapping target for '{source}' is not declared in classes: {target}")
    return errors
