from __future__ import annotations

import json
from pathlib import Path


def load_coco(path: str) -> dict:
    with Path(path).open(encoding="utf-8") as file:
        data = json.load(file)
    required = {"images", "annotations", "categories"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"COCO annotation is missing keys: {', '.join(sorted(missing))}")
    return data


def validate_coco(path: str) -> dict:
    data = load_coco(path)
    images = {image["id"]: image for image in data["images"]}
    categories = {category["id"]: category for category in data["categories"]}
    errors: list[dict] = []
    for annotation in data["annotations"]:
        identifier = annotation.get("id", "unknown")
        if annotation.get("image_id") not in images:
            errors.append({"annotation_id": identifier, "reason": "unknown_image_id"})
        if annotation.get("category_id") not in categories:
            errors.append({"annotation_id": identifier, "reason": "unknown_category_id"})
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4 or any(not isinstance(value, (int, float)) for value in bbox):
            errors.append({"annotation_id": identifier, "reason": "invalid_bbox"})
        elif bbox[2] <= 0 or bbox[3] <= 0:
            errors.append({"annotation_id": identifier, "reason": "non_positive_bbox"})
    names = [category.get("name") for category in data["categories"]]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        errors.append({"reason": "duplicate_category_names", "names": duplicate_names})
    return {
        "status": "passed" if not errors else "failed",
        "images": len(images),
        "annotations": len(data["annotations"]),
        "categories": len(categories),
        "category_ids": sorted(categories),
        "class_names": [categories[key].get("name", str(key)) for key in sorted(categories)],
        "errors": errors,
    }
