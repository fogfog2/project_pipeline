from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .fingerprints import dataset_fingerprint


def _canonical_items(items: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    return sorted(items, key=lambda item: (str(item.get(key, "")), json.dumps(item, sort_keys=True, ensure_ascii=False)))


def build_dataset_snapshot(*, task_kind: str, format: str, manifest_path: str | None, annotation_path: str | None) -> dict[str, Any]:
    """Build a deterministic, path-independent snapshot for a DatasetVersion."""
    content_hash, fingerprints = dataset_fingerprint(manifest_path, annotation_path)
    snapshot: dict[str, Any] = {
        "schema_version": "1.0",
        "task_kind": task_kind,
        "format": format,
        "content_hash": content_hash,
        "source_hashes": sorted(fingerprints.values()),
        "counts": {},
    }
    if format.lower() == "coco" and annotation_path and Path(annotation_path).is_file():
        try:
            data = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
            if isinstance(data, dict):
                categories = [{key: value for key, value in item.items() if key in {"id", "name", "supercategory"}} for item in data.get("categories", []) if isinstance(item, dict)]
                images = [{key: value for key, value in item.items() if key in {"id", "file_name", "width", "height", "license", "date_captured"}} for item in data.get("images", []) if isinstance(item, dict)]
                annotations = [{key: value for key, value in item.items() if key in {"id", "image_id", "category_id", "bbox", "area", "iscrowd", "segmentation"}} for item in data.get("annotations", []) if isinstance(item, dict)]
                snapshot["categories"] = _canonical_items(categories, "id")
                snapshot["images"] = _canonical_items(images, "id")
                snapshot["annotations"] = _canonical_items(annotations, "id")
                snapshot["counts"] = {"categories": len(categories), "images": len(images), "annotations": len(annotations)}
        except (OSError, ValueError, json.JSONDecodeError):
            snapshot["parse_status"] = "unavailable"
    return snapshot


def diff_dataset_snapshots(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Compare two snapshots without exposing source paths."""
    result: dict[str, Any] = {"changed": [], "added": {}, "removed": {}, "modified": {}}
    for collection in ("categories", "images", "annotations"):
        left_items = {str(item.get("id")): item for item in left.get(collection, []) if isinstance(item, dict) and item.get("id") is not None}
        right_items = {str(item.get("id")): item for item in right.get(collection, []) if isinstance(item, dict) and item.get("id") is not None}
        added = sorted(set(right_items) - set(left_items))
        removed = sorted(set(left_items) - set(right_items))
        modified = sorted(key for key in set(left_items) & set(right_items) if left_items[key] != right_items[key])
        if added:
            result["added"][collection] = added
        if removed:
            result["removed"][collection] = removed
        if modified:
            result["modified"][collection] = modified
    result["changed"] = sorted(set(result["added"]) | set(result["removed"]) | set(result["modified"]))
    result["same_content_hash"] = left.get("content_hash") == right.get("content_hash")
    result["counts"] = {"left": left.get("counts", {}), "right": right.get("counts", {})}
    return result
