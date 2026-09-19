from __future__ import annotations

import json
import csv
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
                snapshot["class_names"] = [str(item.get("name", item.get("id"))) for item in _canonical_items(categories, "id")]
                snapshot["images"] = _canonical_items(images, "id")
                snapshot["annotations"] = _canonical_items(annotations, "id")
                snapshot["counts"] = {"categories": len(categories), "images": len(images), "annotations": len(annotations)}
        except (OSError, ValueError, json.JSONDecodeError):
            snapshot["parse_status"] = "unavailable"
    elif format.lower() in {"yolo", "yolo-txt"}:
        root = Path(manifest_path or annotation_path or "").expanduser()
        if root.is_dir():
            images: list[dict[str, Any]] = []
            annotations: list[dict[str, Any]] = []
            class_ids: set[int] = set()
            for image in sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"} and "images" in path.parts):
                relative = image.relative_to(root).as_posix()
                image_id = relative
                images.append({"id": image_id, "file_name": relative})
                parts = list(image.parts)
                image_index = parts.index("images")
                label_path = Path(*parts[:image_index], "labels", *parts[image_index + 1:]).with_suffix(".txt")
                if not label_path.is_file():
                    continue
                for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
                    values = line.split()
                    if len(values) != 5:
                        continue
                    try:
                        class_id = int(values[0]); coordinates = [float(value) for value in values[1:]]
                    except ValueError:
                        continue
                    class_ids.add(class_id)
                    annotations.append({"id": f"{image_id}:{line_number}", "image_id": image_id, "category_id": class_id, "bbox": coordinates, "format": "yolo-normalized"})
            snapshot["images"] = _canonical_items(images, "id")
            snapshot["annotations"] = _canonical_items(annotations, "id")
            snapshot["categories"] = [{"id": class_id, "name": str(class_id)} for class_id in sorted(class_ids)]
            snapshot["class_names"] = [str(class_id) for class_id in sorted(class_ids)]
            snapshot["counts"] = {"images": len(images), "annotations": len(annotations), "categories": len(class_ids)}
        else:
            snapshot["parse_status"] = "unavailable"
    elif format.lower() in {"classification", "classification-folder"} and Path(manifest_path or annotation_path or "").expanduser().is_dir():
        root = Path(manifest_path or annotation_path or "").expanduser()
        if root.is_dir():
            images = []
            class_names = []
            for class_path in sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")):
                class_names.append(class_path.name)
                for image in sorted(path for path in class_path.rglob("*") if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}):
                    relative = image.relative_to(root).as_posix()
                    images.append({"id": relative, "file_name": relative, "label": class_path.name})
            snapshot["images"] = _canonical_items(images, "id")
            snapshot["class_names"] = class_names
            snapshot["counts"] = {"images": len(images), "categories": len(class_names)}
    elif format.lower() in {"classification", "classification-folder"}:
        snapshot["parse_status"] = "unavailable"
    elif format.lower() in {"classification-csv", "csv", "classification"} and manifest_path and Path(manifest_path).is_file():
        try:
            with Path(manifest_path).open(encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
            images = [{"id": str(row.get("path", index)), "file_name": row.get("path", ""), "label": row.get("label", "")} for index, row in enumerate(rows)]
            class_names = sorted({str(item["label"]) for item in images if item.get("label")})
            snapshot["images"] = _canonical_items(images, "id")
            snapshot["class_names"] = class_names
            snapshot["counts"] = {"images": len(images), "categories": len(class_names)}
        except (OSError, csv.Error):
            snapshot["parse_status"] = "unavailable"
    elif format.lower() in {"jsonl", "ndjson", "common-jsonl"} and manifest_path and Path(manifest_path).is_file():
        try:
            items: list[dict[str, Any]] = []
            parse_errors = 0
            with Path(manifest_path).open(encoding="utf-8") as file:
                for index, line in enumerate(file):
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError:
                        parse_errors += 1
                        continue
                    if not isinstance(value, dict):
                        parse_errors += 1
                        continue
                    item = dict(value)
                    item["id"] = str(item.get("id", item.get("image_id", item.get("path", index))))
                    items.append(item)
            snapshot["items"] = _canonical_items(items, "id")
            snapshot["counts"] = {"items": len(items)}
            snapshot["parse_errors"] = parse_errors
            snapshot["parse_status"] = "complete" if parse_errors == 0 else "partial"
            labels = sorted({str(item["label"]) for item in items if item.get("label") is not None})
            if labels:
                snapshot["class_names"] = labels
                snapshot["counts"]["categories"] = len(labels)
        except (OSError, ValueError):
            snapshot["parse_status"] = "unavailable"
    return snapshot


def diff_dataset_snapshots(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Compare two snapshots without exposing source paths."""
    result: dict[str, Any] = {"changed": [], "added": {}, "removed": {}, "modified": {}}
    for collection in ("categories", "images", "annotations", "items"):
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
