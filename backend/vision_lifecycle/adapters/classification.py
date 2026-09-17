from __future__ import annotations

import csv
from pathlib import Path

from .yolo import IMAGE_EXTENSIONS


def validate_classification_folder(root_path: str) -> dict:
    root = Path(root_path)
    if not root.is_dir():
        raise ValueError(f"Classification root does not exist: {root}")
    classes = []
    total = 0
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        count = sum(1 for path in child.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
        classes.append({"name": child.name, "images": count})
        total += count
    return {"status": "passed" if classes else "failed", "format": "classification-folder", "classes": classes, "images": total, "errors": [] if classes else [{"reason": "no_class_directories"}]}


def validate_classification_csv(path: str) -> dict:
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"Classification CSV does not exist: {source}")
    with source.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required = {"path", "label"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("Classification CSV requires path,label headers")
        rows = list(reader)
    labels = sorted({row["label"] for row in rows if row.get("label")})
    missing_paths = [row["path"] for row in rows if not (source.parent / row["path"]).is_file()]
    return {"status": "passed" if not missing_paths else "failed", "format": "classification-csv", "images": len(rows), "class_names": labels, "missing_paths": missing_paths, "errors": []}
