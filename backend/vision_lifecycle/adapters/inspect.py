from __future__ import annotations

from pathlib import Path

from .classification import validate_classification_csv, validate_classification_folder
from .coco import validate_coco
from .jsonl import validate_jsonl
from .yolo import validate_yolo_directory


def inspect_path(path: str) -> dict:
    source = Path(path)
    if source.is_file() and source.suffix.lower() == ".json":
        result = validate_coco(str(source))
        return {"detected_format": "coco", "path": str(source.resolve()), "result": result}
    if source.is_file() and source.suffix.lower() == ".csv":
        result = validate_classification_csv(str(source))
        return {"detected_format": "classification-csv", "path": str(source.resolve()), "result": result}
    if source.is_file() and source.suffix.lower() in {".jsonl", ".ndjson"}:
        result = validate_jsonl(str(source))
        return {"detected_format": "jsonl", "path": str(source.resolve()), "result": result}
    if source.is_dir() and (source / "images").exists() and (source / "labels").exists():
        result = validate_yolo_directory(str(source))
        return {"detected_format": "yolo-txt", "path": str(source.resolve()), "result": result}
    if source.is_dir():
        result = validate_classification_folder(str(source))
        return {"detected_format": "classification-folder", "path": str(source.resolve()), "result": result}
    raise ValueError(f"Path does not exist: {source}")
