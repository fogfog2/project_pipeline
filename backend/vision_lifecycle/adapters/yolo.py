from __future__ import annotations

from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def validate_yolo_directory(root_path: str) -> dict:
    root = Path(root_path)
    if not root.is_dir():
        raise ValueError(f"YOLO dataset directory does not exist: {root}")
    image_files = [path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS and "images" in path.parts]
    missing_labels: list[str] = []
    errors: list[dict] = []
    label_count = 0
    class_ids: set[int] = set()
    for image in image_files:
        parts = list(image.parts)
        index = parts.index("images")
        label_path = Path(*parts[:index], "labels", *parts[index + 1:]).with_suffix(".txt")
        if not label_path.exists():
            missing_labels.append(str(image.relative_to(root)))
            continue
        label_count += 1
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            values = line.split()
            if len(values) != 5:
                errors.append({"file": str(label_path.relative_to(root)), "line": line_number, "reason": "expected_5_values"})
                continue
            try:
                class_id = int(values[0])
                coordinates = [float(value) for value in values[1:]]
            except ValueError:
                errors.append({"file": str(label_path.relative_to(root)), "line": line_number, "reason": "non_numeric_label"})
                continue
            if class_id < 0 or any(value < 0 or value > 1 for value in coordinates) or coordinates[2] <= 0 or coordinates[3] <= 0:
                errors.append({"file": str(label_path.relative_to(root)), "line": line_number, "reason": "invalid_normalized_bbox"})
                continue
            class_ids.add(class_id)
    return {
        "status": "passed" if not errors else "failed",
        "format": "yolo-txt",
        "images": len(image_files),
        "labels": label_count,
        "missing_label_files": missing_labels,
        "class_ids": sorted(class_ids),
        "errors": errors,
        "note": "Missing label files are reported, not automatically treated as background images.",
    }
