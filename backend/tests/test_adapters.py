from pathlib import Path

from vision_lifecycle.adapters.coco import validate_coco
from vision_lifecycle.adapters.yolo import validate_yolo_directory


def test_coco_fixture_has_non_contiguous_categories():
    result = validate_coco("examples/mmdetection/annotations/coco8.json")
    assert result["status"] == "passed"
    assert result["category_ids"] == [1, 3, 9]


def test_yolo_missing_label_is_reported(tmp_path: Path):
    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "labels" / "train").mkdir(parents=True)
    (tmp_path / "images" / "train" / "one.jpg").write_bytes(b"fixture")
    result = validate_yolo_directory(str(tmp_path))
    assert result["missing_label_files"] == ["images/train/one.jpg"]
    assert result["status"] == "passed"
