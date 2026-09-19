from pathlib import Path

from vision_lifecycle.adapters.coco import validate_coco
from vision_lifecycle.adapters.yolo import validate_yolo_directory
from vision_lifecycle.adapters.jsonl import validate_jsonl
from vision_lifecycle.dataset_snapshot import build_dataset_snapshot


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


def test_yolo_and_classification_snapshots_are_portable(tmp_path: Path):
    from PIL import Image

    yolo = tmp_path / "yolo"
    (yolo / "images" / "train").mkdir(parents=True)
    (yolo / "labels" / "train").mkdir(parents=True)
    Image.new("RGB", (4, 3), (1, 2, 3)).save(yolo / "images" / "train" / "one.png")
    (yolo / "labels" / "train" / "one.txt").write_text("2 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    yolo_snapshot = build_dataset_snapshot(task_kind="detection", format="yolo-txt", manifest_path=str(yolo), annotation_path=None)
    assert yolo_snapshot["counts"] == {"images": 1, "annotations": 1, "categories": 1}
    assert yolo_snapshot["images"][0]["id"] == "images/train/one.png"
    assert yolo_snapshot["annotations"][0]["category_id"] == 2

    classification = tmp_path / "classification"
    (classification / "cat").mkdir(parents=True)
    Image.new("RGB", (2, 2), (4, 5, 6)).save(classification / "cat" / "one.png")
    classification_snapshot = build_dataset_snapshot(task_kind="classification", format="classification-folder", manifest_path=str(classification), annotation_path=None)
    assert classification_snapshot["class_names"] == ["cat"]
    assert classification_snapshot["images"][0]["label"] == "cat"


def test_jsonl_manifest_is_validated_and_diffed(tmp_path: Path):
    manifest = tmp_path / "records.jsonl"
    manifest.write_text('{"id":"one","path":"images/one.jpg","label":"cat"}\n{"id":"two","path":"images/two.jpg","label":"dog"}\n', encoding="utf-8")
    result = validate_jsonl(str(manifest))
    assert result["status"] == "passed" and result["record_count"] == 2
    snapshot = build_dataset_snapshot(task_kind="classification", format="jsonl", manifest_path=str(manifest), annotation_path=None)
    assert snapshot["counts"] == {"items": 2, "categories": 2}
    assert snapshot["class_names"] == ["cat", "dog"]
    manifest.write_text('{"id":"one","path":"images/one.jpg","label":"cat"}\n{"id":"two","path":"images/two.jpg","label":"bird"}\n', encoding="utf-8")
    changed = build_dataset_snapshot(task_kind="classification", format="jsonl", manifest_path=str(manifest), annotation_path=None)
    from vision_lifecycle.dataset_snapshot import diff_dataset_snapshots
    assert diff_dataset_snapshots(snapshot, changed)["modified"]["items"] == ["two"]
