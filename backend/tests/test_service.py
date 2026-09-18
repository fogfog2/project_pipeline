from vision_lifecycle.database import Base, SessionLocal, engine
from vision_lifecycle.service import compare_models, seed_demo
from vision_lifecycle.models import ModelVersion
from vision_lifecycle.evaluators.detection import evaluate_coco_predictions


def test_demo_has_comparable_models():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with SessionLocal() as session:
        project = seed_demo(session)
        models = session.query(ModelVersion).filter_by(project_id=project.id).all()
        baseline = next(model for model in models if model.alias == "baseline")
        candidate = next(model for model in models if model.alias == "candidate")
        comparison = compare_models(session, baseline.id, candidate.id)
        assert comparison["compatible"] is True
        assert comparison["delta"]["bbox_mAP"] == 0.014
        assert "latency_ms_p50" not in comparison["delta"]
        assert "target profile" in comparison["metric_compatibility"]["latency_ms_p50"]


def test_detection_evaluator_reads_fixture():
    result = evaluate_coco_predictions(
        "examples/mmdetection/annotations/coco8.json",
        [
            {"image_id": 1, "category_id": 1, "bbox": [120, 85, 118, 290], "score": 0.9},
            {"image_id": 1, "category_id": 3, "bbox": [260, 230, 260, 145], "score": 0.9},
            {"image_id": 2, "category_id": 9, "bbox": [85, 180, 210, 180], "score": 0.9},
        ],
    )
    assert result["bbox_AP50"] == 1.0
    assert result["recall"] == 1.0


def test_detection_evaluator_explains_invalid_predictions():
    result = evaluate_coco_predictions(
        "examples/mmdetection/annotations/coco8.json",
        [
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, -1, 5], "score": 0.9},
            {"image_id": 999, "category_id": 1, "bbox": [0, 0, 1, 1], "score": 0.9},
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 1, 1], "score": float("nan")},
        ],
    )
    assert result["invalid_predictions"] == 3
    assert {item["reason"] for item in result["invalid_prediction_examples"]} == {
        "bbox width and height must be positive", "unknown or invalid image_id", "score must be a finite number",
    }
