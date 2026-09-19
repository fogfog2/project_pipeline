from vision_lifecycle.database import Base, SessionLocal, engine
from vision_lifecycle.service import compare_models, seed_demo
from vision_lifecycle.models import DatasetVersion, ModelVersion, Project, Run
from vision_lifecycle.evaluators.detection import evaluate_coco_predictions


def test_report_keeps_whole_dataset_evaluations_separate():
    from vision_lifecycle.service import evaluation_set_report, safe_export
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with SessionLocal() as session:
        project = Project(name="separate-report")
        session.add(project); session.flush()
        datasets = [DatasetVersion(project_id=project.id, name="validation", version=version, task_kind="classification", format="classification-csv") for version in ("v1", "v2")]
        session.add_all(datasets); session.flush()
        for dataset in datasets:
            session.add(Run(project_id=project.id, dataset_id=dataset.id, kind="evaluation", name=dataset.version, status="completed", metrics={"accuracy": 0.5}))
        session.commit()
        groups = evaluation_set_report(session, project.id)
        assert len(groups) == 2
        assert {group["dataset_version"] for group in groups} == {"v1", "v2"}
        assert all(len(group["runs"]) == 1 for group in groups)
        assert safe_export(session, project.id)["evaluation_set_report"] == groups


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


def test_comparison_ignores_incompatible_or_incomplete_latest_runs():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with SessionLocal() as session:
        project = Project(name="comparison-contract")
        session.add(project); session.flush()
        dataset = DatasetVersion(project_id=project.id, name="eval", version="v1", task_kind="detection", format="coco")
        baseline = ModelVersion(project_id=project.id, name="base", version="v1", family="base", task_kind="detection", format="onnx", alias="baseline", metadata_json={"class_mapping_version": "labels-v1"})
        candidate = ModelVersion(project_id=project.id, name="candidate", version="v1", family="candidate", task_kind="detection", format="onnx", alias="candidate", metadata_json={"class_mapping_version": "labels-v1"})
        session.add_all([dataset, baseline, candidate]); session.flush()
        contract = {"evaluator_version": "eval-v1", "protocol": "coco_full", "scope": "core"}
        session.add_all([
            Run(project_id=project.id, kind="evaluation", name="base compatible", status="completed", dataset_id=dataset.id, model_id=baseline.id, config=contract, metrics={"bbox_mAP": 0.8}),
            Run(project_id=project.id, kind="evaluation", name="candidate compatible", status="completed", dataset_id=dataset.id, model_id=candidate.id, config=contract, metrics={"bbox_mAP": 0.75}),
            Run(project_id=project.id, kind="evaluation", name="candidate failed", status="failed", dataset_id=dataset.id, model_id=candidate.id, config={"evaluator_version": "other"}, metrics={"bbox_mAP": 0.99}),
        ])
        session.commit()
        # Newer results with unknown datasets cannot hide a reproducible pair.
        session.add_all([
            Run(project_id=project.id, kind="evaluation", name="base unknown", status="completed", model_id=baseline.id, config=contract, metrics={"bbox_mAP": 0.99}),
            Run(project_id=project.id, kind="evaluation", name="candidate unknown", status="completed", model_id=candidate.id, config=contract, metrics={"bbox_mAP": 1.0}),
        ])
        session.commit()
        comparison = compare_models(session, baseline.id, candidate.id)
        assert comparison["compatible"] is True
        assert comparison["delta"]["bbox_mAP"] == -0.05
        assert comparison["baseline_evaluation"].name == "base compatible"
        assert comparison["candidate_evaluation"].name == "candidate compatible"


def test_comparison_can_pin_explicit_evaluation_runs():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with SessionLocal() as session:
        from vision_lifecycle.models import ModelVersion, Project, Run
        project = Project(name="explicit-comparison")
        session.add(project); session.flush()
        baseline = ModelVersion(project_id=project.id, name="base", version="v1", family="fixture", task_kind="detection", format="external", alias="baseline", metadata_json={"class_mapping_version": "labels-v1"})
        candidate = ModelVersion(project_id=project.id, name="candidate", version="v1", family="fixture", task_kind="detection", format="external", alias="candidate", metadata_json={"class_mapping_version": "labels-v1"})
        session.add_all([baseline, candidate]); session.flush()
        dataset = DatasetVersion(project_id=project.id, name="eval", version="v1", task_kind="detection", format="coco")
        session.add(dataset); session.flush()
        contract = {"evaluator_version": "v1", "protocol": "coco", "scope": "full"}
        base_run = Run(project_id=project.id, model_id=baseline.id, kind="evaluation", name="base chosen", status="completed", dataset_id=dataset.id, config=contract, metrics={"bbox_mAP": 0.8})
        candidate_run = Run(project_id=project.id, model_id=candidate.id, kind="evaluation", name="candidate chosen", status="completed", dataset_id=dataset.id, config=contract, metrics={"bbox_mAP": 0.7})
        session.add_all([base_run, candidate_run]); session.commit()
        result = compare_models(session, baseline.id, candidate.id, base_run.id, candidate_run.id)
        assert result["baseline_evaluation"].id == base_run.id
        assert result["candidate_evaluation"].id == candidate_run.id
        assert result["delta"]["bbox_mAP"] == -0.1


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
    assert any(item.get("image_id") == 999 for item in result["invalid_prediction_examples"])
