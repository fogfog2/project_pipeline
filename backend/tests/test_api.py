from pathlib import Path
from time import sleep

from fastapi.testclient import TestClient

from vision_lifecycle.database import Base, engine
from vision_lifecycle.main import app
from vision_lifecycle.models import Job
from vision_lifecycle.runner import recover_interrupted, worker_once


def test_demo_api_validates_evaluates_and_redacts():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with TestClient(app) as client:
        demo = client.post("/api/v1/projects/demo")
        assert demo.status_code == 201
        project_id = demo.json()["id"]
        validation = client.post(f"/api/v1/projects/{project_id}/inspect-path", json={"path": "examples/mmdetection/annotations/coco8.json"})
        assert validation.status_code == 200
        assert validation.json()["detected_format"] == "coco"
        models = client.get(f"/api/v1/projects/{project_id}/models").json()
        datasets = client.get(f"/api/v1/projects/{project_id}/datasets").json()
        rtm = next(model for model in models if model["family"] == "RTMDet-tiny")
        evaluation = client.post(f"/api/v1/projects/{project_id}/evaluations/predictions", json={
            "model_id": rtm["id"], "dataset_id": datasets[0]["id"],
            "predictions_path": "examples/mmdetection/predictions/rtmdet-tiny.json",
        })
        assert evaluation.status_code == 201
        assert evaluation.json()["run"]["metrics"]["bbox_AP50"] == 1.0
        assert evaluation.json()["run"]["details"]["per_class"]["person"]["true_positive"] == 1
        full = client.post(f"/api/v1/projects/{project_id}/evaluations/predictions", json={
            "model_id": rtm["id"], "dataset_id": datasets[0]["id"],
            "predictions_path": "examples/mmdetection/predictions/rtmdet-tiny.json", "protocol": "coco_full",
        })
        assert full.status_code == 201
        assert full.json()["run"]["config"]["evaluator_version"] == "coco-full-v1"
        assert full.json()["result"]["bbox_mAP"] == 1.0
        exported = client.get(f"/api/v1/projects/{project_id}/export").json()
        assert "storage_root" not in exported["project"]
        assert "annotation_path" not in exported["datasets"][0]
        assert "artifact_path" not in exported["models"][0]
        graph = client.get(f"/api/v1/projects/{project_id}/lineage")
        assert graph.status_code == 200
        assert any(edge["relation"] == "trained_from" for edge in graph.json()["edges"])
        assert any(node["kind"] == "artifact" for node in graph.json()["nodes"])
        assert any(edge["relation"] == "has_artifact" for edge in graph.json()["edges"])


def test_mock_board_job_completes():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with TestClient(app) as client:
        project_id = client.post("/api/v1/projects/demo").json()["id"]
        created = client.post(f"/api/v1/projects/{project_id}/jobs", json={"runner_id": "mock-board"})
        assert created.status_code == 201
        job_id = created.json()["id"]
        status = "queued"
        for _ in range(10):
            jobs = client.get(f"/api/v1/projects/{project_id}/jobs").json()
            status = next(job["status"] for job in jobs if job["id"] == job_id)
            if status == "completed":
                break
            sleep(0.01)
        assert status == "completed"


def test_restart_marks_active_jobs_interrupted():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    from vision_lifecycle.database import SessionLocal
    from vision_lifecycle.models import Project
    with TestClient(app) as client:
        with SessionLocal() as session:
            project = Project(name="restart-project")
            session.add(project); session.flush()
            item = Job(project_id=project.id, runner_id="mock-board", command=["builtin:mock-board"], input_json={"_runner_args": []}, status="running")
            session.add(item); session.commit(); project_id, job_id = project.id, item.id
        job = {"id": job_id}
        with SessionLocal() as session:
            item = session.get(Job, job["id"])
            item.status = "running"
            session.commit()
        assert recover_interrupted() == 1
        current = next(item for item in client.get(f"/api/v1/projects/{project_id}/jobs").json() if item["id"] == job["id"])
        assert current["status"] == "interrupted"
        retry = client.post(f"/api/v1/projects/{project_id}/jobs/{job['id']}/retry")
        assert retry.status_code == 201
        assert retry.json()["input_json"]["retry_of"] == job["id"]


def test_external_worker_claims_one_queued_job():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    from vision_lifecycle.database import SessionLocal
    with SessionLocal() as session:
        from vision_lifecycle.models import Project
        project = Project(name="worker-project")
        session.add(project); session.flush()
        job = Job(project_id=project.id, runner_id="mock-board", command=["builtin:mock-board"], input_json={"_runner_args": []}, status="queued")
        session.add(job); session.commit(); job_id = job.id
    assert worker_once() is True
    with SessionLocal() as session:
        assert session.get(Job, job_id).status == "completed"
    assert worker_once() is False


def test_onboarding_session_persists_step_evidence():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with TestClient(app) as client:
        project_id = client.post("/api/v1/projects", json={"name": "onboarding-project", "recipe_id": "mmdetection-onboarding"}).json()["id"]
        created = client.post(f"/api/v1/projects/{project_id}/onboarding", json={"recipe_id": "mmdetection-onboarding"})
        assert created.status_code == 201
        assert len(created.json()["steps"]) == 9
        blocked = client.patch(f"/api/v1/projects/{project_id}/onboarding/steps/storage", json={"status": "completed"})
        assert blocked.status_code == 409
        client.post(f"/api/v1/projects/{project_id}/storages", json={"name": "workspace", "root_path": "."})
        updated = client.patch(f"/api/v1/projects/{project_id}/onboarding/steps/storage", json={"status": "completed", "evidence": {"storage_id": "STORE-1"}})
        assert updated.status_code == 200
        loaded = client.get(f"/api/v1/projects/{project_id}/onboarding").json()
        assert next(step for step in loaded["steps"] if step["step_id"] == "storage")["status"] == "completed"


def test_result_import_is_idempotent_and_detects_conflict():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with TestClient(app) as client:
        project_id = client.post("/api/v1/projects/demo").json()["id"]
        manifest = {"schema_version": "1.0", "external_run_id": "EXT-1", "kind": "board", "name": "board result", "metrics": {"latency_ms_p90": 20}, "details": {"measurement_path": "/private/board/result.json", "operator_note": "fixture"}}
        first = client.post(f"/api/v1/projects/{project_id}/results/import", json={"manifest": manifest})
        second = client.post(f"/api/v1/projects/{project_id}/results/import", json={"manifest": manifest})
        changed = client.post(f"/api/v1/projects/{project_id}/results/import", json={"manifest": {**manifest, "metrics": {"latency_ms_p90": 21}}})
        assert first.json()["status"] == "created"
        run_id = first.json()["run"]["id"]
        assert client.get(f"/api/v1/projects/{project_id}/runs").json()[0]["details"]["measurement_path"] == "/private/board/result.json"
        assert second.json()["status"] == "existing"
        assert changed.status_code == 409
        exported = client.get(f"/api/v1/projects/{project_id}/export").json()
        exported_run = next(item for item in exported["runs"] if item["id"] == run_id)
        assert "measurement_path" not in exported_run["details"]


def test_target_profile_can_be_linked_to_board_import():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with TestClient(app) as client:
        project_id = client.post("/api/v1/projects/demo").json()["id"]
        target = client.post(f"/api/v1/projects/{project_id}/targets", json={
            "name": "Jetson Orin NX", "version": "r36", "runtime": "TensorRT", "hardware": {"power_mode": "15W"},
        })
        assert target.status_code == 201
        linked = client.post(f"/api/v1/projects/{project_id}/results/import", json={"manifest": {
            "schema_version": "1.0", "external_run_id": "BOARD-TARGET-1", "kind": "board", "name": "latency",
            "target_profile_id": target.json()["id"], "metrics": {"latency_ms_p50": 4.2},
        }})
        assert linked.status_code == 201
        assert linked.json()["run"]["config"]["target_profile_id"] == target.json()["id"]


def test_quantization_and_board_lineage_requires_project_owned_references():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with TestClient(app) as client:
        project_id = client.post("/api/v1/projects", json={"name": "quant-lineage"}).json()["id"]
        model = client.post(f"/api/v1/projects/{project_id}/models", json={"name": "fp32", "version": "v1", "family": "fixture"}).json()
        calibration = client.post(f"/api/v1/projects/{project_id}/calibration-sets", json={"name": "cal", "version": "v1", "sampling": {"count": 10}}).json()
        quant = client.post(f"/api/v1/projects/{project_id}/quantization-runs", json={"name": "int8", "source_model_id": model["id"], "calibration_set_id": calibration["id"], "method": "ptq", "activation_dtype": "int8"})
        assert quant.status_code == 201
        target = client.post(f"/api/v1/projects/{project_id}/targets", json={"name": "board", "version": "v1"}).json()
        benchmark = client.post(f"/api/v1/projects/{project_id}/board-benchmarks", json={"name": "board-v1", "model_id": model["id"], "target_profile_id": target["id"], "metrics": {"latency_ms_p50": 4.2}, "measurement": {"batch_size": 1, "warmup_runs": 5}})
        assert benchmark.status_code == 201


def test_quantization_loss_and_target_gap_require_compatible_evaluations():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with TestClient(app) as client:
        project_id = client.post("/api/v1/projects", json={"name": "quant-comparison"}).json()["id"]
        dataset = client.post(f"/api/v1/projects/{project_id}/datasets", json={"name": "eval", "version": "v1"}).json()
        base = client.post(f"/api/v1/projects/{project_id}/models", json={"name": "base", "version": "v1", "family": "base", "metadata_json": {"class_mapping_version": "labels-v1"}}).json()
        quant = client.post(f"/api/v1/projects/{project_id}/models", json={"name": "int8", "version": "v1", "family": "int8", "metadata_json": {"class_mapping_version": "labels-v1"}}).json()
        lineage = client.post(f"/api/v1/projects/{project_id}/quantization-runs", json={"name": "ptq", "source_model_id": base["id"], "output_model_id": quant["id"], "source_role": "fp32", "output_role": "quantsim", "method": "ptq"}).json()
        contract = {"evaluator_version": "eval-v1", "protocol": "coco_full", "scope": "core"}
        baseline_run = client.post(f"/api/v1/projects/{project_id}/runs", json={"kind": "evaluation", "name": "fp32", "dataset_id": dataset["id"], "model_id": base["id"], "config": contract, "metrics": {"bbox_mAP": 0.8}}).json()
        quantsim_run = client.post(f"/api/v1/projects/{project_id}/runs", json={"kind": "evaluation", "name": "quantsim", "dataset_id": dataset["id"], "model_id": quant["id"], "config": contract, "metrics": {"bbox_mAP": 0.75}}).json()
        target_run = client.post(f"/api/v1/projects/{project_id}/runs", json={"kind": "evaluation", "name": "target", "dataset_id": dataset["id"], "model_id": quant["id"], "config": contract, "metrics": {"bbox_mAP": 0.7}}).json()
        target = client.post(f"/api/v1/projects/{project_id}/targets", json={"name": "board", "version": "v1"}).json()
        benchmark = client.post(f"/api/v1/projects/{project_id}/board-benchmarks", json={"name": "target-result", "model_id": quant["id"], "target_profile_id": target["id"], "evaluation_run_id": target_run["id"]}).json()
        comparison = client.post(f"/api/v1/projects/{project_id}/quantization-runs/{lineage['id']}/compare", json={"baseline_run_id": baseline_run["id"], "quantsim_run_id": quantsim_run["id"], "target_benchmark_id": benchmark["id"]})
        assert comparison.status_code == 200
        assert comparison.json()["comparison"]["status"] == "COMPLETE"
        assert comparison.json()["comparison"]["quantization_loss"] == 0.05
        assert comparison.json()["comparison"]["target_gap"] == 0.1


def test_classification_evaluation_api():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with TestClient(app) as client:
        project_id = client.post("/api/v1/projects", json={"name": "classification-api", "task_kind": "classification"}).json()["id"]
        model = client.post(f"/api/v1/projects/{project_id}/models", json={"name": "classifier", "version": "v1", "family": "simple-classifier", "task_kind": "classification", "format": "onnx"}).json()
        response = client.post(f"/api/v1/projects/{project_id}/evaluations/classification", json={
            "model_id": model["id"],
            "records": [{"image_id": "1", "ground_truth": "cat", "prediction": "cat"}, {"image_id": "2", "ground_truth": "dog", "prediction": "cat"}],
        })
        assert response.status_code == 201
        assert response.json()["result"]["top1_accuracy"] == 0.5
        runs = client.get(f"/api/v1/projects/{project_id}/runs").json()
        assert runs[0]["details"]["confusion_matrix"]["dog"]["cat"] == 1


def test_classification_dataset_mapping_rejects_unknown_label():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with TestClient(app) as client:
        project_id = client.post("/api/v1/projects", json={"name": "classification-mapping", "task_kind": "classification"}).json()["id"]
        dataset = client.post(f"/api/v1/projects/{project_id}/datasets", json={"name": "labels", "version": "v1", "task_kind": "classification", "class_names": ["cat", "dog"]}).json()
        model = client.post(f"/api/v1/projects/{project_id}/models", json={"name": "classifier", "version": "v1", "family": "fixture", "task_kind": "classification"}).json()
        response = client.post(f"/api/v1/projects/{project_id}/evaluations/classification", json={"model_id": model["id"], "dataset_id": dataset["id"], "records": [{"image_id": "1", "ground_truth": "cat", "prediction": "bird"}]})
        assert response.status_code == 422


def test_release_requires_compatible_baseline_for_regression_gate():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with TestClient(app) as client:
        project_id = client.post("/api/v1/projects", json={"name": "release-contract"}).json()["id"]
        dataset = client.post(f"/api/v1/projects/{project_id}/datasets", json={"name": "eval", "version": "v1"}).json()
        baseline = client.post(f"/api/v1/projects/{project_id}/models", json={"name": "base", "version": "v1", "family": "base", "alias": "baseline"}).json()
        candidate = client.post(f"/api/v1/projects/{project_id}/models", json={"name": "candidate", "version": "v1", "family": "candidate", "alias": "candidate"}).json()
        run = client.post(f"/api/v1/projects/{project_id}/runs", json={"kind": "evaluation", "name": "candidate eval", "dataset_id": dataset["id"], "model_id": candidate["id"], "config": {"evaluator_version": "v2", "protocol": "coco_full", "scope": "full"}, "metrics": {"bbox_AP50": 0.7}}).json()
        release = client.post(f"/api/v1/projects/{project_id}/releases", json={"name": "candidate-release", "model_id": candidate["id"], "evaluation_run_id": run["id"], "baseline_model_id": baseline["id"], "gate_config": {"max_regression": {"bbox_AP50": 0.02}}})
        assert release.status_code == 201
        assert release.json()["decision"] == "INCOMPLETE"
        assert release.json()["gate_result"]["baseline_compatibility"]["status"] == "incomplete"


def test_empty_project_and_references_are_explicit_and_reversible():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with TestClient(app) as client:
        first = client.post("/api/v1/projects", json={"name": "empty-project", "task_kind": "unknown", "mode": "guided", "recipe_id": "mmdetection-onboarding"})
        second = client.post("/api/v1/projects", json={"name": "other-project", "task_kind": "detection"})
        assert first.status_code == second.status_code == 201
        project_id, other_id = first.json()["id"], second.json()["id"]
        assert client.get(f"/api/v1/projects/{project_id}/overview").json()["counts"] == {"datasets": 0, "models": 0, "runs": 0, "jobs": 0}
        dataset = client.post(f"/api/v1/projects/{project_id}/datasets", json={"name": "images", "version": "v1"}).json()
        cross_model = client.post(f"/api/v1/projects/{other_id}/models", json={"name": "wrong", "version": "v1", "family": "x", "source_dataset_id": dataset["id"]})
        assert cross_model.status_code == 422
        model = client.post(f"/api/v1/projects/{project_id}/models", json={"name": "classifier", "version": "v1", "family": "x", "source_dataset_id": dataset["id"]}).json()
        assert client.post(f"/api/v1/projects/{project_id}/models/{model['id']}/archive").json()["status"] == "archived"
        assert client.post(f"/api/v1/projects/{project_id}/models/{model['id']}/archive").json()["status"] == "experimental"
        assert client.post(f"/api/v1/projects/{project_id}/archive").json()["status"] == "archived"
