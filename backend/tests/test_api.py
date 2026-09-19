from pathlib import Path
from time import sleep
import sys

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
        assert exported["schema_version"] == "1.1" and exported["generated_at"] and exported["overview"]["counts"]["runs"] >= 2
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


def test_job_logs_support_cursor_windows():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with TestClient(app) as client:
        project_id = client.post("/api/v1/projects", json={"name": "job-log-cursor"}).json()["id"]
        created = client.post(f"/api/v1/projects/{project_id}/jobs", json={"runner_id": "mock-board"}).json()
        job_id = created["id"]
        for _ in range(100):
            response = client.get(f"/api/v1/projects/{project_id}/jobs/{job_id}/logs?limit=1")
            if response.json()["status"] == "completed":
                break
            sleep(0.01)
        first = client.get(f"/api/v1/projects/{project_id}/jobs/{job_id}/logs?cursor=0&limit=1")
        assert first.status_code == 200
        payload = first.json()
        assert payload["next_cursor"] >= payload["cursor"]
        second = client.get(f"/api/v1/projects/{project_id}/jobs/{job_id}/logs?cursor={payload['next_cursor']}&limit=200")
        assert second.status_code == 200
        assert second.json()["complete"] is True
        assert "Mock board" in first.text + second.text


def test_job_log_websocket_sends_reconnectable_snapshot(monkeypatch):
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    monkeypatch.setenv("VISION_LIFECYCLE_EXTERNAL_WORKER", "true")
    with TestClient(app) as client:
        project_id = client.post("/api/v1/projects", json={"name": "job-log-stream"}).json()["id"]
        created = client.post(f"/api/v1/projects/{project_id}/jobs", json={"runner_id": "mock-board"}).json()
        with client.websocket_connect(f"/api/v1/projects/{project_id}/jobs/{created['id']}/stream") as websocket:
            event = websocket.receive_json()
            assert event["type"] == "snapshot"
            assert event["job_id"] == created["id"]
            assert event["status"] == "queued"
            assert event["next_cursor"] == 0


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


def test_api_restart_keeps_external_queue_claimable():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    from vision_lifecycle.database import SessionLocal
    from vision_lifecycle.models import Project
    with SessionLocal() as session:
        project = Project(name="restart-queue")
        session.add(project); session.flush()
        queued = Job(project_id=project.id, runner_id="external", command=["python"], status="queued")
        running = Job(project_id=project.id, runner_id="external", command=["python"], status="running")
        session.add_all([queued, running]); session.commit(); queued_id, running_id = queued.id, running.id
    assert recover_interrupted(include_queued=False) == 1
    with SessionLocal() as session:
        assert session.get(Job, queued_id).status == "queued"
        assert session.get(Job, running_id).status == "interrupted"


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


def test_external_worker_claim_records_lease_and_prevents_double_claim():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    from vision_lifecycle.database import SessionLocal
    from vision_lifecycle.models import Project
    from vision_lifecycle.runner import claim_next_job, renew_lease

    with SessionLocal() as session:
        project = Project(name="leased-worker-project")
        session.add(project); session.flush()
        job = Job(project_id=project.id, runner_id="mock-board", command=["builtin:mock-board"], input_json={"_runner_args": []}, status="queued")
        session.add(job); session.commit(); job_id = job.id
    first = claim_next_job(worker_id="worker-a", lease_seconds=60)
    second = claim_next_job(worker_id="worker-b", lease_seconds=60)
    assert first and first[0] == job_id
    assert second is None
    assert renew_lease(job_id, "worker-a", lease_seconds=60) is True
    assert renew_lease(job_id, "worker-b", lease_seconds=60) is False
    with SessionLocal() as session:
        claimed = session.get(Job, job_id)
        assert claimed.status == "running"
        assert claimed.lease_owner == "worker-a"
        assert claimed.lease_expires_at is not None
        assert claimed.attempt_count == 1


def test_expired_worker_lease_is_requeued_before_next_claim():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    from datetime import UTC, datetime, timedelta
    from vision_lifecycle.database import SessionLocal
    from vision_lifecycle.models import Project
    from vision_lifecycle.runner import reclaim_expired_leases
    with SessionLocal() as session:
        project = Project(name="expired-lease-project")
        session.add(project); session.flush()
        job = Job(project_id=project.id, runner_id="mock-board", command=["builtin:mock-board"], input_json={"_runner_args": []}, status="running", lease_owner="dead-worker", lease_expires_at=datetime.now(UTC) - timedelta(seconds=1), attempt_count=1)
        session.add(job); session.commit(); job_id = job.id
    assert reclaim_expired_leases() == 1
    with SessionLocal() as session:
        queued = session.get(Job, job_id)
        assert queued.status == "queued"
        assert queued.lease_owner is None
        assert "lease expired" in queued.log


def test_retry_keeps_queued_for_external_worker(monkeypatch):
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    monkeypatch.setenv("VISION_LIFECYCLE_EXTERNAL_WORKER", "true")
    with TestClient(app) as client:
        project_id = client.post("/api/v1/projects", json={"name": "external-retry"}).json()["id"]
        created = client.post(f"/api/v1/projects/{project_id}/jobs", json={"runner_id": "mock-board"}).json()
        from vision_lifecycle.database import SessionLocal
        with SessionLocal() as session:
            job = session.get(Job, created["id"])
            job.status = "failed"
            session.commit()
        retried = client.post(f"/api/v1/projects/{project_id}/jobs/{created['id']}/retry")
        assert retried.status_code == 201
        assert retried.json()["status"] == "queued"


def test_runner_cancellation_stops_profile_process_group():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with TestClient(app) as client:
        project_id = client.post("/api/v1/projects", json={"name": "cancel-profile"}).json()["id"]
        profile = client.post(f"/api/v1/projects/{project_id}/runners", json={
            "name": "sleep runner", "executable": sys.executable,
            "default_args": ["-c", "import time; print('runner-started', flush=True); time.sleep(30)"],
            "timeout_seconds": 60,
        }).json()
        created = client.post(f"/api/v1/projects/{project_id}/jobs", json={"runner_id": profile["id"]}).json()
        job_id = created["id"]
        for _ in range(100):
            current = next(item for item in client.get(f"/api/v1/projects/{project_id}/jobs").json() if item["id"] == job_id)
            if current["status"] == "running":
                break
            sleep(0.01)
        # Simulate an API process that does not own the worker subprocess.
        from vision_lifecycle.runner import _lock, _processes
        with _lock:
            _processes.pop(job_id, None)
        cancelled = client.post(f"/api/v1/projects/{project_id}/jobs/{job_id}/cancel")
        assert cancelled.status_code == 200
        for _ in range(100):
            current = next(item for item in client.get(f"/api/v1/projects/{project_id}/jobs").json() if item["id"] == job_id)
            if current["status"] in {"cancelled", "failed", "timed_out"}:
                break
            sleep(0.02)
        assert current["status"] == "cancelled"
        assert "Cancellation requested" in current["log"]


def test_cancel_uses_process_group_signal(monkeypatch):
    from vision_lifecycle import runner

    class FakeProcess:
        pid = 1234
        def poll(self):
            return None

    calls = []
    monkeypatch.setattr(runner, "_processes", {"JOB-1": FakeProcess()})
    monkeypatch.setattr(runner, "_signal_process_group", lambda process, signal_number: calls.append((process, signal_number)))
    assert runner.cancel("JOB-1") is True
    assert calls and calls[0][0].pid == 1234


def test_cancel_running_job_without_local_process_leaves_worker_cancellation_marker():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    from vision_lifecycle.database import SessionLocal
    from vision_lifecycle.models import Project

    with TestClient(app) as client:
        with SessionLocal() as session:
            project = Project(name="external-cancel-marker")
            session.add(project); session.flush()
            item = Job(project_id=project.id, runner_id="external-runner", command=["python"], status="running")
            session.add(item); session.commit(); project_id, job_id = project.id, item.id
        cancelled = client.post(f"/api/v1/projects/{project_id}/jobs/{job_id}/cancel")
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "cancelling"


def test_runner_timeout_records_terminal_status():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with TestClient(app) as client:
        project_id = client.post("/api/v1/projects", json={"name": "timeout-profile"}).json()["id"]
        profile = client.post(f"/api/v1/projects/{project_id}/runners", json={
            "name": "short runner", "executable": sys.executable,
            "default_args": ["-c", "import time; time.sleep(5)"],
            "timeout_seconds": 1,
        }).json()
        created = client.post(f"/api/v1/projects/{project_id}/jobs", json={"runner_id": profile["id"]}).json()
        job_id = created["id"]
        for _ in range(180):
            current = next(item for item in client.get(f"/api/v1/projects/{project_id}/jobs").json() if item["id"] == job_id)
            if current["status"] == "timed_out":
                break
            sleep(0.02)
        assert current["status"] == "timed_out"
        assert current["result_json"]["timeout_seconds"] == 1


def test_onboarding_session_persists_step_evidence():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with TestClient(app) as client:
        project_id = client.post("/api/v1/projects", json={"name": "onboarding-project", "recipe_id": "mmdetection-onboarding"}).json()["id"]
        created = client.post(f"/api/v1/projects/{project_id}/onboarding", json={"recipe_id": "mmdetection-onboarding"})
        assert created.status_code == 201
        assert len(created.json()["steps"]) == 9
        storage_step = next(step for step in created.json()["steps"] if step["step_id"] == "storage")
        assert storage_step["readiness"]["ready"] is False
        assert "storage mapping" in storage_step["readiness"]["reason"]
        blocked = client.patch(f"/api/v1/projects/{project_id}/onboarding/steps/storage", json={"status": "completed"})
        assert blocked.status_code == 409
        client.post(f"/api/v1/projects/{project_id}/storages", json={"name": "workspace", "root_path": "."})
        updated = client.patch(f"/api/v1/projects/{project_id}/onboarding/steps/storage", json={"status": "completed", "evidence": {"storage_id": "STORE-1"}})
        assert updated.status_code == 200
        loaded = client.get(f"/api/v1/projects/{project_id}/onboarding").json()
        storage_step = next(step for step in loaded["steps"] if step["step_id"] == "storage")
        assert storage_step["status"] == "completed"
        assert storage_step["readiness"]["ready"] is True


def test_project_connection_settings_are_editable_and_persisted():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with TestClient(app) as client:
        created = client.post("/api/v1/projects", json={"name": "connection-settings", "task_kind": "unknown"})
        assert created.status_code == 201
        project_id = created.json()["id"]
        updated = client.patch(f"/api/v1/projects/{project_id}", json={
            "description": "customer vision project",
            "storage_root": "/mnt/vision",
            "git_url": "https://github.com/example/vision",
            "default_branch": "main",
        })
        assert updated.status_code == 200
        assert updated.json()["git_url"].endswith("/vision")
        listed = client.get("/api/v1/projects").json()
        persisted = next(item for item in listed if item["id"] == project_id)
        assert persisted["storage_root"] == "/mnt/vision"
        assert persisted["default_branch"] == "main"


def test_overview_reports_explicit_readiness_checks():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with TestClient(app) as client:
        project_id = client.post("/api/v1/projects", json={"name": "readiness"}).json()["id"]
        overview = client.get(f"/api/v1/projects/{project_id}/overview").json()
        assert overview["readiness"]["status"] == "not_started"
        assert {item["id"] for item in overview["readiness"]["checks"]} == {"storage", "dataset", "model", "evaluation"}


def test_recipe_contract_is_read_only_and_versioned():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with TestClient(app) as client:
        response = client.get("/api/v1/recipes/mmdetection-onboarding")
        assert response.status_code == 200
        body = response.json()
        assert body["version"] == "1.0"
        assert "RTMDet-tiny" in body["definition"]
        assert client.get("/api/v1/recipes/unknown-recipe").status_code == 404


def test_schema_registry_includes_recipe_contract():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with TestClient(app) as client:
        recipe = client.get("/api/v1/schemas").json()["schemas"]["recipe"]
        assert recipe["title"] == "Vision Lifecycle Recipe"
        assert "steps" in recipe["required"]


def test_recipe_list_matches_onboarding_step_engine():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with TestClient(app) as client:
        recipes = {item["id"]: item["steps"] for item in client.get("/api/v1/recipes").json()}
        created = client.post("/api/v1/projects", json={"name": "recipe-step-match", "recipe_id": "mmdetection-onboarding"}).json()
        session = client.post(f"/api/v1/projects/{created['id']}/onboarding", json={"recipe_id": "mmdetection-onboarding"}).json()
        assert recipes["mmdetection-onboarding"] == [item["step_id"] for item in session["steps"]]


def test_plugin_endpoint_exposes_versioned_adapter_registry():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with TestClient(app) as client:
        plugins = client.get("/api/v1/plugins")
        assert plugins.status_code == 200
        onnx = next(item for item in plugins.json() if item["id"] == "onnx")
        assert onnx["kind"] == "inference"
        assert onnx["contract_version"] == "1.0"
        assert "batch-evaluation" in onnx["capabilities"]


def test_classification_evaluation_blocks_changed_dataset_source(tmp_path):
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    manifest = tmp_path / "records.csv"
    manifest.write_text("image,label\ncat.png,cat\n", encoding="utf-8")
    with TestClient(app) as client:
        project_id = client.post("/api/v1/projects", json={"name": "classification-drift", "task_kind": "classification"}).json()["id"]
        dataset = client.post(f"/api/v1/projects/{project_id}/datasets", json={"name": "images", "version": "v1", "task_kind": "classification", "format": "classification", "manifest_path": str(manifest), "class_names": ["cat"]}).json()
        model = client.post(f"/api/v1/projects/{project_id}/models", json={"name": "classifier", "version": "v1", "family": "fixture", "task_kind": "classification", "format": "external"}).json()
        manifest.write_text("image,label\ncat.png,dog\n", encoding="utf-8")
        response = client.post(f"/api/v1/projects/{project_id}/evaluations/classification", json={"model_id": model["id"], "dataset_id": dataset["id"], "records": [{"image_id": "1", "ground_truth": "cat", "prediction": "cat"}]})
        assert response.status_code == 409
        assert "source files changed" in response.text


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


def test_result_manifest_can_link_quantization_and_calibration_lineage():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with TestClient(app) as client:
        project_id = client.post("/api/v1/projects", json={"name": "manifest-lineage"}).json()["id"]
        model = client.post(f"/api/v1/projects/{project_id}/models", json={"name": "fp32", "version": "v1", "family": "fixture"}).json()
        calibration = client.post(f"/api/v1/projects/{project_id}/calibration-sets", json={"name": "cal", "version": "v1", "sampling": {"count": 1}, "preprocessing": {"color": "rgb"}}).json()
        quantization = client.post(f"/api/v1/projects/{project_id}/quantization-runs", json={"name": "int8", "source_model_id": model["id"], "calibration_set_id": calibration["id"], "method": "ptq"}).json()
        imported = client.post(f"/api/v1/projects/{project_id}/results/import", json={"manifest": {
            "schema_version": "1.0", "external_run_id": "QUANT-1", "kind": "quantization", "name": "external int8",
            "quantization_run_id": quantization["id"], "calibration_set_id": calibration["id"], "metrics": {"size_mb": 2.1},
        }})
        assert imported.status_code == 201, imported.text
        assert imported.json()["run"]["config"]["quantization_run_id"] == quantization["id"]
        assert imported.json()["run"]["config"]["calibration_set_id"] == calibration["id"]


def test_training_run_registration_is_idempotent_and_detects_external_conflict():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with TestClient(app) as client:
        project_id = client.post("/api/v1/projects", json={"name": "training-import"}).json()["id"]
        payload = {"kind": "training", "name": "train-v1", "external_run_id": "TRAIN-1", "config": {"commit": "abc"}, "metrics": {"loss": 0.2}}
        first = client.post(f"/api/v1/projects/{project_id}/runs", json=payload)
        same = client.post(f"/api/v1/projects/{project_id}/runs", json=payload)
        changed = client.post(f"/api/v1/projects/{project_id}/runs", json={**payload, "metrics": {"loss": 0.1}})
        assert first.status_code == 201 and same.status_code == 200
        assert same.json()["id"] == first.json()["id"]
        assert changed.status_code == 409


def test_target_profile_can_be_linked_to_board_import():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with TestClient(app) as client:
        project_id = client.post("/api/v1/projects/demo").json()["id"]
        target = client.post(f"/api/v1/projects/{project_id}/targets", json={
            "name": "Jetson Orin NX", "version": "r36", "runtime": "TensorRT", "hardware": {"power_mode": "15W"}, "metadata_json": {"firmware": "r36.2", "accelerator": "DLA0"},
        })
        assert target.status_code == 201
        assert target.json()["metadata_json"]["accelerator"] == "DLA0"
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
        assert benchmark.json()["measurement"]["contract_validation"]["status"] == "incomplete"
        complete = client.post(f"/api/v1/projects/{project_id}/board-benchmarks", json={"name": "board-v2", "model_id": model["id"], "target_profile_id": target["id"], "metrics": {"latency_ms_p50": 4.2}, "measurement": {"source": "external", "scope": "batch", "batch_size": 1, "warmup_runs": 5, "iterations": 100, "units": {"latency_ms_p50": "ms"}}})
        assert complete.status_code == 201
        assert complete.json()["measurement"]["contract_validation"]["status"] == "passed"


def test_quantization_encoding_artifact_is_registered_and_verified(tmp_path):
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    encoding = tmp_path / "encoding.json"
    encoding.write_text('{"scale": 0.25}', encoding="utf-8")
    with TestClient(app) as client:
        project_id = client.post("/api/v1/projects", json={"name": "encoding-artifact"}).json()["id"]
        model = client.post(f"/api/v1/projects/{project_id}/models", json={"name": "fp32", "version": "v1", "family": "fixture"}).json()
        created = client.post(f"/api/v1/projects/{project_id}/quantization-runs", json={"name": "int8", "source_model_id": model["id"], "encoding_path": str(encoding), "method": "ptq"})
        assert created.status_code == 201, created.text
        assert created.json()["metadata_json"]["encoding_artifact_id"]
        verified = client.post(f"/api/v1/projects/{project_id}/quantization-runs/{created.json()['id']}/verify-encoding")
        assert verified.status_code == 200
        assert verified.json()["ok"] is True


def test_board_raw_output_artifact_is_registered_and_verified(tmp_path):
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    raw_output = tmp_path / "board.log"
    raw_output.write_text("latency_ms_p50=4.2\n", encoding="utf-8")
    with TestClient(app) as client:
        project_id = client.post("/api/v1/projects", json={"name": "board-output-artifact"}).json()["id"]
        model = client.post(f"/api/v1/projects/{project_id}/models", json={"name": "int8", "version": "v1", "family": "fixture"}).json()
        target = client.post(f"/api/v1/projects/{project_id}/targets", json={"name": "board", "version": "v1"}).json()
        created = client.post(f"/api/v1/projects/{project_id}/board-benchmarks", json={
            "name": "board-run", "model_id": model["id"], "target_profile_id": target["id"],
            "raw_output_path": str(raw_output), "metrics": {"latency_ms_p50": 4.2},
            "measurement": {"source": "external", "scope": "batch", "batch_size": 1, "warmup_runs": 5, "iterations": 10, "units": {"latency_ms_p50": "ms"}},
        })
        assert created.status_code == 201, created.text
        assert created.json()["raw_output_hash"]
        verified = client.post(f"/api/v1/projects/{project_id}/board-benchmarks/{created.json()['id']}/verify-output")
        assert verified.status_code == 200
        assert verified.json()["ok"] is True


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


def test_training_run_typed_provenance_is_normalized_and_references_are_checked():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with TestClient(app) as client:
        project_id = client.post("/api/v1/projects", json={"name": "typed-training"}).json()["id"]
        other_project_id = client.post("/api/v1/projects", json={"name": "typed-training-other"}).json()["id"]
        dataset = client.post(f"/api/v1/projects/{project_id}/datasets", json={"name": "images", "version": "v1"}).json()
        labels = client.post(f"/api/v1/projects/{project_id}/label-schemas", json={"name": "labels", "version": "v1", "dataset_id": dataset["id"], "classes": [{"id": 1, "name": "person"}]}).json()
        split = client.post(f"/api/v1/projects/{project_id}/splits", json={"name": "split", "version": "v1", "dataset_id": dataset["id"], "definition": {"splits": {"train": ["1"]}}}).json()
        created = client.post(f"/api/v1/projects/{project_id}/runs", json={
            "kind": "training", "name": "rtmdet train", "dataset_id": dataset["id"], "external_run_id": "TRAIN-1",
            "training": {"framework": "MMDetection", "framework_version": "3.3.0", "commit": "abc123", "seed": 42, "split_id": split["id"], "label_schema_id": labels["id"]},
            "config": {"optimizer": "AdamW"}, "metrics": {"loss": 0.12},
        })
        assert created.status_code == 201, created.text
        assert created.json()["details"]["training"]["lineage_status"] == "complete"
        assert created.json()["details"]["training"]["commit"] == "abc123"
        cross_project_split = client.post(f"/api/v1/projects/{other_project_id}/splits", json={"name": "other", "version": "v1", "definition": {"splits": {"train": ["1"]}}}).json()
        rejected = client.post(f"/api/v1/projects/{project_id}/runs", json={
            "kind": "training", "name": "bad", "training": {"split_id": cross_project_split["id"]},
        })
        assert rejected.status_code == 422
        other_dataset = client.post(f"/api/v1/projects/{project_id}/datasets", json={"name": "other-images", "version": "v1"}).json()
        other_split = client.post(f"/api/v1/projects/{project_id}/splits", json={"name": "other-split", "version": "v1", "dataset_id": other_dataset["id"], "definition": {"splits": {"train": ["1"]}}}).json()
        mismatched = client.post(f"/api/v1/projects/{project_id}/runs", json={
            "kind": "training", "name": "mismatched", "dataset_id": dataset["id"], "training": {"split_id": other_split["id"]},
        })
        assert mismatched.status_code == 422


def test_label_schema_parent_history_and_lineage_are_preserved():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with TestClient(app) as client:
        project_id = client.post("/api/v1/projects", json={"name": "label-history"}).json()["id"]
        dataset = client.post(f"/api/v1/projects/{project_id}/datasets", json={"name": "labels", "version": "v1"}).json()
        base = client.post(f"/api/v1/projects/{project_id}/label-schemas", json={
            "name": "labels", "version": "v1", "dataset_id": dataset["id"], "classes": [{"id": 1, "name": "person"}], "mapping": {"person": 1},
        }).json()
        child = client.post(f"/api/v1/projects/{project_id}/label-schemas", json={
            "name": "labels", "version": "v2", "dataset_id": dataset["id"], "parent_label_schema_id": base["id"], "classes": [{"id": 1, "name": "pedestrian"}, {"id": 2, "name": "cyclist"}], "mapping": {"pedestrian": 1, "cyclist": 2},
        })
        assert child.status_code == 201, child.text
        assert child.json()["parent_label_schema_id"] == base["id"]
        graph = client.get(f"/api/v1/projects/{project_id}/lineage").json()
        assert any(edge["relation"] == "supersedes" and edge["source"] == base["id"] and edge["target"] == child.json()["id"] for edge in graph["edges"])


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


def test_release_captures_immutable_evidence_snapshots_and_required_types():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with TestClient(app) as client:
        project_id = client.post("/api/v1/projects", json={"name": "release-evidence"}).json()["id"]
        model = client.post(f"/api/v1/projects/{project_id}/models", json={"name": "candidate", "version": "v1", "family": "fixture"}).json()
        dataset = client.post(f"/api/v1/projects/{project_id}/datasets", json={"name": "eval", "version": "v1"}).json()
        run = client.post(f"/api/v1/projects/{project_id}/runs", json={"kind": "evaluation", "name": "candidate eval", "dataset_id": dataset["id"], "model_id": model["id"], "config": {"evaluator_version": "eval-v1"}, "metrics": {"bbox_AP50": 0.7}}).json()
        release = client.post(f"/api/v1/projects/{project_id}/releases", json={
            "name": "release-with-evidence", "model_id": model["id"], "evaluation_run_id": run["id"],
            "gate_config": {"minimum": {"bbox_AP50": 0.5}}, "evidence": [{"type": "evaluation", "id": run["id"], "required": True}],
        })
        assert release.status_code == 201 and release.json()["decision"] == "PASS"
        evidence = client.get(f"/api/v1/projects/{project_id}/releases/{release.json()['id']}/evidence")
        assert evidence.status_code == 200 and evidence.json()[0]["snapshot"]["entity"]["metrics"]["bbox_AP50"] == 0.7
        from vision_lifecycle.database import SessionLocal
        from vision_lifecycle.models import Run
        with SessionLocal() as session:
            stored_run = session.get(Run, run["id"])
            stored_run.metrics = {"bbox_AP50": 0.1}
            session.commit()
        unchanged = client.get(f"/api/v1/projects/{project_id}/releases/{release.json()['id']}/evidence").json()[0]
        assert unchanged["snapshot"]["entity"]["metrics"]["bbox_AP50"] == 0.7
        incomplete = client.post(f"/api/v1/projects/{project_id}/releases", json={
            "name": "release-missing-board", "model_id": model["id"], "evaluation_run_id": run["id"],
            "gate_config": {"minimum": {"bbox_AP50": 0.0}, "required_evidence": ["evaluation", "board"]},
            "evidence": [{"type": "evaluation", "id": run["id"]}],
        })
        assert incomplete.status_code == 201 and incomplete.json()["decision"] == "INCOMPLETE"
        assert "board" in incomplete.json()["gate_result"]["evidence"]["missing_required"]


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


def test_model_alias_history_records_reason_and_is_exported():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with TestClient(app) as client:
        project_id = client.post("/api/v1/projects", json={"name": "alias-history"}).json()["id"]
        model = client.post(f"/api/v1/projects/{project_id}/models", json={
            "name": "candidate", "version": "v1", "family": "fixture", "alias": "candidate",
        }).json()
        changed = client.patch(f"/api/v1/projects/{project_id}/models/{model['id']}", json={
            "alias": "baseline", "alias_reason": "선정된 기준 모델로 승격",
        })
        assert changed.status_code == 200
        history = client.get(f"/api/v1/projects/{project_id}/models/{model['id']}/alias-history")
        assert history.status_code == 200
        assert history.json()[0]["previous_alias"] == "candidate"
        assert history.json()[0]["new_alias"] == "baseline"
        assert history.json()[0]["reason"] == "선정된 기준 모델로 승격"
        exported = client.get(f"/api/v1/projects/{project_id}/export").json()
        assert exported["model_alias_history"][0]["model_id"] == model["id"]


def test_model_training_run_lineage_is_explicit():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with TestClient(app) as client:
        project_id = client.post("/api/v1/projects", json={"name": "training-lineage"}).json()["id"]
        run = client.post(f"/api/v1/projects/{project_id}/runs", json={"kind": "training", "name": "external train", "status": "completed", "config": {"commit": "abc123"}}).json()
        model = client.post(f"/api/v1/projects/{project_id}/models", json={"name": "trained", "version": "v1", "family": "rtmdet", "source_run_id": run["id"]})
        assert model.status_code == 201
        graph = client.get(f"/api/v1/projects/{project_id}/lineage").json()
        assert {edge["relation"] for edge in graph["edges"] if edge["target"] == model.json()["id"]} >= {"contains", "produced_by"}


def test_audit_events_track_changes_and_export_redacts_paths():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with TestClient(app) as client:
        project = client.post("/api/v1/projects", json={"name": "audit-project"}).json()
        project_id = project["id"]
        dataset = client.post(f"/api/v1/projects/{project_id}/datasets", json={
            "name": "images", "version": "v1", "manifest_path": "/private/images.json",
        }).json()
        model = client.post(f"/api/v1/projects/{project_id}/models", json={
            "name": "candidate", "version": "v1", "family": "fixture", "source_dataset_id": dataset["id"],
        }).json()
        client.patch(f"/api/v1/projects/{project_id}/models/{model['id']}", json={"alias": "candidate", "alias_reason": "initial review"})
        events = client.get(f"/api/v1/projects/{project_id}/audit-events").json()
        assert events and events[0]["entity_type"] == "model"
        assert {event["action"] for event in events} >= {"created", "updated"}
        exported = client.get(f"/api/v1/projects/{project_id}/export").json()
        audit = next(event for event in exported["audit_events"] if event["entity_id"] == dataset["id"])
        assert "/private/images.json" not in str(audit)


def test_archive_impact_lists_lineage_dependencies_before_state_change():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with TestClient(app) as client:
        project_id = client.post("/api/v1/projects", json={"name": "impact-project"}).json()["id"]
        dataset = client.post(f"/api/v1/projects/{project_id}/datasets", json={"name": "images", "version": "v1"}).json()
        model = client.post(f"/api/v1/projects/{project_id}/models", json={"name": "candidate", "version": "v1", "family": "fixture", "source_dataset_id": dataset["id"]}).json()
        run = client.post(f"/api/v1/projects/{project_id}/runs", json={"kind": "training", "name": "train", "dataset_id": dataset["id"], "model_id": model["id"]}).json()
        dataset_impact = client.get(f"/api/v1/projects/{project_id}/datasets/{dataset['id']}/impact").json()
        assert {item["kind"] for item in dataset_impact["dependencies"]} >= {"model", "run"}
        model_impact = client.get(f"/api/v1/projects/{project_id}/models/{model['id']}/impact").json()
        assert any(item["id"] == run["id"] for item in model_impact["dependencies"])
        assert client.get(f"/api/v1/projects/{project_id}/datasets/{dataset['id']}/impact").json()["entity"]["status"] == "draft"


def test_storage_impact_includes_inventory_assets_and_jobs():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with TestClient(app) as client:
        project_id = client.post("/api/v1/projects", json={"name": "storage-impact"}).json()["id"]
        storage = client.post(f"/api/v1/projects/{project_id}/storages", json={"name": "workspace", "root_path": "."}).json()
        inventory = client.post(f"/api/v1/projects/{project_id}/storages/{storage['id']}/inventory", json={"relative_path": "examples", "limit": 2})
        assert inventory.status_code == 200
        client.post(f"/api/v1/projects/{project_id}/storages/{storage['id']}/inventory-job", json={"recursive": False, "limit": 2})
        impact = client.get(f"/api/v1/projects/{project_id}/storages/{storage['id']}/impact")
        assert impact.status_code == 200
        assert {item["kind"] for item in impact.json()["dependencies"]} >= {"asset", "inventory-job"}


def test_evaluation_set_is_persisted_and_must_match_dataset():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with TestClient(app) as client:
        project_id = client.post("/api/v1/projects", json={"name": "evaluation-set-contract"}).json()["id"]
        dataset = client.post(f"/api/v1/projects/{project_id}/datasets", json={
            "name": "eval", "version": "v1", "format": "coco", "annotation_path": "examples/mmdetection/annotations/coco8.json",
        }).json()
        other_dataset = client.post(f"/api/v1/projects/{project_id}/datasets", json={"name": "other", "version": "v1"}).json()
        model = client.post(f"/api/v1/projects/{project_id}/models", json={"name": "candidate", "version": "v1", "family": "fixture", "task_kind": "detection", "source_dataset_id": dataset["id"]}).json()
        evaluation_set = client.post(f"/api/v1/projects/{project_id}/evaluation-sets", json={
            "name": "core", "version": "v1", "dataset_id": dataset["id"], "purpose": "core", "definition": {"items": [1]},
        }).json()
        assert evaluation_set["validation"]["status"] == "passed"
        revalidated = client.post(f"/api/v1/projects/{project_id}/evaluation-sets/{evaluation_set['id']}/validate")
        assert revalidated.status_code == 200
        assert revalidated.json()["validation"]["selected_item_count"] == 1
        valid = client.post(f"/api/v1/projects/{project_id}/evaluations/predictions", json={
            "model_id": model["id"], "dataset_id": dataset["id"], "evaluation_set_id": evaluation_set["id"], "predictions_path": "examples/mmdetection/predictions/rtmdet-tiny.json",
        })
        assert valid.status_code == 201
        assert valid.json()["run"]["config"]["evaluation_set_id"] == evaluation_set["id"]
        assert valid.json()["run"]["details"]["evaluation_set_filter"]["selected_image_count"] == 1
        assert valid.json()["run"]["details"]["invalid_predictions"] == 0
        invalid = client.post(f"/api/v1/projects/{project_id}/evaluations/predictions", json={
            "model_id": model["id"], "dataset_id": other_dataset["id"], "evaluation_set_id": evaluation_set["id"], "predictions_path": "examples/mmdetection/predictions/rtmdet-tiny.json",
        })
        assert invalid.status_code == 422


def test_classification_evaluation_set_filters_records_before_metrics():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with TestClient(app) as client:
        project_id = client.post("/api/v1/projects", json={"name": "classification-evaluation-set", "task_kind": "classification"}).json()["id"]
        dataset = client.post(f"/api/v1/projects/{project_id}/datasets", json={"name": "labels", "version": "v1", "task_kind": "classification", "class_names": ["cat", "dog"]}).json()
        model = client.post(f"/api/v1/projects/{project_id}/models", json={"name": "classifier", "version": "v1", "family": "fixture", "task_kind": "classification"}).json()
        evaluation_set = client.post(f"/api/v1/projects/{project_id}/evaluation-sets", json={
            "name": "core", "version": "v1", "dataset_id": dataset["id"], "definition": {"items": ["1"]},
        }).json()
        response = client.post(f"/api/v1/projects/{project_id}/evaluations/classification", json={
            "model_id": model["id"], "dataset_id": dataset["id"], "evaluation_set_id": evaluation_set["id"],
            "records": [
                {"image_id": "1", "ground_truth": "cat", "prediction": "cat"},
                {"image_id": "2", "ground_truth": "dog", "prediction": "cat"},
            ],
        })
        assert response.status_code == 201, response.text
        assert response.json()["result"]["top1_accuracy"] == 1.0
        assert response.json()["run"]["details"]["evaluation_set_filter"]["evaluated_record_count"] == 1


def test_versioned_contract_schemas_are_available_from_api():
    with TestClient(app) as client:
        response = client.get("/api/v1/schemas")
        assert response.status_code == 200
        body = response.json()
        assert body["schema_version"] == "1.0"
        assert {"dataset", "model", "run", "result_manifest", "release"} <= set(body["schemas"])
        assert "properties" in body["schemas"]["dataset"]


def test_field_failure_batch_preserves_source_lineage_and_artifacts():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with TestClient(app) as client:
        project_id = client.post("/api/v1/projects", json={"name": "field-feedback"}).json()["id"]
        dataset = client.post(f"/api/v1/projects/{project_id}/datasets", json={"name": "eval", "version": "v1"}).json()
        model = client.post(f"/api/v1/projects/{project_id}/models", json={"name": "candidate", "version": "v1", "family": "fixture", "source_dataset_id": dataset["id"]}).json()
        batch = client.post(f"/api/v1/projects/{project_id}/field-batches", json={
            "name": "night failures", "source_model_id": model["id"], "source_dataset_id": dataset["id"],
            "source_path": "examples/mmdetection/dataset.json", "sample_count": 4, "failure_count": 2,
            "metadata_json": {"label_status": "pending"},
        })
        assert batch.status_code == 201
        assert batch.json()["source_model_id"] == model["id"]
        assert batch.json()["metadata_json"]["artifact_ids"]["source_path"]
        listed = client.get(f"/api/v1/projects/{project_id}/field-batches")
        assert listed.status_code == 200 and listed.json()[0]["failure_count"] == 2
        graph = client.get(f"/api/v1/projects/{project_id}/lineage").json()
        assert any(node["id"] == batch.json()["id"] and node["kind"] == "field-batch" for node in graph["nodes"])


def test_split_validation_rejects_duplicate_items_and_group_leakage():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with TestClient(app) as client:
        project_id = client.post("/api/v1/projects", json={"name": "split-validation"}).json()["id"]
        dataset = client.post(f"/api/v1/projects/{project_id}/datasets", json={
            "name": "coco", "version": "v1", "format": "coco", "annotation_path": "examples/mmdetection/annotations/coco8.json",
        }).json()
        invalid = client.post(f"/api/v1/projects/{project_id}/splits", json={
            "name": "leaky", "version": "v1", "dataset_id": dataset["id"], "definition": {
                "assignments": [
                    {"item_id": 1, "split": "train", "group_id": "event-1"},
                    {"item_id": 1, "split": "val", "group_id": "event-1"},
                    {"item_id": 2, "split": "val", "group_id": "event-1"},
                ], "require_complete": True,
            },
        })
        assert invalid.status_code == 201
        assert invalid.json()["validation"]["status"] == "failed"
        assert invalid.json()["validation"]["duplicate_items"] == {"1": ["train", "val"]}
        assert "event-1" in invalid.json()["validation"]["group_leaks"]
        checked = client.post(f"/api/v1/projects/{project_id}/splits/{invalid.json()['id']}/validate")
        assert checked.status_code == 200 and checked.json()["status"] == "draft"
        valid = client.post(f"/api/v1/projects/{project_id}/splits", json={
            "name": "clean", "version": "v1", "dataset_id": dataset["id"], "definition": {
                "splits": {"train": [1], "val": [2]}, "groups": {"1": "event-1", "2": "event-2"}, "require_complete": True,
            },
        })
        assert valid.status_code == 201 and valid.json()["validation"]["status"] == "passed"
        promoted = client.post(f"/api/v1/projects/{project_id}/splits/{valid.json()['id']}/validate")
        assert promoted.status_code == 200 and promoted.json()["status"] == "validated"
