from pathlib import Path
from time import sleep

import pytest
from fastapi.testclient import TestClient

from vision_lifecycle.database import Base, engine
from vision_lifecycle.main import app
from vision_lifecycle.storage import browse, resolve_within, storage_status


def test_storage_browse_stays_within_configured_root(tmp_path: Path):
    root = tmp_path / "root"
    (root / "nested").mkdir(parents=True)
    (root / "nested" / "labels.json").write_text("{}", encoding="utf-8")
    (root / "image.jpg").write_bytes(b"fixture")
    assert storage_status(str(root))["status"] == "available"
    result = browse(str(root), "nested")
    assert result["entries"][0]["relative_path"] == "nested/labels.json"
    with pytest.raises(ValueError, match="escapes"):
        resolve_within(str(root), "../outside")


def test_dataset_hash_blocks_evaluation_after_source_changes(tmp_path: Path):
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    annotation = tmp_path / "instances.json"
    annotation.write_text(Path("examples/mmdetection/annotations/coco8.json").read_text(encoding="utf-8"), encoding="utf-8")
    with TestClient(app) as client:
        project_id = client.post("/api/v1/projects", json={"name": "hash-project"}).json()["id"]
        mapping = client.post(f"/api/v1/projects/{project_id}/storages", json={"name": "workspace", "root_path": str(tmp_path)}).json()
        assert client.post(f"/api/v1/projects/{project_id}/storages/{mapping['id']}/browse", json={}).status_code == 200
        inventory = client.post(f"/api/v1/projects/{project_id}/storages/{mapping['id']}/inventory", json={}).json()
        assert inventory["count"] >= 1
        assert client.get(f"/api/v1/projects/{project_id}/assets").json()[0]["sha256"]
        inventory_job = client.post(f"/api/v1/projects/{project_id}/storages/{mapping['id']}/inventory-job", json={"limit": 100}).json()
        for _ in range(20):
            current = next(item for item in client.get(f"/api/v1/projects/{project_id}/jobs").json() if item["id"] == inventory_job["id"])
            if current["status"] == "completed":
                break
            sleep(0.01)
        assert current["status"] == "completed"
        remapped_root = tmp_path / "remapped"
        remapped_root.mkdir()
        remapped = client.patch(f"/api/v1/projects/{project_id}/storages/{mapping['id']}", json={"root_path": str(remapped_root)})
        assert remapped.status_code == 200
        assert remapped.json()["id"] == mapping["id"]
        dataset = client.post(f"/api/v1/projects/{project_id}/datasets", json={"name": "coco", "version": "v1", "annotation_path": str(annotation)}).json()
        assert dataset["content_hash"].startswith("sha256:")
        preview = client.get(f"/api/v1/projects/{project_id}/datasets/{dataset['id']}/preview").json()
        assert preview["images"] and preview["images"][0]["annotations"]
        label = client.post(f"/api/v1/projects/{project_id}/label-schemas", json={"name": "coco-labels", "version": "v1", "dataset_id": dataset["id"], "classes": [{"id": 1, "name": "person"}]}).json()
        split = client.post(f"/api/v1/projects/{project_id}/splits", json={"name": "default", "version": "v1", "dataset_id": dataset["id"], "definition": {"train": 0.8, "val": 0.2, "seed": 42}}).json()
        eval_set = client.post(f"/api/v1/projects/{project_id}/evaluation-sets", json={"name": "core", "version": "v1", "dataset_id": dataset["id"], "purpose": "core", "definition": {"items": [1]}}).json()
        calibration = client.post(f"/api/v1/projects/{project_id}/calibration-sets", json={"name": "representative", "version": "v1", "dataset_id": dataset["id"], "sampling": {"count": 1}, "preprocessing": {"color": "rgb"}}).json()
        assert all(item["content_hash"].startswith("sha256:") for item in (label, split, eval_set, calibration))
        finalized = client.post(f"/api/v1/projects/{project_id}/datasets/{dataset['id']}/finalize")
        assert finalized.status_code == 200
        assert finalized.json()["status"] == "finalized"
        assert client.post(f"/api/v1/projects/{project_id}/datasets/{dataset['id']}/finalize").status_code == 409
        artifact = tmp_path / "model.onnx"; artifact.write_bytes(b"onnx-fixture")
        config = tmp_path / "model.json"; config.write_text("{}", encoding="utf-8")
        model = client.post(f"/api/v1/projects/{project_id}/models", json={"name": "detector", "version": "v1", "family": "fixture", "source_dataset_id": dataset["id"], "artifact_path": str(artifact), "config_path": str(config)}).json()
        assert model["artifact_sha256"] and model["config_sha256"]
        artifacts = client.get(f"/api/v1/projects/{project_id}/artifacts").json()
        assert {item["kind"] for item in artifacts} >= {"dataset-annotation", "model", "config"}
        artifact_verify = client.post(f"/api/v1/projects/{project_id}/models/{model['id']}/verify-artifacts")
        assert artifact_verify.status_code == 200 and artifact_verify.json()["ok"] is True
        artifact.write_bytes(b"changed-model")
        drift = client.post(f"/api/v1/projects/{project_id}/models/{model['id']}/verify-artifacts")
        assert drift.json()["ok"] is False and any(item["status"] == "drifted" for item in drift.json()["artifacts"])
        replacement = tmp_path / "model-v2.onnx"; replacement.write_bytes(b"onnx-v2")
        updated = client.patch(f"/api/v1/projects/{project_id}/models/{model['id']}", json={"artifact_path": str(replacement)})
        assert updated.status_code == 200
        current_artifacts = client.get(f"/api/v1/projects/{project_id}/artifacts").json()
        assert any(item["status"] == "superseded" for item in current_artifacts if item["kind"] == "model")
        annotation.write_text(annotation.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        response = client.post(f"/api/v1/projects/{project_id}/evaluations/predictions", json={
            "model_id": model["id"], "dataset_id": dataset["id"], "predictions_path": "examples/mmdetection/predictions/rtmdet-tiny.json",
        })
        assert response.status_code == 409
        exported = client.get(f"/api/v1/projects/{project_id}/export")
        assert exported.status_code == 200
        # Path values and the absolute-path keys inside source fingerprints are
        # intentionally absent from a Pages-safe snapshot.
        assert str(tmp_path) not in exported.text
        assert all("source_path" not in item for item in exported.json()["artifacts"])
