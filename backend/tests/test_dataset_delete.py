from fastapi.testclient import TestClient

from vision_lifecycle.database import Base, engine
from vision_lifecycle.main import app


def test_unreferenced_draft_dataset_can_be_deleted_without_touching_source(tmp_path):
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    source = tmp_path / "images"
    source.mkdir()
    with TestClient(app) as client:
        project_id = client.post("/api/v1/projects", json={"name": "draft-delete"}).json()["id"]
        dataset = client.post(f"/api/v1/projects/{project_id}/datasets", json={
            "name": "scratch", "version": "v1", "task_kind": "detection", "format": "folder", "manifest_path": str(source),
        }).json()
        response = client.delete(f"/api/v1/projects/{project_id}/datasets/{dataset['id']}")
        assert response.status_code == 204
        assert source.is_dir()
        assert client.get(f"/api/v1/projects/{project_id}/datasets").json() == []
        events = client.get(f"/api/v1/projects/{project_id}/audit-events").json()
        assert any(event["action"] == "deleted" and event["entity_id"] == dataset["id"] for event in events)


def test_referenced_or_finalized_dataset_cannot_be_deleted():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with TestClient(app) as client:
        project_id = client.post("/api/v1/projects", json={"name": "protected-delete"}).json()["id"]
        dataset = client.post(f"/api/v1/projects/{project_id}/datasets", json={
            "name": "source", "version": "v1", "task_kind": "detection", "format": "unknown",
        }).json()
        model = client.post(f"/api/v1/projects/{project_id}/models", json={
            "name": "candidate", "version": "v1", "family": "fixture", "task_kind": "detection", "source_dataset_id": dataset["id"],
        })
        assert model.status_code == 201
        blocked = client.delete(f"/api/v1/projects/{project_id}/datasets/{dataset['id']}")
        assert blocked.status_code == 409
        assert "dependencies" in blocked.json()["detail"]

        other = client.post(f"/api/v1/projects/{project_id}/datasets", json={
            "name": "final", "version": "v1", "task_kind": "detection", "format": "unknown",
        }).json()
        finalized = client.post(f"/api/v1/projects/{project_id}/datasets/{other['id']}/finalize")
        assert finalized.status_code == 422  # no accessible manifest/annotation
        # A finalized record is protected independently of source accessibility.
        from vision_lifecycle.database import SessionLocal
        from vision_lifecycle.models import DatasetVersion
        with SessionLocal() as session:
            session.get(DatasetVersion, other["id"]).status = "finalized"
            session.commit()
        assert client.delete(f"/api/v1/projects/{project_id}/datasets/{other['id']}").status_code == 409
