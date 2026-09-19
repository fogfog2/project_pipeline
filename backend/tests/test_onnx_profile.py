from pathlib import Path

import pytest

from vision_lifecycle.inference.onnx import OnnxProfile, infer
from vision_lifecycle.database import Base, engine
from vision_lifecycle.main import app
from fastapi.testclient import TestClient


def test_detection_profile_requires_known_shape():
    profile = OnnxProfile.from_dict({"task_kind": "detection", "input_name": "images", "input_size": [640, 640], "outputs": {"boxes": "boxes", "scores": "scores", "labels": "labels"}})
    assert profile.layout == "NCHW"


def test_onnx_model_without_profile_is_not_marked_runnable():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with TestClient(app) as client:
        project_id = client.post("/api/v1/projects", json={"name": "profile-required"}).json()["id"]
        response = client.post(f"/api/v1/projects/{project_id}/models", json={"name": "missing-profile", "version": "v1", "family": "fixture", "task_kind": "detection", "format": "onnx", "runnable": True})
        assert response.status_code == 201
        assert response.json()["runnable"] is False
        assert response.json()["metadata_json"]["adapter_status"] == "required"
        updated = client.patch(f"/api/v1/projects/{project_id}/models/{response.json()['id']}", json={"runnable": True})
        assert updated.status_code == 200
        assert updated.json()["runnable"] is False


def test_onnx_model_rejects_malformed_profile():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with TestClient(app) as client:
        project_id = client.post("/api/v1/projects", json={"name": "bad-profile"}).json()["id"]
        response = client.post(f"/api/v1/projects/{project_id}/models", json={"name": "bad", "version": "v1", "family": "fixture", "format": "onnx", "metadata_json": {"onnx_profile": {"task_kind": "detection", "input_size": [0, 640]}}})
        assert response.status_code == 422
        assert "Invalid ONNX profile" in response.json()["detail"]


def test_checked_in_onnx_fixtures_are_executable():
    result = infer("examples/fixtures/classification_mean.onnx", "examples/fixtures/images/classification.png", {"task_kind": "classification", "input_name": "images", "input_size": [2, 2]})
    assert result["top_indices"] == [2, 1, 0]
    detection = infer("examples/fixtures/detection_constant.onnx", "examples/fixtures/images/detection.jpg", {"task_kind": "detection", "input_name": "images", "input_size": [2, 2], "outputs": {"boxes": "boxes", "scores": "scores", "labels": "labels"}})
    assert detection["predictions"][0]["label"] == 0


def test_detection_profile_applies_threshold_and_class_mapping():
    mapped = infer("examples/fixtures/detection_constant.onnx", "examples/fixtures/images/detection.jpg", {
        "task_kind": "detection", "input_name": "images", "input_size": [2, 2],
        "outputs": {"boxes": "boxes", "scores": "scores", "labels": "labels"},
        "score_threshold": 0.5, "class_mapping": {"0": 7}, "bbox_format": "xywh",
    })
    assert mapped["predictions"][0]["label"] == 7
    filtered = infer("examples/fixtures/detection_constant.onnx", "examples/fixtures/images/detection.jpg", {
        "task_kind": "detection", "input_name": "images", "input_size": [2, 2],
        "outputs": {"boxes": "boxes", "scores": "scores", "labels": "labels"}, "score_threshold": 1.0,
    })
    assert filtered["predictions"] == []


def test_profile_rejects_invalid_postprocess_contract():
    with pytest.raises(ValueError, match="resize_mode"):
        OnnxProfile.from_dict({"task_kind": "detection", "input_name": "images", "input_size": [2, 2], "resize_mode": "crop"})


def test_classification_inference_profile(tmp_path: Path):
    onnx = pytest.importorskip("onnx")
    from PIL import Image
    from onnx import TensorProto, helper

    model_path = tmp_path / "mean.onnx"
    graph = helper.make_graph(
        [helper.make_node("ReduceMean", ["images"], ["scores"], axes=[2, 3], keepdims=0)],
        "mean-model", [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 2, 2])],
        [helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.save(model, model_path)
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (2, 2), (0, 10, 20)).save(image_path)
    result = infer(str(model_path), str(image_path), {"task_kind": "classification", "input_name": "images", "input_size": [2, 2]})
    assert result["top_indices"] == [2, 1, 0]


def test_onnx_batch_classification_evaluation_persists_run_and_input_artifact(tmp_path: Path, monkeypatch):
    onnx = pytest.importorskip("onnx")
    from PIL import Image
    from onnx import TensorProto, helper

    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    model_path = tmp_path / "mean.onnx"
    graph = helper.make_graph(
        [helper.make_node("ReduceMean", ["images"], ["scores"], axes=[2, 3], keepdims=0)],
        "mean-model", [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 2, 2])],
        [helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.save(model, model_path)
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (2, 2), (0, 10, 20)).save(image_path)
    records_path = tmp_path / "records.json"
    records_path.write_text('[{"image_path":"sample.png","image_id":"1","ground_truth":"blue"}]', encoding="utf-8")
    with TestClient(app) as client:
        project_id = client.post("/api/v1/projects", json={"name": "onnx-batch", "task_kind": "classification"}).json()["id"]
        dataset = client.post(f"/api/v1/projects/{project_id}/datasets", json={"name": "colors", "version": "v1", "task_kind": "classification", "format": "classification", "class_names": ["red", "green", "blue"]}).json()
        registered = client.post(f"/api/v1/projects/{project_id}/models", json={"name": "mean", "version": "v1", "family": "fixture-classifier", "task_kind": "classification", "format": "onnx", "artifact_path": str(model_path), "metadata_json": {"onnx_profile": {"task_kind": "classification", "input_name": "images", "input_size": [2, 2]}}}).json()
        response = client.post(f"/api/v1/projects/{project_id}/evaluations/onnx-batch", json={"model_id": registered["id"], "dataset_id": dataset["id"], "records_path": str(records_path)})
        assert response.status_code == 201, response.text
        assert response.json()["result"]["top1_accuracy"] == 1.0
        run = response.json()["run"]
        assert run["details"]["records_artifact_id"]
        assert any(item["kind"] == "onnx-records" for item in client.get(f"/api/v1/projects/{project_id}/artifacts").json())
        monkeypatch.setenv("VISION_LIFECYCLE_EXTERNAL_WORKER", "true")
        queued = client.post(f"/api/v1/projects/{project_id}/evaluations/onnx-batch/jobs", json={"model_id": registered["id"], "dataset_id": dataset["id"], "records_path": str(records_path)})
        assert queued.status_code == 201
        from vision_lifecycle.runner import worker_once
        assert worker_once("onnx-worker") is True
        assert client.get(f"/api/v1/projects/{project_id}/jobs").json()[0]["status"] == "completed"


def test_inference_preview_falls_back_to_managed_model_copy(tmp_path: Path, monkeypatch):
    onnx = pytest.importorskip("onnx")
    from PIL import Image
    from onnx import TensorProto, helper

    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    monkeypatch.setenv("VISION_LIFECYCLE_ARTIFACT_ROOT", str(tmp_path / "managed"))
    model_path = tmp_path / "mean.onnx"
    graph = helper.make_graph(
        [helper.make_node("ReduceMean", ["images"], ["scores"], axes=[2, 3], keepdims=0)],
        "mean-model", [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 2, 2])],
        [helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.save(model, model_path)
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (2, 2), (0, 10, 20)).save(image_path)
    with TestClient(app) as client:
        project_id = client.post("/api/v1/projects", json={"name": "managed-fallback", "task_kind": "classification"}).json()["id"]
        registered = client.post(f"/api/v1/projects/{project_id}/models", json={"name": "mean", "version": "v1", "family": "fixture", "task_kind": "classification", "format": "onnx", "artifact_path": str(model_path), "metadata_json": {"onnx_profile": {"task_kind": "classification", "input_name": "images", "input_size": [2, 2]}}}).json()
        model_path.unlink()
        response = client.post(f"/api/v1/projects/{project_id}/inference-preview", json={"model_id": registered["id"], "image_path": str(image_path)})
        assert response.status_code == 200, response.text
        assert response.json()["artifact_source"] == "managed"


def test_onnx_batch_detection_requires_and_uses_explicit_class_mapping(tmp_path: Path):
    onnx = pytest.importorskip("onnx")
    from PIL import Image
    from onnx import TensorProto, helper

    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    model_path = tmp_path / "detector.onnx"
    constants = [
        helper.make_node("Constant", [], ["boxes"], value=helper.make_tensor("boxes_value", TensorProto.FLOAT, [1, 4], [120, 85, 118, 290])),
        helper.make_node("Constant", [], ["scores"], value=helper.make_tensor("scores_value", TensorProto.FLOAT, [1], [0.99])),
        helper.make_node("Constant", [], ["labels"], value=helper.make_tensor("labels_value", TensorProto.INT64, [1], [0])),
    ]
    graph = helper.make_graph(constants, "constant-detector", [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 2, 2])], [
        helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 4]), helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1]), helper.make_tensor_value_info("labels", TensorProto.INT64, [1]),
    ])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.save(model, model_path)
    image_path = tmp_path / "demo.jpg"
    Image.new("RGB", (2, 2), (1, 2, 3)).save(image_path)
    annotation = tmp_path / "coco.json"
    annotation.write_text(Path("examples/mmdetection/annotations/coco8.json").read_text(encoding="utf-8"), encoding="utf-8")
    records = tmp_path / "records.json"
    records.write_text('[{"image_path":"demo.jpg","image_id":1}]', encoding="utf-8")
    with TestClient(app) as client:
        project_id = client.post("/api/v1/projects", json={"name": "onnx-detection", "task_kind": "detection"}).json()["id"]
        dataset = client.post(f"/api/v1/projects/{project_id}/datasets", json={"name": "coco", "version": "v1", "task_kind": "detection", "format": "coco", "annotation_path": str(annotation)}).json()
        registered = client.post(f"/api/v1/projects/{project_id}/models", json={"name": "detector", "version": "v1", "family": "fixture-detector", "task_kind": "detection", "format": "onnx", "artifact_path": str(model_path), "metadata_json": {"onnx_profile": {"task_kind": "detection", "input_name": "images", "input_size": [2, 2], "outputs": {"boxes": "boxes", "scores": "scores", "labels": "labels"}}, "class_mapping": {"0": 1}}}).json()
        response = client.post(f"/api/v1/projects/{project_id}/evaluations/onnx-batch", json={"model_id": registered["id"], "dataset_id": dataset["id"], "records_path": str(records)})
        assert response.status_code == 201, response.text
        assert response.json()["run"]["details"]["prediction_count"] == 1
