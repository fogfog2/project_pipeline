from pathlib import Path

import pytest

from vision_lifecycle.inference.onnx import OnnxProfile, infer
from vision_lifecycle.database import Base, engine
from vision_lifecycle.main import app
from fastapi.testclient import TestClient


def test_detection_profile_requires_known_shape():
    profile = OnnxProfile.from_dict({"task_kind": "detection", "input_name": "images", "input_size": [640, 640], "outputs": {"boxes": "boxes", "scores": "scores", "labels": "labels"}})
    assert profile.layout == "NCHW"


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


def test_onnx_batch_classification_evaluation_persists_run_and_input_artifact(tmp_path: Path):
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
