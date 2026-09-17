from pathlib import Path

import pytest

from vision_lifecycle.inference.onnx import OnnxProfile, infer


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
