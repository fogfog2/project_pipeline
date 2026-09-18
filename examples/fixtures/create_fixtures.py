"""Create tiny CPU ONNX fixtures used by the onboarding smoke flow."""
from pathlib import Path

from PIL import Image
from onnx import TensorProto, helper, save

ROOT = Path(__file__).parent
(ROOT / "images").mkdir(exist_ok=True)
(ROOT / "records").mkdir(exist_ok=True)

def model(graph):
    value = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    value.ir_version = 10
    return value

classification = helper.make_graph(
    [helper.make_node("ReduceMean", ["images"], ["scores"], axes=[2, 3], keepdims=0)],
    "mean-classifier",
    [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 2, 2])],
    [helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 3])],
)
save(model(classification), ROOT / "classification_mean.onnx")

detection = helper.make_graph(
    [
        helper.make_node("Constant", [], ["boxes"], value=helper.make_tensor("boxes_value", TensorProto.FLOAT, [1, 4], [0.0, 0.0, 2.0, 2.0])),
        helper.make_node("Constant", [], ["scores"], value=helper.make_tensor("scores_value", TensorProto.FLOAT, [1], [0.99])),
        helper.make_node("Constant", [], ["labels"], value=helper.make_tensor("labels_value", TensorProto.INT64, [1], [0])),
    ],
    "constant-detector",
    [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 2, 2])],
    [
        helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 4]),
        helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1]),
        helper.make_tensor_value_info("labels", TensorProto.INT64, [1]),
    ],
)
save(model(detection), ROOT / "detection_constant.onnx")

Image.new("RGB", (2, 2), (0, 10, 20)).save(ROOT / "images" / "classification.png")
Image.new("RGB", (2, 2), (1, 2, 3)).save(ROOT / "images" / "detection.jpg")
(ROOT / "records" / "classification.json").write_text(
    '[{"image_path":"../images/classification.png","image_id":"1","ground_truth":"blue"}]\n',
    encoding="utf-8",
)
(ROOT / "records" / "detection.json").write_text(
    '[{"image_path":"../images/detection.jpg","image_id":1}]\n',
    encoding="utf-8",
)
print("Created classification_mean.onnx and detection_constant.onnx fixtures")
