from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OnnxProfile:
    task_kind: str
    input_name: str
    input_size: tuple[int, int]
    layout: str = "NCHW"
    color: str = "RGB"
    mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    std: tuple[float, float, float] = (1.0, 1.0, 1.0)
    outputs: dict[str, str] | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "OnnxProfile":
        size = value.get("input_size", [0, 0])
        if value.get("task_kind") not in {"classification", "detection"}:
            raise ValueError("ONNX profile task_kind must be classification or detection")
        if not isinstance(size, list) or len(size) != 2 or any(not isinstance(item, int) or item <= 0 for item in size):
            raise ValueError("ONNX profile input_size must be [width, height]")
        if value.get("layout", "NCHW") not in {"NCHW", "NHWC"}:
            raise ValueError("ONNX profile layout must be NCHW or NHWC")
        if value.get("color", "RGB") not in {"RGB", "BGR"}:
            raise ValueError("ONNX profile color must be RGB or BGR")
        return cls(
            task_kind=value["task_kind"], input_name=value["input_name"], input_size=(size[0], size[1]),
            layout=value.get("layout", "NCHW"), color=value.get("color", "RGB"),
            mean=tuple(value.get("mean", [0.0, 0.0, 0.0])), std=tuple(value.get("std", [1.0, 1.0, 1.0])), outputs=value.get("outputs"),
        )


def diagnose() -> dict:
    try:
        import onnxruntime as ort
        return {"available": True, "version": ort.__version__, "providers": ort.get_available_providers()}
    except ImportError:
        return {"available": False, "reason": "Install the onnx extra dependencies to enable local inference."}


def _dependencies():
    try:
        import numpy as np
        import onnxruntime as ort
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("ONNX Runtime inference needs numpy, pillow, and onnxruntime. Install project dependencies first.") from error
    return np, ort, Image


def _input_tensor(image_path: str, profile: OnnxProfile):
    np, _, Image = _dependencies()
    source = Path(image_path)
    if not source.is_file():
        raise ValueError(f"Image does not exist: {source}")
    image = Image.open(source).convert("RGB").resize(profile.input_size)
    values = np.asarray(image, dtype=np.float32)
    if profile.color == "BGR":
        values = values[..., ::-1]
    values = (values - np.asarray(profile.mean, dtype=np.float32)) / np.asarray(profile.std, dtype=np.float32)
    if profile.layout == "NCHW":
        values = values.transpose(2, 0, 1)[None, ...]
    else:
        values = values[None, ...]
    return values


def infer(model_path: str, image_path: str, profile_data: dict[str, Any]) -> dict:
    profile = OnnxProfile.from_dict(profile_data)
    np, ort, _ = _dependencies()
    if not Path(model_path).is_file():
        raise ValueError(f"ONNX model does not exist: {model_path}")
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    known_inputs = {item.name for item in session.get_inputs()}
    if profile.input_name not in known_inputs:
        raise ValueError(f"Profile input '{profile.input_name}' is not in model inputs: {sorted(known_inputs)}")
    output_values = session.run(None, {profile.input_name: _input_tensor(image_path, profile)})
    output_names = [item.name for item in session.get_outputs()]
    results = dict(zip(output_names, output_values, strict=True))
    if profile.task_kind == "classification":
        if len(results) != 1 and not profile.outputs:
            raise ValueError("Classification profile must specify outputs when the model has multiple output tensors")
        output_name = (profile.outputs or {}).get("scores") or output_names[0]
        scores = results.get(output_name)
        if scores is None:
            raise ValueError(f"Classification score output '{output_name}' is unavailable")
        flat = np.asarray(scores).reshape(-1)
        order = flat.argsort()[::-1]
        return {"task_kind": "classification", "provider": "CPUExecutionProvider", "top_indices": order[:5].tolist(), "top_scores": [float(flat[index]) for index in order[:5]]}
    required = {"boxes", "scores", "labels"}
    mapping = profile.outputs or {}
    if not required.issubset(mapping):
        raise ValueError("Detection ONNX profile requires explicit outputs: boxes, scores, labels. Use a task adapter for raw YOLO tensors.")
    missing = [key for key, output_name in mapping.items() if output_name not in results]
    if missing:
        raise ValueError(f"Detection output tensors are missing for: {missing}")
    boxes, scores, labels = (np.asarray(results[mapping[key]]).reshape(-1, 4) if key == "boxes" else np.asarray(results[mapping[key]]).reshape(-1) for key in ("boxes", "scores", "labels"))
    return {"task_kind": "detection", "provider": "CPUExecutionProvider", "predictions": [{"bbox": [float(value) for value in box], "score": float(score), "label": int(label)} for box, score, label in zip(boxes, scores, labels, strict=True)]}
