from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import math


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
    resize_mode: str = "stretch"
    score_threshold: float = 0.0
    nms_iou_threshold: float | None = None
    bbox_format: str = "xywh"
    class_mapping: dict[str, int] | None = None

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
        if value.get("resize_mode", "stretch") not in {"stretch", "letterbox"}:
            raise ValueError("ONNX profile resize_mode must be stretch or letterbox")
        threshold = value.get("score_threshold", 0.0)
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= float(threshold) <= 1:
            raise ValueError("ONNX profile score_threshold must be between 0 and 1")
        nms = value.get("nms_iou_threshold")
        if nms is not None and (isinstance(nms, bool) or not isinstance(nms, (int, float)) or not 0 < float(nms) <= 1):
            raise ValueError("ONNX profile nms_iou_threshold must be between 0 and 1 when provided")
        if value.get("bbox_format", "xywh") not in {"xywh", "xyxy"}:
            raise ValueError("ONNX profile bbox_format must be xywh or xyxy")
        if not isinstance(value.get("input_name"), str) or not value["input_name"]:
            raise ValueError("ONNX profile input_name is required")
        mean = value.get("mean", [0.0, 0.0, 0.0]); std = value.get("std", [1.0, 1.0, 1.0])
        if not isinstance(mean, list) or len(mean) != 3 or not all(isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(float(item)) for item in mean):
            raise ValueError("ONNX profile mean must contain three finite numbers")
        if not isinstance(std, list) or len(std) != 3 or not all(isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(float(item)) and float(item) != 0 for item in std):
            raise ValueError("ONNX profile std must contain three non-zero finite numbers")
        mapping = value.get("class_mapping")
        if mapping is not None and (not isinstance(mapping, dict) or any(not str(key).lstrip("-").isdigit() or not isinstance(label, int) for key, label in mapping.items())):
            raise ValueError("ONNX profile class_mapping must map numeric labels to integer labels")
        return cls(
            task_kind=value["task_kind"], input_name=value["input_name"], input_size=(size[0], size[1]),
            layout=value.get("layout", "NCHW"), color=value.get("color", "RGB"),
            mean=tuple(mean), std=tuple(std), outputs=value.get("outputs"), resize_mode=value.get("resize_mode", "stretch"),
            score_threshold=float(threshold), nms_iou_threshold=float(nms) if nms is not None else None,
            bbox_format=value.get("bbox_format", "xywh"), class_mapping=mapping,
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
    image = Image.open(source).convert("RGB")
    original_size = image.size
    scale = (1.0, 0.0, 0.0)
    if profile.resize_mode == "letterbox":
        ratio = min(profile.input_size[0] / image.width, profile.input_size[1] / image.height)
        resized = image.resize((max(1, round(image.width * ratio)), max(1, round(image.height * ratio))))
        canvas = Image.new("RGB", profile.input_size, (0, 0, 0))
        pad_x = (profile.input_size[0] - resized.width) / 2
        pad_y = (profile.input_size[1] - resized.height) / 2
        canvas.paste(resized, (round(pad_x), round(pad_y)))
        image = canvas
        scale = (ratio, pad_x, pad_y)
    else:
        image = image.resize(profile.input_size)
    values = np.asarray(image, dtype=np.float32)
    if profile.color == "BGR":
        values = values[..., ::-1]
    values = (values - np.asarray(profile.mean, dtype=np.float32)) / np.asarray(profile.std, dtype=np.float32)
    if profile.layout == "NCHW":
        values = values.transpose(2, 0, 1)[None, ...]
    else:
        values = values[None, ...]
    return values, {"original_size": original_size, "scale": scale}


def _nms(predictions: list[dict[str, object]], threshold: float) -> list[dict[str, object]]:
    kept: list[dict[str, object]] = []
    for candidate in sorted(predictions, key=lambda item: float(item["score"]), reverse=True):
        cx, cy, cw, ch = candidate["bbox"]
        overlap = False
        for selected in kept:
            if candidate["label"] != selected["label"]:
                continue
            sx, sy, sw, sh = selected["bbox"]
            ix = max(0.0, min(cx + cw, sx + sw) - max(cx, sx)); iy = max(0.0, min(cy + ch, sy + sh) - max(cy, sy))
            union = cw * ch + sw * sh - ix * iy
            if union > 0 and ix * iy / union > threshold:
                overlap = True; break
        if not overlap:
            kept.append(candidate)
    return kept


def infer(model_path: str, image_path: str, profile_data: dict[str, Any]) -> dict:
    profile = OnnxProfile.from_dict(profile_data)
    np, ort, _ = _dependencies()
    if not Path(model_path).is_file():
        raise ValueError(f"ONNX model does not exist: {model_path}")
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    known_inputs = {item.name for item in session.get_inputs()}
    if profile.input_name not in known_inputs:
        raise ValueError(f"Profile input '{profile.input_name}' is not in model inputs: {sorted(known_inputs)}")
    tensor, transform = _input_tensor(image_path, profile)
    output_values = session.run(None, {profile.input_name: tensor})
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
    predictions: list[dict[str, object]] = []
    ratio, pad_x, pad_y = transform["scale"]
    original_width, original_height = transform["original_size"]
    for box, score, label in zip(boxes, scores, labels, strict=True):
        score_value = float(score)
        if score_value < profile.score_threshold:
            continue
        values = [float(value) for value in box]
        if profile.bbox_format == "xyxy":
            x, y, x2, y2 = values; values = [x, y, x2 - x, y2 - y]
        if profile.resize_mode == "letterbox":
            values = [(values[0] - pad_x) / ratio, (values[1] - pad_y) / ratio, values[2] / ratio, values[3] / ratio]
        values[0] = max(0.0, min(values[0], float(original_width))); values[1] = max(0.0, min(values[1], float(original_height)))
        values[2] = max(0.0, min(values[2], float(original_width) - values[0])); values[3] = max(0.0, min(values[3], float(original_height) - values[1]))
        output_label = int(label)
        if profile.class_mapping is not None:
            if str(output_label) not in profile.class_mapping:
                raise ValueError(f"ONNX profile class_mapping has no label for {output_label}")
            output_label = profile.class_mapping[str(output_label)]
        predictions.append({"bbox": values, "score": score_value, "label": output_label})
    if profile.nms_iou_threshold is not None:
        predictions = _nms(predictions, profile.nms_iou_threshold)
    return {"task_kind": "detection", "provider": "CPUExecutionProvider", "predictions": predictions}
