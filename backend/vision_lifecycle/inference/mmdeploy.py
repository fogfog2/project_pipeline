from __future__ import annotations

from pathlib import Path


def diagnose() -> dict:
    """Report availability without making MMDeploy a required local install."""
    try:
        import mmdeploy_runtime
        return {"available": True, "version": getattr(mmdeploy_runtime, "__version__", "unknown")}
    except ImportError:
        return {
            "available": False,
            "reason": "Install the mmdeploy optional dependency in the target-compatible runner environment.",
        }


def infer(model_directory: str, image_path: str, profile: dict) -> dict:
    """Run a deployed detector through the MMDeploy runtime Python API.

    ``model_directory`` is a converted MMDeploy model bundle directory.  The
    adapter does not guess model outputs: it only accepts the runtime's standard
    detector result and records the supplied target runtime identity.
    """
    try:
        from mmdeploy_runtime import Detector
    except ImportError as error:
        raise RuntimeError("MMDeploy inference requires the mmdeploy optional dependency.") from error
    if not Path(model_directory).is_dir():
        raise ValueError(f"MMDeploy model directory does not exist: {model_directory}")
    if not Path(image_path).is_file():
        raise ValueError(f"Image does not exist: {image_path}")
    device_name = str(profile.get("device_name", "cpu"))
    device_id = int(profile.get("device_id", 0))
    detector = Detector(model_directory, device_name=device_name, device_id=device_id)
    result = detector(image_path)
    if not isinstance(result, tuple) or len(result) < 2:
        raise ValueError("MMDeploy detector did not return (bboxes, labels)")
    bboxes, labels = result[0], result[1]
    if len(bboxes) != len(labels):
        raise ValueError("MMDeploy returned inconsistent boxes and labels")
    predictions = []
    for bbox, label in zip(bboxes, labels, strict=True):
        if len(bbox) < 5:
            raise ValueError("MMDeploy detector bounding box must include xyxy and score")
        predictions.append({
            "bbox_xyxy": [float(value) for value in bbox[:4]],
            "score": float(bbox[4]), "label": int(label),
        })
    return {
        "task_kind": "detection", "provider": f"MMDeploy runtime ({device_name}:{device_id})",
        "predictions": predictions,
    }
