from __future__ import annotations

from pathlib import Path


def diagnose() -> dict:
    try:
        import mmdet
        return {"available": True, "version": mmdet.__version__}
    except ImportError:
        return {"available": False, "reason": "Install the mmdetection optional dependency in the runner environment."}


def infer(config_path: str, checkpoint_path: str, image_path: str) -> dict:
    """Run an MMDetection 3.x detection bundle using its native config pipeline."""
    try:
        from mmdet.apis import DetInferencer
    except ImportError as error:
        raise RuntimeError("MMDetection native inference requires the mmdetection optional dependency.") from error
    for label, path in {"config": config_path, "checkpoint": checkpoint_path, "image": image_path}.items():
        if not Path(path).is_file():
            raise ValueError(f"MMDetection {label} does not exist: {path}")
    inferencer = DetInferencer(model=config_path, weights=checkpoint_path, device="cpu")
    output = inferencer(image_path, no_save_vis=True, return_vis=False)
    prediction = output["predictions"][0]
    boxes, scores, labels = prediction.get("bboxes", []), prediction.get("scores", []), prediction.get("labels", [])
    if not (len(boxes) == len(scores) == len(labels)):
        raise ValueError("MMDetection returned inconsistent boxes, scores, and labels")
    return {
        "task_kind": "detection", "provider": "MMDetection native CPU",
        "predictions": [{"bbox": [float(value) for value in box], "score": float(score), "label": int(label)} for box, score, label in zip(boxes, scores, labels, strict=True)],
    }
