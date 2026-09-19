from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .artifacts import register_artifact
from .audit import record_audit
from .dataset_snapshot import build_dataset_snapshot
from .evaluators.detection import evaluate_coco_full, evaluate_coco_predictions
from .fingerprints import dataset_fingerprint
from .models import DatasetVersion, EvaluationSetVersion, ModelVersion, Run
from .serializers import as_dict


class DatasetSourceChangedError(ValueError):
    """Raised when a finalized dataset no longer matches its source files."""


def _evaluation_image_ids(evaluation_set: EvaluationSetVersion | None) -> set[int] | None:
    if not evaluation_set:
        return None
    definition = evaluation_set.definition or {}
    raw = next((definition[key] for key in ("items", "image_ids") if key in definition), None)
    if raw is None:
        return None
    if not isinstance(raw, list) or not raw:
        raise ValueError("COCO evaluation set items/image_ids must be a non-empty list")
    result: set[int] = set()
    for item in raw:
        value = item.get("image_id", item.get("item_id", item.get("id"))) if isinstance(item, dict) else item
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise ValueError("COCO evaluation set items must contain scalar image IDs")
        try:
            integer = int(str(value))
        except ValueError as error:
            raise ValueError("COCO evaluation set items must use integer image IDs") from error
        if str(integer) != str(value):
            raise ValueError("COCO evaluation set items must use canonical integer image IDs")
        result.add(integer)
    return result


def _ensure_source_current(dataset: DatasetVersion) -> None:
    if not dataset.content_hash or not dataset.content_hash.startswith("sha256:"):
        return
    current_hash, _ = dataset_fingerprint(dataset.manifest_path, dataset.annotation_path)
    if current_hash != dataset.content_hash:
        raise DatasetSourceChangedError("Dataset source files changed since this version was registered; create a new DatasetVersion")


def evaluate_coco_prediction_file(
    session: Session,
    project_id: str,
    *,
    dataset_id: str,
    model_id: str,
    predictions_path: str,
    protocol: str = "onboarding_ap50",
    iou_threshold: float = 0.5,
    evaluator_version: str = "lifecycle-ap50-v1",
    evaluation_set_id: str | None = None,
) -> dict[str, Any]:
    dataset = session.get(DatasetVersion, dataset_id)
    model = session.get(ModelVersion, model_id)
    if not dataset or dataset.project_id != project_id or not model or model.project_id != project_id:
        raise ValueError("Model and dataset must belong to this project")
    evaluation_set = session.get(EvaluationSetVersion, evaluation_set_id) if evaluation_set_id else None
    if evaluation_set_id and (not evaluation_set or evaluation_set.project_id != project_id):
        raise ValueError("Evaluation set must belong to this project")
    if evaluation_set and evaluation_set.dataset_id and evaluation_set.dataset_id != dataset.id:
        raise ValueError("Evaluation set must reference the selected DatasetVersion")
    if dataset.format != "coco" or not dataset.annotation_path:
        raise ValueError("Prediction evaluation currently requires a COCO dataset with annotation_path")
    _ensure_source_current(dataset)
    with Path(predictions_path).open(encoding="utf-8") as file:
        predictions = json.load(file)
    if not isinstance(predictions, list):
        raise ValueError("Prediction JSON must be a list")
    image_ids = _evaluation_image_ids(evaluation_set)
    if protocol == "onboarding_ap50":
        result = evaluate_coco_predictions(dataset.annotation_path, predictions, iou_threshold, image_ids=image_ids)
    elif protocol == "coco_full":
        result = evaluate_coco_full(dataset.annotation_path, predictions, image_ids=image_ids)
    else:
        raise ValueError("protocol must be onboarding_ap50 or coco_full")
    metric_scope = result.get("metric_scope", "")
    details = {key: value for key, value in result.items() if key != "metric_scope"}
    if evaluation_set:
        details["evaluation_set_filter"] = {"evaluation_set_id": evaluation_set.id, "selected_image_count": len(image_ids) if image_ids is not None else "all"}
    metrics = {key: value for key, value in details.items() if isinstance(value, (int, float))}
    run = Run(
        project_id=project_id, kind="evaluation", name=f"{model.family} prediction import", status="completed",
        dataset_id=dataset.id, model_id=model.id,
        config={"evaluator_version": "coco-full-v1" if protocol == "coco_full" and evaluator_version == "lifecycle-ap50-v1" else evaluator_version, "protocol": protocol, "iou_threshold": iou_threshold, "scope": metric_scope, **({"evaluation_set_id": evaluation_set.id} if evaluation_set else {})},
        metrics=metrics, details=details, environment={"source": "external-prediction-json"}, notes="Per-class output retained in evaluation import result.",
    )
    session.add(run)
    session.flush()
    artifact = register_artifact(session, project_id, kind="prediction", logical_name=f"{model.name}/{model.version}/{dataset.name}/{dataset.version}/predictions", owner_type="run", owner_id=run.id, source_path=predictions_path, notes=f"{protocol} evaluation input")
    details["prediction_artifact_id"] = artifact.id
    run.details = details
    record_audit(session, project_id, "run", run.id, "registered", after={"kind": run.kind, "name": run.name, "status": run.status, "dataset_id": run.dataset_id, "model_id": run.model_id})
    session.commit()
    session.refresh(run)
    return {"run": as_dict(run), "result": {"metric_scope": metric_scope, **details}}
