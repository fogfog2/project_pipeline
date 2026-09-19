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
from .evaluators.classification import evaluate_classification
from .fingerprints import dataset_fingerprint
from .inference.onnx import infer as infer_onnx
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


def evaluate_onnx_batch_file(session: Session, project_id: str, *, model_id: str, dataset_id: str, records_path: str, evaluator_version: str = "onnx-batch-v1", top_k: int = 5, evaluation_set_id: str | None = None) -> dict[str, Any]:
    """Run an explicitly profiled ONNX model over records and persist one Run."""
    dataset = session.get(DatasetVersion, dataset_id)
    model = session.get(ModelVersion, model_id)
    if not dataset or dataset.project_id != project_id or not model or model.project_id != project_id:
        raise ValueError("Model and dataset must belong to this project")
    evaluation_set = session.get(EvaluationSetVersion, evaluation_set_id) if evaluation_set_id else None
    if evaluation_set_id and (not evaluation_set or evaluation_set.project_id != project_id):
        raise ValueError("Evaluation set must belong to this project")
    if evaluation_set and evaluation_set.dataset_id and evaluation_set.dataset_id != dataset.id:
        raise ValueError("Evaluation set must reference the selected DatasetVersion")
    if model.format != "onnx" or not model.artifact_path:
        raise ValueError("ONNX batch evaluation requires a registered ONNX model artifact")
    profile = model.metadata_json.get("onnx_profile")
    if not profile:
        raise ValueError("Model metadata must include an explicit onnx_profile")
    _ensure_source_current(dataset)
    with Path(records_path).open(encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list) or not records:
        raise ValueError("ONNX records JSON must be a non-empty list")
    if any(not isinstance(record, dict) or not record.get("image_path") for record in records):
        raise ValueError("Every ONNX record must include image_path")
    resolved_records = []
    for record in records:
        image_path = Path(str(record["image_path"]))
        if not image_path.is_absolute():
            image_path = Path(records_path).parent / image_path
        resolved_records.append({**record, "image_path": str(image_path)})
    item_keys = None
    if evaluation_set:
        raw = next((evaluation_set.definition.get(key) for key in ("items", "image_ids") if key in evaluation_set.definition), None)
        if raw is not None:
            if not isinstance(raw, list) or not raw:
                raise ValueError("Evaluation set must contain at least one item")
            item_keys = {str(item.get("image_id", item.get("item_id", item.get("id"))) if isinstance(item, dict) else item) for item in raw}
            resolved_records = [record for record in resolved_records if str(record.get("image_id", record.get("item_id", record.get("id", "")))) in item_keys]
            if not resolved_records:
                raise ValueError("Evaluation set selected no ONNX records")
    predictions = [infer_onnx(model.artifact_path, record["image_path"], profile) for record in resolved_records]
    if model.task_kind == "classification":
        class_names = [str(value) for value in dataset.class_names]
        output_records = []
        for index, (record, prediction) in enumerate(zip(resolved_records, predictions, strict=True)):
            top_indices = prediction.get("top_indices", [])
            labels = [class_names[item] if item < len(class_names) else str(item) for item in top_indices[:top_k]]
            if not labels:
                raise ValueError(f"ONNX classification returned no scores for record {index}")
            output_records.append({"image_id": record.get("image_id", str(index)), "ground_truth": record.get("ground_truth"), "prediction": labels[0], "top_k": labels})
        if any(item["ground_truth"] is None for item in output_records):
            raise ValueError("Classification ONNX records must include ground_truth")
        result = evaluate_classification(output_records)
        metrics = {key: result[key] for key in ("top1_accuracy", "topk_accuracy", "macro_precision", "macro_recall", "macro_f1", "records")}
        details = {**result, "records_path": records_path, "inference_provider": predictions[0].get("provider", "unknown")}
    elif model.task_kind == "detection":
        if dataset.format != "coco" or not dataset.annotation_path:
            raise ValueError("Detection ONNX batch evaluation requires a COCO dataset with annotation_path")
        mapping = model.metadata_json.get("class_mapping")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("Detection ONNX model metadata must include explicit class_mapping from model labels to COCO category IDs")
        coco_predictions = []
        for index, (record, prediction) in enumerate(zip(resolved_records, predictions, strict=True)):
            image_id = record.get("image_id", index)
            for item in prediction.get("predictions", []):
                label = str(item.get("label"))
                if label not in mapping and item.get("label") not in mapping:
                    raise ValueError(f"Detection class mapping is missing model label {label}")
                category_id = mapping.get(label, mapping.get(item.get("label")))
                coco_predictions.append({"image_id": image_id, "category_id": int(category_id), "bbox": item["bbox"], "score": item["score"]})
        result = evaluate_coco_full(dataset.annotation_path, coco_predictions, image_ids=_evaluation_image_ids(evaluation_set))
        metrics = {key: value for key, value in result.items() if isinstance(value, (int, float))}
        details = {key: value for key, value in result.items() if key != "metric_scope"}
        details.update({"records_path": records_path, "inference_provider": predictions[0].get("provider", "unknown"), "prediction_count": len(coco_predictions)})
    else:
        raise ValueError(f"ONNX batch evaluation does not support task_kind {model.task_kind}")
    if evaluation_set:
        details["evaluation_set_filter"] = {"evaluation_set_id": evaluation_set.id, "selected_item_count": len(item_keys) if item_keys is not None else "all", "evaluated_record_count": len(resolved_records)}
    run = Run(project_id=project_id, kind="evaluation", name=f"{model.family} ONNX batch evaluation", status="completed", dataset_id=dataset.id, model_id=model.id, config={"evaluator_version": evaluator_version, "protocol": "onnx-batch", "task_kind": model.task_kind, "top_k": top_k, "scope": result.get("metric_scope", "classification"), **({"evaluation_set_id": evaluation_set.id} if evaluation_set else {})}, metrics=metrics, details=details, environment={"provider": details.get("inference_provider", "unknown"), "runtime": "onnxruntime-cpu"}, notes="ONNX batch inference and evaluation from an explicit image record manifest.")
    session.add(run)
    session.flush()
    artifact = register_artifact(session, project_id, kind="onnx-records", logical_name=f"{model.name}/{model.version}/{dataset.name}/{dataset.version}/onnx-records", owner_type="run", owner_id=run.id, source_path=records_path, notes="ONNX batch input records")
    run.details = {**details, "records_artifact_id": artifact.id}
    record_audit(session, project_id, "run", run.id, "registered", after={"kind": run.kind, "name": run.name, "status": run.status, "dataset_id": run.dataset_id, "model_id": run.model_id})
    session.commit()
    session.refresh(run)
    return {"run": as_dict(run), "result": {"metric_scope": result.get("metric_scope", "classification"), **details}}
