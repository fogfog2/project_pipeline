from __future__ import annotations

from contextlib import asynccontextmanager
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from .database import SessionLocal, init_database
from .adapters.coco import load_coco, validate_coco
from .adapters.inspect import inspect_path
from .evaluators.detection import evaluate_coco_full, evaluate_coco_predictions
from .evaluators.classification import evaluate_classification
from .inference.onnx import diagnose as diagnose_onnx, infer as infer_onnx
from .inference.mmdetection import diagnose as diagnose_mmdetection, infer as infer_mmdetection
from .inference.mmdeploy import diagnose as diagnose_mmdeploy, infer as infer_mmdeploy
from .importer import manifest_hash, validate_result_manifest
from .models import Artifact, AuditEvent, BoardBenchmark, CalibrationSetVersion, DataAsset, DatasetVersion, EvaluationSetVersion, FieldDataBatch, Job, LabelSchemaVersion, ModelAliasHistory, ModelVersion, OnboardingSession, Project, QuantizationRun, Release, ReleaseEvidence, Run, RunnerProfile, SplitVersion, StepProgress, StorageMapping, TargetProfile
from .release_gate import GateConfigError, evaluate_gate
from .runner import cancel, launch, recover_interrupted
from .schemas import ArtifactCreate, BoardBenchmarkCreate, ClassificationEvaluationCreate, ComparisonRequest, DatasetCreate, DatasetUpdate, FieldDataBatchCreate, InferencePreviewRequest, JobCreate, ModelCreate, ModelUpdate, OnboardingCreate, OnnxBatchEvaluationCreate, PathInspectRequest, PredictionEvaluationCreate, ProjectCreate, ProjectUpdate, QuantizationComparisonRequest, QuantizationRunCreate, ReleaseCreate, ResultImportCreate, RunCreate, RunnerProfileCreate, StepProgressUpdate, StorageBrowseRequest, StorageInventoryRequest, StorageMappingCreate, StorageMappingUpdate, TargetProfileCreate, VersionDefinitionCreate
from . import schemas as contract_schemas
from .serializers import as_dict
from .service import agent_request, compare_models, lineage, overview, safe_export, seed_demo
from .artifacts import register_artifact, verify_artifact
from .fingerprints import dataset_fingerprint, file_sha256
from .dataset_snapshot import build_dataset_snapshot, diff_dataset_snapshots
from .split_validation import validate_split_definition
from .calibration_validation import inspect_calibration_statistics, validate_calibration_definition
from .evaluation_validation import validate_evaluation_definition
from .board_validation import validate_board_measurement
from .storage import browse as browse_storage, inventory as inventory_storage, storage_status
from .audit import record_audit


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_database()
    # Queued jobs belong to the external worker queue and must remain claimable
    # after an API restart. Only jobs that were actively running are marked
    # interrupted; they are never started again automatically.
    recover_interrupted(include_queued=False)
    yield


app = FastAPI(title="Vision Lifecycle API", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_methods=["*"], allow_headers=["*"])


def get_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def require_project(session: Session, project_id: str) -> Project:
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    return project


@app.get("/api/v1/health")
def health():
    return {"status": "ok", "mode": "local", "worker_mode": os.environ.get("VISION_LIFECYCLE_EXTERNAL_WORKER", "false").lower() == "true", "version": "0.1.0"}


@app.get("/api/v1/plugins")
def plugins():
    return [
        {"id": "coco", "kind": "dataset", "tasks": ["detection"], "capabilities": ["inspect", "validate", "prediction-evaluation", "onnx-batch-evaluation"]},
        {"id": "yolo-txt", "kind": "dataset", "tasks": ["detection"], "capabilities": ["inspect", "validate"]},
        {"id": "classification", "kind": "dataset", "tasks": ["classification"], "capabilities": ["inspect", "validate", "onnx-batch-evaluation"]},
        {"id": "mmdetection", "kind": "model", "tasks": ["detection"], "capabilities": ["register", "native-inference", "external-result-import"]},
        {"id": "mmdeploy", "kind": "model", "tasks": ["detection"], "capabilities": ["runtime-inference", "target-profile"]},
        {"id": "mock-board", "kind": "runner", "tasks": ["detection", "classification"], "capabilities": ["run", "result-contract"]},
    ]


@app.get("/api/v1/schemas")
def list_contract_schemas():
    """Return the versioned contracts shared by API, CLI and onboarding agents."""
    return {
        "schema_version": "1.0",
        "schemas": {
            "project": contract_schemas.ProjectCreate.model_json_schema(),
            "dataset": contract_schemas.DatasetCreate.model_json_schema(),
            "model": contract_schemas.ModelCreate.model_json_schema(),
            "run": contract_schemas.RunCreate.model_json_schema(),
            "result_manifest": contract_schemas.ResultImportCreate.model_json_schema(),
            "release": contract_schemas.ReleaseCreate.model_json_schema(),
            "recipe": contract_schemas.RECIPE_SCHEMA,
        },
    }


@app.get("/api/v1/recipes")
def list_recipes():
    return [
        {"id": "blank", "name": "내 프로젝트 연결", "task_kind": "unknown", "steps": ["project", "data", "model", "evaluation", "report"]},
        {"id": "mmdetection-onboarding", "name": "MMDetection RTMDet·YOLOX 실습", "task_kind": "detection", "steps": ["project", "data", "rtmdet", "evaluation", "yolox", "comparison"]},
        {"id": "classification-onboarding", "name": "분류 모델 실습", "task_kind": "classification", "steps": ["project", "data", "model", "evaluation", "comparison"]},
    ]


@app.get("/api/v1/recipes/{recipe_id}")
def recipe_definition(recipe_id: str):
    """Expose the reviewable recipe contract without executing any action."""
    allowed = {"blank", "mmdetection-onboarding", "classification-onboarding"}
    if recipe_id not in allowed:
        raise HTTPException(404, "Recipe not found")
    path = Path(__file__).resolve().parents[2] / "recipes" / recipe_id / "recipe.yaml"
    if not path.is_file():
        raise HTTPException(404, "Recipe definition is not installed")
    return {"id": recipe_id, "format": "yaml", "version": "1.0", "definition": path.read_text(encoding="utf-8")}


_RECIPE_STEPS = {
    "blank": ["project", "storage", "data", "contracts", "model", "evaluation", "comparison", "report"],
    "mmdetection-onboarding": ["project", "storage", "data", "contracts", "rtmdet", "evaluation", "yolox", "comparison", "report"],
    "classification-onboarding": ["project", "storage", "data", "contracts", "model", "evaluation", "comparison", "report"],
}


def _step_readiness(session: Session, project_id: str, step_id: str) -> tuple[bool, str]:
    project = session.get(Project, project_id)
    if step_id == "project":
        return bool(project), "project exists"
    if step_id == "storage":
        ready = session.scalar(select(StorageMapping.id).where(StorageMapping.project_id == project_id, StorageMapping.status == "available")) is not None
        return ready, "available storage mapping is required"
    if step_id == "data":
        ready = session.scalar(select(DatasetVersion.id).where(DatasetVersion.project_id == project_id, DatasetVersion.status != "archived")) is not None
        return ready, "a non-archived DatasetVersion is required"
    if step_id == "contracts":
        ready = any(session.scalar(select(entity.id).where(entity.project_id == project_id)) is not None for entity in (LabelSchemaVersion, SplitVersion, EvaluationSetVersion, CalibrationSetVersion))
        return ready, "at least one versioned data contract is required"
    if step_id in {"model", "rtmdet", "yolox"}:
        query = select(ModelVersion).where(ModelVersion.project_id == project_id, ModelVersion.status != "archived")
        models = session.scalars(query).all()
        if step_id == "rtmdet":
            models = [item for item in models if "rtmdet" in item.family.lower()]
        elif step_id == "yolox":
            models = [item for item in models if "yolox" in item.family.lower()]
        return bool(models), f"a matching {step_id} ModelVersion is required"
    if step_id == "evaluation":
        return session.scalar(select(Run.id).where(Run.project_id == project_id, Run.kind == "evaluation", Run.status == "completed")) is not None, "a completed evaluation run is required"
    if step_id == "comparison":
        return all(session.scalar(select(ModelVersion.id).where(ModelVersion.project_id == project_id, ModelVersion.alias == alias)) is not None for alias in ("baseline", "candidate")), "baseline and candidate aliases are required"
    if step_id == "report":
        return session.scalar(select(Release.id).where(Release.project_id == project_id)) is not None, "a Release decision is required"
    return True, "ready"


def _onboarding_step_views(session: Session, project_id: str, onboarding_id: str) -> list[dict]:
    steps = session.scalars(select(StepProgress).where(StepProgress.session_id == onboarding_id).order_by(StepProgress.created_at)).all()
    views = []
    for item in steps:
        ready, reason = _step_readiness(session, project_id, item.step_id)
        view = as_dict(item)
        view["readiness"] = {"ready": ready, "reason": reason}
        views.append(view)
    return views


@app.get("/api/v1/projects/{project_id}/onboarding")
def get_onboarding(project_id: str, session: Session = Depends(get_session)):
    require_project(session, project_id)
    onboarding = session.scalar(select(OnboardingSession).where(OnboardingSession.project_id == project_id))
    if not onboarding:
        return {"session": None, "steps": []}
    return {"session": as_dict(onboarding), "steps": _onboarding_step_views(session, project_id, onboarding.id)}


@app.post("/api/v1/projects/{project_id}/onboarding", status_code=201)
def create_onboarding(project_id: str, payload: OnboardingCreate, session: Session = Depends(get_session)):
    project = require_project(session, project_id)
    existing = session.scalar(select(OnboardingSession).where(OnboardingSession.project_id == project_id))
    if existing:
        return {"session": as_dict(existing), "steps": _onboarding_step_views(session, project_id, existing.id)}
    recipe_id = payload.recipe_id if payload.recipe_id in _RECIPE_STEPS else "blank"
    onboarding = OnboardingSession(project_id=project.id, recipe_id=recipe_id, recipe_version=payload.recipe_version)
    session.add(onboarding); session.flush()
    steps = [StepProgress(session_id=onboarding.id, step_id=step) for step in _RECIPE_STEPS[recipe_id]]
    session.add_all(steps); session.commit(); session.refresh(onboarding)
    return {"session": as_dict(onboarding), "steps": _onboarding_step_views(session, project_id, onboarding.id)}


@app.patch("/api/v1/projects/{project_id}/onboarding/steps/{step_id}")
def update_onboarding_step(project_id: str, step_id: str, payload: StepProgressUpdate, session: Session = Depends(get_session)):
    require_project(session, project_id)
    onboarding = session.scalar(select(OnboardingSession).where(OnboardingSession.project_id == project_id))
    if not onboarding:
        raise HTTPException(404, "Onboarding session not found")
    if payload.status not in {"not_started", "ready", "running", "blocked", "completed", "skipped", "needs_revalidation"}:
        raise HTTPException(422, "Unsupported onboarding step status")
    step = session.scalar(select(StepProgress).where(StepProgress.session_id == onboarding.id, StepProgress.step_id == step_id))
    if not step:
        raise HTTPException(404, "Onboarding step not found")
    if payload.status == "completed":
        ready, reason = _step_readiness(session, project_id, step_id)
        if not ready:
            raise HTTPException(409, f"Step is not ready: {reason}")
    step.status = payload.status; step.evidence = payload.evidence
    steps = session.scalars(select(StepProgress).where(StepProgress.session_id == onboarding.id)).all()
    onboarding.status = "completed" if steps and all(item.status in {"completed", "skipped"} for item in steps) else "active"
    session.commit(); session.refresh(step)
    view = as_dict(step)
    ready, reason = _step_readiness(session, project_id, step.step_id)
    view["readiness"] = {"ready": ready, "reason": reason}
    return view


@app.get("/api/v1/environment")
def environment_status():
    return {"onnxruntime": diagnose_onnx(), "mmdetection": diagnose_mmdetection(), "mmdeploy": diagnose_mmdeploy()}


@app.get("/api/v1/projects")
def list_projects(session: Session = Depends(get_session)):
    return [as_dict(x) for x in session.scalars(select(Project).order_by(Project.created_at.desc())).all()]


@app.post("/api/v1/projects", status_code=201)
def create_project(payload: ProjectCreate, session: Session = Depends(get_session)):
    if session.scalar(select(Project).where(Project.name == payload.name)):
        raise HTTPException(409, "A project with this name already exists")
    project = Project(**payload.model_dump())
    session.add(project); session.flush()
    record_audit(session, project.id, "project", project.id, "created", after={"name": project.name, "task_kind": project.task_kind, "mode": project.mode})
    session.commit(); session.refresh(project)
    return as_dict(project)


@app.patch("/api/v1/projects/{project_id}")
def update_project(project_id: str, payload: ProjectUpdate, session: Session = Depends(get_session)):
    project = require_project(session, project_id)
    if payload.name and payload.name != project.name and session.scalar(select(Project).where(Project.name == payload.name)):
        raise HTTPException(409, "A project with this name already exists")
    values = payload.model_dump(exclude_unset=True)
    before = {key: getattr(project, key) for key in values}
    for key, value in values.items():
        setattr(project, key, value)
    record_audit(session, project_id, "project", project.id, "updated", before=before, after={key: getattr(project, key) for key in values})
    session.commit(); session.refresh(project)
    return as_dict(project)


@app.post("/api/v1/projects/{project_id}/archive")
def archive_project(project_id: str, session: Session = Depends(get_session)):
    project = require_project(session, project_id)
    previous_status = project.status
    project.status = "archived" if project.status != "archived" else "active"
    record_audit(session, project_id, "project", project.id, "archived" if project.status == "archived" else "restored", before={"status": previous_status}, after={"status": project.status})
    session.commit(); session.refresh(project)
    return as_dict(project)


@app.post("/api/v1/projects/demo", status_code=201)
def create_demo(session: Session = Depends(get_session)):
    return as_dict(seed_demo(session))


@app.get("/api/v1/projects/{project_id}/overview")
def project_overview(project_id: str, session: Session = Depends(get_session)):
    try:
        data = overview(session, project_id)
    except LookupError as error:
        raise HTTPException(404, str(error)) from error
    return {key: [as_dict(v) for v in value] if isinstance(value, list) and value and hasattr(value[0], "__table__") else as_dict(value) if hasattr(value, "__table__") else value for key, value in data.items()}


@app.get("/api/v1/projects/{project_id}/agent-request")
def get_agent_request(project_id: str, session: Session = Depends(get_session)):
    try:
        return agent_request(session, project_id)
    except LookupError as error:
        raise HTTPException(404, str(error)) from error


@app.get("/api/v1/projects/{project_id}/lineage")
def project_lineage(project_id: str, session: Session = Depends(get_session)):
    try:
        return lineage(session, project_id)
    except LookupError as error:
        raise HTTPException(404, str(error)) from error


@app.get("/api/v1/projects/{project_id}/datasets")
def list_datasets(project_id: str, session: Session = Depends(get_session)):
    require_project(session, project_id)
    return [as_dict(x) for x in session.scalars(select(DatasetVersion).where(DatasetVersion.project_id == project_id)).all()]


@app.post("/api/v1/projects/{project_id}/datasets", status_code=201)
def create_dataset(project_id: str, payload: DatasetCreate, session: Session = Depends(get_session)):
    require_project(session, project_id)
    duplicate = session.scalar(select(DatasetVersion).where(DatasetVersion.project_id == project_id, DatasetVersion.name == payload.name, DatasetVersion.version == payload.version))
    if duplicate:
        raise HTTPException(409, "This DatasetVersion already exists in the project")
    if payload.parent_dataset_id:
        parent = session.get(DatasetVersion, payload.parent_dataset_id)
        if not parent or parent.project_id != project_id:
            raise HTTPException(422, "Parent DatasetVersion must belong to this project")
    content_hash, fingerprints = dataset_fingerprint(payload.manifest_path, payload.annotation_path)
    validation = {**payload.validation, "source_fingerprints": fingerprints} if fingerprints else payload.validation
    snapshot = build_dataset_snapshot(task_kind=payload.task_kind, format=payload.format, manifest_path=payload.manifest_path, annotation_path=payload.annotation_path)
    values = payload.model_dump(exclude={"validation"})
    if not values.get("class_names") and snapshot.get("class_names"):
        values["class_names"] = snapshot["class_names"]
    if not values.get("sample_count") and snapshot.get("counts", {}).get("images"):
        values["sample_count"] = snapshot["counts"]["images"]
    dataset = DatasetVersion(project_id=project_id, **values, content_hash=content_hash, snapshot=snapshot, validation=validation)
    session.add(dataset); session.flush()
    if payload.annotation_path:
        register_artifact(session, project_id, kind="dataset-annotation", logical_name=f"{payload.name}/{payload.version}/annotation", owner_type="dataset", owner_id=dataset.id, source_path=payload.annotation_path)
    if payload.manifest_path:
        register_artifact(session, project_id, kind="dataset-manifest", logical_name=f"{payload.name}/{payload.version}/manifest", owner_type="dataset", owner_id=dataset.id, source_path=payload.manifest_path)
    record_audit(session, project_id, "dataset", dataset.id, "created", after={"name": dataset.name, "version": dataset.version, "status": dataset.status, "format": dataset.format})
    session.commit(); session.refresh(dataset)
    return as_dict(dataset)


@app.patch("/api/v1/projects/{project_id}/datasets/{dataset_id}")
def update_dataset(project_id: str, dataset_id: str, payload: DatasetUpdate, session: Session = Depends(get_session)):
    require_project(session, project_id)
    dataset = session.get(DatasetVersion, dataset_id)
    if not dataset or dataset.project_id != project_id:
        raise HTTPException(404, "Dataset not found")
    if dataset.status == "finalized":
        raise HTTPException(409, "Finalized DatasetVersion is immutable; create a new version")
    values = payload.model_dump(exclude_unset=True)
    before = {key: getattr(dataset, key) for key in values}
    for key, value in values.items():
        setattr(dataset, key, value)
    if "annotation_path" in values or "manifest_path" in values:
        dataset.content_hash, fingerprints = dataset_fingerprint(dataset.manifest_path, dataset.annotation_path)
        dataset.snapshot = build_dataset_snapshot(task_kind=dataset.task_kind, format=dataset.format, manifest_path=dataset.manifest_path, annotation_path=dataset.annotation_path)
        if not dataset.class_names and dataset.snapshot.get("class_names"):
            dataset.class_names = dataset.snapshot["class_names"]
        if not dataset.sample_count and dataset.snapshot.get("counts", {}).get("images"):
            dataset.sample_count = dataset.snapshot["counts"]["images"]
        dataset.validation = {**dataset.validation, "source_fingerprints": fingerprints}
        if "annotation_path" in values and dataset.annotation_path:
            register_artifact(session, project_id, kind="dataset-annotation", logical_name=f"{dataset.name}/{dataset.version}/annotation", owner_type="dataset", owner_id=dataset.id, source_path=dataset.annotation_path)
        if "manifest_path" in values and dataset.manifest_path:
            register_artifact(session, project_id, kind="dataset-manifest", logical_name=f"{dataset.name}/{dataset.version}/manifest", owner_type="dataset", owner_id=dataset.id, source_path=dataset.manifest_path)
    record_audit(session, project_id, "dataset", dataset.id, "updated", before=before, after={key: getattr(dataset, key) for key in values})
    session.commit(); session.refresh(dataset)
    return as_dict(dataset)


@app.post("/api/v1/projects/{project_id}/datasets/{dataset_id}/archive")
def archive_dataset(project_id: str, dataset_id: str, session: Session = Depends(get_session)):
    require_project(session, project_id)
    dataset = session.get(DatasetVersion, dataset_id)
    if not dataset or dataset.project_id != project_id:
        raise HTTPException(404, "Dataset not found")
    previous_status = dataset.status
    dataset.status = "draft" if dataset.status == "archived" else "archived"
    record_audit(session, project_id, "dataset", dataset.id, "archived" if dataset.status == "archived" else "restored", before={"status": previous_status}, after={"status": dataset.status})
    session.commit(); session.refresh(dataset)
    return as_dict(dataset)


@app.get("/api/v1/projects/{project_id}/datasets/{dataset_id}/impact")
def dataset_impact(project_id: str, dataset_id: str, session: Session = Depends(get_session)):
    require_project(session, project_id)
    dataset = session.get(DatasetVersion, dataset_id)
    if not dataset or dataset.project_id != project_id:
        raise HTTPException(404, "Dataset not found")
    models = session.scalars(select(ModelVersion).where(ModelVersion.project_id == project_id, ModelVersion.source_dataset_id == dataset_id)).all()
    runs = session.scalars(select(Run).where(Run.project_id == project_id, Run.dataset_id == dataset_id)).all()
    versions = []
    for entity_type, entity in (("label_schema", LabelSchemaVersion), ("split", SplitVersion), ("evaluation_set", EvaluationSetVersion), ("calibration_set", CalibrationSetVersion)):
        versions.extend({"kind": entity_type, "id": item.id, "name": item.name, "version": item.version, "status": item.status} for item in session.scalars(select(entity).where(entity.project_id == project_id, entity.dataset_id == dataset_id)).all())
    dependencies = ([{"kind": "model", "id": item.id, "name": f"{item.name} {item.version}", "status": item.status} for item in models]
        + [{"kind": "run", "id": item.id, "name": item.name, "status": item.status} for item in runs]
        + versions)
    return {"entity": {"kind": "dataset", "id": dataset.id, "name": f"{dataset.name} {dataset.version}", "status": dataset.status}, "dependencies": dependencies, "blocking": [item for item in dependencies if item["status"] not in {"archived", "failed", "cancelled"}]}


@app.post("/api/v1/projects/{project_id}/datasets/{dataset_id}/finalize")
def finalize_dataset(project_id: str, dataset_id: str, session: Session = Depends(get_session)):
    require_project(session, project_id)
    dataset = session.get(DatasetVersion, dataset_id)
    if not dataset or dataset.project_id != project_id:
        raise HTTPException(404, "Dataset not found")
    if dataset.status == "archived":
        raise HTTPException(409, "Restore the DatasetVersion before finalizing")
    if dataset.status == "finalized":
        raise HTTPException(409, "DatasetVersion is already finalized; create a new version for changes")
    content_hash, fingerprints = dataset_fingerprint(dataset.manifest_path, dataset.annotation_path)
    if not content_hash:
        raise HTTPException(422, "Finalize requires at least one accessible manifest or annotation file")
    dataset.content_hash = content_hash
    dataset.snapshot = build_dataset_snapshot(task_kind=dataset.task_kind, format=dataset.format, manifest_path=dataset.manifest_path, annotation_path=dataset.annotation_path)
    dataset.validation = {**dataset.validation, "source_fingerprints": fingerprints, "finalized_at": "local"}
    dataset.status = "finalized"
    record_audit(session, project_id, "dataset", dataset.id, "finalized", before={"status": "draft"}, after={"status": dataset.status, "content_hash": dataset.content_hash})
    session.commit(); session.refresh(dataset)
    return as_dict(dataset)


@app.get("/api/v1/projects/{project_id}/datasets/{dataset_id}/preview")
def preview_dataset(project_id: str, dataset_id: str, limit: int = 12, session: Session = Depends(get_session)):
    require_project(session, project_id)
    dataset = session.get(DatasetVersion, dataset_id)
    if not dataset or dataset.project_id != project_id:
        raise HTTPException(404, "Dataset not found")
    if dataset.format != "coco" or not dataset.annotation_path:
        raise HTTPException(422, "Preview currently requires a COCO annotation path")
    if limit < 1 or limit > 100:
        raise HTTPException(422, "limit must be between 1 and 100")
    try:
        source = load_coco(dataset.annotation_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise HTTPException(422, str(error)) from error
    categories = {item["id"]: item.get("name", str(item["id"])) for item in source["categories"]}
    by_image: dict[int, list[dict]] = {}
    for annotation in source["annotations"]:
        by_image.setdefault(annotation["image_id"], []).append({"id": annotation.get("id"), "category_id": annotation.get("category_id"), "category_name": categories.get(annotation.get("category_id"), "unknown"), "bbox": annotation.get("bbox"), "iscrowd": annotation.get("iscrowd", 0)})
    images = [{"id": image.get("id"), "file_name": image.get("file_name"), "width": image.get("width"), "height": image.get("height"), "annotations": by_image.get(image.get("id"), [])} for image in source["images"][:limit]]
    return {"dataset_id": dataset.id, "categories": categories, "images": images, "truncated": len(source["images"]) > limit}


@app.get("/api/v1/projects/{project_id}/datasets/{dataset_id}/diff")
def diff_dataset(project_id: str, dataset_id: str, against_id: str, session: Session = Depends(get_session)):
    require_project(session, project_id)
    left = session.get(DatasetVersion, dataset_id)
    right = session.get(DatasetVersion, against_id)
    if not left or left.project_id != project_id or not right or right.project_id != project_id:
        raise HTTPException(404, "DatasetVersion not found")
    return {"left": as_dict(left), "right": as_dict(right), "diff": diff_dataset_snapshots(left.snapshot or {}, right.snapshot or {})}


@app.post("/api/v1/projects/{project_id}/datasets/validate-coco")
def validate_dataset_coco(project_id: str, annotation_path: str, session: Session = Depends(get_session)):
    require_project(session, project_id)
    try:
        return validate_coco(annotation_path)
    except (OSError, ValueError) as error:
        raise HTTPException(422, str(error)) from error


@app.post("/api/v1/projects/{project_id}/inspect-path")
def inspect_dataset_path(project_id: str, payload: PathInspectRequest, session: Session = Depends(get_session)):
    require_project(session, project_id)
    try:
        return inspect_path(payload.path)
    except (OSError, ValueError) as error:
        raise HTTPException(422, str(error)) from error


@app.get("/api/v1/projects/{project_id}/storages")
def list_storages(project_id: str, session: Session = Depends(get_session)):
    require_project(session, project_id)
    return [as_dict(item) for item in session.scalars(select(StorageMapping).where(StorageMapping.project_id == project_id).order_by(StorageMapping.name)).all()]


@app.post("/api/v1/projects/{project_id}/storages", status_code=201)
def create_storage(project_id: str, payload: StorageMappingCreate, session: Session = Depends(get_session)):
    require_project(session, project_id)
    if session.scalar(select(StorageMapping).where(StorageMapping.project_id == project_id, StorageMapping.name == payload.name)):
        raise HTTPException(409, "A storage mapping with this name already exists")
    validation = storage_status(payload.root_path)
    mapping = StorageMapping(project_id=project_id, **payload.model_dump(), status=validation["status"], last_validation=validation)
    session.add(mapping); session.flush()
    record_audit(session, project_id, "storage", mapping.id, "created", after={"name": mapping.name, "status": mapping.status, "read_only": mapping.read_only, "root_path": mapping.root_path})
    session.commit(); session.refresh(mapping)
    return as_dict(mapping)


@app.patch("/api/v1/projects/{project_id}/storages/{storage_id}")
def update_storage(project_id: str, storage_id: str, payload: StorageMappingUpdate, session: Session = Depends(get_session)):
    require_project(session, project_id)
    mapping = session.get(StorageMapping, storage_id)
    if not mapping or mapping.project_id != project_id:
        raise HTTPException(404, "Storage mapping not found")
    values = payload.model_dump(exclude_unset=True)
    before = {key: getattr(mapping, key) for key in values}
    for key, value in values.items():
        setattr(mapping, key, value)
    if "root_path" in values:
        validation = storage_status(mapping.root_path)
        mapping.status = validation["status"]
        mapping.last_validation = validation
    record_audit(session, project_id, "storage", mapping.id, "updated", before=before, after={key: getattr(mapping, key) for key in values})
    session.commit(); session.refresh(mapping)
    return as_dict(mapping)


@app.post("/api/v1/projects/{project_id}/storages/{storage_id}/validate")
def validate_storage(project_id: str, storage_id: str, session: Session = Depends(get_session)):
    require_project(session, project_id)
    mapping = session.get(StorageMapping, storage_id)
    if not mapping or mapping.project_id != project_id:
        raise HTTPException(404, "Storage mapping not found")
    validation = storage_status(mapping.root_path)
    mapping.status = validation["status"]
    mapping.last_validation = validation
    session.commit(); session.refresh(mapping)
    return as_dict(mapping)


@app.post("/api/v1/projects/{project_id}/storages/{storage_id}/archive")
def archive_storage(project_id: str, storage_id: str, session: Session = Depends(get_session)):
    require_project(session, project_id)
    mapping = session.get(StorageMapping, storage_id)
    if not mapping or mapping.project_id != project_id:
        raise HTTPException(404, "Storage mapping not found")
    previous_status = mapping.status
    if mapping.status == "archived":
        validation = storage_status(mapping.root_path)
        mapping.status = validation["status"]
        mapping.last_validation = validation
    else:
        mapping.status = "archived"
    record_audit(session, project_id, "storage", mapping.id, "archived" if mapping.status == "archived" else "restored", before={"status": previous_status}, after={"status": mapping.status, "root_path": mapping.root_path})
    session.commit(); session.refresh(mapping)
    return as_dict(mapping)


@app.get("/api/v1/projects/{project_id}/storages/{storage_id}/impact")
def storage_impact(project_id: str, storage_id: str, session: Session = Depends(get_session)):
    require_project(session, project_id)
    mapping = session.get(StorageMapping, storage_id)
    if not mapping or mapping.project_id != project_id:
        raise HTTPException(404, "Storage mapping not found")
    assets = session.scalars(select(DataAsset).where(DataAsset.project_id == project_id, DataAsset.storage_id == storage_id)).all()
    jobs = session.scalars(select(Job).where(Job.project_id == project_id, Job.runner_id == "builtin:storage-inventory")).all()
    jobs = [job for job in jobs if isinstance(job.input_json, dict) and job.input_json.get("storage_id") == storage_id]
    dependencies = ([{"kind": "asset", "id": item.id, "name": item.relative_path, "status": item.status} for item in assets]
        + [{"kind": "inventory-job", "id": item.id, "name": item.runner_id, "status": item.status} for item in jobs])
    return {"entity": {"kind": "storage", "id": mapping.id, "name": mapping.name, "status": mapping.status}, "dependencies": dependencies, "blocking": [item for item in dependencies if item["status"] not in {"archived", "failed", "cancelled", "completed"}]}


@app.post("/api/v1/projects/{project_id}/storages/{storage_id}/browse")
def browse_storage_mapping(project_id: str, storage_id: str, payload: StorageBrowseRequest, session: Session = Depends(get_session)):
    require_project(session, project_id)
    mapping = session.get(StorageMapping, storage_id)
    if not mapping or mapping.project_id != project_id:
        raise HTTPException(404, "Storage mapping not found")
    if mapping.status == "archived":
        raise HTTPException(409, "Storage mapping is archived; restore it before browsing")
    try:
        return browse_storage(mapping.root_path, payload.relative_path, payload.limit)
    except (OSError, ValueError) as error:
        raise HTTPException(422, str(error)) from error


@app.get("/api/v1/projects/{project_id}/assets")
def list_assets(project_id: str, session: Session = Depends(get_session)):
    require_project(session, project_id)
    return [as_dict(item) for item in session.scalars(select(DataAsset).where(DataAsset.project_id == project_id).order_by(DataAsset.relative_path)).all()]


@app.get("/api/v1/projects/{project_id}/field-batches")
def list_field_batches(project_id: str, session: Session = Depends(get_session)):
    require_project(session, project_id)
    return [as_dict(item) for item in session.scalars(select(FieldDataBatch).where(FieldDataBatch.project_id == project_id).order_by(FieldDataBatch.created_at.desc())).all()]


@app.post("/api/v1/projects/{project_id}/field-batches", status_code=201)
def create_field_batch(project_id: str, payload: FieldDataBatchCreate, session: Session = Depends(get_session)):
    require_project(session, project_id)
    for entity, label in ((ModelVersion, "Source model"), (DatasetVersion, "Source dataset"), (DatasetVersion, "Candidate dataset")):
        identifier = {"Source model": payload.source_model_id, "Source dataset": payload.source_dataset_id, "Candidate dataset": payload.candidate_dataset_id}[label]
        _project_entity(session, entity, identifier, project_id, label)
    metadata = dict(payload.metadata_json)
    item = FieldDataBatch(project_id=project_id, **payload.model_dump(exclude={"metadata_json"}), metadata_json=metadata)
    session.add(item); session.flush()
    artifact_ids: dict[str, str] = {}
    for key, kind in (("source_path", "field-source"), ("prediction_path", "field-prediction"), ("label_path", "field-label")):
        path = getattr(item, key)
        if path:
            artifact = register_artifact(session, project_id, kind=kind, logical_name=f"field/{item.name}/{key}", owner_type="field-batch", owner_id=item.id, source_path=path)
            session.flush()
            artifact_ids[key] = artifact.id
    if artifact_ids:
        item.metadata_json = {**metadata, "artifact_ids": artifact_ids}
    record_audit(session, project_id, "field-batch", item.id, "registered", after={"name": item.name, "source_model_id": item.source_model_id, "source_dataset_id": item.source_dataset_id, "candidate_dataset_id": item.candidate_dataset_id, "sample_count": item.sample_count, "failure_count": item.failure_count})
    session.commit(); session.refresh(item)
    return as_dict(item)


@app.post("/api/v1/projects/{project_id}/storages/{storage_id}/inventory")
def inventory_storage_mapping(project_id: str, storage_id: str, payload: StorageInventoryRequest, session: Session = Depends(get_session)):
    require_project(session, project_id)
    mapping = session.get(StorageMapping, storage_id)
    if not mapping or mapping.project_id != project_id:
        raise HTTPException(404, "Storage mapping not found")
    if mapping.status == "archived":
        raise HTTPException(409, "Storage mapping is archived; restore it before inventory")
    try:
        entries = inventory_storage(mapping.root_path, payload.relative_path, payload.recursive, payload.limit)
    except (OSError, ValueError) as error:
        raise HTTPException(422, str(error)) from error
    assets = []
    for entry in entries:
        existing = session.scalar(select(DataAsset).where(DataAsset.project_id == project_id, DataAsset.storage_id == storage_id, DataAsset.relative_path == entry["relative_path"]))
        if existing:
            existing.size_bytes = entry["size_bytes"]
            existing.sha256 = entry["sha256"]
            existing.metadata_json = {"suffix": entry["suffix"]}
            existing.status = "discovered"
            assets.append(existing)
        else:
            asset = DataAsset(project_id=project_id, storage_id=storage_id, relative_path=entry["relative_path"], size_bytes=entry["size_bytes"], sha256=entry["sha256"], metadata_json={"suffix": entry["suffix"]})
            session.add(asset)
            assets.append(asset)
    session.commit()
    return {"count": len(assets), "truncated": len(assets) >= payload.limit, "assets": [as_dict(item) for item in assets]}


@app.post("/api/v1/projects/{project_id}/storages/{storage_id}/inventory-job", status_code=201)
def create_inventory_job(project_id: str, storage_id: str, payload: StorageInventoryRequest, session: Session = Depends(get_session)):
    require_project(session, project_id)
    mapping = session.get(StorageMapping, storage_id)
    if not mapping or mapping.project_id != project_id:
        raise HTTPException(404, "Storage mapping not found")
    if mapping.status == "archived":
        raise HTTPException(409, "Storage mapping is archived; restore it before starting inventory")
    job = Job(project_id=project_id, runner_id="builtin:storage-inventory", command=["builtin:storage-inventory"], input_json={"storage_id": storage_id, "relative_path": payload.relative_path, "recursive": payload.recursive, "limit": payload.limit}, status="queued")
    session.add(job); session.commit(); session.refresh(job)
    if os.environ.get("VISION_LIFECYCLE_EXTERNAL_WORKER", "false").lower() != "true":
        launch(job, [])
    return as_dict(job)


def _version_hash(value: dict) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _evaluation_item_keys(evaluation_set: EvaluationSetVersion | None) -> set[str] | None:
    """Return explicit item identifiers from an evaluation-set definition.

    An evaluation set without an item list intentionally means the whole linked
    dataset.  Once ``items`` or ``image_ids`` is present, even an empty list is
    meaningful and selects no records.  Item objects may carry ``image_id``,
    ``item_id`` or ``id`` so the same contract can be used by image manifests
    and classification record files.
    """
    if evaluation_set is None:
        return None
    definition = evaluation_set.definition if isinstance(evaluation_set.definition, dict) else {}
    raw = next((definition[key] for key in ("items", "image_ids") if key in definition), None)
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise HTTPException(422, "Evaluation set definition items/image_ids must be a list")
    keys: set[str] = set()
    for item in raw:
        value = item
        if isinstance(item, dict):
            value = item.get("image_id", item.get("item_id", item.get("id")))
        if value is None or isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise HTTPException(422, "Evaluation set items must contain scalar image/item identifiers")
        keys.add(str(value))
    if not keys:
        raise HTTPException(422, "Evaluation set must contain at least one item")
    return keys


def _evaluation_image_ids(evaluation_set: EvaluationSetVersion | None) -> set[int] | None:
    keys = _evaluation_item_keys(evaluation_set)
    if keys is None:
        return None
    values: set[int] = set()
    for key in keys:
        try:
            value = int(key)
        except ValueError as error:
            raise HTTPException(422, "COCO evaluation set items must use integer image IDs") from error
        if str(value) != key:
            raise HTTPException(422, "COCO evaluation set items must use canonical integer image IDs")
        values.add(value)
    return values


def _evidence_safe(value):
    if isinstance(value, dict):
        result = {}
        for key, nested in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in ("path", "command", "secret", "token", "password", "credential")):
                continue
            result[key] = _evidence_safe(nested)
        return result
    if isinstance(value, list):
        return [_evidence_safe(item) for item in value]
    if isinstance(value, str) and value.startswith("/"):
        return "[redacted-absolute-path]"
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _capture_release_evidence(session: Session, project_id: str, release_id: str, item: dict) -> ReleaseEvidence:
    evidence_type = str(item.get("type", item.get("evidence_type", ""))).lower()
    source_id = str(item.get("id", item.get("source_id", "")))
    required = bool(item.get("required", False))
    entities = {"evaluation": Run, "run": Run, "board": BoardBenchmark, "benchmark": BoardBenchmark, "quantization": QuantizationRun, "artifact": Artifact}
    entity_type = entities.get(evidence_type)
    if not entity_type or not source_id:
        raise HTTPException(422, "Release evidence needs type (evaluation, board, quantization, artifact) and source id")
    entity = session.get(entity_type, source_id)
    if not entity or entity.project_id != project_id:
        raise HTTPException(422, f"Release evidence {evidence_type}:{source_id} must belong to this project")
    if evidence_type in {"evaluation", "run"} and entity.kind != "evaluation":
        raise HTTPException(422, "Release evaluation evidence must reference an evaluation Run")
    if evidence_type in {"board", "benchmark"} and not isinstance(entity, BoardBenchmark):
        raise HTTPException(422, "Release board evidence must reference a BoardBenchmark")
    normalized_type = {"run": "evaluation", "benchmark": "board"}.get(evidence_type, evidence_type)
    snapshot = _evidence_safe({"type": normalized_type, "source_id": source_id, "entity": as_dict(entity)})
    return ReleaseEvidence(project_id=project_id, release_id=release_id, evidence_type=normalized_type, source_id=source_id, required=required, status="captured", content_hash=_version_hash(snapshot), snapshot=snapshot)


@app.get("/api/v1/projects/{project_id}/artifacts")
def list_artifacts(project_id: str, session: Session = Depends(get_session)):
    require_project(session, project_id)
    return [as_dict(item) for item in session.scalars(select(Artifact).where(Artifact.project_id == project_id).order_by(Artifact.created_at.desc())).all()]


@app.post("/api/v1/projects/{project_id}/artifacts", status_code=201)
def create_artifact(project_id: str, payload: ArtifactCreate, session: Session = Depends(get_session)):
    require_project(session, project_id)
    artifact = register_artifact(session, project_id, **payload.model_dump())
    session.commit(); session.refresh(artifact)
    return as_dict(artifact)


def _version_dataset(session: Session, project_id: str, dataset_id: str | None):
    if not dataset_id:
        return None
    dataset = session.get(DatasetVersion, dataset_id)
    if not dataset or dataset.project_id != project_id:
        raise HTTPException(422, "Dataset reference must belong to this project")
    return dataset


@app.get("/api/v1/projects/{project_id}/label-schemas")
def list_label_schemas(project_id: str, session: Session = Depends(get_session)):
    require_project(session, project_id)
    return [as_dict(item) for item in session.scalars(select(LabelSchemaVersion).where(LabelSchemaVersion.project_id == project_id).order_by(LabelSchemaVersion.created_at.desc())).all()]


@app.post("/api/v1/projects/{project_id}/label-schemas", status_code=201)
def create_label_schema(project_id: str, payload: VersionDefinitionCreate, session: Session = Depends(get_session)):
    require_project(session, project_id)
    _version_dataset(session, project_id, payload.dataset_id)
    parent = None
    if payload.parent_label_schema_id:
        parent = session.get(LabelSchemaVersion, payload.parent_label_schema_id)
        if not parent or parent.project_id != project_id:
            raise HTTPException(422, "Parent label schema must belong to this project")
        if parent.dataset_id and payload.dataset_id and parent.dataset_id != payload.dataset_id:
            raise HTTPException(422, "Parent label schema must use the selected DatasetVersion")
    value = {"name": payload.name, "version": payload.version, "classes": payload.classes, "mapping": payload.mapping, "dataset_id": payload.dataset_id, "parent_label_schema_id": payload.parent_label_schema_id}
    item = LabelSchemaVersion(project_id=project_id, parent_label_schema_id=payload.parent_label_schema_id, dataset_id=payload.dataset_id, name=payload.name, version=payload.version, classes=payload.classes, mapping=payload.mapping, status=payload.status, content_hash=_version_hash(value))
    session.add(item); session.commit(); session.refresh(item); return as_dict(item)


@app.get("/api/v1/projects/{project_id}/splits")
def list_splits(project_id: str, session: Session = Depends(get_session)):
    require_project(session, project_id)
    return [as_dict(item) for item in session.scalars(select(SplitVersion).where(SplitVersion.project_id == project_id).order_by(SplitVersion.created_at.desc())).all()]


@app.post("/api/v1/projects/{project_id}/splits", status_code=201)
def create_split(project_id: str, payload: VersionDefinitionCreate, session: Session = Depends(get_session)):
    require_project(session, project_id)
    dataset = _version_dataset(session, project_id, payload.dataset_id)
    value = {"name": payload.name, "version": payload.version, "definition": payload.definition, "dataset_id": payload.dataset_id}
    validation = validate_split_definition(payload.definition, dataset.snapshot if dataset else None)
    item = SplitVersion(project_id=project_id, dataset_id=payload.dataset_id, name=payload.name, version=payload.version, definition=payload.definition, validation=validation, status=payload.status, content_hash=_version_hash(value))
    session.add(item); session.commit(); session.refresh(item); return as_dict(item)


@app.post("/api/v1/projects/{project_id}/splits/{split_id}/validate")
def validate_split(project_id: str, split_id: str, session: Session = Depends(get_session)):
    require_project(session, project_id)
    item = session.get(SplitVersion, split_id)
    if not item or item.project_id != project_id:
        raise HTTPException(404, "Split version not found")
    dataset = _version_dataset(session, project_id, item.dataset_id)
    item.validation = validate_split_definition(item.definition, dataset.snapshot if dataset else None)
    if item.validation["status"] == "passed" and item.status == "draft":
        item.status = "validated"
    session.commit(); session.refresh(item)
    return as_dict(item)


@app.get("/api/v1/projects/{project_id}/evaluation-sets")
def list_evaluation_sets(project_id: str, session: Session = Depends(get_session)):
    require_project(session, project_id)
    return [as_dict(item) for item in session.scalars(select(EvaluationSetVersion).where(EvaluationSetVersion.project_id == project_id).order_by(EvaluationSetVersion.created_at.desc())).all()]


@app.post("/api/v1/projects/{project_id}/evaluation-sets", status_code=201)
def create_evaluation_set(project_id: str, payload: VersionDefinitionCreate, session: Session = Depends(get_session)):
    require_project(session, project_id)
    dataset = _version_dataset(session, project_id, payload.dataset_id)
    value = {"name": payload.name, "version": payload.version, "purpose": payload.purpose, "definition": payload.definition, "dataset_id": payload.dataset_id}
    validation = validate_evaluation_definition(payload.definition, dataset.snapshot if dataset else None)
    item = EvaluationSetVersion(project_id=project_id, dataset_id=payload.dataset_id, name=payload.name, version=payload.version, purpose=payload.purpose, definition=payload.definition, validation=validation, status=payload.status, content_hash=_version_hash(value))
    session.add(item); session.commit(); session.refresh(item); return as_dict(item)


@app.post("/api/v1/projects/{project_id}/evaluation-sets/{evaluation_set_id}/validate")
def validate_evaluation_set(project_id: str, evaluation_set_id: str, session: Session = Depends(get_session)):
    require_project(session, project_id)
    item = session.get(EvaluationSetVersion, evaluation_set_id)
    if not item or item.project_id != project_id:
        raise HTTPException(404, "Evaluation set not found")
    dataset = _version_dataset(session, project_id, item.dataset_id)
    item.validation = validate_evaluation_definition(item.definition, dataset.snapshot if dataset else None)
    if item.validation["status"] == "passed" and item.status == "draft":
        item.status = "validated"
    session.commit(); session.refresh(item)
    return as_dict(item)


@app.get("/api/v1/projects/{project_id}/calibration-sets")
def list_calibration_sets(project_id: str, session: Session = Depends(get_session)):
    require_project(session, project_id)
    return [as_dict(item) for item in session.scalars(select(CalibrationSetVersion).where(CalibrationSetVersion.project_id == project_id).order_by(CalibrationSetVersion.created_at.desc())).all()]


@app.post("/api/v1/projects/{project_id}/calibration-sets", status_code=201)
def create_calibration_set(project_id: str, payload: VersionDefinitionCreate, session: Session = Depends(get_session)):
    require_project(session, project_id); _version_dataset(session, project_id, payload.dataset_id)
    dataset = _version_dataset(session, project_id, payload.dataset_id)
    value = {"name": payload.name, "version": payload.version, "sampling": payload.sampling, "preprocessing": payload.preprocessing, "dataset_id": payload.dataset_id}
    validation = validate_calibration_definition(payload.sampling, payload.preprocessing, dataset.snapshot if dataset else None)
    item = CalibrationSetVersion(project_id=project_id, dataset_id=payload.dataset_id, name=payload.name, version=payload.version, sampling=payload.sampling, preprocessing=payload.preprocessing, validation=validation, status=payload.status, content_hash=_version_hash(value))
    session.add(item); session.commit(); session.refresh(item); return as_dict(item)


@app.post("/api/v1/projects/{project_id}/calibration-sets/{calibration_id}/validate")
def validate_calibration_set(project_id: str, calibration_id: str, session: Session = Depends(get_session)):
    require_project(session, project_id)
    item = session.get(CalibrationSetVersion, calibration_id)
    if not item or item.project_id != project_id:
        raise HTTPException(404, "Calibration set not found")
    dataset = _version_dataset(session, project_id, item.dataset_id)
    item.validation = validate_calibration_definition(item.sampling, item.preprocessing, dataset.snapshot if dataset else None)
    if item.validation["status"] == "passed" and item.status == "draft":
        item.status = "validated"
    session.commit(); session.refresh(item)
    return as_dict(item)


@app.post("/api/v1/projects/{project_id}/calibration-sets/{calibration_id}/inspect")
def inspect_calibration_set(project_id: str, calibration_id: str, session: Session = Depends(get_session)):
    """Collect bounded decoded-image evidence for a calibration contract."""
    require_project(session, project_id)
    item = session.get(CalibrationSetVersion, calibration_id)
    if not item or item.project_id != project_id:
        raise HTTPException(404, "Calibration set not found")
    dataset = _version_dataset(session, project_id, item.dataset_id)
    if not dataset:
        raise HTTPException(422, "Calibration statistics require a linked DatasetVersion")
    statistics = inspect_calibration_statistics(
        sampling=item.sampling,
        preprocessing=item.preprocessing,
        dataset_annotation_path=dataset.annotation_path,
        dataset_manifest_path=dataset.manifest_path,
        snapshot=dataset.snapshot,
    )
    item.validation = {**(item.validation or {}), "statistics": statistics}
    session.commit(); session.refresh(item)
    return {"calibration_set": as_dict(item), "statistics": statistics}


def _project_entity(session: Session, entity, identifier: str | None, project_id: str, label: str):
    if not identifier:
        return None
    value = session.get(entity, identifier)
    if not value or value.project_id != project_id:
        raise HTTPException(422, f"{label} reference must belong to this project")
    return value


def _ensure_dataset_source_current(dataset: DatasetVersion) -> None:
    """Reject official evaluation when a finalized source changed on disk."""
    if not dataset.content_hash or not dataset.content_hash.startswith("sha256:"):
        return
    current_hash, _ = dataset_fingerprint(dataset.manifest_path, dataset.annotation_path)
    if current_hash != dataset.content_hash:
        raise HTTPException(409, "Dataset source files changed since this version was registered; create a new DatasetVersion")


@app.get("/api/v1/projects/{project_id}/quantization-runs")
def list_quantization_runs(project_id: str, session: Session = Depends(get_session)):
    require_project(session, project_id)
    return [as_dict(item) for item in session.scalars(select(QuantizationRun).where(QuantizationRun.project_id == project_id).order_by(QuantizationRun.created_at.desc())).all()]


@app.post("/api/v1/projects/{project_id}/quantization-runs", status_code=201)
def create_quantization_run(project_id: str, payload: QuantizationRunCreate, session: Session = Depends(get_session)):
    require_project(session, project_id)
    _project_entity(session, ModelVersion, payload.source_model_id, project_id, "Source model")
    _project_entity(session, ModelVersion, payload.output_model_id, project_id, "Output model")
    _project_entity(session, CalibrationSetVersion, payload.calibration_set_id, project_id, "Calibration set")
    values = payload.model_dump()
    item = QuantizationRun(project_id=project_id, **values)
    session.add(item); session.flush()
    if payload.encoding_path:
        encoding_artifact = register_artifact(
            session,
            project_id,
            kind="quantization-encoding",
            logical_name=f"{item.name}/{item.id}/encoding",
            owner_type="quantization",
            owner_id=item.id,
            source_path=payload.encoding_path,
            notes="Quantization encoding/scale artifact; verify before comparison.",
        )
        session.flush()
        item.metadata_json = {**(item.metadata_json or {}), "encoding_artifact_id": encoding_artifact.id}
    session.commit(); session.refresh(item); return as_dict(item)


@app.post("/api/v1/projects/{project_id}/quantization-runs/{quantization_id}/verify-encoding")
def verify_quantization_encoding(project_id: str, quantization_id: str, session: Session = Depends(get_session)):
    require_project(session, project_id)
    item = session.get(QuantizationRun, quantization_id)
    if not item or item.project_id != project_id:
        raise HTTPException(404, "Quantization run not found")
    artifact_id = (item.metadata_json or {}).get("encoding_artifact_id")
    artifact = session.get(Artifact, artifact_id) if artifact_id else None
    if not artifact or artifact.project_id != project_id:
        raise HTTPException(422, "Quantization run has no registered encoding artifact")
    result = verify_artifact(artifact)
    session.commit()
    return {"quantization_id": item.id, "ok": result["status"] in {"verified", "managed"}, "artifact": result}


@app.post("/api/v1/projects/{project_id}/quantization-runs/{quantization_id}/compare")
def compare_quantization_run(project_id: str, quantization_id: str, payload: QuantizationComparisonRequest, session: Session = Depends(get_session)):
    require_project(session, project_id)
    quantization = session.get(QuantizationRun, quantization_id)
    baseline = session.get(Run, payload.baseline_run_id)
    quantsim = session.get(Run, payload.quantsim_run_id)
    benchmark = session.get(BoardBenchmark, payload.target_benchmark_id)
    if not quantization or quantization.project_id != project_id or not baseline or baseline.project_id != project_id or not quantsim or quantsim.project_id != project_id or not benchmark or benchmark.project_id != project_id:
        raise HTTPException(422, "Quantization comparison references must belong to this project")
    reasons: list[str] = []
    if baseline.kind != "evaluation" or quantsim.kind != "evaluation" or baseline.status != "completed" or quantsim.status != "completed":
        reasons.append("baseline and QuantSim runs must be completed evaluation runs")
    if baseline.model_id != quantization.source_model_id:
        reasons.append("baseline run model does not match the quantization source model")
    if quantization.output_model_id and quantsim.model_id != quantization.output_model_id:
        reasons.append("QuantSim run model does not match the quantization output model")
    target_run = session.get(Run, benchmark.evaluation_run_id) if benchmark.evaluation_run_id else None
    if not target_run or target_run.project_id != project_id or target_run.kind != "evaluation" or target_run.status != "completed":
        reasons.append("target benchmark needs a completed linked evaluation run")
    else:
        for label, candidate in (("QuantSim", quantsim), ("target", target_run)):
            for key in ("dataset_id",):
                if candidate.dataset_id != baseline.dataset_id:
                    reasons.append(f"{label} {key} does not match baseline")
            for key in ("evaluator_version", "protocol", "scope", "evaluation_set_id"):
                if candidate.config.get(key) != baseline.config.get(key):
                    reasons.append(f"{label} {key} does not match baseline")
        models = [session.get(ModelVersion, identifier) for identifier in (baseline.model_id, quantsim.model_id, target_run.model_id)]
        mappings = [model.metadata_json.get("class_mapping_version") if model else None for model in models]
        if not mappings[0] or len(set(mappings)) != 1:
            reasons.append("baseline, QuantSim, and target class_mapping_version must be identical and explicit")
    base_value = baseline.metrics.get(payload.metric)
    quant_value = quantsim.metrics.get(payload.metric)
    target_value = target_run.metrics.get(payload.metric) if target_run else None
    if not isinstance(base_value, (int, float)) or not isinstance(quant_value, (int, float)) or not isinstance(target_value, (int, float)):
        reasons.append(f"metric {payload.metric} is missing from all compatible evaluations")
    result: dict[str, object] = {
        "status": "INCOMPLETE" if reasons else "COMPLETE",
        "metric": payload.metric,
        "higher_is_better": payload.higher_is_better,
        "roles": {"source": quantization.source_role, "quantsim": quantization.output_role, "target": "board"},
        "reasons": reasons,
    }
    if not reasons:
        if payload.higher_is_better:
            result["quantization_loss"] = round(float(base_value) - float(quant_value), 6)
            result["target_gap"] = round(float(base_value) - float(target_value), 6)
        else:
            result["quantization_loss"] = round(float(quant_value) - float(base_value), 6)
            result["target_gap"] = round(float(target_value) - float(base_value), 6)
    comparison_run = Run(
        project_id=project_id, kind="quantization-comparison", name=f"{quantization.name} comparison", status="completed" if not reasons else "incomplete",
        dataset_id=baseline.dataset_id, model_id=quantsim.model_id, parent_run_id=baseline.id,
        config={"metric": payload.metric, "higher_is_better": payload.higher_is_better, "quantsim_run_id": quantsim.id, "target_benchmark_id": benchmark.id},
        metrics={key: value for key, value in result.items() if key in {"quantization_loss", "target_gap"} and isinstance(value, (int, float))}, details=result,
        environment={"source": "quantization-lineage-comparison"}, notes="Comparison is incomplete unless all evaluation contracts match.",
    )
    session.add(comparison_run); session.commit(); session.refresh(comparison_run)
    return {"comparison": result, "run": as_dict(comparison_run)}


@app.get("/api/v1/projects/{project_id}/board-benchmarks")
def list_board_benchmarks(project_id: str, session: Session = Depends(get_session)):
    require_project(session, project_id)
    return [as_dict(item) for item in session.scalars(select(BoardBenchmark).where(BoardBenchmark.project_id == project_id).order_by(BoardBenchmark.created_at.desc())).all()]


@app.post("/api/v1/projects/{project_id}/board-benchmarks", status_code=201)
def create_board_benchmark(project_id: str, payload: BoardBenchmarkCreate, session: Session = Depends(get_session)):
    require_project(session, project_id)
    _project_entity(session, ModelVersion, payload.model_id, project_id, "Model")
    _project_entity(session, TargetProfile, payload.target_profile_id, project_id, "Target profile")
    evaluation = _project_entity(session, Run, payload.evaluation_run_id, project_id, "Evaluation run")
    if evaluation and evaluation.kind != "evaluation":
        raise HTTPException(422, "Board benchmark evaluation_run_id must reference an evaluation run")
    values = payload.model_dump()
    values["measurement"] = {**payload.measurement, "contract_validation": validate_board_measurement(payload.metrics, payload.measurement)}
    if values["measurement"]["contract_validation"]["status"] == "failed":
        raise HTTPException(422, f"Invalid board measurement: {values['measurement']['contract_validation']['errors']}")
    item = BoardBenchmark(project_id=project_id, **values)
    session.add(item); session.flush()
    if payload.raw_output_path:
        output_artifact = register_artifact(
            session,
            project_id,
            kind="board-output",
            logical_name=f"{item.name}/{item.id}/raw-output",
            owner_type="board",
            owner_id=item.id,
            source_path=payload.raw_output_path,
            notes="Raw board measurement output; keep separate from summary metrics.",
        )
        session.flush()
        item.raw_output_hash = output_artifact.sha256 or payload.raw_output_hash
        item.measurement = {**(item.measurement or {}), "raw_output_artifact_id": output_artifact.id}
    session.commit(); session.refresh(item); return as_dict(item)


@app.post("/api/v1/projects/{project_id}/board-benchmarks/{benchmark_id}/verify-output")
def verify_board_output(project_id: str, benchmark_id: str, session: Session = Depends(get_session)):
    require_project(session, project_id)
    item = session.get(BoardBenchmark, benchmark_id)
    if not item or item.project_id != project_id:
        raise HTTPException(404, "Board benchmark not found")
    artifact_id = (item.measurement or {}).get("raw_output_artifact_id")
    artifact = session.get(Artifact, artifact_id) if artifact_id else None
    if not artifact or artifact.project_id != project_id:
        raise HTTPException(422, "Board benchmark has no registered raw output artifact")
    result = verify_artifact(artifact)
    session.commit()
    return {"benchmark_id": item.id, "ok": result["status"] in {"verified", "managed"}, "artifact": result}


@app.get("/api/v1/projects/{project_id}/models")
def list_models(project_id: str, session: Session = Depends(get_session)):
    require_project(session, project_id)
    return [as_dict(x) for x in session.scalars(select(ModelVersion).where(ModelVersion.project_id == project_id)).all()]


@app.get("/api/v1/projects/{project_id}/models/{model_id}/alias-history")
def model_alias_history(project_id: str, model_id: str, session: Session = Depends(get_session)):
    require_project(session, project_id)
    model = session.get(ModelVersion, model_id)
    if not model or model.project_id != project_id:
        raise HTTPException(404, "Model not found")
    return [as_dict(item) for item in session.scalars(select(ModelAliasHistory).where(ModelAliasHistory.project_id == project_id, ModelAliasHistory.model_id == model_id).order_by(ModelAliasHistory.created_at.desc())).all()]


@app.post("/api/v1/projects/{project_id}/models", status_code=201)
def create_model(project_id: str, payload: ModelCreate, session: Session = Depends(get_session)):
    require_project(session, project_id)
    duplicate = session.scalar(select(ModelVersion).where(ModelVersion.project_id == project_id, ModelVersion.name == payload.name, ModelVersion.version == payload.version))
    if duplicate:
        raise HTTPException(409, "This ModelVersion already exists in the project")
    if payload.source_dataset_id:
        source_dataset = session.get(DatasetVersion, payload.source_dataset_id)
        if not source_dataset or source_dataset.project_id != project_id:
            raise HTTPException(422, "Source dataset must belong to this project")
    if payload.source_run_id:
        source_run = session.get(Run, payload.source_run_id)
        if not source_run or source_run.project_id != project_id:
            raise HTTPException(422, "Source run must belong to this project")
    values = payload.model_dump()
    values["artifact_sha256"] = file_sha256(values["artifact_path"]) if values.get("artifact_path") and Path(values["artifact_path"]).is_file() else None
    values["config_sha256"] = file_sha256(values["config_path"]) if values.get("config_path") and Path(values["config_path"]).is_file() else None
    model = ModelVersion(project_id=project_id, **values)
    session.add(model); session.flush()
    if values.get("artifact_path"):
        register_artifact(session, project_id, kind="model", logical_name=f"{values['name']}/{values['version']}/artifact", owner_type="model", owner_id=model.id, source_path=values["artifact_path"], sha256=values.get("artifact_sha256"))
    if values.get("config_path"):
        register_artifact(session, project_id, kind="config", logical_name=f"{values['name']}/{values['version']}/config", owner_type="model", owner_id=model.id, source_path=values["config_path"], sha256=values.get("config_sha256"))
    record_audit(session, project_id, "model", model.id, "created", after={"name": model.name, "version": model.version, "family": model.family, "alias": model.alias, "format": model.format, "precision": model.precision})
    session.commit(); session.refresh(model)
    return as_dict(model)


@app.patch("/api/v1/projects/{project_id}/models/{model_id}")
def update_model(project_id: str, model_id: str, payload: ModelUpdate, session: Session = Depends(get_session)):
    require_project(session, project_id)
    model = session.get(ModelVersion, model_id)
    if not model or model.project_id != project_id:
        raise HTTPException(404, "Model not found")
    values = payload.model_dump(exclude_unset=True)
    before = {key: getattr(model, key) for key in values if key != "alias_reason"}
    alias_reason = values.pop("alias_reason", "")
    previous_alias = model.alias
    for key, value in values.items():
        setattr(model, key, value)
    if "alias" in values and values["alias"] != previous_alias:
        session.add(ModelAliasHistory(project_id=project_id, model_id=model.id, previous_alias=previous_alias, new_alias=model.alias, reason=alias_reason))
    if "artifact_path" in values:
        model.artifact_sha256 = file_sha256(model.artifact_path) if model.artifact_path and Path(model.artifact_path).is_file() else None
        if model.artifact_path:
            register_artifact(session, project_id, kind="model", logical_name=f"{model.name}/{model.version}/artifact", owner_type="model", owner_id=model.id, source_path=model.artifact_path, sha256=model.artifact_sha256)
    if "config_path" in values:
        model.config_sha256 = file_sha256(model.config_path) if model.config_path and Path(model.config_path).is_file() else None
        if model.config_path:
            register_artifact(session, project_id, kind="config", logical_name=f"{model.name}/{model.version}/config", owner_type="model", owner_id=model.id, source_path=model.config_path, sha256=model.config_sha256)
    record_audit(session, project_id, "model", model.id, "updated", before=before, after={key: getattr(model, key) for key in values if key != "alias_reason"}, details={"alias_reason": alias_reason} if alias_reason else {})
    session.commit(); session.refresh(model)
    return as_dict(model)


@app.post("/api/v1/projects/{project_id}/models/{model_id}/archive")
def archive_model(project_id: str, model_id: str, session: Session = Depends(get_session)):
    require_project(session, project_id)
    model = session.get(ModelVersion, model_id)
    if not model or model.project_id != project_id:
        raise HTTPException(404, "Model not found")
    if model.alias == "production":
        raise HTTPException(409, "Production model must be unaliased before archiving")
    previous_status = model.status
    model.status = "experimental" if model.status == "archived" else "archived"
    record_audit(session, project_id, "model", model.id, "archived" if model.status == "archived" else "restored", before={"status": previous_status}, after={"status": model.status})
    session.commit(); session.refresh(model)
    return as_dict(model)


@app.get("/api/v1/projects/{project_id}/models/{model_id}/impact")
def model_impact(project_id: str, model_id: str, session: Session = Depends(get_session)):
    require_project(session, project_id)
    model = session.get(ModelVersion, model_id)
    if not model or model.project_id != project_id:
        raise HTTPException(404, "Model not found")
    runs = session.scalars(select(Run).where(Run.project_id == project_id, Run.model_id == model_id)).all()
    quantizations = session.scalars(select(QuantizationRun).where(QuantizationRun.project_id == project_id, (QuantizationRun.source_model_id == model_id) | (QuantizationRun.output_model_id == model_id))).all()
    boards = session.scalars(select(BoardBenchmark).where(BoardBenchmark.project_id == project_id, BoardBenchmark.model_id == model_id)).all()
    releases = session.scalars(select(Release).where(Release.project_id == project_id, (Release.model_id == model_id) | (Release.baseline_model_id == model_id))).all()
    dependencies = ([{"kind": "run", "id": item.id, "name": item.name, "status": item.status} for item in runs]
        + [{"kind": "quantization", "id": item.id, "name": item.name, "status": item.status} for item in quantizations]
        + [{"kind": "board", "id": item.id, "name": item.name, "status": item.status} for item in boards]
        + [{"kind": "release", "id": item.id, "name": item.name, "status": item.decision} for item in releases])
    return {"entity": {"kind": "model", "id": model.id, "name": f"{model.name} {model.version}", "status": model.status}, "dependencies": dependencies, "blocking": [item for item in dependencies if item["status"] not in {"archived", "failed", "cancelled", "INCOMPLETE"}]}


@app.post("/api/v1/projects/{project_id}/models/{model_id}/verify-artifacts")
def verify_model_artifacts(project_id: str, model_id: str, session: Session = Depends(get_session)):
    require_project(session, project_id)
    model = session.get(ModelVersion, model_id)
    if not model or model.project_id != project_id:
        raise HTTPException(404, "Model not found")
    artifacts = session.scalars(select(Artifact).where(Artifact.project_id == project_id, Artifact.owner_type == "model", Artifact.owner_id == model.id, Artifact.status != "superseded")).all()
    results = [verify_artifact(artifact) for artifact in artifacts]
    session.commit()
    return {"model_id": model.id, "ok": all(item["status"] in {"verified", "directory", "managed", "registered"} for item in results), "artifacts": results}


def _resolve_model_artifact(session: Session, model: ModelVersion, kind: str, configured_path: str | None) -> tuple[str | None, str]:
    if configured_path and (Path(configured_path).is_file() or Path(configured_path).is_dir()):
        return configured_path, "source"
    artifact = session.scalar(select(Artifact).where(Artifact.project_id == model.project_id, Artifact.owner_type == "model", Artifact.owner_id == model.id, Artifact.kind == kind, Artifact.status != "superseded").order_by(Artifact.created_at.desc()))
    if artifact and artifact.managed_path and (Path(artifact.managed_path).is_file() or Path(artifact.managed_path).is_dir()):
        return artifact.managed_path, "managed"
    return configured_path, "missing"


@app.post("/api/v1/projects/{project_id}/inference-preview")
def inference_preview(project_id: str, payload: InferencePreviewRequest, session: Session = Depends(get_session)):
    require_project(session, project_id)
    model = session.get(ModelVersion, payload.model_id)
    if not model or model.project_id != project_id:
        raise HTTPException(422, "Model must belong to this project")
    for path_value, expected_hash, label in ((model.artifact_path, model.artifact_sha256, "model artifact"), (model.config_path, model.config_sha256, "model config")):
        if expected_hash and path_value and Path(path_value).is_file() and file_sha256(path_value) != expected_hash:
            raise HTTPException(409, f"{label} changed after registration; verify or create a new ModelVersion")
    try:
        artifact_path, artifact_source = _resolve_model_artifact(session, model, "model", model.artifact_path)
        config_path, config_source = _resolve_model_artifact(session, model, "config", model.config_path)
        if model.format == "onnx" and artifact_path:
            profile = model.metadata_json.get("onnx_profile")
            if not profile:
                raise ValueError("Model metadata must include an explicit onnx_profile")
            result = infer_onnx(artifact_path, payload.image_path, profile)
            result["artifact_source"] = artifact_source
            return result
        if model.format == "mmdetection-pytorch" and artifact_path and config_path:
            result = infer_mmdetection(config_path, artifact_path, payload.image_path)
            result["artifact_source"] = {"model": artifact_source, "config": config_source}
            return result
        if model.format == "mmdeploy" and artifact_path:
            result = infer_mmdeploy(artifact_path, payload.image_path, model.metadata_json.get("mmdeploy_profile", {}))
            result["artifact_source"] = artifact_source
            return result
        raise ValueError("Inference preview requires an ONNX profile, a MMDetection config/checkpoint bundle, or an MMDeploy model directory")
    except (OSError, RuntimeError, ValueError) as error:
        raise HTTPException(422, str(error)) from error


@app.get("/api/v1/projects/{project_id}/runs")
def list_runs(project_id: str, session: Session = Depends(get_session)):
    require_project(session, project_id)
    return [as_dict(x) for x in session.scalars(select(Run).where(Run.project_id == project_id).order_by(Run.created_at.desc())).all()]


@app.post("/api/v1/projects/{project_id}/runs", status_code=201)
def create_run(project_id: str, payload: RunCreate, response: Response, session: Session = Depends(get_session)):
    require_project(session, project_id)
    for entity, label in ((session.get(DatasetVersion, payload.dataset_id) if payload.dataset_id else None, "Dataset"), (session.get(ModelVersion, payload.model_id) if payload.model_id else None, "Model"), (session.get(Run, payload.parent_run_id) if payload.parent_run_id else None, "Parent run")):
        if entity and entity.project_id != project_id:
            raise HTTPException(422, f"{label} must belong to this project")
        if (payload.dataset_id and label == "Dataset" or payload.model_id and label == "Model" or payload.parent_run_id and label == "Parent run") and not entity:
            raise HTTPException(422, f"{label} not found")
    if payload.training and payload.kind != "training":
        raise HTTPException(422, "Typed training provenance is only valid for a training Run")
    training_details = None
    if payload.training:
        references = ((payload.training.split_id, SplitVersion, "Split"), (payload.training.label_schema_id, LabelSchemaVersion, "Label schema"))
        for entity_id, entity_type, label in references:
            if not entity_id:
                continue
            entity = session.get(entity_type, entity_id)
            if not entity or entity.project_id != project_id:
                raise HTTPException(422, f"{label} reference must belong to this project")
            if payload.dataset_id and getattr(entity, "dataset_id", None) and entity.dataset_id != payload.dataset_id:
                raise HTTPException(422, f"{label} reference must use the selected training DatasetVersion")
        missing_fields = [field for field, value in (("dataset_id", payload.dataset_id), ("split_id", payload.training.split_id), ("label_schema_id", payload.training.label_schema_id)) if not value and field not in payload.training.unknown_fields]
        training_details = {**payload.training.model_dump(exclude_none=True), "lineage_status": "complete" if not missing_fields else "partial", "missing_fields": missing_fields}
    values = payload.model_dump(exclude={"training"})
    if training_details:
        values["details"] = {**values.get("details", {}), "training": training_details}
    if payload.external_run_id:
        existing = session.scalar(select(Run).where(Run.project_id == project_id, Run.external_run_id == payload.external_run_id))
        import_hash = _version_hash(values)
        if existing:
            if existing.import_hash == import_hash:
                response.status_code = 200
                return as_dict(existing)
            raise HTTPException(409, "An external run with this ID exists but its content differs")
        values["import_hash"] = import_hash
    run = Run(project_id=project_id, **values)
    session.add(run); session.flush()
    record_audit(session, project_id, "run", run.id, "registered", after={"kind": run.kind, "name": run.name, "status": run.status, "external_run_id": run.external_run_id, "dataset_id": run.dataset_id, "model_id": run.model_id})
    session.commit(); session.refresh(run)
    return as_dict(run)


@app.post("/api/v1/projects/{project_id}/results/import", status_code=201)
def import_result(project_id: str, payload: ResultImportCreate, session: Session = Depends(get_session)):
    require_project(session, project_id)
    try:
        manifest = validate_result_manifest(payload.manifest)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    dataset_id, model_id = manifest.get("dataset_id"), manifest.get("model_id")
    if dataset_id and (not (dataset := session.get(DatasetVersion, dataset_id)) or dataset.project_id != project_id):
        raise HTTPException(422, "Result manifest dataset_id does not belong to this project")
    if model_id and (not (model := session.get(ModelVersion, model_id)) or model.project_id != project_id):
        raise HTTPException(422, "Result manifest model_id does not belong to this project")
    target_profile_id = manifest.get("target_profile_id")
    if target_profile_id and (not (target := session.get(TargetProfile, target_profile_id)) or target.project_id != project_id):
        raise HTTPException(422, "Result manifest target_profile_id does not belong to this project")
    quantization_run_id = manifest.get("quantization_run_id")
    if quantization_run_id and (not (quantization := session.get(QuantizationRun, quantization_run_id)) or quantization.project_id != project_id):
        raise HTTPException(422, "Result manifest quantization_run_id does not belong to this project")
    calibration_set_id = manifest.get("calibration_set_id")
    if calibration_set_id and (not (calibration := session.get(CalibrationSetVersion, calibration_set_id)) or calibration.project_id != project_id):
        raise HTTPException(422, "Result manifest calibration_set_id does not belong to this project")
    board_benchmark_id = manifest.get("board_benchmark_id")
    if board_benchmark_id and (not (benchmark := session.get(BoardBenchmark, board_benchmark_id)) or benchmark.project_id != project_id):
        raise HTTPException(422, "Result manifest board_benchmark_id does not belong to this project")
    parent_run_id = manifest.get("parent_run_id")
    if parent_run_id and (not (parent := session.get(Run, parent_run_id)) or parent.project_id != project_id):
        raise HTTPException(422, "Result manifest parent_run_id does not belong to this project")
    fingerprint = manifest_hash(manifest)
    existing = session.scalar(select(Run).where(Run.project_id == project_id, Run.external_run_id == manifest["external_run_id"]))
    if existing:
        if existing.import_hash == fingerprint:
            return {"status": "existing", "run": as_dict(existing)}
        raise HTTPException(409, "An external run with this ID exists but its manifest content differs")
    details = dict(manifest.get("details", {}))
    run = Run(
        project_id=project_id, kind=manifest["kind"], name=manifest["name"], status=manifest.get("status", "completed"),
        dataset_id=dataset_id, model_id=model_id, parent_run_id=parent_run_id,
        external_run_id=manifest["external_run_id"], import_hash=fingerprint,
        config={**manifest.get("config", {}), **({key: value for key, value in (("target_profile_id", target_profile_id), ("quantization_run_id", quantization_run_id), ("calibration_set_id", calibration_set_id), ("board_benchmark_id", board_benchmark_id)) if value})}, metrics=manifest.get("metrics", {}), details=details, environment=manifest.get("environment", {}), notes=manifest.get("notes", ""),
    )
    session.add(run); session.flush()
    record_audit(session, project_id, "run", run.id, "registered", after={"kind": run.kind, "name": run.name, "status": run.status, "dataset_id": run.dataset_id, "model_id": run.model_id})
    session.commit(); session.refresh(run)
    return {"status": "created", "run": as_dict(run)}


@app.post("/api/v1/projects/{project_id}/evaluations/predictions", status_code=201)
def evaluate_predictions(project_id: str, payload: PredictionEvaluationCreate, session: Session = Depends(get_session)):
    require_project(session, project_id)
    dataset = session.get(DatasetVersion, payload.dataset_id)
    model = session.get(ModelVersion, payload.model_id)
    if not dataset or dataset.project_id != project_id or not model or model.project_id != project_id:
        raise HTTPException(422, "Model and dataset must belong to this project")
    evaluation_set = _project_entity(session, EvaluationSetVersion, payload.evaluation_set_id, project_id, "Evaluation set")
    if evaluation_set and evaluation_set.dataset_id and evaluation_set.dataset_id != dataset.id:
        raise HTTPException(422, "Evaluation set must reference the selected DatasetVersion")
    if dataset.format != "coco" or not dataset.annotation_path:
        raise HTTPException(422, "Prediction evaluation currently requires a COCO dataset with annotation_path")
    _ensure_dataset_source_current(dataset)
    try:
        import json
        from pathlib import Path
        with Path(payload.predictions_path).open(encoding="utf-8") as file:
            predictions = json.load(file)
        if not isinstance(predictions, list):
            raise ValueError("Prediction JSON must be a list")
        evaluation_image_ids = _evaluation_image_ids(evaluation_set)
        if payload.protocol == "onboarding_ap50":
            result = evaluate_coco_predictions(dataset.annotation_path, predictions, payload.iou_threshold, image_ids=evaluation_image_ids)
        elif payload.protocol == "coco_full":
            result = evaluate_coco_full(dataset.annotation_path, predictions, image_ids=evaluation_image_ids)
        else:
            raise ValueError("protocol must be onboarding_ap50 or coco_full")
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        raise HTTPException(422, str(error)) from error
    metric_scope = result.get("metric_scope", "")
    details = {key: value for key, value in result.items() if key != "metric_scope"}
    if evaluation_set:
        details["evaluation_set_filter"] = {
            "evaluation_set_id": evaluation_set.id,
            "selected_image_count": len(evaluation_image_ids) if evaluation_image_ids is not None else "all",
        }
    metrics = {key: value for key, value in details.items() if isinstance(value, (int, float))}
    run = Run(
        project_id=project_id, kind="evaluation", name=f"{model.family} prediction import", status="completed",
        dataset_id=dataset.id, model_id=model.id,
        config={"evaluator_version": "coco-full-v1" if payload.protocol == "coco_full" and payload.evaluator_version == "lifecycle-ap50-v1" else payload.evaluator_version, "protocol": payload.protocol, "iou_threshold": payload.iou_threshold, "scope": metric_scope, **({"evaluation_set_id": evaluation_set.id} if evaluation_set else {})},
        metrics=metrics, details=details,
        environment={"source": "external-prediction-json"}, notes="Per-class output retained in evaluation import result.",
    )
    session.add(run); session.flush()
    prediction_artifact = register_artifact(
        session, project_id, kind="prediction", logical_name=f"{model.name}/{model.version}/{dataset.name}/{dataset.version}/predictions",
        owner_type="run", owner_id=run.id, source_path=payload.predictions_path, notes=f"{payload.protocol} evaluation input",
    )
    details["prediction_artifact_id"] = prediction_artifact.id
    run.details = details
    record_audit(session, project_id, "run", run.id, "registered", after={"kind": run.kind, "name": run.name, "status": run.status, "dataset_id": run.dataset_id, "model_id": run.model_id})
    session.commit(); session.refresh(run)
    return {"run": as_dict(run), "result": {"metric_scope": metric_scope, **details}}


@app.post("/api/v1/projects/{project_id}/evaluations/classification", status_code=201)
def evaluate_classification_records(project_id: str, payload: ClassificationEvaluationCreate, session: Session = Depends(get_session)):
    require_project(session, project_id)
    model = session.get(ModelVersion, payload.model_id)
    if not model or model.project_id != project_id or model.task_kind != "classification":
        raise HTTPException(422, "A classification model from this project is required")
    if payload.dataset_id:
        dataset = session.get(DatasetVersion, payload.dataset_id)
        if not dataset or dataset.project_id != project_id:
            raise HTTPException(422, "Dataset must belong to this project")
        if dataset.class_names:
            known = {str(value) for value in dataset.class_names}
            supplied = {str(record.get(key)) for record in payload.records for key in ("ground_truth", "prediction") if record.get(key) is not None}
            unknown = sorted(supplied - known)
            if unknown:
                raise HTTPException(422, f"Classification labels are not in the Dataset class mapping: {', '.join(unknown)}")
        _ensure_dataset_source_current(dataset)
    evaluation_set = _project_entity(session, EvaluationSetVersion, payload.evaluation_set_id, project_id, "Evaluation set")
    if evaluation_set and evaluation_set.dataset_id and evaluation_set.dataset_id != payload.dataset_id:
        raise HTTPException(422, "Evaluation set must reference the selected DatasetVersion")
    try:
        evaluation_item_keys = _evaluation_item_keys(evaluation_set)
        records = payload.records
        if evaluation_item_keys is not None:
            records = [record for record in records if str(record.get("image_id", record.get("item_id", record.get("id", "")))) in evaluation_item_keys]
            if not records:
                raise ValueError("Evaluation set selected no classification records")
        result = evaluate_classification(records)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    if evaluation_set:
        result["evaluation_set_filter"] = {
            "evaluation_set_id": evaluation_set.id,
            "selected_item_count": len(evaluation_item_keys) if evaluation_item_keys is not None else "all",
            "evaluated_record_count": result.get("records", 0),
        }
    metric_keys = {"top1_accuracy", "topk_accuracy", "macro_precision", "macro_recall", "macro_f1", "records", "expected_calibration_error"}
    run = Run(
        project_id=project_id, kind="evaluation", name=f"{model.family} classification evaluation", status="completed",
        dataset_id=payload.dataset_id, model_id=model.id,
        config={"evaluator_version": payload.evaluator_version, "task_kind": "classification", **({"evaluation_set_id": evaluation_set.id} if evaluation_set else {})},
        metrics={key: result[key] for key in metric_keys if key in result}, details=result, environment={"source": "external-classification-records"},
        notes="Per-class and confusion matrix output is returned by this import response.",
    )
    session.add(run); session.flush()
    record_audit(session, project_id, "run", run.id, "registered", after={"kind": run.kind, "name": run.name, "status": run.status, "dataset_id": run.dataset_id, "model_id": run.model_id})
    session.commit(); session.refresh(run)
    return {"run": as_dict(run), "result": result}


@app.post("/api/v1/projects/{project_id}/evaluations/onnx-batch", status_code=201)
def evaluate_onnx_batch(project_id: str, payload: OnnxBatchEvaluationCreate, session: Session = Depends(get_session)):
    """Run an explicitly profiled ONNX model over image records and persist one evaluation Run."""
    require_project(session, project_id)
    dataset = session.get(DatasetVersion, payload.dataset_id)
    model = session.get(ModelVersion, payload.model_id)
    if not dataset or dataset.project_id != project_id or not model or model.project_id != project_id:
        raise HTTPException(422, "Model and dataset must belong to this project")
    evaluation_set = _project_entity(session, EvaluationSetVersion, payload.evaluation_set_id, project_id, "Evaluation set")
    if evaluation_set and evaluation_set.dataset_id and evaluation_set.dataset_id != dataset.id:
        raise HTTPException(422, "Evaluation set must reference the selected DatasetVersion")
    if model.format != "onnx" or not model.artifact_path:
        raise HTTPException(422, "ONNX batch evaluation requires a registered ONNX model artifact")
    profile = model.metadata_json.get("onnx_profile")
    if not profile:
        raise HTTPException(422, "Model metadata must include an explicit onnx_profile")
    _ensure_dataset_source_current(dataset)
    try:
        records_path = Path(payload.records_path)
        with records_path.open(encoding="utf-8") as file:
            records = json.load(file)
        if not isinstance(records, list) or not records:
            raise ValueError("ONNX records JSON must be a non-empty list")
        if any(not isinstance(record, dict) or not record.get("image_path") for record in records):
            raise ValueError("Every ONNX record must include image_path")
        resolved_records = []
        for record in records:
            image_path = Path(str(record["image_path"]))
            if not image_path.is_absolute():
                image_path = records_path.parent / image_path
            resolved_records.append({**record, "image_path": str(image_path)})
        evaluation_item_keys = _evaluation_item_keys(evaluation_set)
        if evaluation_item_keys is not None:
            resolved_records = [record for record in resolved_records if str(record.get("image_id", record.get("item_id", record.get("id", "")))) in evaluation_item_keys]
            if not resolved_records:
                raise ValueError("Evaluation set selected no ONNX records")
        predictions = [infer_onnx(model.artifact_path, record["image_path"], profile) for record in resolved_records]
        if model.task_kind == "classification":
            class_names = [str(value) for value in dataset.class_names]
            output_records = []
            for index, (record, prediction) in enumerate(zip(resolved_records, predictions, strict=True)):
                top_indices = prediction.get("top_indices", [])
                top_labels = [class_names[item] if item < len(class_names) else str(item) for item in top_indices[:payload.top_k]]
                if not top_labels:
                    raise ValueError(f"ONNX classification returned no scores for record {index}")
                output_records.append({"image_id": record.get("image_id", str(index)), "ground_truth": record.get("ground_truth"), "prediction": top_labels[0], "top_k": top_labels})
            if any(item["ground_truth"] is None for item in output_records):
                raise ValueError("Classification ONNX records must include ground_truth")
            result = evaluate_classification(output_records)
            metrics = {key: result[key] for key in ("top1_accuracy", "topk_accuracy", "macro_precision", "macro_recall", "macro_f1", "records")}
            details = {**result, "records_path": payload.records_path, "inference_provider": predictions[0].get("provider", "unknown")}
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
            details.update({"records_path": payload.records_path, "inference_provider": predictions[0].get("provider", "unknown"), "prediction_count": len(coco_predictions)})
        else:
            raise ValueError(f"ONNX batch evaluation does not support task_kind {model.task_kind}")
        if evaluation_set:
            details["evaluation_set_filter"] = {
                "evaluation_set_id": evaluation_set.id,
                "selected_item_count": len(evaluation_item_keys) if evaluation_item_keys is not None else "all",
                "evaluated_record_count": len(resolved_records),
            }
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        raise HTTPException(422, str(error)) from error
    run = Run(project_id=project_id, kind="evaluation", name=f"{model.family} ONNX batch evaluation", status="completed", dataset_id=dataset.id, model_id=model.id, config={"evaluator_version": payload.evaluator_version, "protocol": "onnx-batch", "task_kind": model.task_kind, "top_k": payload.top_k, "scope": result.get("metric_scope", "classification"), **({"evaluation_set_id": evaluation_set.id} if evaluation_set else {})}, metrics=metrics, details=details, environment={"provider": details.get("inference_provider", "unknown"), "runtime": "onnxruntime-cpu"}, notes="ONNX batch inference and evaluation from an explicit image record manifest.")
    session.add(run); session.flush()
    prediction_artifact = register_artifact(session, project_id, kind="onnx-records", logical_name=f"{model.name}/{model.version}/{dataset.name}/{dataset.version}/onnx-records", owner_type="run", owner_id=run.id, source_path=payload.records_path, notes="ONNX batch input records")
    session.flush()
    run.details = {**details, "records_artifact_id": prediction_artifact.id}
    record_audit(session, project_id, "run", run.id, "registered", after={"kind": run.kind, "name": run.name, "status": run.status, "dataset_id": run.dataset_id, "model_id": run.model_id})
    session.commit(); session.refresh(run)
    return {"run": as_dict(run), "result": {"metric_scope": result.get("metric_scope", "classification"), **details}}


@app.post("/api/v1/comparisons")
def comparison(payload: ComparisonRequest, session: Session = Depends(get_session)):
    try:
        result = compare_models(session, payload.baseline_model_id, payload.candidate_model_id)
    except LookupError as error:
        raise HTTPException(404, str(error)) from error
    return {key: as_dict(value) if hasattr(value, "__table__") else value for key, value in result.items()}


@app.get("/api/v1/projects/{project_id}/jobs")
def list_jobs(project_id: str, session: Session = Depends(get_session)):
    require_project(session, project_id)
    return [as_dict(x) for x in session.scalars(select(Job).where(Job.project_id == project_id).order_by(Job.created_at.desc())).all()]


@app.get("/api/v1/projects/{project_id}/jobs/{job_id}/logs")
def job_logs(project_id: str, job_id: str, cursor: int = 0, limit: int = 200, session: Session = Depends(get_session)):
    """Read an append-only log window so clients can reconnect without replaying it all."""
    require_project(session, project_id)
    job = session.get(Job, job_id)
    if not job or job.project_id != project_id:
        raise HTTPException(404, "Job not found")
    if cursor < 0 or limit < 1 or limit > 2_000:
        raise HTTPException(422, "cursor must be non-negative and limit must be between 1 and 2000")
    source = job.log or ""
    start = min(cursor, len(source))
    # Cursor is a byte-independent character offset because the stored log is
    # Unicode text. Returning the next offset makes the contract safe for
    # Korean output as well as ASCII runner output.
    end = min(len(source), start + limit * 4_096)
    window = source[start:end]
    if end < len(source):
        boundary = window.rfind("\n")
        if boundary > 0:
            end = start + boundary + 1
            window = source[start:end]
    return {"job_id": job.id, "status": job.status, "cursor": start, "next_cursor": end, "complete": end >= len(source), "text": window, "result": job.result_json or {}}


@app.get("/api/v1/projects/{project_id}/runners")
def list_runners(project_id: str, session: Session = Depends(get_session)):
    require_project(session, project_id)
    profiles = session.scalars(select(RunnerProfile).where(RunnerProfile.project_id == project_id).order_by(RunnerProfile.name)).all()
    return [{"id": "mock-board", "name": "Mock board runner", "enabled": True, "builtin": True}, *[as_dict(profile) for profile in profiles]]


@app.get("/api/v1/projects/{project_id}/targets")
def list_targets(project_id: str, session: Session = Depends(get_session)):
    require_project(session, project_id)
    return [as_dict(item) for item in session.scalars(select(TargetProfile).where(TargetProfile.project_id == project_id).order_by(TargetProfile.name)).all()]


@app.post("/api/v1/projects/{project_id}/targets", status_code=201)
def create_target(project_id: str, payload: TargetProfileCreate, session: Session = Depends(get_session)):
    require_project(session, project_id)
    target = TargetProfile(project_id=project_id, **payload.model_dump())
    session.add(target); session.commit(); session.refresh(target)
    return as_dict(target)


@app.post("/api/v1/projects/{project_id}/runners", status_code=201)
def create_runner(project_id: str, payload: RunnerProfileCreate, session: Session = Depends(get_session)):
    require_project(session, project_id)
    profile = RunnerProfile(project_id=project_id, **payload.model_dump())
    session.add(profile); session.commit(); session.refresh(profile)
    return as_dict(profile)


@app.post("/api/v1/projects/{project_id}/jobs", status_code=201)
def create_job(project_id: str, payload: JobCreate, session: Session = Depends(get_session)):
    require_project(session, project_id)
    if payload.runner_id != "mock-board":
        profile = session.get(RunnerProfile, payload.runner_id)
        if not profile or profile.project_id != project_id or not profile.enabled:
            raise HTTPException(422, "Runner profile is not available for this project")
        command = [profile.executable, *profile.default_args, *payload.args]
    else:
        command = ["builtin:mock-board"]
    job = Job(project_id=project_id, runner_id=payload.runner_id, command=command, input_json={**payload.input_json, "_runner_args": payload.args}, status="queued")
    session.add(job); session.commit(); session.refresh(job)
    if os.environ.get("VISION_LIFECYCLE_EXTERNAL_WORKER", "false").lower() != "true":
        launch(job, payload.args)
    return as_dict(job)


@app.post("/api/v1/projects/{project_id}/jobs/{job_id}/cancel")
def cancel_job(project_id: str, job_id: str, session: Session = Depends(get_session)):
    job = session.get(Job, job_id)
    if not job or job.project_id != project_id:
        raise HTTPException(404, "Job not found")
    if job.status not in {"queued", "running"}:
        raise HTTPException(409, f"Job cannot be cancelled from {job.status}")
    # Persist the intent before signalling the subprocess. The runner can
    # exit immediately, so writing this first prevents a cancellation race
    # from being reported as a completed job.
    if job.status == "running":
        job.status = "cancelling"
        job.log = f"{job.log}Cancellation requested; stopping runner process group.\n"
        session.commit()
        if cancel(job_id):
            session.refresh(job)
            return as_dict(job)
        # The process may have exited between the status read and signal. The
        # runner will have written its terminal status if it owned the job.
        session.refresh(job)
        if job.status not in {"cancelling", "running"}:
            return as_dict(job)
        # A different worker process may own the subprocess. Leave the
        # cancellation marker for that worker's DB watcher to observe.
        return as_dict(job)
    job.status = "cancelled"
    job.log = f"{job.log}Cancelled before process start.\n"
    session.commit(); session.refresh(job)
    return as_dict(job)


@app.post("/api/v1/projects/{project_id}/jobs/{job_id}/retry", status_code=201)
def retry_job(project_id: str, job_id: str, session: Session = Depends(get_session)):
    original = session.get(Job, job_id)
    if not original or original.project_id != project_id:
        raise HTTPException(404, "Job not found")
    if original.status not in {"failed", "timed_out", "cancelled", "interrupted"}:
        raise HTTPException(409, f"Job cannot be retried from {original.status}")
    args = original.input_json.get("_runner_args", []) if isinstance(original.input_json, dict) else []
    retry_input = {key: value for key, value in (original.input_json or {}).items() if key != "_runner_args"}
    retry_input["retry_of"] = original.id
    new_job = Job(project_id=project_id, runner_id=original.runner_id, command=original.command, input_json={**retry_input, "_runner_args": args}, status="queued")
    session.add(new_job); session.commit(); session.refresh(new_job)
    if os.environ.get("VISION_LIFECYCLE_EXTERNAL_WORKER", "false").lower() != "true":
        launch(new_job, args)
    return as_dict(new_job)


@app.get("/api/v1/projects/{project_id}/releases")
def list_releases(project_id: str, session: Session = Depends(get_session)):
    require_project(session, project_id)
    return [as_dict(item) for item in session.scalars(select(Release).where(Release.project_id == project_id).order_by(Release.created_at.desc())).all()]


@app.post("/api/v1/projects/{project_id}/releases", status_code=201)
def create_release(project_id: str, payload: ReleaseCreate, session: Session = Depends(get_session)):
    require_project(session, project_id)
    model = session.get(ModelVersion, payload.model_id)
    if not model or model.project_id != project_id:
        raise HTTPException(422, "Model must belong to this project")
    if payload.baseline_model_id:
        baseline_model = session.get(ModelVersion, payload.baseline_model_id)
        if not baseline_model or baseline_model.project_id != project_id:
            raise HTTPException(422, "Baseline model must belong to this project")
    evaluation = session.get(Run, payload.evaluation_run_id) if payload.evaluation_run_id else None
    if evaluation and (evaluation.project_id != project_id or evaluation.model_id != model.id or evaluation.kind != "evaluation"):
        raise HTTPException(422, "Evaluation must belong to the selected model and project")
    baseline_metrics = None
    baseline_compatibility = {"status": "not_requested"}
    if payload.baseline_model_id:
        baseline_compatibility = {"status": "incomplete", "reason": "A completed baseline evaluation with the same evaluation contract is required."}
        if evaluation:
            baseline_model = session.get(ModelVersion, payload.baseline_model_id)
            runs = session.scalars(select(Run).where(Run.project_id == project_id, Run.model_id == payload.baseline_model_id, Run.kind == "evaluation", Run.status == "completed", Run.dataset_id == evaluation.dataset_id).order_by(Run.created_at.desc())).all()
            expected_mapping = model.metadata_json.get("class_mapping_version")
            baseline_mapping = baseline_model.metadata_json.get("class_mapping_version") if baseline_model else None
            for candidate in runs:
                if any(candidate.config.get(key) != evaluation.config.get(key) for key in ("evaluator_version", "protocol", "scope", "evaluation_set_id")):
                    continue
                if expected_mapping and expected_mapping != baseline_mapping:
                    continue
                baseline_metrics = candidate.metrics
                baseline_compatibility = {"status": "compatible", "run_id": candidate.id}
                break
    try:
        result = evaluate_gate(evaluation.metrics if evaluation else None, baseline_metrics, payload.gate_config)
    except GateConfigError as error:
        raise HTTPException(422, str(error)) from error
    if payload.baseline_model_id and baseline_compatibility["status"] != "compatible" and payload.gate_config.get("max_regression"):
        result["status"] = "INCOMPLETE"
        result["reason"] = baseline_compatibility["reason"]
    captured_evidence: list[ReleaseEvidence] = []
    try:
        for evidence in payload.evidence:
            captured_evidence.append(_capture_release_evidence(session, project_id, "pending", evidence))
    except HTTPException:
        raise
    required_types = {str(value) for value in payload.gate_config.get("required_evidence", []) if isinstance(value, str)}
    captured_types = {item.evidence_type for item in captured_evidence}
    missing_evidence = sorted(required_types - captured_types)
    if missing_evidence:
        result["status"] = "INCOMPLETE"
        result["reason"] = f"필수 Release evidence가 없습니다: {', '.join(missing_evidence)}"
    result["evidence"] = {"captured": sorted(captured_types), "missing_required": missing_evidence}
    result["baseline_compatibility"] = baseline_compatibility
    values = payload.model_dump(exclude={"evidence"})
    release = Release(project_id=project_id, **values, gate_result=result, decision=result["status"])
    session.add(release); session.flush()
    for evidence in captured_evidence:
        evidence.release_id = release.id
        session.add(evidence)
    record_audit(session, project_id, "release", release.id, "created", after={"name": release.name, "model_id": release.model_id, "decision": release.decision}, details={"evidence_count": len(captured_evidence), "required_evidence_missing": missing_evidence})
    session.commit(); session.refresh(release)
    response = as_dict(release)
    response["evidence"] = [as_dict(item) for item in session.scalars(select(ReleaseEvidence).where(ReleaseEvidence.release_id == release.id)).all()]
    return response


@app.get("/api/v1/projects/{project_id}/releases/{release_id}/evidence")
def list_release_evidence(project_id: str, release_id: str, session: Session = Depends(get_session)):
    require_project(session, project_id)
    release = session.get(Release, release_id)
    if not release or release.project_id != project_id:
        raise HTTPException(404, "Release not found")
    return [as_dict(item) for item in session.scalars(select(ReleaseEvidence).where(ReleaseEvidence.project_id == project_id, ReleaseEvidence.release_id == release_id).order_by(ReleaseEvidence.created_at)).all()]


@app.get("/api/v1/projects/{project_id}/audit-events")
def list_audit_events(project_id: str, limit: int = 200, session: Session = Depends(get_session)):
    require_project(session, project_id)
    bounded_limit = max(1, min(limit, 1000))
    return [as_dict(item) for item in session.scalars(
        select(AuditEvent).where(AuditEvent.project_id == project_id).order_by(AuditEvent.created_at.desc()).limit(bounded_limit)
    ).all()]


@app.get("/api/v1/projects/{project_id}/export")
def export_project(project_id: str, session: Session = Depends(get_session)):
    try:
        return safe_export(session, project_id)
    except LookupError as error:
        raise HTTPException(404, str(error)) from error


@app.get("/")
def local_ui():
    index = Path(__file__).parents[2] / "frontend" / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"message": "Frontend sources are in ./frontend. Run npm install && npm run dev."}
