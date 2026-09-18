from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from .database import SessionLocal, init_database
from .adapters.coco import validate_coco
from .adapters.inspect import inspect_path
from .evaluators.detection import evaluate_coco_full, evaluate_coco_predictions
from .evaluators.classification import evaluate_classification
from .inference.onnx import diagnose as diagnose_onnx, infer as infer_onnx
from .inference.mmdetection import diagnose as diagnose_mmdetection, infer as infer_mmdetection
from .inference.mmdeploy import diagnose as diagnose_mmdeploy, infer as infer_mmdeploy
from .importer import manifest_hash, validate_result_manifest
from .models import DatasetVersion, Job, ModelVersion, Project, Release, Run, RunnerProfile, StorageMapping, TargetProfile
from .release_gate import GateConfigError, evaluate_gate
from .runner import cancel, launch
from .schemas import ClassificationEvaluationCreate, ComparisonRequest, DatasetCreate, DatasetUpdate, InferencePreviewRequest, JobCreate, ModelCreate, ModelUpdate, PathInspectRequest, PredictionEvaluationCreate, ProjectCreate, ProjectUpdate, ReleaseCreate, ResultImportCreate, RunCreate, RunnerProfileCreate, StorageBrowseRequest, StorageMappingCreate, TargetProfileCreate
from .serializers import as_dict
from .service import agent_request, compare_models, overview, safe_export, seed_demo
from .fingerprints import dataset_fingerprint
from .storage import browse as browse_storage, storage_status


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_database()
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
    return {"status": "ok", "mode": "local", "version": "0.1.0"}


@app.get("/api/v1/plugins")
def plugins():
    return [
        {"id": "coco", "kind": "dataset", "tasks": ["detection"], "capabilities": ["inspect", "validate", "prediction-evaluation"]},
        {"id": "yolo-txt", "kind": "dataset", "tasks": ["detection"], "capabilities": ["inspect", "validate"]},
        {"id": "classification", "kind": "dataset", "tasks": ["classification"], "capabilities": ["inspect", "validate"]},
        {"id": "mmdetection", "kind": "model", "tasks": ["detection"], "capabilities": ["register", "native-inference", "external-result-import"]},
        {"id": "mmdeploy", "kind": "model", "tasks": ["detection"], "capabilities": ["runtime-inference", "target-profile"]},
        {"id": "mock-board", "kind": "runner", "tasks": ["detection", "classification"], "capabilities": ["run", "result-contract"]},
    ]


@app.get("/api/v1/recipes")
def list_recipes():
    return [
        {"id": "blank", "name": "내 프로젝트 연결", "task_kind": "unknown", "steps": ["project", "data", "model", "evaluation", "report"]},
        {"id": "mmdetection-onboarding", "name": "MMDetection RTMDet·YOLOX 실습", "task_kind": "detection", "steps": ["project", "data", "rtmdet", "evaluation", "yolox", "comparison"]},
        {"id": "classification-onboarding", "name": "분류 모델 실습", "task_kind": "classification", "steps": ["project", "data", "model", "evaluation", "comparison"]},
    ]


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
    session.add(project); session.commit(); session.refresh(project)
    return as_dict(project)


@app.patch("/api/v1/projects/{project_id}")
def update_project(project_id: str, payload: ProjectUpdate, session: Session = Depends(get_session)):
    project = require_project(session, project_id)
    if payload.name and payload.name != project.name and session.scalar(select(Project).where(Project.name == payload.name)):
        raise HTTPException(409, "A project with this name already exists")
    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(project, key, value)
    session.commit(); session.refresh(project)
    return as_dict(project)


@app.post("/api/v1/projects/{project_id}/archive")
def archive_project(project_id: str, session: Session = Depends(get_session)):
    project = require_project(session, project_id)
    project.status = "archived" if project.status != "archived" else "active"
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
    content_hash, fingerprints = dataset_fingerprint(payload.manifest_path, payload.annotation_path)
    validation = {**payload.validation, "source_fingerprints": fingerprints} if fingerprints else payload.validation
    dataset = DatasetVersion(project_id=project_id, **payload.model_dump(exclude={"validation"}), content_hash=content_hash, validation=validation)
    session.add(dataset); session.commit(); session.refresh(dataset)
    return as_dict(dataset)


@app.patch("/api/v1/projects/{project_id}/datasets/{dataset_id}")
def update_dataset(project_id: str, dataset_id: str, payload: DatasetUpdate, session: Session = Depends(get_session)):
    require_project(session, project_id)
    dataset = session.get(DatasetVersion, dataset_id)
    if not dataset or dataset.project_id != project_id:
        raise HTTPException(404, "Dataset not found")
    if dataset.status == "finalized":
        raise HTTPException(409, "Finalized DatasetVersion is immutable; create a new version")
    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(dataset, key, value)
    session.commit(); session.refresh(dataset)
    return as_dict(dataset)


@app.post("/api/v1/projects/{project_id}/datasets/{dataset_id}/archive")
def archive_dataset(project_id: str, dataset_id: str, session: Session = Depends(get_session)):
    require_project(session, project_id)
    dataset = session.get(DatasetVersion, dataset_id)
    if not dataset or dataset.project_id != project_id:
        raise HTTPException(404, "Dataset not found")
    dataset.status = "draft" if dataset.status == "archived" else "archived"
    session.commit(); session.refresh(dataset)
    return as_dict(dataset)


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
    dataset.validation = {**dataset.validation, "source_fingerprints": fingerprints, "finalized_at": "local"}
    dataset.status = "finalized"
    session.commit(); session.refresh(dataset)
    return as_dict(dataset)


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
    session.add(mapping); session.commit(); session.refresh(mapping)
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


@app.post("/api/v1/projects/{project_id}/storages/{storage_id}/browse")
def browse_storage_mapping(project_id: str, storage_id: str, payload: StorageBrowseRequest, session: Session = Depends(get_session)):
    require_project(session, project_id)
    mapping = session.get(StorageMapping, storage_id)
    if not mapping or mapping.project_id != project_id:
        raise HTTPException(404, "Storage mapping not found")
    try:
        return browse_storage(mapping.root_path, payload.relative_path, payload.limit)
    except (OSError, ValueError) as error:
        raise HTTPException(422, str(error)) from error


@app.get("/api/v1/projects/{project_id}/models")
def list_models(project_id: str, session: Session = Depends(get_session)):
    require_project(session, project_id)
    return [as_dict(x) for x in session.scalars(select(ModelVersion).where(ModelVersion.project_id == project_id)).all()]


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
    model = ModelVersion(project_id=project_id, **payload.model_dump())
    session.add(model); session.commit(); session.refresh(model)
    return as_dict(model)


@app.patch("/api/v1/projects/{project_id}/models/{model_id}")
def update_model(project_id: str, model_id: str, payload: ModelUpdate, session: Session = Depends(get_session)):
    require_project(session, project_id)
    model = session.get(ModelVersion, model_id)
    if not model or model.project_id != project_id:
        raise HTTPException(404, "Model not found")
    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(model, key, value)
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
    model.status = "experimental" if model.status == "archived" else "archived"
    session.commit(); session.refresh(model)
    return as_dict(model)


@app.post("/api/v1/projects/{project_id}/inference-preview")
def inference_preview(project_id: str, payload: InferencePreviewRequest, session: Session = Depends(get_session)):
    require_project(session, project_id)
    model = session.get(ModelVersion, payload.model_id)
    if not model or model.project_id != project_id:
        raise HTTPException(422, "Model must belong to this project")
    try:
        if model.format == "onnx" and model.artifact_path:
            profile = model.metadata_json.get("onnx_profile")
            if not profile:
                raise ValueError("Model metadata must include an explicit onnx_profile")
            return infer_onnx(model.artifact_path, payload.image_path, profile)
        if model.format == "mmdetection-pytorch" and model.artifact_path and model.config_path:
            return infer_mmdetection(model.config_path, model.artifact_path, payload.image_path)
        if model.format == "mmdeploy" and model.artifact_path:
            return infer_mmdeploy(model.artifact_path, payload.image_path, model.metadata_json.get("mmdeploy_profile", {}))
        raise ValueError("Inference preview requires an ONNX profile, a MMDetection config/checkpoint bundle, or an MMDeploy model directory")
    except (OSError, RuntimeError, ValueError) as error:
        raise HTTPException(422, str(error)) from error


@app.get("/api/v1/projects/{project_id}/runs")
def list_runs(project_id: str, session: Session = Depends(get_session)):
    require_project(session, project_id)
    return [as_dict(x) for x in session.scalars(select(Run).where(Run.project_id == project_id).order_by(Run.created_at.desc())).all()]


@app.post("/api/v1/projects/{project_id}/runs", status_code=201)
def create_run(project_id: str, payload: RunCreate, session: Session = Depends(get_session)):
    require_project(session, project_id)
    for entity, label in ((session.get(DatasetVersion, payload.dataset_id) if payload.dataset_id else None, "Dataset"), (session.get(ModelVersion, payload.model_id) if payload.model_id else None, "Model"), (session.get(Run, payload.parent_run_id) if payload.parent_run_id else None, "Parent run")):
        if entity and entity.project_id != project_id:
            raise HTTPException(422, f"{label} must belong to this project")
        if (payload.dataset_id and label == "Dataset" or payload.model_id and label == "Model" or payload.parent_run_id and label == "Parent run") and not entity:
            raise HTTPException(422, f"{label} not found")
    run = Run(project_id=project_id, **payload.model_dump())
    session.add(run); session.commit(); session.refresh(run)
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
    parent_run_id = manifest.get("parent_run_id")
    if parent_run_id and (not (parent := session.get(Run, parent_run_id)) or parent.project_id != project_id):
        raise HTTPException(422, "Result manifest parent_run_id does not belong to this project")
    fingerprint = manifest_hash(manifest)
    existing = session.scalar(select(Run).where(Run.project_id == project_id, Run.external_run_id == manifest["external_run_id"]))
    if existing:
        if existing.import_hash == fingerprint:
            return {"status": "existing", "run": as_dict(existing)}
        raise HTTPException(409, "An external run with this ID exists but its manifest content differs")
    run = Run(
        project_id=project_id, kind=manifest["kind"], name=manifest["name"], status=manifest.get("status", "completed"),
        dataset_id=dataset_id, model_id=model_id, parent_run_id=parent_run_id,
        external_run_id=manifest["external_run_id"], import_hash=fingerprint,
        config={**manifest.get("config", {}), **({"target_profile_id": target_profile_id} if target_profile_id else {})}, metrics=manifest.get("metrics", {}), environment=manifest.get("environment", {}), notes=manifest.get("notes", ""),
    )
    session.add(run); session.commit(); session.refresh(run)
    return {"status": "created", "run": as_dict(run)}


@app.post("/api/v1/projects/{project_id}/evaluations/predictions", status_code=201)
def evaluate_predictions(project_id: str, payload: PredictionEvaluationCreate, session: Session = Depends(get_session)):
    require_project(session, project_id)
    dataset = session.get(DatasetVersion, payload.dataset_id)
    model = session.get(ModelVersion, payload.model_id)
    if not dataset or dataset.project_id != project_id or not model or model.project_id != project_id:
        raise HTTPException(422, "Model and dataset must belong to this project")
    if dataset.format != "coco" or not dataset.annotation_path:
        raise HTTPException(422, "Prediction evaluation currently requires a COCO dataset with annotation_path")
    if dataset.content_hash and dataset.content_hash.startswith("sha256:"):
        current_hash, _ = dataset_fingerprint(dataset.manifest_path, dataset.annotation_path)
        if current_hash != dataset.content_hash:
            raise HTTPException(409, "Dataset source files changed since this version was registered; create a new DatasetVersion")
    try:
        import json
        from pathlib import Path
        with Path(payload.predictions_path).open(encoding="utf-8") as file:
            predictions = json.load(file)
        if not isinstance(predictions, list):
            raise ValueError("Prediction JSON must be a list")
        if payload.protocol == "onboarding_ap50":
            result = evaluate_coco_predictions(dataset.annotation_path, predictions, payload.iou_threshold)
        elif payload.protocol == "coco_full":
            result = evaluate_coco_full(dataset.annotation_path, predictions)
        else:
            raise ValueError("protocol must be onboarding_ap50 or coco_full")
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        raise HTTPException(422, str(error)) from error
    run = Run(
        project_id=project_id, kind="evaluation", name=f"{model.family} prediction import", status="completed",
        dataset_id=dataset.id, model_id=model.id,
        config={"evaluator_version": "coco-full-v1" if payload.protocol == "coco_full" and payload.evaluator_version == "lifecycle-ap50-v1" else payload.evaluator_version, "protocol": payload.protocol, "iou_threshold": payload.iou_threshold, "scope": result.pop("metric_scope")},
        metrics={key: value for key, value in result.items() if isinstance(value, (int, float))},
        environment={"source": "external-prediction-json"}, notes="Per-class output retained in evaluation import result.",
    )
    session.add(run); session.commit(); session.refresh(run)
    return {"run": as_dict(run), "result": result}


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
    try:
        result = evaluate_classification(payload.records)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    metric_keys = {"top1_accuracy", "topk_accuracy", "macro_precision", "macro_recall", "macro_f1", "records"}
    run = Run(
        project_id=project_id, kind="evaluation", name=f"{model.family} classification evaluation", status="completed",
        dataset_id=payload.dataset_id, model_id=model.id,
        config={"evaluator_version": payload.evaluator_version, "task_kind": "classification"},
        metrics={key: result[key] for key in metric_keys}, environment={"source": "external-classification-records"},
        notes="Per-class and confusion matrix output is returned by this import response.",
    )
    session.add(run); session.commit(); session.refresh(run)
    return {"run": as_dict(run), "result": result}


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
    job = Job(project_id=project_id, runner_id=payload.runner_id, command=command, input_json=payload.input_json, status="queued")
    session.add(job); session.commit(); session.refresh(job)
    launch(job, payload.args)
    return as_dict(job)


@app.post("/api/v1/projects/{project_id}/jobs/{job_id}/cancel")
def cancel_job(project_id: str, job_id: str, session: Session = Depends(get_session)):
    job = session.get(Job, job_id)
    if not job or job.project_id != project_id:
        raise HTTPException(404, "Job not found")
    if job.status not in {"queued", "running"}:
        raise HTTPException(409, f"Job cannot be cancelled from {job.status}")
    if cancel(job_id):
        job.status = "cancelling"
        session.commit()
        return as_dict(job)
    job.status = "cancelled"
    job.log = f"{job.log}Cancelled before process start.\n"
    session.commit(); session.refresh(job)
    return as_dict(job)


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
    if payload.baseline_model_id:
        baseline_run = session.scalar(select(Run).where(Run.project_id == project_id, Run.model_id == payload.baseline_model_id, Run.kind == "evaluation", Run.dataset_id == (evaluation.dataset_id if evaluation else None)).order_by(Run.created_at.desc()))
        baseline_metrics = baseline_run.metrics if baseline_run else None
    try:
        result = evaluate_gate(evaluation.metrics if evaluation else None, baseline_metrics, payload.gate_config)
    except GateConfigError as error:
        raise HTTPException(422, str(error)) from error
    release = Release(project_id=project_id, **payload.model_dump(), gate_result=result, decision=result["status"])
    session.add(release); session.commit(); session.refresh(release)
    return as_dict(release)


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
