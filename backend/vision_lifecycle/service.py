from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from .audit import record_audit
from .importer import manifest_hash, validate_result_manifest
from .models import Artifact, AuditEvent, BoardBenchmark, CalibrationSetVersion, DataAsset, DatasetVersion, EvaluationSetVersion, FieldDataBatch, Job, LabelSchemaVersion, ModelAliasHistory, ModelVersion, Project, QuantizationRun, Release, ReleaseEvidence, Run, SplitVersion, StorageMapping, TargetProfile
from .dataset_snapshot import build_dataset_snapshot
from .serializers import as_dict


class ResultManifestConflict(ValueError):
    """Raised when an external run ID is reused with different content."""


class ResultManifestReferenceError(ValueError):
    """Raised when a manifest points outside its project or to a missing entity."""


class DatasetDeletionConflict(ValueError):
    """Raised when a draft dataset is finalized or still referenced."""

    def __init__(self, message: str, dependencies: list[dict] | None = None):
        super().__init__(message)
        self.dependencies = dependencies or []


def delete_draft_dataset(session: Session, project_id: str, dataset_id: str) -> dict:
    """Remove an unreferenced draft dataset from the registry, preserving files."""
    dataset = session.get(DatasetVersion, dataset_id)
    if not dataset or dataset.project_id != project_id:
        raise LookupError("Dataset not found")
    if dataset.status != "draft":
        raise DatasetDeletionConflict("Only a draft DatasetVersion can be deleted; archive or create a new version instead")
    dependencies: list[dict] = []
    models = session.scalars(select(ModelVersion).where(ModelVersion.project_id == project_id, ModelVersion.source_dataset_id == dataset_id)).all()
    runs = session.scalars(select(Run).where(Run.project_id == project_id, Run.dataset_id == dataset_id)).all()
    for entity_type, entity in (("label_schema", LabelSchemaVersion), ("split", SplitVersion), ("evaluation_set", EvaluationSetVersion), ("calibration_set", CalibrationSetVersion)):
        dependencies.extend({"kind": entity_type, "id": item.id, "name": item.name} for item in session.scalars(select(entity).where(entity.project_id == project_id, entity.dataset_id == dataset_id)).all())
    dependencies.extend({"kind": "model", "id": item.id, "name": f"{item.name} {item.version}"} for item in models)
    dependencies.extend({"kind": "run", "id": item.id, "name": item.name} for item in runs)
    if dependencies:
        raise DatasetDeletionConflict("DatasetVersion is referenced and cannot be deleted", dependencies)
    artifacts = session.scalars(select(Artifact).where(Artifact.project_id == project_id, Artifact.owner_type == "dataset", Artifact.owner_id == dataset.id)).all()
    record_audit(session, project_id, "dataset", dataset.id, "deleted", before={"name": dataset.name, "version": dataset.version, "status": dataset.status}, details={"source_files_preserved": True, "artifact_count": len(artifacts)})
    for artifact in artifacts:
        session.delete(artifact)
    session.delete(dataset)
    session.commit()
    return {"deleted": True, "id": dataset_id, "source_files_preserved": True}


def import_result_manifest(session: Session, project_id: str, value: dict) -> dict:
    """Validate and persist an external result through the shared service path."""
    manifest = validate_result_manifest(value)
    references = (
        ("dataset_id", DatasetVersion),
        ("model_id", ModelVersion),
        ("target_profile_id", TargetProfile),
        ("quantization_run_id", QuantizationRun),
        ("calibration_set_id", CalibrationSetVersion),
        ("board_benchmark_id", BoardBenchmark),
        ("parent_run_id", Run),
    )
    for field, entity_type in references:
        identifier = manifest.get(field)
        if identifier:
            entity = session.get(entity_type, identifier)
            if not entity or entity.project_id != project_id:
                raise ResultManifestReferenceError(f"{field} does not belong to this project")
    fingerprint = manifest_hash(manifest)
    existing = session.scalar(select(Run).where(Run.project_id == project_id, Run.external_run_id == manifest["external_run_id"]))
    if existing:
        if existing.import_hash == fingerprint:
            return {"status": "existing", "run": as_dict(existing)}
        raise ResultManifestConflict("An external run with this ID exists but its manifest content differs")
    linked_config = {key: value for key, value in (("target_profile_id", manifest.get("target_profile_id")), ("quantization_run_id", manifest.get("quantization_run_id")), ("calibration_set_id", manifest.get("calibration_set_id")), ("board_benchmark_id", manifest.get("board_benchmark_id"))) if value}
    run = Run(
        project_id=project_id,
        kind=manifest["kind"],
        name=manifest["name"],
        status=manifest.get("status", "completed"),
        dataset_id=manifest.get("dataset_id"),
        model_id=manifest.get("model_id"),
        parent_run_id=manifest.get("parent_run_id"),
        external_run_id=manifest["external_run_id"],
        import_hash=fingerprint,
        config={**manifest.get("config", {}), **linked_config},
        metrics=manifest.get("metrics", {}),
        details=dict(manifest.get("details", {})),
        environment=manifest.get("environment", {}),
        notes=manifest.get("notes", ""),
    )
    session.add(run)
    session.flush()
    record_audit(session, project_id, "run", run.id, "registered", after={"kind": run.kind, "name": run.name, "status": run.status, "dataset_id": run.dataset_id, "model_id": run.model_id})
    session.commit()
    session.refresh(run)
    return {"status": "created", "run": as_dict(run)}


def overview(session: Session, project_id: str) -> dict:
    project = session.get(Project, project_id)
    if not project:
        raise LookupError("Project not found")

    def scrub(value):
        if isinstance(value, dict):
            cleaned = {}
            for key, nested in value.items():
                lowered = str(key).lower()
                if any(token in lowered for token in ("path", "command", "environment", "secret", "token", "password", "credential")):
                    continue
                cleaned[key] = scrub(nested)
            return cleaned
        if isinstance(value, list):
            return [scrub(nested) for nested in value]
        if isinstance(value, str) and value.startswith("/"):
            return "[redacted-absolute-path]"
        return value
    datasets = session.scalars(select(DatasetVersion).where(DatasetVersion.project_id == project_id)).all()
    models = session.scalars(select(ModelVersion).where(ModelVersion.project_id == project_id)).all()
    runs = session.scalars(select(Run).where(Run.project_id == project_id).order_by(Run.created_at.desc())).all()
    jobs = session.scalars(select(Job).where(Job.project_id == project_id).order_by(Job.created_at.desc())).all()
    artifacts = session.scalars(select(Artifact).where(Artifact.project_id == project_id, Artifact.status != "superseded")).all()
    model_checks: list[dict] = []
    for model in models:
        owned = {item.kind for item in artifacts if item.owner_type == "model" and item.owner_id == model.id}
        checks = {
            "dataset": bool(model.source_dataset_id),
            "training_run": bool(model.source_run_id and any(run.id == model.source_run_id and run.kind == "training" for run in runs)),
            "model_artifact": bool(model.artifact_path or "model" in owned),
            "config": bool(model.config_path or "config" in owned),
            "profile": bool(model.format not in {"onnx", "mmdeploy"} or model.metadata_json.get("onnx_profile") or model.metadata_json.get("mmdeploy_profile")),
        }
        model_checks.append({"model_id": model.id, "name": f"{model.family} {model.version}", "checks": checks, "missing": [key for key, value in checks.items() if not value]})
    incomplete = sum(bool(item["missing"]) for item in model_checks)
    model_completeness = 0 if not model_checks else round((len(model_checks) - incomplete) / len(model_checks) * 100)
    storage_ready = any(item.status == "available" for item in session.scalars(select(StorageMapping).where(StorageMapping.project_id == project_id)).all())
    finalized_dataset = any(item.status == "finalized" for item in datasets)
    completed_evaluation = any(item.kind == "evaluation" and item.status == "completed" for item in runs)
    readiness_checks = [
        {"id": "storage", "label": "Storage mapping", "status": "ready" if storage_ready else "missing"},
        {"id": "dataset", "label": "Finalized DatasetVersion", "status": "ready" if finalized_dataset else "missing"},
        {"id": "model", "label": "Model provenance", "status": "ready" if model_checks and not incomplete else "partial" if model_checks else "missing", "models": model_checks},
        {"id": "evaluation", "label": "Completed evaluation", "status": "ready" if completed_evaluation else "missing"},
    ]
    ready_count = sum(item["status"] == "ready" for item in readiness_checks)
    readiness = {"status": "ready" if ready_count == len(readiness_checks) else "partial" if ready_count else "not_started", "ready_count": ready_count, "total": len(readiness_checks), "checks": readiness_checks}
    return {
        "project": project,
        "counts": {"datasets": len(datasets), "models": len(models), "runs": len(runs), "jobs": len(jobs)},
        "baseline": next((m for m in models if m.alias == "baseline"), None),
        "candidate": next((m for m in models if m.alias == "candidate"), None),
        "lineage_completeness": model_completeness,
        "readiness": readiness,
        "recent_runs": runs[:5],
        "next_actions": ([
            "1단계: 이미지·annotation 또는 기존 manifest 경로를 연결하고 형식을 검사하세요.",
            "2단계: class mapping과 evaluation set을 확인한 뒤 DatasetVersion을 확정하세요.",
            "3단계: 외부 학습 결과와 모델 config를 직접 선택해 연결하세요.",
        ] if not datasets else [
            "평가 세트를 확정하고 연결할 모델을 직접 선택하세요.",
            "모델의 config, class mapping, 전처리 profile을 확인한 뒤 샘플 추론을 실행하세요.",
            "동일 평가 세트에서 baseline과 candidate를 비교하세요.",
        ]),
    }


def compare_models(session: Session, baseline_id: str, candidate_id: str) -> dict:
    baseline = session.get(ModelVersion, baseline_id)
    candidate = session.get(ModelVersion, candidate_id)
    if not baseline or not candidate:
        raise LookupError("Model not found")
    base_runs = session.scalars(select(Run).where(Run.model_id == baseline_id, Run.kind == "evaluation", Run.status == "completed").order_by(Run.created_at.desc())).all()
    cand_runs = session.scalars(select(Run).where(Run.model_id == candidate_id, Run.kind == "evaluation", Run.status == "completed").order_by(Run.created_at.desc())).all()
    mapping_matches = bool(
        baseline.metadata_json.get("class_mapping_version")
        and baseline.metadata_json.get("class_mapping_version") == candidate.metadata_json.get("class_mapping_version")
    )
    # Choose the newest pair that shares the complete evaluation contract. A
    # newer failed/incompatible run must not hide an older reproducible pair.
    base = None
    cand = None
    for base_candidate in base_runs:
        for cand_candidate in cand_runs:
            if (
                base_candidate.dataset_id == cand_candidate.dataset_id
                and base_candidate.config == cand_candidate.config
                and base_candidate.config.get("evaluator_version")
                and base_candidate.config.get("evaluator_version") == cand_candidate.config.get("evaluator_version")
                and mapping_matches
            ):
                base, cand = base_candidate, cand_candidate
                break
        if base and cand:
            break
    if not base and base_runs:
        base = base_runs[0]
    if not cand and cand_runs:
        cand = cand_runs[0]
    compatible = bool(base and cand and base.dataset_id == cand.dataset_id and base.config == cand.config and base.config.get("evaluator_version") and base.config.get("evaluator_version") == cand.config.get("evaluator_version") and mapping_matches)
    delta = {}
    metric_compatibility: dict[str, str] = {}
    if compatible:
        # Accuracy shares dataset/evaluator/class mapping compatibility. Runtime
        # measurements need their own measurement contract; a CPU timing must
        # never be compared with a board timing merely because model lineage
        # happens to match.
        latency_contract_keys = ("target_profile_id", "batch_size", "warmup_runs", "measurement_scope")
        latency_contract_matches = all(
            base.config.get(key) == cand.config.get(key) and base.config.get(key) is not None
            for key in latency_contract_keys
        ) and base.environment.get("runtime") == cand.environment.get("runtime") and base.environment.get("runtime") is not None
        for metric, value in cand.metrics.items():
            if metric in base.metrics and isinstance(value, (int, float)):
                if metric.startswith("latency_") and not latency_contract_matches:
                    metric_compatibility[metric] = "target profile, runtime, batch, warm-up, measurement scope가 일치하지 않습니다."
                    continue
                delta[metric] = round(value - base.metrics[metric], 4)
                metric_compatibility[metric] = "compatible"
    return {
        "baseline": baseline,
        "candidate": candidate,
        "baseline_evaluation": base,
        "candidate_evaluation": cand,
        "compatible": compatible,
        "reason": None if compatible else "완료된 평가 중 동일 dataset version, evaluator/protocol 설정, class mapping version의 쌍이 필요합니다.",
        "delta": delta,
        "metric_compatibility": metric_compatibility,
    }


def agent_request(session: Session, project_id: str) -> dict:
    project = session.get(Project, project_id)
    if not project:
        raise LookupError("Project not found")
    datasets = session.scalars(select(DatasetVersion).where(DatasetVersion.project_id == project_id)).all()
    models = session.scalars(select(ModelVersion).where(ModelVersion.project_id == project_id)).all()
    missing: list[str] = []
    if not datasets:
        missing.append("등록된 DatasetVersion이 없습니다.")
    if not models:
        missing.append("등록된 ModelVersion이 없습니다.")
    for model in models:
        if not model.config_path:
            missing.append(f"{model.name} {model.version}: config 경로가 없습니다.")
        if not model.source_dataset_id:
            missing.append(f"{model.name} {model.version}: 학습 데이터 lineage가 없습니다.")
    supplied_paths = {
        "storage_root": project.storage_root,
        "datasets": [{"name": item.name, "version": item.version, "annotation_path": item.annotation_path, "manifest_path": item.manifest_path} for item in datasets],
        "models": [{"name": item.name, "version": item.version, "artifact_path": item.artifact_path, "config_path": item.config_path} for item in models],
    }
    prompt = (
        "skills/vision-lifecycle-onboard/SKILL.md를 사용해서 이 Vision AI 프로젝트를 lifecycle registry에 연결해줘.\n"
        f"프로젝트: {project.name} ({project.task_kind})\n"
        f"제공된 경로와 자료: {supplied_paths}\n"
        f"확인할 누락 정보: {missing or ['없음']}\n"
        "파일을 먼저 조사하고, class mapping과 provenance를 추정하지 말고, 샘플 검증 뒤 등록 결과와 남은 제한을 보고해줘."
    )
    return {"prompt": prompt, "missing": missing, "paths": supplied_paths}


def lineage(session: Session, project_id: str) -> dict:
    project = session.get(Project, project_id)
    if not project:
        raise LookupError("Project not found")
    datasets = session.scalars(select(DatasetVersion).where(DatasetVersion.project_id == project_id)).all()
    label_schemas = session.scalars(select(LabelSchemaVersion).where(LabelSchemaVersion.project_id == project_id)).all()
    models = session.scalars(select(ModelVersion).where(ModelVersion.project_id == project_id)).all()
    runs = session.scalars(select(Run).where(Run.project_id == project_id)).all()
    quants = session.scalars(select(QuantizationRun).where(QuantizationRun.project_id == project_id)).all()
    boards = session.scalars(select(BoardBenchmark).where(BoardBenchmark.project_id == project_id)).all()
    artifacts = session.scalars(select(Artifact).where(Artifact.project_id == project_id)).all()
    field_batches = session.scalars(select(FieldDataBatch).where(FieldDataBatch.project_id == project_id)).all()
    releases = session.scalars(select(Release).where(Release.project_id == project_id)).all()
    release_ids = [item.id for item in releases]
    evidences = session.scalars(select(ReleaseEvidence).where(ReleaseEvidence.project_id == project_id)).all()
    targets = session.scalars(select(TargetProfile).where(TargetProfile.project_id == project_id)).all()
    nodes = [{"id": project.id, "kind": "project", "label": project.name}]
    nodes += [{"id": item.id, "kind": "dataset", "label": f"{item.name} {item.version}", "status": item.status} for item in datasets]
    nodes += [{"id": item.id, "kind": "label-schema", "label": f"{item.name} {item.version}", "status": item.status} for item in label_schemas]
    nodes += [{"id": item.id, "kind": "model", "label": f"{item.family} {item.version}", "status": item.status} for item in models]
    nodes += [{"id": item.id, "kind": "run", "label": item.name, "status": item.status} for item in runs]
    nodes += [{"id": item.id, "kind": "quantization", "label": item.name, "status": item.status} for item in quants]
    nodes += [{"id": item.id, "kind": "board", "label": item.name, "status": item.status} for item in boards]
    nodes += [{"id": item.id, "kind": "target", "label": f"{item.name} {item.version}"} for item in targets]
    nodes += [{"id": item.id, "kind": "release", "label": item.name, "status": item.decision} for item in releases]
    nodes += [{"id": item.id, "kind": "release-evidence", "label": f"{item.evidence_type}: {item.source_id}", "status": item.status} for item in evidences]
    nodes += [{"id": item.id, "kind": "artifact", "label": item.logical_name, "status": item.status} for item in artifacts]
    nodes += [{"id": item.id, "kind": "field-batch", "label": item.name, "status": item.status} for item in field_batches]
    edges = []
    for dataset in datasets:
        edges.append({"source": project.id, "target": dataset.id, "relation": "contains"})
    for label_schema in label_schemas:
        edges.append({"source": project.id, "target": label_schema.id, "relation": "contains"})
        if label_schema.dataset_id:
            edges.append({"source": label_schema.dataset_id, "target": label_schema.id, "relation": "defines_labels"})
        if label_schema.parent_label_schema_id:
            edges.append({"source": label_schema.parent_label_schema_id, "target": label_schema.id, "relation": "supersedes"})
    for model in models:
        edges.append({"source": project.id, "target": model.id, "relation": "contains"})
        if model.source_dataset_id:
            edges.append({"source": model.source_dataset_id, "target": model.id, "relation": "trained_from"})
        if model.source_run_id:
            edges.append({"source": model.source_run_id, "target": model.id, "relation": "produced_by"})
    for run in runs:
        if run.model_id:
            edges.append({"source": run.model_id, "target": run.id, "relation": run.kind})
        if run.dataset_id:
            edges.append({"source": run.dataset_id, "target": run.id, "relation": "evaluated_on"})
        for key, relation in (("quantization_run_id", "result_of"), ("calibration_set_id", "uses_calibration"), ("board_benchmark_id", "summarizes")):
            reference_id = (run.config or {}).get(key)
            if reference_id:
                edges.append({"source": reference_id, "target": run.id, "relation": relation})
    for quant in quants:
        edges.append({"source": quant.source_model_id, "target": quant.id, "relation": "quantized"})
        if quant.output_model_id:
            edges.append({"source": quant.id, "target": quant.output_model_id, "relation": "produced"})
        if quant.calibration_set_id:
            edges.append({"source": quant.calibration_set_id, "target": quant.id, "relation": "calibrated_by"})
    for board in boards:
        edges.append({"source": board.model_id, "target": board.id, "relation": "measured"})
        edges.append({"source": board.target_profile_id, "target": board.id, "relation": "on_target"})
        if board.evaluation_run_id:
            edges.append({"source": board.evaluation_run_id, "target": board.id, "relation": "validated_by"})
    for release in releases:
        edges.append({"source": release.model_id, "target": release.id, "relation": "released"})
        if release.evaluation_run_id:
            edges.append({"source": release.evaluation_run_id, "target": release.id, "relation": "evidence"})
        if release.baseline_model_id:
            edges.append({"source": release.baseline_model_id, "target": release.id, "relation": "baseline"})
    for evidence in evidences:
        if evidence.release_id in release_ids:
            edges.append({"source": evidence.id, "target": evidence.release_id, "relation": "evidence"})
    for artifact in artifacts:
        owner_id = artifact.owner_id or project.id
        if artifact.owner_id:
            edges.append({"source": owner_id, "target": artifact.id, "relation": "has_artifact"})
        else:
            edges.append({"source": project.id, "target": artifact.id, "relation": "contains"})
    for batch in field_batches:
        if batch.source_model_id:
            edges.append({"source": batch.source_model_id, "target": batch.id, "relation": "field_from_model"})
        if batch.source_dataset_id:
            edges.append({"source": batch.source_dataset_id, "target": batch.id, "relation": "field_from_dataset"})
        if batch.candidate_dataset_id:
            edges.append({"source": batch.id, "target": batch.candidate_dataset_id, "relation": "candidate_dataset"})
    return {"project_id": project_id, "nodes": nodes, "edges": edges}


def safe_export(session: Session, project_id: str) -> dict:
    """Create a Pages-safe, portable registry snapshot shared by API and CLI."""
    from .serializers import as_dict

    project = session.get(Project, project_id)
    if not project:
        raise LookupError("Project not found")

    def scrub(value):
        if isinstance(value, dict):
            cleaned = {}
            for key, nested in value.items():
                lowered = str(key).lower()
                if any(token in lowered for token in ("path", "command", "environment", "secret", "token", "password", "credential")):
                    continue
                cleaned[key] = scrub(nested)
            return cleaned
        if isinstance(value, list):
            return [scrub(nested) for nested in value]
        if isinstance(value, str) and value.startswith("/"):
            return "[redacted-absolute-path]"
        return value

    def safe(item):
        data = as_dict(item)
        for key in {"storage_root", "root_path", "source_path", "managed_path", "artifact_path", "config_path", "annotation_path", "manifest_path", "raw_output_path", "command", "environment_names", "working_directory"}:
            data.pop(key, None)
        # A dataset fingerprint is useful for tamper detection, but its internal
        # detail keys are source paths.  Pages exports must stay portable and
        # must not disclose machine/NAS layouts.
        if isinstance(data.get("validation"), dict):
            validation = dict(data["validation"])
            fingerprints = validation.pop("source_fingerprints", None)
            if fingerprints:
                validation["source_fingerprint_count"] = len(fingerprints)
            data["validation"] = validation
        if isinstance(data.get("last_validation"), dict):
            data["last_validation"] = {
                key: value for key, value in data["last_validation"].items()
                if key in {"status", "readable", "writable", "reason"}
            }
        for nested_key in ("details", "snapshot", "before_json", "after_json"):
            if isinstance(data.get(nested_key), (dict, list)):
                data[nested_key] = scrub(data[nested_key])
        return data

    summary = overview(session, project_id)
    overview_export = {
        "counts": summary["counts"],
        "lineage_completeness": summary["lineage_completeness"],
        "readiness": summary.get("readiness", {}),
        "baseline": safe(summary["baseline"]) if summary["baseline"] else None,
        "candidate": safe(summary["candidate"]) if summary["candidate"] else None,
        "recent_runs": [safe(item) for item in summary["recent_runs"]],
        "next_actions": summary["next_actions"],
    }
    return {
        "schema_version": "1.1", "generated_at": datetime.now(UTC).isoformat(), "project": safe(project), "overview": overview_export,
        "datasets": [safe(x) for x in session.scalars(select(DatasetVersion).where(DatasetVersion.project_id == project_id)).all()],
        "models": [safe(x) for x in session.scalars(select(ModelVersion).where(ModelVersion.project_id == project_id)).all()],
        "model_alias_history": [safe(x) for x in session.scalars(select(ModelAliasHistory).where(ModelAliasHistory.project_id == project_id).order_by(ModelAliasHistory.created_at.desc())).all()],
        "runs": [safe(x) for x in session.scalars(select(Run).where(Run.project_id == project_id)).all()],
        "storages": [safe(x) for x in session.scalars(select(StorageMapping).where(StorageMapping.project_id == project_id)).all()],
        "assets": [safe(x) for x in session.scalars(select(DataAsset).where(DataAsset.project_id == project_id)).all()],
        "field_batches": [safe(x) for x in session.scalars(select(FieldDataBatch).where(FieldDataBatch.project_id == project_id)).all()],
        "artifacts": [safe(x) for x in session.scalars(select(Artifact).where(Artifact.project_id == project_id)).all()],
        "label_schemas": [safe(x) for x in session.scalars(select(LabelSchemaVersion).where(LabelSchemaVersion.project_id == project_id)).all()],
        "splits": [safe(x) for x in session.scalars(select(SplitVersion).where(SplitVersion.project_id == project_id)).all()],
        "evaluation_sets": [safe(x) for x in session.scalars(select(EvaluationSetVersion).where(EvaluationSetVersion.project_id == project_id)).all()],
        "calibration_sets": [safe(x) for x in session.scalars(select(CalibrationSetVersion).where(CalibrationSetVersion.project_id == project_id)).all()],
        "quantization_runs": [safe(x) for x in session.scalars(select(QuantizationRun).where(QuantizationRun.project_id == project_id)).all()],
        "board_benchmarks": [safe(x) for x in session.scalars(select(BoardBenchmark).where(BoardBenchmark.project_id == project_id)).all()],
        "targets": [safe(x) for x in session.scalars(select(TargetProfile).where(TargetProfile.project_id == project_id)).all()],
        "releases": [safe(x) for x in session.scalars(select(Release).where(Release.project_id == project_id)).all()],
        "release_evidence": [safe(x) for x in session.scalars(select(ReleaseEvidence).where(ReleaseEvidence.project_id == project_id)).all()],
        "audit_events": [safe(x) for x in session.scalars(select(AuditEvent).where(AuditEvent.project_id == project_id).order_by(AuditEvent.created_at.desc())).all()],
        "redactions": ["storage_root", "root_path", "source_path", "managed_path", "artifact_path", "config_path", "annotation_path", "manifest_path", "raw_output_path", "command", "environment_names", "working_directory"],
    }


def seed_demo(session: Session) -> Project:
    existing = session.scalar(select(Project).where(Project.name == "MMDetection RTMDet · YOLOX Demo"))
    if existing:
        return existing
    project = Project(
        name="MMDetection RTMDet · YOLOX Demo",
        description="공통 COCO subset에서 RTMDet-tiny와 YOLOX-s를 비교하는 온보딩 예제",
        task_kind="detection",
        storage_root="examples/mmdetection/assets",
        git_url="https://github.com/open-mmlab/mmdetection",
        default_branch="v3.3.0",
    )
    session.add(project)
    session.flush()
    dataset = DatasetVersion(
        project_id=project.id,
        name="coco-mini-lifecycle-demo",
        version="v1",
        task_kind="detection",
        format="coco",
        manifest_path="examples/mmdetection/dataset.json",
        annotation_path="examples/mmdetection/annotations/coco8.json",
        content_hash="demo-coco-mini-v1",
        snapshot=build_dataset_snapshot(task_kind="detection", format="coco", manifest_path="examples/mmdetection/dataset.json", annotation_path="examples/mmdetection/annotations/coco8.json"),
        sample_count=2,
        class_names=["person", "bicycle", "car"],
        status="finalized",
        validation={"status": "passed", "images": 2, "annotations": 3, "missing_labels": 0},
    )
    session.add(dataset)
    session.flush()
    rtm = ModelVersion(
        project_id=project.id, name="RTMDet", version="tiny-coco-demo", family="RTMDet-tiny",
        task_kind="detection", format="mmdetection-pytorch", precision="fp32",
        artifact_path="artifacts/rtmdet_tiny.pth", config_path="configs/rtmdet_tiny_8xb32-300e_coco.py",
        source_dataset_id=dataset.id, alias="candidate", runnable=False,
        metadata_json={"framework": "MMDetection", "framework_version": "3.3.0", "runner": "mmdetection-native", "class_mapping_version": "coco-mini-v1"},
    )
    yolox = ModelVersion(
        project_id=project.id, name="YOLOX", version="s-coco-demo", family="YOLOX-s",
        task_kind="detection", format="mmdetection-pytorch", precision="fp32",
        artifact_path="artifacts/yolox_s.pth", config_path="configs/yolox_s_8xb8-300e_coco.py",
        source_dataset_id=dataset.id, alias="baseline", runnable=False,
        metadata_json={"framework": "MMDetection", "framework_version": "3.3.0", "runner": "mmdetection-native", "class_mapping_version": "coco-mini-v1"},
    )
    session.add_all([rtm, yolox])
    session.flush()
    for model in (rtm, yolox):
        session.add(Artifact(project_id=project.id, kind="model", logical_name=f"{model.name}/{model.version}/artifact", owner_type="model", owner_id=model.id, source_path=model.artifact_path, status="unavailable", notes="공식 checkpoint는 별도 준비 script로 연결합니다."))
        session.add(Artifact(project_id=project.id, kind="config", logical_name=f"{model.name}/{model.version}/config", owner_type="model", owner_id=model.id, source_path=model.config_path, status="unavailable", notes="MMDetection config 예시 경로"))
    for model, map_value, ap50, latency in [(rtm, 0.412, 0.621, 11.8), (yolox, 0.398, 0.607, 14.3)]:
        session.add(Run(
            project_id=project.id, kind="evaluation", name=f"{model.family} COCO8 evaluation", status="completed",
            dataset_id=dataset.id, model_id=model.id,
            config={"evaluator_version": "coco-v1", "iou": "0.50:0.95", "samples": 8},
            metrics={"bbox_mAP": map_value, "bbox_AP50": ap50, "latency_ms_p50": latency},
            environment={"provider": "CPUExecutionProvider", "source": "demo"},
            notes="온보딩 fixture: 전체 COCO benchmark 결과가 아닙니다.",
        ))
    session.commit()
    return project
