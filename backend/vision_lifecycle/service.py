from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import CalibrationSetVersion, DataAsset, DatasetVersion, EvaluationSetVersion, Job, LabelSchemaVersion, ModelVersion, Project, Release, Run, SplitVersion, StorageMapping, TargetProfile


def overview(session: Session, project_id: str) -> dict:
    project = session.get(Project, project_id)
    if not project:
        raise LookupError("Project not found")
    datasets = session.scalars(select(DatasetVersion).where(DatasetVersion.project_id == project_id)).all()
    models = session.scalars(select(ModelVersion).where(ModelVersion.project_id == project_id)).all()
    runs = session.scalars(select(Run).where(Run.project_id == project_id).order_by(Run.created_at.desc())).all()
    jobs = session.scalars(select(Job).where(Job.project_id == project_id).order_by(Job.created_at.desc())).all()
    incomplete = sum(not m.source_dataset_id or not m.config_path for m in models)
    return {
        "project": project,
        "counts": {"datasets": len(datasets), "models": len(models), "runs": len(runs), "jobs": len(jobs)},
        "baseline": next((m for m in models if m.alias == "baseline"), None),
        "candidate": next((m for m in models if m.alias == "candidate"), None),
        "lineage_completeness": 0 if not models else round((len(models) - incomplete) / len(models) * 100),
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
    base_runs = session.scalars(select(Run).where(Run.model_id == baseline_id, Run.kind == "evaluation").order_by(Run.created_at.desc())).all()
    cand_runs = session.scalars(select(Run).where(Run.model_id == candidate_id, Run.kind == "evaluation").order_by(Run.created_at.desc())).all()
    base = base_runs[0] if base_runs else None
    cand = cand_runs[0] if cand_runs else None
    mapping_matches = bool(
        baseline.metadata_json.get("class_mapping_version")
        and baseline.metadata_json.get("class_mapping_version") == candidate.metadata_json.get("class_mapping_version")
    )
    compatible = bool(base and cand and base.dataset_id == cand.dataset_id and base.config.get("evaluator_version") == cand.config.get("evaluator_version") and mapping_matches)
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
        "reason": None if compatible else "동일 dataset version, evaluator version, class mapping version의 평가 결과가 필요합니다.",
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


def safe_export(session: Session, project_id: str) -> dict:
    """Create a Pages-safe, portable registry snapshot shared by API and CLI."""
    from .serializers import as_dict

    project = session.get(Project, project_id)
    if not project:
        raise LookupError("Project not found")

    def safe(item):
        data = as_dict(item)
        for key in {"storage_root", "root_path", "artifact_path", "config_path", "annotation_path", "manifest_path", "command", "environment_names", "working_directory"}:
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
        return data

    return {
        "schema_version": "1.0", "project": safe(project),
        "datasets": [safe(x) for x in session.scalars(select(DatasetVersion).where(DatasetVersion.project_id == project_id)).all()],
        "models": [safe(x) for x in session.scalars(select(ModelVersion).where(ModelVersion.project_id == project_id)).all()],
        "runs": [safe(x) for x in session.scalars(select(Run).where(Run.project_id == project_id)).all()],
        "storages": [safe(x) for x in session.scalars(select(StorageMapping).where(StorageMapping.project_id == project_id)).all()],
        "assets": [safe(x) for x in session.scalars(select(DataAsset).where(DataAsset.project_id == project_id)).all()],
        "label_schemas": [safe(x) for x in session.scalars(select(LabelSchemaVersion).where(LabelSchemaVersion.project_id == project_id)).all()],
        "splits": [safe(x) for x in session.scalars(select(SplitVersion).where(SplitVersion.project_id == project_id)).all()],
        "evaluation_sets": [safe(x) for x in session.scalars(select(EvaluationSetVersion).where(EvaluationSetVersion.project_id == project_id)).all()],
        "calibration_sets": [safe(x) for x in session.scalars(select(CalibrationSetVersion).where(CalibrationSetVersion.project_id == project_id)).all()],
        "targets": [safe(x) for x in session.scalars(select(TargetProfile).where(TargetProfile.project_id == project_id)).all()],
        "releases": [safe(x) for x in session.scalars(select(Release).where(Release.project_id == project_id)).all()],
        "redactions": ["storage_root", "root_path", "artifact_path", "config_path", "annotation_path", "manifest_path", "command", "environment_names", "working_directory"],
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
