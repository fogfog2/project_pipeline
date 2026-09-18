from __future__ import annotations

import argparse
import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

from .adapters.coco import validate_coco
from .adapters.inspect import inspect_path
from .artifacts import register_artifact
from .database import SessionLocal, engine, init_database
from .fingerprints import dataset_fingerprint, file_sha256
from .dataset_snapshot import build_dataset_snapshot
from .importer import manifest_hash, validate_result_manifest
from .models import Artifact, BoardBenchmark, CalibrationSetVersion, DatasetVersion, ModelVersion, Project, QuantizationRun, Run, StorageMapping, TargetProfile
from .runner import run_worker
from .serializers import as_dict
from .service import safe_export, seed_demo
from . import schemas as contract_schemas


def _json(value: object) -> None:
    print(json.dumps(value, default=str, ensure_ascii=False, indent=2))


def _read_json(path: str) -> dict:
    with Path(path).open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError("JSON input must be an object")
    return value


def _backup_manifest(session) -> dict:
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "projects": [{"id": item.id, "name": item.name} for item in session.scalars(select(Project)).all()],
        "storages": [{"id": item.id, "name": item.name, "status": item.status} for item in session.scalars(select(StorageMapping)).all()],
        "artifacts": [{"id": item.id, "kind": item.kind, "sha256": item.sha256, "status": item.status} for item in session.scalars(select(Artifact)).all()],
    }


def _verify_backup_manifest(db_path: Path, manifest_path: Path) -> dict:
    if not manifest_path.is_file():
        return {"status": "unverified", "reason": "backup manifest not found"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    connection = sqlite3.connect(str(db_path))
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise ValueError(f"Restored SQLite integrity check failed: {integrity}")
        project_ids = {row[0] for row in connection.execute("SELECT id FROM projects")}
        storage_ids = {row[0] for row in connection.execute("SELECT id FROM storage_mappings")}
        artifact_rows = {row[0]: row[1] for row in connection.execute("SELECT id, sha256 FROM artifacts")}
    finally:
        connection.close()
    expected_projects = {item["id"] for item in manifest.get("projects", [])}
    expected_storages = {item["id"] for item in manifest.get("storages", [])}
    expected_artifacts = {item["id"]: item.get("sha256") for item in manifest.get("artifacts", [])}
    if project_ids != expected_projects or storage_ids != expected_storages or artifact_rows != expected_artifacts:
        raise ValueError("Restored registry does not match the backup manifest IDs/hashes")
    return {"status": "verified", "projects": len(project_ids), "storages": len(storage_ids), "artifacts": len(artifact_rows)}


def main() -> None:
    parser = argparse.ArgumentParser(prog="visionops", description="Vision lifecycle registry CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="Create the local SQLite registry")
    sub.add_parser("demo", help="Load the RTMDet and YOLOX onboarding demo")
    sub.add_parser("projects", help="List registered projects")
    schema = sub.add_parser("schemas", help="Print versioned registry JSON Schemas")
    schema.add_argument("--output", help="Optional JSON file for the complete schema registry")
    inspect = sub.add_parser("inspect", help="Detect a dataset path without registering it")
    inspect.add_argument("path")
    validate = sub.add_parser("validate-coco", help="Validate a COCO annotation file")
    validate.add_argument("annotation_path")
    create_project = sub.add_parser("create-project", help="Register a project from a JSON object")
    create_project.add_argument("manifest")
    create_dataset = sub.add_parser("create-dataset", help="Register a DatasetVersion from a JSON object")
    create_dataset.add_argument("project_id")
    create_dataset.add_argument("manifest")
    create_model = sub.add_parser("create-model", help="Register a ModelVersion from a JSON object")
    create_model.add_argument("project_id")
    create_model.add_argument("manifest")
    import_result = sub.add_parser("import-result", help="Register an external result manifest")
    import_result.add_argument("project_id")
    import_result.add_argument("manifest", help="JSON result manifest")
    artifact = sub.add_parser("register-artifact", help="Register and hash a file provenance record")
    artifact.add_argument("project_id")
    artifact.add_argument("manifest", help="JSON object with kind, logical_name, and optional source_path")
    export = sub.add_parser("export", help="Write a Pages-safe project snapshot")
    export.add_argument("project_id")
    export.add_argument("--output", required=True, help="Destination JSON file")
    worker = sub.add_parser("worker", help="Run the external job worker")
    worker.add_argument("--once", action="store_true", help="Claim at most one queued job and exit")
    worker.add_argument("--poll-seconds", type=float, default=1.0)
    backup = sub.add_parser("backup", help="Create a SQLite registry backup")
    backup.add_argument("--output", required=True, help="Destination SQLite file")
    backup.add_argument("--manifest", help="Optional backup manifest JSON (defaults to <output>.manifest.json)")
    restore = sub.add_parser("restore", help="Restore the local registry from a SQLite backup")
    restore.add_argument("--input", required=True, help="Source SQLite backup file")
    restore.add_argument("--manifest", help="Optional backup manifest JSON (defaults to <input>.manifest.json)")
    args = parser.parse_args()

    init_database()
    if args.command == "inspect":
        _json(inspect_path(args.path)); return
    if args.command == "validate-coco":
        _json(validate_coco(args.annotation_path)); return
    if args.command == "schemas":
        registry = {
            "schema_version": "1.0",
            "schemas": {
                "project": contract_schemas.ProjectCreate.model_json_schema(),
                "dataset": contract_schemas.DatasetCreate.model_json_schema(),
                "model": contract_schemas.ModelCreate.model_json_schema(),
                "run": contract_schemas.RunCreate.model_json_schema(),
                "result_manifest": contract_schemas.ResultImportCreate.model_json_schema(),
                "release": contract_schemas.ReleaseCreate.model_json_schema(),
            },
        }
        if args.output:
            output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"Wrote schema registry: {output}")
        else:
            _json(registry)
        return
    if args.command == "worker":
        run_worker(poll_seconds=args.poll_seconds, once=args.once); return
    db_path = Path(os.environ.get("VISION_LIFECYCLE_DB", ".vision-lifecycle/registry.db"))
    if args.command == "backup":
        output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
        source = sqlite3.connect(str(db_path)); destination = sqlite3.connect(str(output))
        try:
            source.backup(destination)
        finally:
            destination.close(); source.close()
        manifest_path = Path(args.manifest or f"{output}.manifest.json")
        with SessionLocal() as session:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(_backup_manifest(session), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote registry backup: {output}")
        print(f"Wrote backup manifest: {manifest_path}")
        return
    if args.command == "restore":
        source_path = Path(args.input)
        if not source_path.is_file():
            raise ValueError(f"Backup file does not exist: {source_path}")
        engine.dispose(); db_path.parent.mkdir(parents=True, exist_ok=True)
        source = sqlite3.connect(str(source_path)); destination = sqlite3.connect(str(db_path))
        try:
            source.backup(destination)
        finally:
            destination.close(); source.close()
        init_database()
        manifest_path = Path(args.manifest or f"{source_path}.manifest.json")
        verification = _verify_backup_manifest(db_path, manifest_path)
        print(f"Restored registry backup: {source_path} ({verification['status']})")
        return
    with SessionLocal() as session:
        if args.command == "init":
            print("Registry initialized at .vision-lifecycle/registry.db")
        elif args.command == "demo":
            _json(as_dict(seed_demo(session)))
        elif args.command == "projects":
            _json([as_dict(p) for p in session.scalars(select(Project)).all()])
        elif args.command == "create-project":
            payload = _read_json(args.manifest)
            if session.scalar(select(Project).where(Project.name == payload.get("name"))):
                raise ValueError("A project with this name already exists")
            project = Project(**payload); session.add(project); session.commit(); session.refresh(project); _json(as_dict(project))
        elif args.command == "create-dataset":
            if not session.get(Project, args.project_id):
                raise ValueError("Project not found")
            payload = _read_json(args.manifest)
            content_hash, fingerprints = dataset_fingerprint(payload.get("manifest_path"), payload.get("annotation_path"))
            if fingerprints:
                payload["validation"] = {**payload.get("validation", {}), "source_fingerprints": fingerprints}
            payload["content_hash"] = content_hash
            payload["snapshot"] = build_dataset_snapshot(task_kind=payload.get("task_kind", "detection"), format=payload.get("format", "coco"), manifest_path=payload.get("manifest_path"), annotation_path=payload.get("annotation_path"))
            if not payload.get("class_names") and payload["snapshot"].get("class_names"):
                payload["class_names"] = payload["snapshot"]["class_names"]
            if not payload.get("sample_count") and payload["snapshot"].get("counts", {}).get("images"):
                payload["sample_count"] = payload["snapshot"]["counts"]["images"]
            dataset = DatasetVersion(project_id=args.project_id, **payload)
            session.add(dataset); session.flush()
            if payload.get("annotation_path"):
                register_artifact(session, args.project_id, kind="dataset-annotation", logical_name=f"{dataset.name}/{dataset.version}/annotation", owner_type="dataset", owner_id=dataset.id, source_path=payload["annotation_path"])
            if payload.get("manifest_path"):
                register_artifact(session, args.project_id, kind="dataset-manifest", logical_name=f"{dataset.name}/{dataset.version}/manifest", owner_type="dataset", owner_id=dataset.id, source_path=payload["manifest_path"])
            session.commit(); session.refresh(dataset); _json(as_dict(dataset))
        elif args.command == "create-model":
            if not session.get(Project, args.project_id):
                raise ValueError("Project not found")
            payload = _read_json(args.manifest)
            payload["artifact_sha256"] = file_sha256(payload["artifact_path"]) if payload.get("artifact_path") and Path(payload["artifact_path"]).is_file() else None
            payload["config_sha256"] = file_sha256(payload["config_path"]) if payload.get("config_path") and Path(payload["config_path"]).is_file() else None
            model = ModelVersion(project_id=args.project_id, **payload)
            session.add(model); session.flush()
            if payload.get("artifact_path"):
                register_artifact(session, args.project_id, kind="model", logical_name=f"{model.name}/{model.version}/artifact", owner_type="model", owner_id=model.id, source_path=payload["artifact_path"], sha256=payload["artifact_sha256"])
            if payload.get("config_path"):
                register_artifact(session, args.project_id, kind="config", logical_name=f"{model.name}/{model.version}/config", owner_type="model", owner_id=model.id, source_path=payload["config_path"], sha256=payload["config_sha256"])
            session.commit(); session.refresh(model); _json(as_dict(model))
        elif args.command == "import-result":
            if not session.get(Project, args.project_id):
                raise ValueError("Project not found")
            manifest = validate_result_manifest(_read_json(args.manifest))
            references = (
                ("dataset_id", DatasetVersion), ("model_id", ModelVersion), ("target_profile_id", TargetProfile),
                ("quantization_run_id", QuantizationRun), ("calibration_set_id", CalibrationSetVersion),
                ("board_benchmark_id", BoardBenchmark), ("parent_run_id", Run),
            )
            for field, entity_type in references:
                identifier = manifest.get(field)
                if identifier:
                    entity = session.get(entity_type, identifier)
                    if not entity or entity.project_id != args.project_id:
                        raise ValueError(f"{field} does not belong to this project")
            fingerprint = manifest_hash(manifest)
            existing = session.scalar(select(Run).where(Run.project_id == args.project_id, Run.external_run_id == manifest["external_run_id"]))
            if existing:
                if existing.import_hash == fingerprint:
                    _json({"status": "existing", "run": as_dict(existing)}); return
                raise ValueError("An external run with this ID exists but its manifest content differs")
            config = {**manifest.get("config", {})}
            for field in ("target_profile_id", "quantization_run_id", "calibration_set_id", "board_benchmark_id"):
                if manifest.get(field):
                    config[field] = manifest[field]
            run = Run(
                project_id=args.project_id, kind=manifest["kind"], name=manifest["name"], status=manifest.get("status", "completed"),
                dataset_id=manifest.get("dataset_id"), model_id=manifest.get("model_id"), parent_run_id=manifest.get("parent_run_id"),
                external_run_id=manifest["external_run_id"], import_hash=fingerprint, config=config,
                metrics=manifest.get("metrics", {}), details=manifest.get("details", {}), environment=manifest.get("environment", {}), notes=manifest.get("notes", ""),
            )
            session.add(run); session.commit(); session.refresh(run); _json({"status": "created", "run": as_dict(run)})
        elif args.command == "register-artifact":
            if not session.get(Project, args.project_id):
                raise ValueError("Project not found")
            artifact = register_artifact(session, args.project_id, **_read_json(args.manifest))
            session.commit(); session.refresh(artifact); _json(as_dict(artifact))
        elif args.command == "export":
            output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(safe_export(session, args.project_id), default=str, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"Wrote safe snapshot: {output}")


if __name__ == "__main__":
    main()
