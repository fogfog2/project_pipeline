from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path

from sqlalchemy import select

from .adapters.coco import validate_coco
from .adapters.inspect import inspect_path
from .artifacts import register_artifact
from .database import SessionLocal, engine, init_database
from .fingerprints import dataset_fingerprint, file_sha256
from .models import DatasetVersion, ModelVersion, Project
from .runner import run_worker
from .serializers import as_dict
from .service import safe_export, seed_demo


def _json(value: object) -> None:
    print(json.dumps(value, default=str, ensure_ascii=False, indent=2))


def _read_json(path: str) -> dict:
    with Path(path).open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError("JSON input must be an object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(prog="visionops", description="Vision lifecycle registry CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="Create the local SQLite registry")
    sub.add_parser("demo", help="Load the RTMDet and YOLOX onboarding demo")
    sub.add_parser("projects", help="List registered projects")
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
    restore = sub.add_parser("restore", help="Restore the local registry from a SQLite backup")
    restore.add_argument("--input", required=True, help="Source SQLite backup file")
    args = parser.parse_args()

    init_database()
    if args.command == "inspect":
        _json(inspect_path(args.path)); return
    if args.command == "validate-coco":
        _json(validate_coco(args.annotation_path)); return
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
        print(f"Wrote registry backup: {output}"); return
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
        print(f"Restored registry backup: {source_path}"); return
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
            dataset = DatasetVersion(project_id=args.project_id, **payload)
            session.add(dataset); session.flush()
            if payload.get("annotation_path"):
                register_artifact(session, args.project_id, kind="dataset-annotation", logical_name=f"{dataset.name}/{dataset.version}/annotation", source_path=payload["annotation_path"])
            if payload.get("manifest_path"):
                register_artifact(session, args.project_id, kind="dataset-manifest", logical_name=f"{dataset.name}/{dataset.version}/manifest", source_path=payload["manifest_path"])
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
                register_artifact(session, args.project_id, kind="model", logical_name=f"{model.name}/{model.version}/artifact", source_path=payload["artifact_path"], sha256=payload["artifact_sha256"])
            if payload.get("config_path"):
                register_artifact(session, args.project_id, kind="config", logical_name=f"{model.name}/{model.version}/config", source_path=payload["config_path"], sha256=payload["config_sha256"])
            session.commit(); session.refresh(model); _json(as_dict(model))
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
