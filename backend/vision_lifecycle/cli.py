from __future__ import annotations

import argparse
import json
from pathlib import Path

from sqlalchemy import select

from .adapters.coco import validate_coco
from .adapters.inspect import inspect_path
from .database import SessionLocal, init_database
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
    export = sub.add_parser("export", help="Write a Pages-safe project snapshot")
    export.add_argument("project_id")
    export.add_argument("--output", required=True, help="Destination JSON file")
    worker = sub.add_parser("worker", help="Run the external job worker")
    worker.add_argument("--once", action="store_true", help="Claim at most one queued job and exit")
    worker.add_argument("--poll-seconds", type=float, default=1.0)
    args = parser.parse_args()

    init_database()
    if args.command == "inspect":
        _json(inspect_path(args.path)); return
    if args.command == "validate-coco":
        _json(validate_coco(args.annotation_path)); return
    if args.command == "worker":
        run_worker(poll_seconds=args.poll_seconds, once=args.once); return
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
            dataset = DatasetVersion(project_id=args.project_id, **_read_json(args.manifest))
            session.add(dataset); session.commit(); session.refresh(dataset); _json(as_dict(dataset))
        elif args.command == "create-model":
            if not session.get(Project, args.project_id):
                raise ValueError("Project not found")
            model = ModelVersion(project_id=args.project_id, **_read_json(args.manifest))
            session.add(model); session.commit(); session.refresh(model); _json(as_dict(model))
        elif args.command == "export":
            output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(safe_export(session, args.project_id), default=str, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"Wrote safe snapshot: {output}")


if __name__ == "__main__":
    main()
