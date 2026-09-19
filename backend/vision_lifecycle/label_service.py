"""Label registration contract shared by HTTP and offline CLI clients."""
import hashlib
import json

from sqlalchemy.orm import Session

from .label_validation import validate_label_schema
from .models import DatasetVersion, LabelSchemaVersion, Project
from .schemas import VersionDefinitionCreate


def register_label_schema(session: Session, project_id: str, payload: VersionDefinitionCreate) -> LabelSchemaVersion:
    if not session.get(Project, project_id):
        raise ValueError("Project not found")
    if payload.dataset_id:
        dataset = session.get(DatasetVersion, payload.dataset_id)
        if not dataset or dataset.project_id != project_id:
            raise ValueError("Dataset reference must belong to this project")
    errors = validate_label_schema(payload.classes, payload.mapping)
    if errors:
        raise ValueError(f"Invalid label schema: {errors}")
    if payload.parent_label_schema_id:
        parent = session.get(LabelSchemaVersion, payload.parent_label_schema_id)
        if not parent or parent.project_id != project_id:
            raise ValueError("Parent label schema must belong to this project")
        if parent.dataset_id and payload.dataset_id and parent.dataset_id != payload.dataset_id:
            raise ValueError("Parent label schema must use the selected DatasetVersion")
    fields = payload.model_dump(include={"name", "version", "classes", "mapping", "dataset_id", "parent_label_schema_id"})
    digest = "sha256:" + hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
    item = LabelSchemaVersion(project_id=project_id, **fields, status=payload.status, content_hash=digest)
    session.add(item)
    session.commit()
    session.refresh(item)
    return item
