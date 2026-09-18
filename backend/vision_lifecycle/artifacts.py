from __future__ import annotations

import os
import shutil
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from .fingerprints import file_sha256
from .models import Artifact


def register_artifact(
    session: Session,
    project_id: str,
    *,
    kind: str,
    logical_name: str,
    owner_type: str | None = None,
    owner_id: str | None = None,
    source_path: str | None,
    sha256: str | None = None,
    notes: str = "",
) -> Artifact:
    """Register a portable logical file reference and verify it when accessible."""
    if owner_type and owner_id:
        previous = session.scalars(select(Artifact).where(Artifact.project_id == project_id, Artifact.owner_type == owner_type, Artifact.owner_id == owner_id, Artifact.kind == kind, Artifact.status != "superseded")).all()
        for item in previous:
            item.status = "superseded"
    path = Path(source_path).expanduser() if source_path else None
    digest = sha256
    size = 0
    status = "registered"
    if path and path.is_file():
        digest = digest or file_sha256(path)
        size = path.stat().st_size
        status = "verified"
    elif path and path.is_dir():
        status = "directory"
    elif source_path:
        status = "unavailable"
    managed_path = None
    if path and (path.is_file() or path.is_dir()):
        try:
            managed_root = Path(os.environ.get("VISION_LIFECYCLE_ARTIFACT_ROOT", ".vision-lifecycle/artifacts"))
            managed_path = managed_root / project_id / kind / f"{digest or 'directory'}-{path.name}"
            managed_path.parent.mkdir(parents=True, exist_ok=True)
            if path.is_dir():
                shutil.copytree(path, managed_path, dirs_exist_ok=True)
            else:
                shutil.copy2(path, managed_path)
        except OSError as error:
            notes = f"{notes} Managed artifact copy failed: {error}".strip()
            managed_path = None
    artifact = Artifact(
        project_id=project_id,
        kind=kind,
        logical_name=logical_name,
        owner_type=owner_type,
        owner_id=owner_id,
        source_path=source_path,
        managed_path=str(managed_path) if managed_path else None,
        sha256=digest,
        size_bytes=size,
        status=status,
        notes=notes,
    )
    session.add(artifact)
    return artifact


def verify_artifact(artifact: Artifact) -> dict[str, object]:
    """Refresh an artifact's accessibility, size, and content status."""
    path = Path(artifact.source_path).expanduser() if artifact.source_path else None
    if not path:
        managed = Path(artifact.managed_path).expanduser() if artifact.managed_path else None
        if managed and (managed.is_file() or managed.is_dir()):
            artifact.status = "managed"
            return {"id": artifact.id, "status": artifact.status, "managed_path": str(managed), "reason": "managed artifact copy is available"}
        artifact.status = "registered"
        return {"id": artifact.id, "status": artifact.status, "reason": "source path is not configured"}
    if path.is_dir():
        if not path.exists() and artifact.managed_path and Path(artifact.managed_path).is_dir():
            artifact.status = "managed"
            return {"id": artifact.id, "status": artifact.status, "managed_path": artifact.managed_path, "reason": "source directory is unavailable; managed copy is available"}
        artifact.status = "directory"
        return {"id": artifact.id, "status": artifact.status, "reason": "directory references are not hashed"}
    if not path.is_file():
        if artifact.managed_path and (Path(artifact.managed_path).is_file() or Path(artifact.managed_path).is_dir()):
            artifact.status = "managed"
            return {"id": artifact.id, "status": artifact.status, "managed_path": artifact.managed_path, "reason": "source file is unavailable; managed copy is available"}
        artifact.status = "unavailable"
        return {"id": artifact.id, "status": artifact.status, "reason": "source file is unavailable"}
    digest = file_sha256(path)
    artifact.size_bytes = path.stat().st_size
    if artifact.sha256 and artifact.sha256 != digest:
        artifact.status = "drifted"
        return {"id": artifact.id, "status": artifact.status, "expected_sha256": artifact.sha256, "actual_sha256": digest}
    artifact.sha256 = digest
    artifact.status = "verified"
    return {"id": artifact.id, "status": artifact.status, "sha256": digest, "size_bytes": artifact.size_bytes}
