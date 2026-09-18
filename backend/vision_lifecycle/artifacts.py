from __future__ import annotations

from pathlib import Path

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
    artifact = Artifact(
        project_id=project_id,
        kind=kind,
        logical_name=logical_name,
        owner_type=owner_type,
        owner_id=owner_id,
        source_path=source_path,
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
        artifact.status = "registered"
        return {"id": artifact.id, "status": artifact.status, "reason": "source path is not configured"}
    if path.is_dir():
        artifact.status = "directory"
        return {"id": artifact.id, "status": artifact.status, "reason": "directory references are not hashed"}
    if not path.is_file():
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
