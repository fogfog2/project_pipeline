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
        source_path=source_path,
        sha256=digest,
        size_bytes=size,
        status=status,
        notes=notes,
    )
    session.add(artifact)
    return artifact
