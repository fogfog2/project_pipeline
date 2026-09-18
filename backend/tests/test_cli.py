import sys
from pathlib import Path

from vision_lifecycle import cli
from vision_lifecycle.database import Base, SessionLocal, engine
from vision_lifecycle.models import Artifact, Project


def test_backup_restore_writes_and_verifies_registry_manifest(tmp_path: Path, monkeypatch):
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with SessionLocal() as session:
        project = Project(name="backup-project")
        session.add(project)
        session.flush()
        session.add(Artifact(project_id=project.id, kind="fixture", logical_name="model", sha256="a" * 64, size_bytes=1))
        session.commit()
        project_id = project.id
    backup = tmp_path / "registry.sqlite"
    monkeypatch.setattr(sys, "argv", ["visionops", "backup", "--output", str(backup)])
    cli.main()
    manifest = Path(f"{backup}.manifest.json")
    assert backup.is_file() and manifest.is_file()
    with SessionLocal() as session:
        session.get(Project, project_id).name = "changed-before-restore"
        session.commit()
    monkeypatch.setattr(sys, "argv", ["visionops", "restore", "--input", str(backup)])
    cli.main()
    with SessionLocal() as session:
        assert session.get(Project, project_id).name == "backup-project"
