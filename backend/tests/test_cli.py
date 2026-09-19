import sys
import os
import sqlite3
import subprocess
from pathlib import Path

from vision_lifecycle import cli
from vision_lifecycle.database import Base, SessionLocal, engine
from vision_lifecycle.models import Artifact, ModelVersion, Project, Run


def test_backup_restore_writes_and_verifies_registry_manifest(tmp_path: Path, monkeypatch):
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with SessionLocal() as session:
        project = Project(name="backup-project")
        session.add(project)
        session.flush()
        managed = tmp_path / "managed" / "model.bin"
        managed.parent.mkdir()
        managed.write_bytes(b"model")
        session.add(Artifact(project_id=project.id, kind="fixture", logical_name="model", sha256="a" * 64, size_bytes=5, managed_path=str(managed)))
        session.commit()
        project_id = project.id
    backup = tmp_path / "registry.sqlite"
    monkeypatch.setattr(sys, "argv", ["visionops", "backup", "--output", str(backup)])
    cli.main()
    manifest = Path(f"{backup}.manifest.json")
    assert backup.is_file() and manifest.is_file()
    assert list(Path(f"{backup}.artifacts").rglob("model.bin"))
    managed.unlink()
    with SessionLocal() as session:
        session.get(Project, project_id).name = "changed-before-restore"
        session.commit()
    monkeypatch.setattr(sys, "argv", ["visionops", "restore", "--input", str(backup)])
    cli.main()
    with SessionLocal() as session:
        assert session.get(Project, project_id).name == "backup-project"
    assert managed.read_bytes() == b"model"


def test_cli_import_result_uses_same_manifest_contract(tmp_path: Path, monkeypatch, capsys):
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with SessionLocal() as session:
        project = Project(name="cli-import-project")
        session.add(project); session.flush()
        model = ModelVersion(project_id=project.id, name="model", version="v1", family="fixture", task_kind="detection", format="external")
        session.add(model); session.commit(); project_id, model_id = project.id, model.id
    manifest = tmp_path / "result.json"
    manifest.write_text(f'{{"schema_version":"1.0","external_run_id":"CLI-1","kind":"evaluation","name":"cli eval","model_id":"{model_id}","metrics":{{"bbox_AP50":0.5}}}}', encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["visionops", "import-result", project_id, str(manifest)])
    cli.main()
    output = capsys.readouterr().out
    assert '"status": "created"' in output
    with SessionLocal() as session:
        run = session.query(Run).filter_by(external_run_id="CLI-1").one()
        assert run.model_id == model_id and run.metrics["bbox_AP50"] == 0.5


def test_cli_delete_dataset_uses_safe_draft_policy(tmp_path: Path, monkeypatch, capsys):
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    source = tmp_path / "source"
    source.mkdir()
    with SessionLocal() as session:
        project = Project(name="cli-delete-project")
        session.add(project); session.flush()
        from vision_lifecycle.models import DatasetVersion
        dataset = DatasetVersion(project_id=project.id, name="scratch", version="v1", task_kind="unknown", format="folder", manifest_path=str(source))
        session.add(dataset); session.commit(); project_id, dataset_id = project.id, dataset.id
    monkeypatch.setattr(sys, "argv", ["visionops", "delete-dataset", project_id, dataset_id])
    cli.main()
    assert '"source_files_preserved": true' in capsys.readouterr().out
    assert source.is_dir()
    with SessionLocal() as session:
        assert session.get(DatasetVersion, dataset_id) is None


def test_cli_migrate_creates_versioned_registry(tmp_path: Path):
    db_path = tmp_path / "migrated.sqlite"
    environment = {**os.environ, "VISION_LIFECYCLE_DB": str(db_path), "PYTHONPATH": str(Path(__file__).parents[1])}
    result = subprocess.run([sys.executable, "-m", "vision_lifecycle.cli", "migrate"], cwd=Path(__file__).parents[2], env=environment, capture_output=True, text=True, check=True)
    assert "Registry migrations applied" in result.stdout
    with sqlite3.connect(db_path) as connection:
        revision = connection.execute("SELECT version_num FROM alembic_version").fetchone()[0]
        assert revision == "177ea08cc805"
        assert connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='projects'").fetchone()
