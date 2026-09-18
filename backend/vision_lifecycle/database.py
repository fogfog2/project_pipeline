from __future__ import annotations

from pathlib import Path
import os

from sqlalchemy import create_engine, event, inspect
from sqlalchemy.orm import DeclarativeBase, sessionmaker


class Base(DeclarativeBase):
    pass


def database_url(path: str | None = None) -> str:
    db_path = Path(path or os.environ.get("VISION_LIFECYCLE_DB", ".vision-lifecycle/registry.db"))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path.resolve()}"


engine = create_engine(database_url(), connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


@event.listens_for(engine, "connect")
def _enable_sqlite_integrity(connection, _):
    connection.execute("PRAGMA foreign_keys=ON")


def init_database() -> None:
    from . import models  # noqa: F401

    Base.metadata.create_all(engine)
    # Lightweight compatibility migration for the local SQLite starter. Once a
    # team moves to a shared database, the same metadata can be managed by
    # Alembic without changing the public API.
    run_columns = {column["name"] for column in inspect(engine).get_columns("runs")}
    project_columns = {column["name"] for column in inspect(engine).get_columns("projects")}
    additions = {"external_run_id": "VARCHAR(200)", "import_hash": "VARCHAR(128)", "details": "JSON NOT NULL DEFAULT '{}'"}
    with engine.begin() as connection:
        for name, declaration in additions.items():
            if name not in run_columns:
                connection.exec_driver_sql(f"ALTER TABLE runs ADD COLUMN {name} {declaration}")
        artifact_columns = {column["name"] for column in inspect(engine).get_columns("artifacts")}
        for name, declaration in {"owner_type": "VARCHAR(40)", "owner_id": "VARCHAR(40)"}.items():
            if name not in artifact_columns:
                connection.exec_driver_sql(f"ALTER TABLE artifacts ADD COLUMN {name} {declaration}")
        quant_columns = {column["name"] for column in inspect(engine).get_columns("quantization_runs")}
        for name, declaration in {"source_role": "VARCHAR(40) NOT NULL DEFAULT 'fp32'", "output_role": "VARCHAR(40) NOT NULL DEFAULT 'quantized'"}.items():
            if name not in quant_columns:
                connection.exec_driver_sql(f"ALTER TABLE quantization_runs ADD COLUMN {name} {declaration}")
        project_additions = {"mode": "VARCHAR(40) NOT NULL DEFAULT 'user'", "recipe_id": "VARCHAR(120)", "status": "VARCHAR(40) NOT NULL DEFAULT 'active'"}
        for name, declaration in project_additions.items():
            if name not in project_columns:
                connection.exec_driver_sql(f"ALTER TABLE projects ADD COLUMN {name} {declaration}")
        dataset_columns = {column["name"] for column in inspect(engine).get_columns("dataset_versions")}
        for name, declaration in {"parent_dataset_id": "VARCHAR(40)", "snapshot": "JSON NOT NULL DEFAULT '{}'"}.items():
            if name not in dataset_columns:
                connection.exec_driver_sql(f"ALTER TABLE dataset_versions ADD COLUMN {name} {declaration}")
        model_columns = {column["name"] for column in inspect(engine).get_columns("model_versions")}
        if "status" not in model_columns:
            connection.exec_driver_sql("ALTER TABLE model_versions ADD COLUMN status VARCHAR(40) NOT NULL DEFAULT 'experimental'")
        model_columns = {column["name"] for column in inspect(engine).get_columns("model_versions")}
        if "artifact_sha256" not in model_columns:
            connection.exec_driver_sql("ALTER TABLE model_versions ADD COLUMN artifact_sha256 VARCHAR(64)")
        if "config_sha256" not in model_columns:
            connection.exec_driver_sql("ALTER TABLE model_versions ADD COLUMN config_sha256 VARCHAR(64)")
