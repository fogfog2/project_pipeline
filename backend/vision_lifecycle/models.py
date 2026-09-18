from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:12]}"


class Timestamped:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC))


class Project(Base, Timestamped):
    __tablename__ = "projects"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("PRJ"))
    name: Mapped[str] = mapped_column(String(200), unique=True)
    description: Mapped[str] = mapped_column(Text, default="")
    task_kind: Mapped[str] = mapped_column(String(50), default="detection")
    storage_root: Mapped[str | None] = mapped_column(Text, nullable=True)
    git_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    default_branch: Mapped[str | None] = mapped_column(String(200), nullable=True)
    mode: Mapped[str] = mapped_column(String(40), default="user")
    recipe_id: Mapped[str | None] = mapped_column(String(120), nullable=True)
    status: Mapped[str] = mapped_column(String(40), default="active")


class DatasetVersion(Base, Timestamped):
    __tablename__ = "dataset_versions"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("DS"))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    version: Mapped[str] = mapped_column(String(100))
    task_kind: Mapped[str] = mapped_column(String(50))
    format: Mapped[str] = mapped_column(String(100))
    manifest_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    annotation_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    sample_count: Mapped[int] = mapped_column(default=0)
    class_names: Mapped[list] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(50), default="draft")
    validation: Mapped[dict] = mapped_column(JSON, default=dict)


class StorageMapping(Base, Timestamped):
    __tablename__ = "storage_mappings"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("STORE"))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    name: Mapped[str] = mapped_column(String(120))
    root_path: Mapped[str] = mapped_column(Text)
    read_only: Mapped[bool] = mapped_column(default=True)
    status: Mapped[str] = mapped_column(String(40), default="unavailable")
    last_validation: Mapped[dict] = mapped_column(JSON, default=dict)
    notes: Mapped[str] = mapped_column(Text, default="")


class DataAsset(Base, Timestamped):
    """Discovered file using a storage id and portable relative path."""
    __tablename__ = "data_assets"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("ASSET"))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    storage_id: Mapped[str] = mapped_column(ForeignKey("storage_mappings.id"), index=True)
    relative_path: Mapped[str] = mapped_column(Text)
    kind: Mapped[str] = mapped_column(String(40), default="file")
    size_bytes: Mapped[int] = mapped_column(default=0)
    sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(40), default="discovered")
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)


class LabelSchemaVersion(Base, Timestamped):
    __tablename__ = "label_schema_versions"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("LABEL"))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    dataset_id: Mapped[str | None] = mapped_column(ForeignKey("dataset_versions.id"), nullable=True)
    name: Mapped[str] = mapped_column(String(200))
    version: Mapped[str] = mapped_column(String(100))
    classes: Mapped[list] = mapped_column(JSON, default=list)
    mapping: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(40), default="draft")
    content_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)


class SplitVersion(Base, Timestamped):
    __tablename__ = "split_versions"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("SPLIT"))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    dataset_id: Mapped[str | None] = mapped_column(ForeignKey("dataset_versions.id"), nullable=True)
    name: Mapped[str] = mapped_column(String(200))
    version: Mapped[str] = mapped_column(String(100))
    definition: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(40), default="draft")
    content_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)


class EvaluationSetVersion(Base, Timestamped):
    __tablename__ = "evaluation_set_versions"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("EVALSET"))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    dataset_id: Mapped[str | None] = mapped_column(ForeignKey("dataset_versions.id"), nullable=True)
    name: Mapped[str] = mapped_column(String(200))
    version: Mapped[str] = mapped_column(String(100))
    purpose: Mapped[str] = mapped_column(String(80), default="core")
    definition: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(40), default="draft")
    content_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)


class CalibrationSetVersion(Base, Timestamped):
    __tablename__ = "calibration_set_versions"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("CAL"))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    dataset_id: Mapped[str | None] = mapped_column(ForeignKey("dataset_versions.id"), nullable=True)
    name: Mapped[str] = mapped_column(String(200))
    version: Mapped[str] = mapped_column(String(100))
    sampling: Mapped[dict] = mapped_column(JSON, default=dict)
    preprocessing: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(40), default="draft")
    content_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)


class ModelVersion(Base, Timestamped):
    __tablename__ = "model_versions"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("MODEL"))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    version: Mapped[str] = mapped_column(String(100))
    family: Mapped[str] = mapped_column(String(100))
    task_kind: Mapped[str] = mapped_column(String(50))
    format: Mapped[str] = mapped_column(String(50))
    precision: Mapped[str] = mapped_column(String(50), default="fp32")
    artifact_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    config_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    artifact_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    config_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source_dataset_id: Mapped[str | None] = mapped_column(ForeignKey("dataset_versions.id"), nullable=True)
    source_run_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    alias: Mapped[str | None] = mapped_column(String(50), nullable=True)
    runnable: Mapped[bool] = mapped_column(default=False)
    status: Mapped[str] = mapped_column(String(40), default="experimental")
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)


class Run(Base, Timestamped):
    __tablename__ = "runs"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("RUN"))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    kind: Mapped[str] = mapped_column(String(50))
    name: Mapped[str] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(50), default="completed")
    dataset_id: Mapped[str | None] = mapped_column(ForeignKey("dataset_versions.id"), nullable=True)
    model_id: Mapped[str | None] = mapped_column(ForeignKey("model_versions.id"), nullable=True)
    parent_run_id: Mapped[str | None] = mapped_column(ForeignKey("runs.id"), nullable=True)
    external_run_id: Mapped[str | None] = mapped_column(String(200), nullable=True, index=True)
    import_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    environment: Mapped[dict] = mapped_column(JSON, default=dict)
    notes: Mapped[str] = mapped_column(Text, default="")


class Job(Base, Timestamped):
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("JOB"))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    runner_id: Mapped[str] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(50), default="queued")
    command: Mapped[list] = mapped_column(JSON, default=list)
    input_json: Mapped[dict] = mapped_column(JSON, default=dict)
    result_json: Mapped[dict] = mapped_column(JSON, default=dict)
    log: Mapped[str] = mapped_column(Text, default="")


class RunnerProfile(Base, Timestamped):
    """A user-approved local executable profile.

    Environment values are never stored here; only the names inherited from the
    service environment are retained.
    """
    __tablename__ = "runner_profiles"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("RUNNER"))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    executable: Mapped[str] = mapped_column(Text)
    default_args: Mapped[list] = mapped_column(JSON, default=list)
    working_directory: Mapped[str | None] = mapped_column(Text, nullable=True)
    environment_names: Mapped[list] = mapped_column(JSON, default=list)
    timeout_seconds: Mapped[int] = mapped_column(default=3600)
    enabled: Mapped[bool] = mapped_column(default=True)


class TargetProfile(Base, Timestamped):
    """Versioned deployment target; credentials and commands do not belong here."""
    __tablename__ = "target_profiles"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("TARGET"))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    version: Mapped[str] = mapped_column(String(100), default="v1")
    target_kind: Mapped[str] = mapped_column(String(80), default="board")
    runtime: Mapped[str] = mapped_column(String(120), default="unknown")
    hardware: Mapped[dict] = mapped_column(JSON, default=dict)
    notes: Mapped[str] = mapped_column(Text, default="")


class Release(Base, Timestamped):
    __tablename__ = "releases"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("REL"))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    model_id: Mapped[str] = mapped_column(ForeignKey("model_versions.id"))
    evaluation_run_id: Mapped[str | None] = mapped_column(ForeignKey("runs.id"), nullable=True)
    baseline_model_id: Mapped[str | None] = mapped_column(ForeignKey("model_versions.id"), nullable=True)
    gate_config: Mapped[dict] = mapped_column(JSON, default=dict)
    gate_result: Mapped[dict] = mapped_column(JSON, default=dict)
    decision: Mapped[str] = mapped_column(String(50), default="NOT_CONFIGURED")
    notes: Mapped[str] = mapped_column(Text, default="")
