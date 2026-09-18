from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ProjectCreate(BaseModel):
    name: str
    description: str = ""
    task_kind: str = "detection"
    storage_root: str | None = None
    git_url: str | None = None
    default_branch: str | None = None
    mode: str = "user"
    recipe_id: str | None = None


class DatasetCreate(BaseModel):
    name: str
    version: str
    task_kind: str = "detection"
    format: str = "coco"
    manifest_path: str | None = None
    annotation_path: str | None = None
    sample_count: int = 0
    class_names: list[str] = Field(default_factory=list)
    status: str = "draft"
    validation: dict[str, Any] = Field(default_factory=dict)


class ProjectUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    task_kind: str | None = None
    storage_root: str | None = None
    git_url: str | None = None
    default_branch: str | None = None


class DatasetUpdate(BaseModel):
    name: str | None = None
    annotation_path: str | None = None
    manifest_path: str | None = None
    class_names: list[str] | None = None
    validation: dict[str, Any] | None = None


class StorageMappingCreate(BaseModel):
    name: str
    root_path: str
    read_only: bool = True
    notes: str = ""


class StorageMappingUpdate(BaseModel):
    root_path: str | None = None
    read_only: bool | None = None
    notes: str | None = None


class StorageBrowseRequest(BaseModel):
    relative_path: str = ""
    limit: int = Field(default=200, ge=1, le=500)


class StorageInventoryRequest(BaseModel):
    relative_path: str = ""
    recursive: bool = True
    limit: int = Field(default=1000, ge=1, le=10_000)


class VersionDefinitionCreate(BaseModel):
    name: str
    version: str
    dataset_id: str | None = None
    classes: list[dict[str, Any]] = Field(default_factory=list)
    mapping: dict[str, Any] = Field(default_factory=dict)
    definition: dict[str, Any] = Field(default_factory=dict)
    purpose: str = "core"
    sampling: dict[str, Any] = Field(default_factory=dict)
    preprocessing: dict[str, Any] = Field(default_factory=dict)
    status: str = "draft"


class QuantizationRunCreate(BaseModel):
    name: str
    source_model_id: str
    output_model_id: str | None = None
    calibration_set_id: str | None = None
    method: str = "unknown"
    weight_dtype: str = "unknown"
    activation_dtype: str = "unknown"
    encoding_path: str | None = None
    status: str = "registered"
    metadata_json: dict[str, Any] = Field(default_factory=dict)


class BoardBenchmarkCreate(BaseModel):
    name: str
    model_id: str
    target_profile_id: str
    evaluation_run_id: str | None = None
    status: str = "registered"
    metrics: dict[str, float] = Field(default_factory=dict)
    measurement: dict[str, Any] = Field(default_factory=dict)
    raw_output_hash: str | None = None
    notes: str = ""


class OnboardingCreate(BaseModel):
    recipe_id: str = "blank"
    recipe_version: str = "v1"


class StepProgressUpdate(BaseModel):
    status: str
    evidence: dict[str, Any] = Field(default_factory=dict)


class ModelUpdate(BaseModel):
    name: str | None = None
    alias: str | None = None
    artifact_path: str | None = None
    config_path: str | None = None
    metadata_json: dict[str, Any] | None = None


class ModelCreate(BaseModel):
    name: str
    version: str
    family: str
    task_kind: str = "detection"
    format: str = "pytorch"
    precision: str = "fp32"
    artifact_path: str | None = None
    config_path: str | None = None
    source_dataset_id: str | None = None
    source_run_id: str | None = None
    alias: str | None = None
    runnable: bool = False
    metadata_json: dict[str, Any] = Field(default_factory=dict)


class RunCreate(BaseModel):
    kind: str
    name: str
    status: str = "completed"
    dataset_id: str | None = None
    model_id: str | None = None
    parent_run_id: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    environment: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""


class ComparisonRequest(BaseModel):
    baseline_model_id: str
    candidate_model_id: str


class JobCreate(BaseModel):
    runner_id: str
    args: list[str] = Field(default_factory=list)
    input_json: dict[str, Any] = Field(default_factory=dict)


class RunnerProfileCreate(BaseModel):
    name: str
    executable: str
    default_args: list[str] = Field(default_factory=list)
    working_directory: str | None = None
    environment_names: list[str] = Field(default_factory=list)
    timeout_seconds: int = Field(default=3600, ge=1, le=86_400)
    enabled: bool = True


class PredictionEvaluationCreate(BaseModel):
    model_id: str
    dataset_id: str
    predictions_path: str
    evaluator_version: str = "lifecycle-ap50-v1"
    protocol: str = "onboarding_ap50"
    iou_threshold: float = 0.5


class TargetProfileCreate(BaseModel):
    name: str
    version: str = "v1"
    target_kind: str = "board"
    runtime: str = "unknown"
    hardware: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""


class PathInspectRequest(BaseModel):
    path: str


class ReleaseCreate(BaseModel):
    name: str
    model_id: str
    evaluation_run_id: str | None = None
    baseline_model_id: str | None = None
    gate_config: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""


class InferencePreviewRequest(BaseModel):
    model_id: str
    image_path: str


class ResultImportCreate(BaseModel):
    manifest: dict[str, Any]


class ClassificationEvaluationCreate(BaseModel):
    model_id: str
    dataset_id: str | None = None
    evaluator_version: str = "classification-v1"
    records: list[dict[str, Any]]
