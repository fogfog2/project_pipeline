"""Versioned adapter capability registry used by the API and onboarding agents.

The registry describes integration points; it does not infer an unknown format
or execute a command. A new adapter must declare its task, supported actions,
and contract version before it can be advertised to the UI.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AdapterSpec:
    id: str
    kind: str
    tasks: tuple[str, ...]
    capabilities: tuple[str, ...]
    contract_version: str = "1.0"

    def as_dict(self) -> dict:
        value = asdict(self)
        value["tasks"] = list(self.tasks)
        value["capabilities"] = list(self.capabilities)
        return value


ADAPTERS: tuple[AdapterSpec, ...] = (
    AdapterSpec("coco", "dataset", ("detection",), ("inspect", "validate", "prediction-evaluation", "onnx-batch-evaluation")),
    AdapterSpec("yolo-txt", "dataset", ("detection",), ("inspect", "validate")),
    AdapterSpec("classification", "dataset", ("classification",), ("inspect", "validate", "onnx-batch-evaluation")),
    AdapterSpec("mmdetection", "inference", ("detection",), ("register", "native-inference", "external-result-import")),
    AdapterSpec("onnx", "inference", ("classification", "detection"), ("preview", "batch-evaluation", "cpu")),
    AdapterSpec("mmdeploy", "inference", ("detection",), ("runtime-inference", "target-profile")),
    AdapterSpec("classification-metrics", "evaluator", ("classification",), ("top-k", "confusion", "per-class", "calibration")),
    AdapterSpec("coco-metrics", "evaluator", ("detection",), ("ap", "ap50", "ap75", "per-class", "invalid-input-report")),
    AdapterSpec("local-filesystem", "storage", ("unknown", "classification", "detection"), ("browse", "inventory", "hash", "remap")),
    AdapterSpec("result-manifest-v1", "result-importer", ("unknown", "classification", "detection"), ("idempotency", "conflict-detection", "lineage")),
    AdapterSpec("mock-board", "runner", ("classification", "detection"), ("run", "result-contract")),
)


def list_adapters() -> list[dict]:
    return [item.as_dict() for item in ADAPTERS]


def get_adapter(adapter_id: str) -> AdapterSpec | None:
    return next((item for item in ADAPTERS if item.id == adapter_id), None)
