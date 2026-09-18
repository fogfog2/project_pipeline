from __future__ import annotations

import hashlib
import json
from typing import Any


def manifest_hash(manifest: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_result_manifest(value: dict[str, Any]) -> dict[str, Any]:
    required = {"schema_version", "external_run_id", "kind", "name"}
    missing = required - value.keys()
    if missing:
        raise ValueError(f"Result manifest is missing required keys: {', '.join(sorted(missing))}")
    if value["schema_version"] != "1.0":
        raise ValueError(f"Unsupported result manifest schema: {value['schema_version']}")
    if value["kind"] not in {"training", "export", "quantization", "evaluation", "board"}:
        raise ValueError("Result manifest kind must be training, export, quantization, evaluation, or board")
    if not isinstance(value.get("metrics", {}), dict) or not isinstance(value.get("config", {}), dict) or not isinstance(value.get("details", {}), dict):
        raise ValueError("Result manifest metrics, config, and details must be objects")
    for field in ("target_profile_id", "quantization_run_id", "calibration_set_id", "board_benchmark_id"):
        if field in value and value[field] is not None and not isinstance(value[field], str):
            raise ValueError(f"{field} must be an ID string when provided")
    return value
