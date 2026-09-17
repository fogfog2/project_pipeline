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
    if not isinstance(value.get("metrics", {}), dict) or not isinstance(value.get("config", {}), dict):
        raise ValueError("Result manifest metrics and config must be objects")
    if "target_profile_id" in value and not isinstance(value["target_profile_id"], str):
        raise ValueError("target_profile_id must be a target profile ID when provided")
    return value
