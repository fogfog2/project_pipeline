from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def validate_jsonl(path: str) -> dict[str, Any]:
    """Validate the common one-JSON-object-per-line manifest format."""
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"JSONL manifest does not exist: {source}")
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                errors.append({"line": line_number, "reason": f"invalid JSON: {error.msg}"})
                continue
            if not isinstance(value, dict):
                errors.append({"line": line_number, "reason": "record must be a JSON object"})
                continue
            records.append(value)
    return {
        "status": "passed" if not errors else "failed",
        "format": "jsonl",
        "record_count": len(records),
        "error_count": len(errors),
        "errors": errors[:50],
    }
