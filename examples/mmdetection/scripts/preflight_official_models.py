#!/usr/bin/env python3
"""Check user-provided RTMDet/YOLOX artifacts before registry registration."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

MODELS = {
    "rtmdet-tiny": {"family": "RTMDet-tiny", "mim_config": "rtmdet_tiny_8xb32-300e_coco"},
    "yolox-s": {"family": "YOLOX-s", "mim_config": "yolox_s_8xb8-300e_coco"},
}


def _file_info(value: str) -> dict[str, Any]:
    path = Path(value).expanduser()
    if not path.is_file():
        return {"path": str(path), "exists": False}
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "exists": True, "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def inspect_bundle(model: str, config: str, checkpoint: str) -> dict[str, Any]:
    if model not in MODELS:
        raise ValueError(f"model must be one of: {', '.join(sorted(MODELS))}")
    required = {name: importlib.util.find_spec(name) is not None for name in ("mmdet", "mmengine", "mmcv")}
    config_info, checkpoint_info = _file_info(config), _file_info(checkpoint)
    missing = [key for key, item in (("config", config_info), ("checkpoint", checkpoint_info)) if not item["exists"]]
    missing_dependencies = [name for name, present in required.items() if not present]
    status = "ready" if not missing and not missing_dependencies else "missing_artifacts" if missing else "missing_dependencies"
    return {"schema_version": "1.0", "status": status, "model": model, "family": MODELS[model]["family"], "mim_config": MODELS[model]["mim_config"], "framework": "MMDetection", "framework_version": "3.3.0", "artifacts": {"config": config_info, "checkpoint": checkpoint_info}, "dependencies": required, "missing": missing, "missing_dependencies": missing_dependencies, "next": "Register config/checkpoint paths in the project model form." if status == "ready" else "Run prepare-official-models.sh, then activate the MMDetection 3.3 environment."}


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight official RTMDet/YOLOX MMDetection artifacts")
    parser.add_argument("model", choices=sorted(MODELS))
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = inspect_bundle(args.model, args.config, args.checkpoint)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = Path(args.output).expanduser(); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
