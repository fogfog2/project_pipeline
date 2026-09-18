from __future__ import annotations

import hashlib
import json
from pathlib import Path


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_fingerprint(manifest_path: str | None, annotation_path: str | None) -> tuple[str | None, dict[str, str]]:
    paths = {path for path in (manifest_path, annotation_path) if path and Path(path).is_file()}
    details = {str(Path(path).resolve()): file_sha256(path) for path in sorted(paths)}
    if not details:
        return None, {}
    encoded = json.dumps(details, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}", details
