from __future__ import annotations

import os
from pathlib import Path

from .fingerprints import file_sha256


def storage_status(root_path: str) -> dict:
    root = Path(root_path).expanduser()
    if not root.exists():
        return {"status": "unavailable", "reason": "Path does not exist"}
    if not root.is_dir():
        return {"status": "unavailable", "reason": "Path is not a directory"}
    return {
        "status": "available" if os.access(root, os.R_OK) else "unavailable",
        "resolved_root": str(root.resolve()),
        "readable": os.access(root, os.R_OK),
        "writable": os.access(root, os.W_OK),
    }


def resolve_within(root_path: str, relative_path: str = "") -> Path:
    root = Path(root_path).expanduser().resolve(strict=True)
    target = (root / relative_path).resolve(strict=False)
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError("Requested path escapes the configured storage root") from error
    return target


def browse(root_path: str, relative_path: str = "", limit: int = 200) -> dict:
    target = resolve_within(root_path, relative_path)
    if not target.exists():
        raise ValueError("Requested path does not exist")
    if not target.is_dir():
        raise ValueError("Requested path is not a directory")
    root = Path(root_path).expanduser().resolve(strict=True)
    entries: list[dict] = []
    for child in sorted(target.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))[:limit]:
        try:
            resolved = child.resolve(strict=False)
            relative = resolved.relative_to(root)
        except ValueError:
            continue
        entries.append({
            "name": child.name,
            "relative_path": str(relative),
            "kind": "directory" if child.is_dir() else "file",
            "size_bytes": child.stat().st_size if child.is_file() else None,
        })
    return {"relative_path": str(target.relative_to(root)), "entries": entries, "truncated": len(list(target.iterdir())) > limit}


def inventory(root_path: str, relative_path: str = "", recursive: bool = True, limit: int = 1000) -> list[dict]:
    target = resolve_within(root_path, relative_path)
    if not target.exists() or not target.is_dir():
        raise ValueError("Inventory path is not an existing directory")
    root = Path(root_path).expanduser().resolve(strict=True)
    iterator = target.rglob("*") if recursive else target.iterdir()
    result: list[dict] = []
    for child in sorted(iterator, key=lambda item: str(item).lower()):
        if len(result) >= limit:
            break
        if not child.is_file():
            continue
        resolved = child.resolve(strict=True)
        try:
            relative = resolved.relative_to(root)
        except ValueError:
            continue
        result.append({"relative_path": str(relative), "kind": "file", "size_bytes": resolved.stat().st_size, "sha256": file_sha256(resolved), "suffix": resolved.suffix.lower()})
    return result
