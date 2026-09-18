from __future__ import annotations

from collections import defaultdict
from typing import Any


def _members(definition: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize supported split definitions to item/split/group records.

    The portable contract accepts either ``splits: {train: [ids], val: [ids]}``
    or ``assignments: [{item_id, split, group_id?}]``.  It deliberately does
    not derive groups from filenames or paths.
    """
    result: list[dict[str, Any]] = []
    assignments = definition.get("assignments")
    if assignments is not None:
        if not isinstance(assignments, list):
            return result
        for item in assignments:
            if isinstance(item, dict):
                result.append({"item_id": item.get("item_id", item.get("image_id", item.get("id"))), "split": item.get("split"), "group_id": item.get("group_id")})
        return result
    splits = definition.get("splits", {})
    if not isinstance(splits, dict):
        return result
    groups = definition.get("groups", {})
    groups = groups if isinstance(groups, dict) else {}
    for split_name, values in splits.items():
        if not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, dict):
                item_id = value.get("item_id", value.get("image_id", value.get("id")))
                group_id = value.get("group_id")
            else:
                item_id = value
                group_id = groups.get(str(value))
            result.append({"item_id": item_id, "split": split_name, "group_id": group_id})
    return result


def validate_split_definition(definition: dict[str, Any], snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate duplicate membership and group leakage without guessing groups."""
    definition = definition if isinstance(definition, dict) else {}
    records = _members(definition)
    errors: list[dict[str, Any]] = []
    if not records:
        errors.append({"reason": "no split assignments were provided"})
    item_splits: dict[str, list[str]] = defaultdict(list)
    group_splits: dict[str, set[str]] = defaultdict(set)
    counts: dict[str, int] = defaultdict(int)
    for index, record in enumerate(records):
        item_id = record.get("item_id")
        split_name = record.get("split")
        if item_id is None or split_name in (None, ""):
            errors.append({"index": index, "reason": "each assignment needs item_id and split"})
            continue
        item_key = str(item_id)
        split_key = str(split_name)
        item_splits[item_key].append(split_key)
        counts[split_key] += 1
        if record.get("group_id") is not None:
            group_splits[str(record["group_id"])].add(split_key)
    duplicates = {item: sorted(set(splits)) for item, splits in item_splits.items() if len(set(splits)) > 1}
    for item, splits in duplicates.items():
        errors.append({"item_id": item, "splits": splits, "reason": "item appears in multiple splits"})
    group_leaks = {group: sorted(splits) for group, splits in group_splits.items() if len(splits) > 1}
    for group, splits in group_leaks.items():
        errors.append({"group_id": group, "splits": splits, "reason": "group appears in multiple splits"})
    known_items: set[str] = set()
    if isinstance(snapshot, dict):
        known_items = {str(item.get("id")) for item in snapshot.get("images", []) if isinstance(item, dict) and item.get("id") is not None}
    unknown_items = sorted(set(item_splits) - known_items) if known_items else []
    for item in unknown_items:
        errors.append({"item_id": item, "reason": "item is not present in the DatasetVersion snapshot"})
    unassigned_items = sorted(known_items - set(item_splits)) if known_items else []
    if definition.get("require_complete") and unassigned_items:
        errors.append({"items": unassigned_items[:100], "count": len(unassigned_items), "reason": "Dataset items are not assigned to a split"})
    return {
        "status": "passed" if not errors and records else "failed" if errors else "incomplete",
        "errors": errors,
        "split_counts": dict(sorted(counts.items())),
        "assigned_items": len(item_splits),
        "known_items": len(known_items),
        "unassigned_items": unassigned_items,
        "duplicate_items": duplicates,
        "group_leaks": group_leaks,
    }
