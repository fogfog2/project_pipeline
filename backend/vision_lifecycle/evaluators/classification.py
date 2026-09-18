from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any


def _confidence_calibration(records: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [record for record in records if "confidence" in record]
    if not scored:
        return {"status": "not_available", "records": 0, "bins": []}
    values = [float(record["confidence"]) for record in scored]
    bins: list[dict[str, Any]] = []
    total_error = 0.0
    for index in range(10):
        lower = index / 10
        upper = 1.0 if index == 9 else (index + 1) / 10
        members = [record for record in scored if lower <= float(record["confidence"]) <= upper and (index == 9 or float(record["confidence"]) < upper)]
        accuracy = sum(str(record["ground_truth"]) == str(record["prediction"]) for record in members) / len(members) if members else 0.0
        mean_confidence = sum(float(record["confidence"]) for record in members) / len(members) if members else 0.0
        gap = abs(accuracy - mean_confidence) if members else 0.0
        total_error += len(members) / len(scored) * gap
        bins.append({"lower": round(lower, 2), "upper": round(upper, 2), "count": len(members), "accuracy": round(accuracy, 6), "confidence": round(mean_confidence, 6), "gap": round(gap, 6)})
    return {"status": "computed", "records": len(scored), "missing_confidence": len(records) - len(scored), "ece": round(total_error, 6), "bins": bins}


def evaluate_classification(records: list[dict[str, Any]], *, include_slices: bool = True) -> dict:
    """Evaluate imported classification predictions using explicit labels.

    Each record needs `image_id`, `ground_truth`, and `prediction`. Optional
    `top_k` may contain labels ordered from most to least likely.
    """
    if not records:
        raise ValueError("Classification evaluation requires at least one record")
    errors: list[dict] = []
    labels: set[str] = set()
    matrix: dict[str, Counter] = defaultdict(Counter)
    top_k_hits = 0
    seen_image_ids: set[str] = set()
    valid_records: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        required = {"image_id", "ground_truth", "prediction"}
        missing = required - record.keys()
        if missing:
            errors.append({"index": index, "reason": f"missing: {', '.join(sorted(missing))}"})
            continue
        image_id = str(record["image_id"])
        if not image_id:
            errors.append({"index": index, "reason": "image_id must not be empty"})
            continue
        if image_id in seen_image_ids:
            errors.append({"index": index, "reason": "duplicate image_id"})
            continue
        seen_image_ids.add(image_id)
        truth, prediction = str(record["ground_truth"]), str(record["prediction"])
        if not truth or not prediction:
            errors.append({"index": index, "reason": "ground_truth and prediction must not be empty"})
            continue
        top_k = record.get("top_k", [prediction])
        if not isinstance(top_k, list) or not top_k:
            errors.append({"index": index, "reason": "top_k must be a non-empty list"})
            continue
        if "confidence" in record:
            confidence = record["confidence"]
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(float(confidence)) or not 0 <= float(confidence) <= 1:
                errors.append({"index": index, "reason": "confidence must be a finite number between 0 and 1"})
                continue
        labels.update([truth, prediction]); matrix[truth][prediction] += 1
        if truth in [str(value) for value in top_k]:
            top_k_hits += 1
        valid_records.append(record)
    valid = len(records) - len(errors)
    if not valid:
        raise ValueError("No valid classification records")
    per_class: dict[str, dict] = {}
    precision_values = []; recall_values = []; f1_values = []
    for label in sorted(labels):
        true_positive = matrix[label][label]
        false_positive = sum(matrix[truth][label] for truth in labels if truth != label)
        false_negative = sum(matrix[label][prediction] for prediction in labels if prediction != label)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precision_values.append(precision); recall_values.append(recall); f1_values.append(f1)
        per_class[label] = {"precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6), "support": sum(matrix[label].values())}
    correct = sum(matrix[label][label] for label in labels)
    result = {
        "top1_accuracy": round(correct / valid, 6),
        "topk_accuracy": round(top_k_hits / valid, 6),
        "macro_precision": round(sum(precision_values) / len(precision_values), 6),
        "macro_recall": round(sum(recall_values) / len(recall_values), 6),
        "macro_f1": round(sum(f1_values) / len(f1_values), 6),
        "records": valid,
        "invalid_records": errors,
        "per_class": per_class,
        "confusion_matrix": {truth: dict(predictions) for truth, predictions in matrix.items()},
        "confidence_calibration": _confidence_calibration(valid_records),
    }
    if result["confidence_calibration"].get("status") == "computed":
        result["expected_calibration_error"] = result["confidence_calibration"]["ece"]
    if include_slices:
        slice_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in valid_records:
            names = record.get("slices", record.get("slice", []))
            names = names if isinstance(names, list) else [names]
            for name in names:
                if name not in (None, ""):
                    slice_records[str(name)].append(record)
        result["slice_metrics"] = {
            name: {
                key: subset.get(key)
                for key in ("top1_accuracy", "topk_accuracy", "macro_precision", "macro_recall", "macro_f1", "records", "expected_calibration_error")
                if key in subset
            }
            for name, subset_records in sorted(slice_records.items())
            for subset in [evaluate_classification(subset_records, include_slices=False)]
        }
    return result
