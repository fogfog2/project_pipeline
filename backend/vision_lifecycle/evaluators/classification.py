from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def evaluate_classification(records: list[dict[str, Any]]) -> dict:
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
        labels.update([truth, prediction]); matrix[truth][prediction] += 1
        if truth in [str(value) for value in top_k]:
            top_k_hits += 1
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
    return {
        "top1_accuracy": round(correct / valid, 6),
        "topk_accuracy": round(top_k_hits / valid, 6),
        "macro_precision": round(sum(precision_values) / len(precision_values), 6),
        "macro_recall": round(sum(recall_values) / len(recall_values), 6),
        "macro_f1": round(sum(f1_values) / len(f1_values), 6),
        "records": valid,
        "invalid_records": errors,
        "per_class": per_class,
        "confusion_matrix": {truth: dict(predictions) for truth, predictions in matrix.items()},
    }
