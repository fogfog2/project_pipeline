from __future__ import annotations

from collections import defaultdict
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from ..adapters.coco import load_coco


def _iou(box_a: list[float], box_b: list[float]) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return 0.0 if union <= 0 else intersection / union


def _average_precision(flags: list[tuple[float, int]], ground_truth_count: int) -> float:
    if ground_truth_count == 0:
        return 0.0
    flags.sort(reverse=True)
    true_positive = false_positive = 0
    recalls: list[float] = []
    precisions: list[float] = []
    for _, matched in flags:
        true_positive += matched
        false_positive += 1 - matched
        recalls.append(true_positive / ground_truth_count)
        precisions.append(true_positive / (true_positive + false_positive))
    area = 0.0
    previous = 0.0
    for recall_threshold in [index / 100 for index in range(101)]:
        precision = max((p for r, p in zip(recalls, precisions) if r >= recall_threshold), default=0.0)
        area += precision * (recall_threshold - previous)
        previous = recall_threshold
    return area


def evaluate_coco_predictions(annotation_path: str, predictions: list[dict], iou_threshold: float = 0.5) -> dict:
    """A small, dependency-free COCO-style AP@IoU evaluator for onboarding fixtures.

    Full COCO AP@[.50:.95] remains an integration point for pycocotools/MMDetection.
    """
    coco = load_coco(annotation_path)
    categories = {category["id"]: category["name"] for category in coco["categories"]}
    ground_truth: dict[int, dict[tuple[int, int], list[dict]]] = defaultdict(lambda: defaultdict(list))
    for annotation in coco["annotations"]:
        if annotation.get("iscrowd", 0):
            continue
        ground_truth[annotation["category_id"]][(annotation["image_id"], annotation["category_id"])].append(annotation)
    grouped_predictions: dict[int, list[dict]] = defaultdict(list)
    invalid_predictions = 0
    for prediction in predictions:
        if prediction.get("category_id") not in categories or not isinstance(prediction.get("bbox"), list) or len(prediction["bbox"]) != 4:
            invalid_predictions += 1
            continue
        grouped_predictions[prediction["category_id"]].append(prediction)
    per_class: dict[str, dict] = {}
    aps: list[float] = []
    total_tp = total_fp = total_gt = 0
    for category_id, class_name in categories.items():
        matched: set[tuple[int, int]] = set()
        flags: list[tuple[float, int]] = []
        category_gt = sum(len(items) for items in ground_truth[category_id].values())
        for prediction in sorted(grouped_predictions[category_id], key=lambda item: item.get("score", 0), reverse=True):
            key = (prediction.get("image_id"), category_id)
            candidates = ground_truth[category_id].get(key, [])
            best_index, best_iou = None, 0.0
            for index, annotation in enumerate(candidates):
                if (category_id, annotation["id"]) in matched:
                    continue
                score = _iou(prediction["bbox"], annotation["bbox"])
                if score > best_iou:
                    best_iou, best_index = score, index
            is_tp = int(best_index is not None and best_iou >= iou_threshold)
            if is_tp:
                matched.add((category_id, candidates[best_index]["id"]))
            flags.append((float(prediction.get("score", 0)), is_tp))
        tp = sum(flag for _, flag in flags)
        fp = len(flags) - tp
        total_tp += tp; total_fp += fp; total_gt += category_gt
        ap = _average_precision(flags, category_gt)
        aps.append(ap)
        per_class[class_name] = {"category_id": category_id, "ap50": round(ap, 6), "ground_truth": category_gt, "true_positive": tp, "false_positive": fp}
    return {
        "metric_scope": "onboarding AP50 only",
        "bbox_AP50": round(sum(aps) / len(aps), 6) if aps else 0.0,
        "precision": round(total_tp / (total_tp + total_fp), 6) if total_tp + total_fp else 0.0,
        "recall": round(total_tp / total_gt, 6) if total_gt else 0.0,
        "invalid_predictions": invalid_predictions,
        "per_class": per_class,
    }


def evaluate_coco_full(annotation_path: str, predictions: list[dict]) -> dict:
    """Evaluate a COCO prediction JSON with the official COCO bbox protocol.

    This intentionally stays separate from the small AP50 evaluator used during
    onboarding.  A caller must explicitly select ``coco-full-v1`` so a quick
    fixture result can never be presented as AP@[.50:.95].
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as error:  # pragma: no cover - dependency declaration is tested by packaging
        raise RuntimeError("Full COCO evaluation requires pycocotools. Reinstall the vision-lifecycle environment.") from error

    source = Path(annotation_path)
    if not source.is_file():
        raise ValueError(f"COCO annotation does not exist: {annotation_path}")
    coco_gt = COCO(str(source))
    category_ids = {category["id"] for category in coco_gt.dataset.get("categories", [])}
    valid: list[dict] = []
    invalid = 0
    for item in predictions:
        if (
            item.get("category_id") not in category_ids
            or not isinstance(item.get("image_id"), int)
            or not isinstance(item.get("bbox"), list)
            or len(item["bbox"]) != 4
            or not isinstance(item.get("score"), (int, float))
        ):
            invalid += 1
            continue
        valid.append({key: item[key] for key in ("image_id", "category_id", "bbox", "score")})
    if not valid:
        return {
            "metric_scope": "official COCO bbox AP@[.50:.95] (no valid predictions)",
            "bbox_mAP": 0.0, "bbox_AP50": 0.0, "bbox_AP75": 0.0, "bbox_AR100": 0.0,
            "invalid_predictions": invalid, "per_class": {},
        }

    def run(category_id: int | None = None) -> list[float]:
        with NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8", delete=False) as file:
            json.dump(valid, file)
            result_path = Path(file.name)
        try:
            coco_dt = coco_gt.loadRes(str(result_path))
            evaluator = COCOeval(coco_gt, coco_dt, "bbox")
            if category_id is not None:
                evaluator.params.catIds = [category_id]
            # pycocotools prints its fixed summary. The API response should only
            # carry structured metrics, so keep the worker/server log clean.
            with redirect_stdout(io.StringIO()):
                evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
            return [float(value) for value in evaluator.stats]
        finally:
            result_path.unlink(missing_ok=True)

    stats = run()
    per_class: dict[str, dict] = {}
    for category in coco_gt.dataset.get("categories", []):
        class_stats = run(int(category["id"]))
        per_class[str(category["name"])] = {
            "category_id": category["id"], "bbox_mAP": round(class_stats[0], 6),
            "bbox_AP50": round(class_stats[1], 6), "bbox_AP75": round(class_stats[2], 6),
        }
    return {
        "metric_scope": "official COCO bbox AP@[.50:.95]",
        "bbox_mAP": round(stats[0], 6), "bbox_AP50": round(stats[1], 6),
        "bbox_AP75": round(stats[2], 6), "bbox_AR100": round(stats[8], 6),
        "invalid_predictions": invalid, "per_class": per_class,
    }
