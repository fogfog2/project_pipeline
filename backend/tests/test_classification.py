from vision_lifecycle.evaluators.classification import evaluate_classification


def test_classification_metrics_and_confusion_matrix():
    result = evaluate_classification([
        {"image_id": "1", "ground_truth": "cat", "prediction": "cat", "top_k": ["cat", "dog"]},
        {"image_id": "2", "ground_truth": "dog", "prediction": "cat", "top_k": ["cat", "dog"]},
    ])
    assert result["top1_accuracy"] == 0.5
    assert result["topk_accuracy"] == 1.0
    assert result["confusion_matrix"]["dog"]["cat"] == 1


def test_classification_invalid_records_are_reported():
    result = evaluate_classification([
        {"image_id": "1", "ground_truth": "cat", "prediction": "cat", "top_k": "cat"},
        {"image_id": "1", "ground_truth": "dog", "prediction": "dog"},
        {"image_id": "2", "ground_truth": "dog", "prediction": "dog"},
    ])
    assert result["records"] == 1
    assert {item["reason"] for item in result["invalid_records"]} == {"top_k must be a non-empty list", "duplicate image_id"}
