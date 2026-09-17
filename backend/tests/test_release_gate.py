from vision_lifecycle.release_gate import evaluate_gate


def test_gate_fails_metric_regression():
    result = evaluate_gate(
        {"bbox_AP50": 0.60, "latency_ms_p90": 20},
        {"bbox_AP50": 0.63},
        {"minimum": {"bbox_AP50": 0.55}, "maximum": {"latency_ms_p90": 25}, "max_regression": {"bbox_AP50": 0.02}},
    )
    assert result["status"] == "FAIL"


def test_gate_marks_missing_metric_incomplete():
    result = evaluate_gate({"bbox_AP50": 0.6}, None, {"maximum": {"latency_ms_p90": 25}})
    assert result["status"] == "INCOMPLETE"
