import pytest

from vision_lifecycle.release_gate import GateConfigError, evaluate_gate


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


def test_gate_uses_lower_is_better_for_latency():
    result = evaluate_gate({"latency_ms_p90": 30}, {"latency_ms_p90": 20}, {"max_regression": {"latency_ms_p90": 2}})
    assert result["status"] == "FAIL"
    assert result["checks"][0]["value"] == 10


def test_gate_rejects_unknown_rules_and_empty_rules():
    with pytest.raises(GateConfigError):
        evaluate_gate({"accuracy": 1}, None, {"unsupported": {"accuracy": 1}})
    with pytest.raises(GateConfigError):
        evaluate_gate({"accuracy": 1}, None, {"minimum": {}})


def test_gate_requires_critical_class_metrics_and_checks_thresholds():
    config = {"critical_classes": {"person": {"ap50": 0.8, "recall": {"minimum": 0.7, "maximum": 1.0}}}}
    missing = evaluate_gate({"bbox_AP50": 0.9}, None, config, candidate_details={"per_class": {}})
    assert missing["status"] == "INCOMPLETE"
    failed = evaluate_gate({"bbox_AP50": 0.9}, None, config, candidate_details={"per_class": {"person": {"ap50": 0.75, "recall": 0.8}}})
    assert failed["status"] == "FAIL"
    passed = evaluate_gate({"bbox_AP50": 0.9}, None, config, candidate_details={"per_class": {"person": {"ap50": 0.85, "recall": 0.8}}})
    assert passed["status"] == "PASS"
