from __future__ import annotations

from typing import Any


def evaluate_gate(candidate: dict[str, Any] | None, baseline: dict[str, Any] | None, config: dict[str, Any]) -> dict:
    """Evaluate explicit metric rules without silently treating missing data as pass.

    Config shape: {"minimum": {"bbox_AP50": 0.4}, "maximum":
    {"latency_ms_p90": 25}, "max_regression": {"bbox_mAP": 0.01}}.
    """
    if not config:
        return {"status": "NOT_CONFIGURED", "checks": [], "reason": "No gate rules are configured."}
    if candidate is None:
        return {"status": "INCOMPLETE", "checks": [], "reason": "Candidate evaluation is missing."}
    checks: list[dict] = []
    incomplete = False
    for name, minimum in config.get("minimum", {}).items():
        value = candidate.get(name)
        if value is None:
            incomplete = True; checks.append({"metric": name, "status": "missing", "rule": f">= {minimum}"})
        else:
            checks.append({"metric": name, "value": value, "rule": f">= {minimum}", "status": "pass" if value >= minimum else "fail"})
    for name, maximum in config.get("maximum", {}).items():
        value = candidate.get(name)
        if value is None:
            incomplete = True; checks.append({"metric": name, "status": "missing", "rule": f"<= {maximum}"})
        else:
            checks.append({"metric": name, "value": value, "rule": f"<= {maximum}", "status": "pass" if value <= maximum else "fail"})
    regressions = config.get("max_regression", {})
    if regressions and baseline is None:
        return {"status": "INCOMPLETE", "checks": checks, "reason": "Baseline evaluation is required for regression rules."}
    for name, maximum in regressions.items():
        current = candidate.get(name)
        prior = baseline.get(name) if baseline else None
        if current is None or prior is None:
            incomplete = True; checks.append({"metric": name, "status": "missing", "rule": f"regression <= {maximum}"})
        else:
            regression = prior - current
            checks.append({"metric": name, "value": regression, "rule": f"regression <= {maximum}", "status": "pass" if regression <= maximum else "fail"})
    if incomplete:
        status = "INCOMPLETE"
    elif any(check["status"] == "fail" for check in checks):
        status = "FAIL"
    else:
        status = "PASS"
    return {"status": status, "checks": checks}
