from __future__ import annotations

from typing import Any
import math


class GateConfigError(ValueError):
    """Raised when a release gate contains an unsupported or unsafe rule."""


_SECTIONS = {"minimum", "maximum", "max_regression", "direction"}
_LOWER_IS_BETTER_PREFIXES = ("latency", "memory", "model_size", "power", "temperature", "startup", "failure", "error")


def _number(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise GateConfigError(f"Gate value for {name} must be a finite number")
    return float(value)


def _validate_config(config: dict[str, Any]) -> dict[str, dict[str, float]]:
    if not isinstance(config, dict):
        raise GateConfigError("Gate config must be an object")
    unknown = set(config) - _SECTIONS
    if unknown:
        raise GateConfigError(f"Unsupported gate sections: {', '.join(sorted(unknown))}")
    rules: dict[str, dict[str, float]] = {}
    for section in ("minimum", "maximum", "max_regression"):
        values = config.get(section, {})
        if not isinstance(values, dict):
            raise GateConfigError(f"Gate section {section} must be an object")
        rules[section] = {str(name): _number(value, str(name)) for name, value in values.items()}
    directions = config.get("direction", {})
    if not isinstance(directions, dict) or any(value not in {"higher_is_better", "lower_is_better"} for value in directions.values()):
        raise GateConfigError("Gate direction must be higher_is_better or lower_is_better")
    rules["direction"] = directions
    if not any(rules[section] for section in ("minimum", "maximum", "max_regression")):
        raise GateConfigError("Gate config has no metric rules")
    return rules


def _direction(metric: str, directions: dict[str, str]) -> str:
    return directions.get(metric, "lower_is_better" if metric.lower().startswith(_LOWER_IS_BETTER_PREFIXES) else "higher_is_better")


def evaluate_gate(candidate: dict[str, Any] | None, baseline: dict[str, Any] | None, config: dict[str, Any]) -> dict:
    """Evaluate explicit metric rules without silently treating missing data as pass.

    Config shape: {"minimum": {"bbox_AP50": 0.4}, "maximum":
    {"latency_ms_p90": 25}, "max_regression": {"bbox_mAP": 0.01}}.
    """
    if not config:
        return {"status": "NOT_CONFIGURED", "checks": [], "reason": "No gate rules are configured."}
    rules = _validate_config(config)
    if candidate is None:
        return {"status": "INCOMPLETE", "checks": [], "reason": "Candidate evaluation is missing."}
    checks: list[dict] = []
    incomplete = False
    for name, minimum in rules["minimum"].items():
        value = candidate.get(name)
        if value is None:
            incomplete = True; checks.append({"metric": name, "status": "missing", "rule": f">= {minimum}"})
        else:
            checks.append({"metric": name, "value": value, "rule": f">= {minimum}", "status": "pass" if value >= minimum else "fail"})
    for name, maximum in rules["maximum"].items():
        value = candidate.get(name)
        if value is None:
            incomplete = True; checks.append({"metric": name, "status": "missing", "rule": f"<= {maximum}"})
        else:
            checks.append({"metric": name, "value": value, "rule": f"<= {maximum}", "status": "pass" if value <= maximum else "fail"})
    regressions = rules["max_regression"]
    if regressions and baseline is None:
        return {"status": "INCOMPLETE", "checks": checks, "reason": "Baseline evaluation is required for regression rules."}
    for name, maximum in regressions.items():
        current = candidate.get(name)
        prior = baseline.get(name) if baseline else None
        if current is None or prior is None:
            incomplete = True; checks.append({"metric": name, "status": "missing", "rule": f"regression <= {maximum}"})
        else:
            regression = prior - current if _direction(name, rules["direction"]) == "higher_is_better" else current - prior
            checks.append({"metric": name, "value": regression, "rule": f"regression <= {maximum}", "status": "pass" if regression <= maximum else "fail"})
    if incomplete:
        status = "INCOMPLETE"
    elif any(check["status"] == "fail" for check in checks):
        status = "FAIL"
    else:
        status = "PASS"
    return {"status": status, "checks": checks}
