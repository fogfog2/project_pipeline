from __future__ import annotations

from typing import Any
import math


class GateConfigError(ValueError):
    """Raised when a release gate contains an unsupported or unsafe rule."""


_SECTIONS = {"minimum", "maximum", "max_regression", "direction"}
_METADATA_KEYS = {"required_evidence", "required_evidence_refs", "critical_classes"}
_LOWER_IS_BETTER_PREFIXES = ("latency", "memory", "model_size", "power", "temperature", "startup", "failure", "error")


def _number(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise GateConfigError(f"Gate value for {name} must be a finite number")
    return float(value)


def _validate_config(config: dict[str, Any]) -> dict[str, dict[str, float]]:
    if not isinstance(config, dict):
        raise GateConfigError("Gate config must be an object")
    unknown = set(config) - _SECTIONS - _METADATA_KEYS
    if unknown:
        raise GateConfigError(f"Unsupported gate sections: {', '.join(sorted(unknown))}")
    if "required_evidence" in config and (not isinstance(config["required_evidence"], list) or any(not isinstance(value, str) for value in config["required_evidence"])):
        raise GateConfigError("required_evidence must be a list of strings")
    refs = config.get("required_evidence_refs", [])
    if not isinstance(refs, list) or any(not isinstance(value, dict) or not isinstance(value.get("type"), str) or not isinstance(value.get("id"), str) or not value["type"] or not value["id"] for value in refs):
        raise GateConfigError("required_evidence_refs must be a list of objects with type and id strings")
    critical = config.get("critical_classes", {})
    if not isinstance(critical, dict):
        raise GateConfigError("critical_classes must be an object")
    for class_name, class_rules in critical.items():
        if not isinstance(class_name, str) or not isinstance(class_rules, dict) or not class_rules:
            raise GateConfigError("critical_classes entries must map a class name to metric rules")
        for metric, rule in class_rules.items():
            if isinstance(rule, (int, float)) and not isinstance(rule, bool):
                _number(rule, f"{class_name}.{metric}")
            elif isinstance(rule, dict):
                if set(rule) - {"minimum", "maximum"} or not any(key in rule for key in ("minimum", "maximum")):
                    raise GateConfigError(f"Critical class rule for {class_name}.{metric} must contain minimum or maximum")
                for key in ("minimum", "maximum"):
                    if key in rule:
                        _number(rule[key], f"{class_name}.{metric}.{key}")
            else:
                raise GateConfigError(f"Critical class rule for {class_name}.{metric} must be a number or object")
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
    if not any(rules[section] for section in ("minimum", "maximum", "max_regression")) and not critical:
        raise GateConfigError("Gate config has no metric rules")
    return rules


def _direction(metric: str, directions: dict[str, str]) -> str:
    return directions.get(metric, "lower_is_better" if metric.lower().startswith(_LOWER_IS_BETTER_PREFIXES) else "higher_is_better")


def evaluate_gate(candidate: dict[str, Any] | None, baseline: dict[str, Any] | None, config: dict[str, Any], *, candidate_details: dict[str, Any] | None = None, baseline_details: dict[str, Any] | None = None) -> dict:
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
    critical = config.get("critical_classes", {})
    per_class = (candidate_details or {}).get("per_class", {}) if isinstance(candidate_details, dict) else {}
    for class_name, class_rules in critical.items():
        class_metrics = per_class.get(class_name) if isinstance(per_class, dict) else None
        for metric, raw_rule in class_rules.items():
            rule = {"minimum": raw_rule} if isinstance(raw_rule, (int, float)) and not isinstance(raw_rule, bool) else raw_rule
            value = class_metrics.get(metric) if isinstance(class_metrics, dict) else None
            if value is None:
                incomplete = True
                checks.append({"metric": f"{class_name}.{metric}", "status": "missing", "rule": rule})
                continue
            if "minimum" in rule:
                checks.append({"metric": f"{class_name}.{metric}", "value": value, "rule": f">= {rule['minimum']}", "status": "pass" if value >= rule["minimum"] else "fail"})
            if "maximum" in rule:
                checks.append({"metric": f"{class_name}.{metric}", "value": value, "rule": f"<= {rule['maximum']}", "status": "pass" if value <= rule["maximum"] else "fail"})
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
