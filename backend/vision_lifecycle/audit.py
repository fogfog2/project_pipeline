from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy.orm import Session

from .models import AuditEvent


def _json_safe(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(nested) for nested in value]
    return value


def record_audit(
    session: Session,
    project_id: str,
    entity_type: str,
    entity_id: str,
    action: str,
    *,
    before: dict[str, Any] | None = None,
    after: dict[str, Any] | None = None,
    details: dict[str, Any] | None = None,
    actor: str = "local-operator",
) -> AuditEvent:
    """Stage an immutable audit event in the current transaction."""
    event = AuditEvent(
        project_id=project_id,
        entity_type=entity_type,
        entity_id=entity_id,
        action=action,
        actor=actor,
        before_json=_json_safe(before or {}),
        after_json=_json_safe(after or {}),
        details=_json_safe(details or {}),
    )
    session.add(event)
    return event
