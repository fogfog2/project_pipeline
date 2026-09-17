from __future__ import annotations

from sqlalchemy.inspection import inspect


def as_dict(item):
    return {column.key: getattr(item, column.key) for column in inspect(item).mapper.column_attrs}
