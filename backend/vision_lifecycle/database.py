from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import DeclarativeBase, sessionmaker


class Base(DeclarativeBase):
    pass


def database_url(path: str | None = None) -> str:
    db_path = Path(path or ".vision-lifecycle/registry.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path.resolve()}"


engine = create_engine(database_url(), connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_database() -> None:
    from . import models  # noqa: F401

    Base.metadata.create_all(engine)
    # Lightweight compatibility migration for the local SQLite starter. Once a
    # team moves to a shared database, the same metadata can be managed by
    # Alembic without changing the public API.
    columns = {column["name"] for column in inspect(engine).get_columns("runs")}
    additions = {"external_run_id": "VARCHAR(200)", "import_hash": "VARCHAR(128)"}
    with engine.begin() as connection:
        for name, declaration in additions.items():
            if name not in columns:
                connection.exec_driver_sql(f"ALTER TABLE runs ADD COLUMN {name} {declaration}")
