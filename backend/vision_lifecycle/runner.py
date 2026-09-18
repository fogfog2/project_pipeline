from __future__ import annotations

import os
import subprocess
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .database import SessionLocal
from sqlalchemy import select, update

from .models import DataAsset, Job, RunnerProfile, StorageMapping
from .storage import inventory as inventory_storage


_processes: dict[str, subprocess.Popen[str]] = {}
_lock = threading.Lock()


def recover_interrupted(*, include_queued: bool = True) -> int:
    """Mark jobs from a previous process as interrupted.

    The API service treats queued work as interrupted because it does not own
    execution. An external worker leaves queued work available to claim.
    """
    changed = 0
    with SessionLocal() as session:
        statuses = ["running", "cancelling"] + (["queued"] if include_queued else [])
        jobs = session.query(Job).filter(Job.status.in_(statuses)).all()
        for job in jobs:
            job.status = "interrupted"
            job.log = f"{job.log}Service restarted; manual retry is required.\n"
            changed += 1
        session.commit()
    return changed


def _append_log(job_id: str, message: str, *, status: str | None = None, result: dict[str, Any] | None = None) -> None:
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        if not job:
            return
        job.log = f"{job.log}{message}"
        if status:
            job.status = status
        if result is not None:
            job.result_json = result
        session.commit()


def _run_mock_board(job_id: str) -> None:
    _append_log(job_id, "Mock board runner started.\n", status="running")
    result = {
        "source": "mock-board",
        "measured_at": datetime.now(UTC).isoformat(),
        "warning": "This is an integration fixture, not a hardware measurement.",
        "metrics": {"latency_ms_p50": 12.4, "latency_ms_p90": 15.1, "memory_mb_peak": 148.0},
    }
    _append_log(job_id, "Mock board result collected.\n", status="completed", result=result)


def _run_inventory(job_id: str) -> None:
    _append_log(job_id, "Storage inventory started.\n", status="running")
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        mapping_id = (job.input_json or {}).get("storage_id") if job else None
        mapping = session.get(StorageMapping, mapping_id) if mapping_id else None
        relative_path = (job.input_json or {}).get("relative_path", "") if job else ""
        recursive = bool((job.input_json or {}).get("recursive", True)) if job else True
        limit = int((job.input_json or {}).get("limit", 1000)) if job else 1000
        project_id = job.project_id if job else None
    if not mapping or not project_id:
        _append_log(job_id, "Storage mapping not found.\n", status="failed")
        return
    try:
        entries = inventory_storage(mapping.root_path, relative_path, recursive, limit)
        imported = 0
        with SessionLocal() as session:
            for entry in entries:
                current = session.get(Job, job_id)
                if current and current.status == "cancelling":
                    _append_log(job_id, f"Inventory cancelled after {imported} files.\n", status="cancelled", result={"count": imported})
                    return
                asset = session.query(DataAsset).filter_by(project_id=project_id, storage_id=mapping.id, relative_path=entry["relative_path"]).first()
                if asset:
                    asset.size_bytes = entry["size_bytes"]; asset.sha256 = entry["sha256"]; asset.status = "discovered"
                else:
                    session.add(DataAsset(project_id=project_id, storage_id=mapping.id, relative_path=entry["relative_path"], size_bytes=entry["size_bytes"], sha256=entry["sha256"], metadata_json={"suffix": entry["suffix"]}))
                imported += 1
                if imported % 100 == 0:
                    session.commit(); _append_log(job_id, f"Inventory processed {imported} files.\n")
            session.commit()
        _append_log(job_id, "Storage inventory completed.\n", status="completed", result={"count": imported, "truncated": imported >= limit})
    except (OSError, ValueError) as error:
        _append_log(job_id, f"Storage inventory failed: {error}\n", status="failed")


def _run_profile(job_id: str, profile_id: str, args: list[str]) -> None:
    with SessionLocal() as session:
        profile = session.get(RunnerProfile, profile_id)
        job = session.get(Job, job_id)
        if not profile or not job:
            return
        command = [profile.executable, *profile.default_args, *args]
        working_directory = profile.working_directory
        timeout = profile.timeout_seconds
        allowed_environment = {name: os.environ[name] for name in profile.environment_names if name in os.environ}
    if working_directory and not Path(working_directory).is_dir():
        _append_log(job_id, f"Working directory does not exist: {working_directory}\n", status="failed")
        return
    environment = {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", ""), "LANG": os.environ.get("LANG", "C.UTF-8"), **allowed_environment}
    _append_log(job_id, f"Runner started: {command[0]}\n", status="running")
    try:
        process = subprocess.Popen(command, cwd=working_directory or None, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        with _lock:
            _processes[job_id] = process
        output, _ = process.communicate(timeout=timeout)
        with _lock:
            _processes.pop(job_id, None)
        with SessionLocal() as session:
            current = session.get(Job, job_id)
            was_cancelled = bool(current and current.status == "cancelling")
        current_status = "cancelled" if was_cancelled else "completed" if process.returncode == 0 else "failed"
        _append_log(job_id, output or "", status=current_status, result={"exit_code": process.returncode, "runner": profile_id})
    except subprocess.TimeoutExpired:
        with _lock:
            process = _processes.pop(job_id, None)
        if process:
            process.kill()
            output, _ = process.communicate()
            _append_log(job_id, output or "", status="timed_out", result={"runner": profile_id, "timeout_seconds": timeout})
        else:
            _append_log(job_id, "Runner timed out.\n", status="timed_out")
    except OSError as error:
        _append_log(job_id, f"Runner failed to start: {error}\n", status="failed")


def run_job(job_id: str, runner_id: str, args: list[str]) -> None:
    if runner_id == "mock-board":
        _run_mock_board(job_id)
    elif runner_id == "builtin:storage-inventory":
        _run_inventory(job_id)
    else:
        _run_profile(job_id, runner_id, args)


def claim_next_job() -> tuple[str, str, list[str]] | None:
    """Atomically claim one queued job for an external worker process."""
    with SessionLocal() as session:
        candidate = session.scalar(select(Job).where(Job.status == "queued").order_by(Job.created_at).limit(1))
        if not candidate:
            return None
        result = session.execute(update(Job).where(Job.id == candidate.id, Job.status == "queued").values(status="running"))
        if result.rowcount != 1:
            session.rollback()
            return None
        args = candidate.input_json.get("_runner_args", []) if isinstance(candidate.input_json, dict) else []
        session.commit()
        return candidate.id, candidate.runner_id, args


def worker_once() -> bool:
    claimed = claim_next_job()
    if not claimed:
        return False
    run_job(*claimed)
    return True


def run_worker(*, poll_seconds: float = 1.0, once: bool = False) -> None:
    recover_interrupted(include_queued=False)
    while True:
        worked = worker_once()
        if once:
            return
        if not worked:
            time.sleep(max(0.1, poll_seconds))


def launch(job: Job, args: list[str]) -> None:
    threading.Thread(target=run_job, args=(job.id, job.runner_id, args), daemon=True).start()


def cancel(job_id: str) -> bool:
    with _lock:
        process = _processes.get(job_id)
    if not process:
        return False
    process.terminate()
    return True
