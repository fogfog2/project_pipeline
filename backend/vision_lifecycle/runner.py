from __future__ import annotations

import os
import signal
import socket
import subprocess
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from .database import SessionLocal
from .evaluation_service import evaluate_coco_prediction_file
from sqlalchemy import select, update
from sqlalchemy.orm.exc import StaleDataError

from .models import DataAsset, Job, RunnerProfile, StorageMapping
from .storage import inventory as inventory_storage


_processes: dict[str, subprocess.Popen[str]] = {}
_lock = threading.Lock()


def _signal_process_group(process: subprocess.Popen[str], signal_number: int) -> None:
    """Signal a runner and its descendants when the platform supports it."""
    if process.poll() is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(os.getpgid(process.pid), signal_number)
            return
        except (OSError, ProcessLookupError):
            # The process may have exited between poll() and killpg(). Fall
            # through to the direct signal so cancellation remains best effort.
            pass
    try:
        if signal_number == getattr(signal, "SIGTERM", 15):
            process.terminate()
        else:
            process.kill()
    except (OSError, ProcessLookupError):
        pass


def _stop_process(process: subprocess.Popen[str], *, grace_seconds: float = 2.0) -> str:
    """Stop a runner process group and return the final captured output."""
    _signal_process_group(process, getattr(signal, "SIGTERM", 15))
    try:
        output, _ = process.communicate(timeout=max(0.1, grace_seconds))
        return output or ""
    except subprocess.TimeoutExpired:
        _signal_process_group(process, getattr(signal, "SIGKILL", 9))
        output, _ = process.communicate()
        return output or ""


def _watch_cancellation(job_id: str, process: subprocess.Popen[str], stop_event: threading.Event) -> None:
    """Bridge DB cancellation requests to a process owned by another worker."""
    while not stop_event.wait(0.2):
        with SessionLocal() as session:
            current = session.get(Job, job_id)
            cancelling = bool(current and current.status == "cancelling")
        if cancelling:
            _signal_process_group(process, getattr(signal, "SIGTERM", 15))
            return


def recover_interrupted(*, include_queued: bool = True) -> int:
    """Mark jobs from a previous process as interrupted.

    API startup normally passes ``include_queued=False`` so external workers
    can still claim jobs that were queued before the API restarted. Callers
    that own the whole queue may opt into marking queued work interrupted.
    """
    changed = 0
    with SessionLocal() as session:
        statuses = ["running", "cancelling"] + (["queued"] if include_queued else [])
        jobs = session.query(Job).filter(Job.status.in_(statuses)).all()
        for job in jobs:
            job.status = "interrupted"
            job.lease_owner = None
            job.lease_expires_at = None
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
            if status in {"completed", "failed", "cancelled", "timed_out", "interrupted"}:
                job.lease_owner = None
                job.lease_expires_at = None
        if result is not None:
            job.result_json = result
        try:
            session.commit()
        except StaleDataError:
            # A service restart or test/process-local database reset may make
            # the object stale between the read and commit. Logs must never
            # turn into an unhandled daemon-thread exception; the next append
            # or status poll can recover the authoritative row if it exists.
            session.rollback()


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


def _run_prediction_evaluation(job_id: str) -> None:
    _append_log(job_id, "COCO prediction evaluation started.\n", status="running")
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        payload = dict(job.input_json or {}) if job else {}
        project_id = job.project_id if job else None
    if not project_id:
        _append_log(job_id, "Evaluation job project was not found.\n", status="failed")
        return
    try:
        with SessionLocal() as session:
            result = evaluate_coco_prediction_file(
                session,
                project_id,
                dataset_id=str(payload["dataset_id"]),
                model_id=str(payload["model_id"]),
                predictions_path=str(payload["predictions_path"]),
                protocol=str(payload.get("protocol", "onboarding_ap50")),
                iou_threshold=float(payload.get("iou_threshold", 0.5)),
                evaluator_version=str(payload.get("evaluator_version", "lifecycle-ap50-v1")),
                evaluation_set_id=payload.get("evaluation_set_id"),
            )
        run = result.get("run", {})
        _append_log(job_id, "COCO prediction evaluation completed.\n", status="completed", result={"run_id": run.get("id"), "metrics": result.get("result", {})})
    except (KeyError, OSError, RuntimeError, ValueError, TypeError) as error:
        _append_log(job_id, f"COCO prediction evaluation failed: {error}\n", status="failed")


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
        process = subprocess.Popen(
            command,
            cwd=working_directory or None,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            # A runner commonly launches a shell, framework process, or board
            # utility. Keeping a separate session lets cancellation and
            # timeout terminate the complete process tree instead of leaving
            # grandchildren alive in the background.
            start_new_session=(os.name == "posix"),
        )
        with _lock:
            _processes[job_id] = process
        cancel_watch_stop = threading.Event()
        cancel_watch = threading.Thread(target=_watch_cancellation, args=(job_id, process, cancel_watch_stop), daemon=True)
        cancel_watch.start()
        try:
            output, _ = process.communicate(timeout=timeout)
        finally:
            cancel_watch_stop.set()
            cancel_watch.join(timeout=0.5)
        with _lock:
            _processes.pop(job_id, None)
        with SessionLocal() as session:
            current = session.get(Job, job_id)
            was_cancelled = bool(current and current.status == "cancelling")
        current_status = "cancelled" if was_cancelled else "completed" if process.returncode == 0 else "failed"
        _append_log(job_id, output or "", status=current_status, result={"exit_code": process.returncode, "runner": profile_id})
    except subprocess.TimeoutExpired as error:
        with _lock:
            process = _processes.pop(job_id, None)
        if process:
            output = _stop_process(process)
            with SessionLocal() as session:
                current = session.get(Job, job_id)
                was_cancelled = bool(current and current.status == "cancelling")
            status = "cancelled" if was_cancelled else "timed_out"
            result = {"runner": profile_id, "timeout_seconds": timeout}
            _append_log(job_id, output or "", status=status, result=result)
        else:
            _append_log(job_id, "Runner timed out.\n", status="timed_out")
    except OSError as error:
        _append_log(job_id, f"Runner failed to start: {error}\n", status="failed")


def run_job(job_id: str, runner_id: str, args: list[str]) -> None:
    if runner_id == "mock-board":
        _run_mock_board(job_id)
    elif runner_id == "builtin:storage-inventory":
        _run_inventory(job_id)
    elif runner_id == "builtin:evaluate-coco-predictions":
        _run_prediction_evaluation(job_id)
    else:
        _run_profile(job_id, runner_id, args)


def worker_identity() -> str:
    return f"{socket.gethostname()}:{os.getpid()}:{uuid4().hex[:8]}"


def reclaim_expired_leases(*, now: datetime | None = None) -> int:
    """Return abandoned running jobs to the queue after their lease expires.

    Only ``running`` jobs are requeued. A job already marked ``cancelling`` is
    left for its owning worker so a late process termination cannot create a
    duplicate execution.
    """
    current_time = now or datetime.now(UTC)
    with SessionLocal() as session:
        jobs = session.scalars(select(Job).where(Job.status == "running", Job.lease_expires_at.is_not(None), Job.lease_expires_at <= current_time)).all()
        for job in jobs:
            job.status = "queued"
            job.lease_owner = None
            job.lease_expires_at = None
            job.log = f"{job.log}Worker lease expired; job returned to queue.\n"
        session.commit()
        return len(jobs)


def renew_lease(job_id: str, worker_id: str, *, lease_seconds: int = 300) -> bool:
    """Extend a lease only while this worker still owns the running job."""
    expires = datetime.now(UTC) + timedelta(seconds=max(10, lease_seconds))
    with SessionLocal() as session:
        result = session.execute(
            update(Job)
            .where(Job.id == job_id, Job.status.in_(["running", "cancelling"]), Job.lease_owner == worker_id)
            .values(lease_expires_at=expires)
        )
        session.commit()
        return result.rowcount == 1


def _lease_heartbeat(job_id: str, worker_id: str, stop_event: threading.Event, lease_seconds: int = 300) -> None:
    interval = max(1.0, min(30.0, lease_seconds / 3))
    while not stop_event.wait(interval):
        if not renew_lease(job_id, worker_id, lease_seconds=lease_seconds):
            return


def claim_next_job(worker_id: str | None = None, lease_seconds: int = 300) -> tuple[str, str, list[str]] | None:
    """Atomically claim one queued job and record its worker lease."""
    reclaim_expired_leases()
    owner = worker_id or worker_identity()
    expires = datetime.now(UTC) + timedelta(seconds=max(10, lease_seconds))
    with SessionLocal() as session:
        candidate = session.scalar(select(Job).where(Job.status == "queued").order_by(Job.created_at).limit(1))
        if not candidate:
            return None
        result = session.execute(update(Job).where(Job.id == candidate.id, Job.status == "queued").values(status="running", lease_owner=owner, lease_expires_at=expires, attempt_count=Job.attempt_count + 1))
        if result.rowcount != 1:
            session.rollback()
            return None
        args = candidate.input_json.get("_runner_args", []) if isinstance(candidate.input_json, dict) else []
        session.commit()
        return candidate.id, candidate.runner_id, args


def worker_once(worker_id: str | None = None) -> bool:
    owner = worker_id or worker_identity()
    claimed = claim_next_job(worker_id=owner)
    if not claimed:
        return False
    stop_event = threading.Event()
    heartbeat = threading.Thread(target=_lease_heartbeat, args=(claimed[0], owner, stop_event), daemon=True)
    heartbeat.start()
    try:
        run_job(*claimed)
    finally:
        stop_event.set()
        heartbeat.join(timeout=1.0)
    return True


def run_worker(*, poll_seconds: float = 1.0, once: bool = False) -> None:
    recover_interrupted(include_queued=False)
    reclaim_expired_leases()
    worker_id = worker_identity()
    while True:
        worked = worker_once(worker_id)
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
    # Keep cancellation semantics identical to timeout handling: a runner
    # may spawn a shell, framework process, or board utility. Terminating only
    # the immediate parent leaves those descendants behind and can corrupt a
    # later retry. POSIX runners are started in their own session, so signal
    # the whole process group here.
    _signal_process_group(process, getattr(signal, "SIGTERM", 15))
    return True
