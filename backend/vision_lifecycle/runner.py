from __future__ import annotations

import os
import subprocess
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .database import SessionLocal
from .models import Job, RunnerProfile


_processes: dict[str, subprocess.Popen[str]] = {}
_lock = threading.Lock()


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


def launch(job: Job, args: list[str]) -> None:
    if job.runner_id == "mock-board":
        target = _run_mock_board
        target_args = (job.id,)
    else:
        target = _run_profile
        target_args = (job.id, job.runner_id, args)
    threading.Thread(target=target, args=target_args, daemon=True).start()


def cancel(job_id: str) -> bool:
    with _lock:
        process = _processes.get(job_id)
    if not process:
        return False
    process.terminate()
    return True
