from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCRIPT = REPO_ROOT / "reproducibility" / "run_live_audit.py"
FRONTEND_DIST = REPO_ROOT / "reproducibility_site" / "dist"

DEFAULT_ORIGINS = [
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "http://127.0.0.1:4173",
    "http://localhost:4173",
]

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "REPRO_ALLOWED_ORIGINS",
        ",".join(DEFAULT_ORIGINS),
    ).split(",")
    if origin.strip()
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RunRecord:
    id: str
    status: str = "queued"
    current_step: int = 0
    current_step_title: str = "Waiting"
    started_at: str | None = None
    completed_at: str | None = None
    return_code: int | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    process: subprocess.Popen[str] | None = None
    event_lock: threading.Lock = field(default_factory=threading.Lock)

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "current_step": self.current_step,
            "current_step_title": self.current_step_title,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "return_code": self.return_code,
            "event_count": len(self.events),
        }


app = FastAPI(
    title="Breast DCE-MRI Reproducibility API",
    version="0.1.1",
    description=(
        "Local API for a fixed reviewer-facing reproducibility audit. "
        "It does not expose arbitrary shell execution and does not run clinical software."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

runs: dict[str, RunRecord] = {}
runs_lock = threading.Lock()


def add_event(record: RunRecord, event_type: str, **payload: Any) -> None:
    with record.event_lock:
        event = {
            "seq": len(record.events),
            "type": event_type,
            "timestamp": utc_now(),
            **payload,
        }
        record.events.append(event)


def classify_log(line: str) -> str:
    upper = line.upper()
    if "[PASS]" in upper or "ALL CHECKS PASSED" in upper:
        return "pass"
    if "[FAIL]" in upper or "[ERROR]" in upper or "FAILED" in upper:
        return "error"
    if "[WARN]" in upper or "WARNING" in upper:
        return "warn"
    return "info"


def execute_audit(record: RunRecord) -> None:
    """Run the fixed audit in a normal background thread.

    This intentionally uses subprocess.Popen instead of asyncio subprocesses.
    That makes process launching reliable on Windows/PowerShell regardless of
    the event-loop policy selected by Uvicorn.
    """
    record.status = "running"
    record.started_at = utc_now()
    add_event(record, "run_started", status="running")

    if not AUDIT_SCRIPT.exists():
        record.status = "failed"
        record.completed_at = utc_now()
        add_event(
            record,
            "log",
            text=f"[ERROR] Audit script not found: {AUDIT_SCRIPT}",
            kind="error",
        )
        add_event(
            record,
            "run_failed",
            status="failed",
            message="Audit script not found.",
        )
        return

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    try:
        process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(AUDIT_SCRIPT),
            ],
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        record.process = process

        assert process.stdout is not None

        for raw_line in process.stdout:
            line = raw_line.rstrip("\r\n")

            if line.startswith("@@STAGE|"):
                parts = line.split("|", 2)
                if len(parts) == 3:
                    try:
                        step = int(parts[1])
                    except ValueError:
                        step = record.current_step

                    title = parts[2]
                    record.current_step = step
                    record.current_step_title = title
                    add_event(
                        record,
                        "stage",
                        step=step,
                        title=title,
                    )
                continue

            if line.startswith("@@AUDIT_DONE|"):
                continue

            add_event(
                record,
                "log",
                text=line,
                kind=classify_log(line),
            )

        return_code = process.wait()
        record.return_code = return_code
        record.process = None
        record.completed_at = utc_now()

        if record.status == "cancelled":
            add_event(record, "run_cancelled", status="cancelled")
            return

        if return_code == 0:
            record.status = "completed"
            record.current_step = 5
            record.current_step_title = "Canonical Audit"
            add_event(record, "run_completed", status="completed")
        else:
            record.status = "failed"
            add_event(
                record,
                "run_failed",
                status="failed",
                message=f"Audit exited with code {return_code}.",
            )

    except Exception as exc:
        record.process = None
        record.status = "failed"
        record.completed_at = utc_now()

        add_event(
            record,
            "log",
            text=f"[ERROR] Backend could not start the audit: {type(exc).__name__}: {exc}",
            kind="error",
        )
        add_event(
            record,
            "run_failed",
            status="failed",
            message=f"{type(exc).__name__}: {exc}",
        )


@app.get("/api/health")
def health() -> dict[str, Any]:
    active = next(
        (
            record.public()
            for record in runs.values()
            if record.status in {"queued", "running"}
        ),
        None,
    )

    return {
        "ok": AUDIT_SCRIPT.exists(),
        "mode": "local-reproducibility-audit",
        "audit_script": str(AUDIT_SCRIPT.relative_to(REPO_ROOT)),
        "frontend_build_available": FRONTEND_DIST.exists(),
        "active_run": active,
    }


@app.post("/api/runs", status_code=201)
def start_run() -> dict[str, Any]:
    with runs_lock:
        active = next(
            (
                record
                for record in runs.values()
                if record.status in {"queued", "running"}
            ),
            None,
        )

        if active:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "A reproducibility audit is already running.",
                    "run_id": active.id,
                },
            )

        run_id = uuid.uuid4().hex[:12]
        record = RunRecord(id=run_id)
        runs[run_id] = record

        worker = threading.Thread(
            target=execute_audit,
            args=(record,),
            name=f"repro-audit-{run_id}",
            daemon=True,
        )
        worker.start()

        return record.public()


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    record = runs.get(run_id)

    if not record:
        raise HTTPException(status_code=404, detail="Run not found.")

    return record.public()


@app.post("/api/runs/{run_id}/cancel")
def cancel_run(run_id: str) -> dict[str, Any]:
    record = runs.get(run_id)

    if not record:
        raise HTTPException(status_code=404, detail="Run not found.")

    if record.status not in {"queued", "running"}:
        return record.public()

    record.status = "cancelled"
    record.completed_at = utc_now()

    process = record.process
    if process is not None and process.poll() is None:
        try:
            process.terminate()
        except Exception as exc:
            add_event(
                record,
                "log",
                text=f"[WARN] Could not terminate audit cleanly: {exc}",
                kind="warn",
            )

    add_event(
        record,
        "log",
        text="[WARN] Audit cancelled by user.",
        kind="warn",
    )

    return record.public()


@app.get("/api/runs/{run_id}/events")
async def run_events(run_id: str) -> StreamingResponse:
    record = runs.get(run_id)

    if not record:
        raise HTTPException(status_code=404, detail="Run not found.")

    async def event_stream():
        cursor = 0
        heartbeat_ticks = 0

        while True:
            batch: list[dict[str, Any]] = []

            with record.event_lock:
                if cursor < len(record.events):
                    batch = record.events[cursor:]
                    cursor = len(record.events)

            for event in batch:
                yield f"data: {json.dumps(event, separators=(',', ':'))}\n\n"

            if record.status in {"completed", "failed", "cancelled"}:
                with record.event_lock:
                    caught_up = cursor >= len(record.events)

                if caught_up:
                    break

            await asyncio.sleep(0.15)

            heartbeat_ticks += 1
            if heartbeat_ticks >= 60:
                heartbeat_ticks = 0
                yield ": heartbeat\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if FRONTEND_DIST.exists():
    app.mount(
        "/",
        StaticFiles(
            directory=str(FRONTEND_DIST),
            html=True,
        ),
        name="frontend",
    )
