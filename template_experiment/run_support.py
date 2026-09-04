"""Durable checkpoints and visible, resumable API collection for synthesis runs."""

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import tempfile
import threading
import time


def progress(message):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def atomic_text(path, text):
    """Replace one artifact atomically; the JSONL checkpoint is the source of truth."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_jsonl(path, rows):
    atomic_text(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                              for row in rows))


def read_jsonl(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()] if path.exists() else []


def indexed(rows, fields):
    result = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in result:
            raise ValueError(f"Duplicate checkpoint identity: {key}")
        result[key] = row
    return result


def verify_identity(prior, expected, fields):
    for field in fields:
        if prior.get(field) != expected.get(field):
            raise ValueError(f"Saved request differs in {field}; use a new output directory "
                             "for changed prompts, evidence, or models. Existing outputs are untouched.")


@contextmanager
def output_lock(output_dir):
    """Prevent two processes from charging for the same pending request."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".collection.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another collection is using {output_dir}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


@contextmanager
def heartbeat(label, interval=15):
    stopped = threading.Event()
    started = time.monotonic()

    def report():
        while not stopped.wait(interval):
            progress(f"{label}: still waiting for response ({time.monotonic() - started:.0f}s elapsed)")

    worker = threading.Thread(target=report, daemon=True)
    worker.start()
    try:
        yield
    finally:
        stopped.set()
        worker.join()


def collect_responses(jobs, *, client_factory, extract, finalize, checkpoint, label,
                      identity_field, heartbeat_interval=15, response_details=None):
    """Resume missing requests; never resample a received but invalid response.

    A request can still be billed if the process dies after the provider accepts it but
    before its response is saved. Such in-flight attempts are explicitly recorded.
    """
    saved = sum(row.get("status") in {"complete", "contract_failed"} for row in jobs)
    progress(f"{label}: {saved}/{len(jobs)} finished responses saved; "
             f"{sum(not _received(row) for row in jobs)} API requests remaining")
    client = None
    for index, row in enumerate(jobs, 1):
        tag = f"{label} [{index}/{len(jobs)}] {identity_field}={row[identity_field]}"
        if row.get("status") in {"complete", "contract_failed"}:
            progress(f"{tag}: reused saved response ({row['status']})")
            continue
        if not _received(row):
            if client is None:
                client = client_factory()
            attempts = row.setdefault("attempts", [])
            attempt = {"started_utc": datetime.now(timezone.utc).isoformat(), "status": "in_progress"}
            attempts.append(attempt)
            row.update(status="request_in_progress", error="")
            checkpoint()
            started = time.monotonic()
            progress(f"{tag}: requesting response; attempt {len(attempts)}")
            try:
                with heartbeat(tag, heartbeat_interval):
                    response = client.invoke(row["prompt"])
                raw = extract(response)
            except BaseException as exc:
                attempt.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                               finished_utc=datetime.now(timezone.utc).isoformat(),
                               elapsed_seconds=round(time.monotonic() - started, 3),
                               error_type=type(exc).__name__)
                row.update(status="request_failed", error=type(exc).__name__)
                checkpoint()
                progress(f"{tag}: {type(exc).__name__}; checkpoint saved. Rerun the same command to resume.")
                raise
            attempt.update(status="response_received", finished_utc=datetime.now(timezone.utc).isoformat(),
                           elapsed_seconds=round(time.monotonic() - started, 3))
            row.update(raw_response=raw, response_received=True, status="response_received",
                       generated_utc=attempt["finished_utc"])
            if response_details is not None:
                row.update(response_details(response))
            # Persist the paid response before parsing/validation can fail.
            checkpoint()
            progress(f"{tag}: response received and saved ({attempt['elapsed_seconds']:.1f}s)")
        try:
            finalize(row)
        except BaseException as exc:
            row.update(status="processing_failed", error=type(exc).__name__)
            checkpoint()
            progress(f"{tag}: local processing failed; raw response saved, no API retry needed")
            raise
        checkpoint()
        progress(f"{tag}: {row['status']}; checkpoint saved")


def _received(row):
    return bool(row.get("response_received") or row.get("raw_response")
                or row.get("status") in {"complete", "contract_failed", "response_received", "processing_failed"})
