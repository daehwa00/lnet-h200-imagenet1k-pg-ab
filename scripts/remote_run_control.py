"""Git-ref-backed owner control for long-running public-repository jobs."""

from __future__ import annotations

# ruff: noqa: EM101, RUF022, S603, S607, T201, TRY003
# pyright: reportExplicitAny=false
import json
import re
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "lnet.remote_run_control.v1"
MAX_BODY_BYTES = 4 * 1024
HEX_40 = re.compile(r"^[0-9a-f]{40}$")
REF_PATTERN = re.compile(r"^refs/heads/[A-Za-z0-9._/-]+$")


class StopRequestedError(RuntimeError):
    """Raised only at an application-defined safe stopping boundary."""

    def __init__(self, record: dict[str, Any]) -> None:
        super().__init__("repository owner requested a graceful stop")
        self.record = record


@dataclass(slots=True)
class GitHubRunControl:
    """Poll one strict JSON file through an uncached mutable Git branch ref.

    The deployment commit stays immutable.  Only the dedicated control branch is
    mutable, and every record is bound to both the campaign and deployment SHA.
    After a valid stop is observed it remains latched for this process.
    """

    repo_url: str
    ref: str
    control_path: str
    campaign_id: str
    target_commit: str
    repo_root: Path
    poll_seconds: float = 15.0
    timeout_seconds: float = 5.0
    max_unreachable_seconds: float = 15 * 60.0
    stopped_generation: int | None = None
    updates_seen: int = 0
    active_variant: str | None = None
    last_generation: int | None = None
    last_remote_sha: str | None = None
    last_record: dict[str, Any] | None = None
    last_poll_monotonic: float = float("-inf")
    last_success_monotonic: float | None = None
    failures: int = 0
    stop_latched: bool = False
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False)
    _shutdown_event: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        parsed = urllib.parse.urlsplit(self.repo_url)
        if (
            parsed.scheme != "https"
            or parsed.hostname != "github.com"
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or not parsed.path.endswith(".git")
        ):
            raise ValueError("run-control repository must be a public HTTPS GitHub clone URL")
        if (
            REF_PATTERN.fullmatch(self.ref) is None
            or ".." in self.ref
            or "//" in self.ref
            or self.ref.endswith("/")
        ):
            raise ValueError("invalid run-control branch ref")
        path = PurePosixPath(self.control_path)
        if (
            path.is_absolute()
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
            or "\\" in self.control_path
            or "\0" in self.control_path
        ):
            raise ValueError("invalid run-control repository path")
        if (
            not self.campaign_id
            or HEX_40.fullmatch(self.target_commit) is None
            or self.poll_seconds < 1.0
            or self.timeout_seconds <= 0.0
            or self.max_unreachable_seconds < self.poll_seconds
            or isinstance(self.stopped_generation, bool)
            or (self.stopped_generation is not None and self.stopped_generation < 1)
        ):
            raise ValueError("invalid run-control identity or timing")
        self.repo_root = self.repo_root.resolve()

    @property
    def stop_requested(self) -> bool:
        return self._stop_event.is_set()

    def set_variant(self, variant: str) -> None:
        self.active_variant = variant
        self.updates_seen = 0

    def require_start(
        self,
        *,
        attempts: int = 3,
        delay_seconds: float = 2.0,
    ) -> dict[str, Any]:
        """Fail closed before expensive work unless a current run record is valid."""
        if attempts < 1:
            raise ValueError("startup attempts must be positive")
        for attempt in range(attempts):
            record = self.poll(force=True)
            if record is not None:
                return record
            if attempt + 1 < attempts:
                time.sleep(delay_seconds)
        raise RuntimeError("run-control branch is unavailable or invalid at startup")

    def start_background(self) -> None:
        """Poll off the training thread; callbacks only inspect an Event."""
        if self._thread is not None:
            return
        self._shutdown_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="lnet-run-control",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._shutdown_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=self.timeout_seconds + 1.0)

    def after_optimizer_step(self) -> None:
        """Raise after a completed update without doing network I/O here."""
        self.updates_seen += 1
        self.raise_if_stopped()

    def raise_if_stopped(self) -> None:
        if not self._stop_event.is_set():
            return
        record = self.last_record or self._synthetic_stop("stop_requested")
        raise StopRequestedError(record)

    def poll(self, *, force: bool = False) -> dict[str, Any] | None:
        self.raise_if_stopped()
        now = time.monotonic()
        if not force and now - self.last_poll_monotonic < self.poll_seconds:
            return None
        self.last_poll_monotonic = now
        try:
            record = self._fetch()
            self._validate_generation(record)
        except (OSError, subprocess.SubprocessError, TimeoutError, TypeError, ValueError) as error:
            return self._degraded(now, error)
        self.failures = 0
        self.last_success_monotonic = now
        self.last_generation = int(record["generation"])
        self.last_record = record
        if record["action"] == "stop":
            self._latch_stop(record)
        return record

    def _poll_loop(self) -> None:
        while not self._shutdown_event.wait(self.poll_seconds):
            try:
                self.poll(force=True)
            except StopRequestedError:
                return

    def _degraded(self, now: float, error: Exception) -> None:
        self.failures += 1
        if self.failures == 1 or self.failures % 10 == 0:
            print(
                "RUN_CONTROL_DEGRADED="
                + json.dumps(
                    {
                        "error": type(error).__name__,
                        "failures": self.failures,
                        "variant": self.active_variant,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if (
            self.last_success_monotonic is not None
            and now - self.last_success_monotonic >= self.max_unreachable_seconds
        ):
            self._latch_stop(self._synthetic_stop("control_unreachable"))

    def _latch_stop(self, record: dict[str, Any]) -> None:
        self.last_record = record
        self.stop_latched = True
        self._stop_event.set()
        raise StopRequestedError(record)

    def _synthetic_stop(self, reason: str) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "campaign_id": self.campaign_id,
            "target_commit": self.target_commit,
            "generation": self.last_generation or self.stopped_generation or 0,
            "action": "stop",
            "updated_at": datetime.now().astimezone().isoformat(),
            "reason": reason,
        }

    def _fetch(self) -> dict[str, Any]:
        remote_sha = self._remote_sha()
        if remote_sha == self.last_remote_sha and self.last_record is not None:
            return self.last_record
        repository = urllib.parse.urlsplit(self.repo_url).path.removesuffix(".git").lstrip("/")
        control_path = urllib.parse.quote(self.control_path, safe="/")
        raw_url = (
            f"https://raw.githubusercontent.com/{repository}/{remote_sha}/{control_path}"
        )
        request = urllib.request.Request(  # noqa: S310 -- host and commit are validated.
            raw_url,
            headers={"Cache-Control": "no-cache", "User-Agent": "lnet-owner-control/1"},
        )
        with urllib.request.urlopen(  # noqa: S310 -- request URL is constructed above.
            request,
            timeout=self.timeout_seconds,
        ) as response:
            body = response.read(MAX_BODY_BYTES + 1)
        if len(body) > MAX_BODY_BYTES:
            raise ValueError("run-control document is too large")
        payload = json.loads(body)
        record = self._validate_payload(payload)
        self.last_remote_sha = remote_sha
        return record

    def _remote_sha(self) -> str:
        completed = subprocess.run(
            ["git", "ls-remote", "--exit-code", self.repo_url, self.ref],
            check=True,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
        )
        fields = completed.stdout.split()
        if len(fields) != 2 or fields[1] != self.ref or HEX_40.fullmatch(fields[0]) is None:
            raise ValueError("run-control branch returned an invalid ref")
        return fields[0]

    def _validate_payload(self, payload: object) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise TypeError("run-control payload must be an object")
        required = {
            "schema",
            "campaign_id",
            "target_commit",
            "generation",
            "action",
            "updated_at",
            "reason",
        }
        if set(payload) != required:
            raise ValueError("run-control payload keys changed")
        if (
            payload["schema"] != SCHEMA
            or payload["campaign_id"] != self.campaign_id
            or payload["target_commit"] != self.target_commit
        ):
            raise ValueError("run-control identity changed")
        generation = payload["generation"]
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 1
            or payload["action"] not in {"run", "stop"}
            or not isinstance(payload["reason"], str)
            or not payload["reason"]
            or len(payload["reason"]) > 500
            or not isinstance(payload["updated_at"], str)
        ):
            raise ValueError("run-control values are invalid")
        updated = datetime.fromisoformat(payload["updated_at"])
        if updated.tzinfo is None:
            raise ValueError("run-control updated_at must contain a timezone")
        return payload

    def _validate_generation(self, record: dict[str, Any]) -> None:
        generation = int(record["generation"])
        if self.stopped_generation is not None and generation <= self.stopped_generation:
            self._latch_stop(self._synthetic_stop("previous_stop_remains_latched"))
        if self.last_generation is not None:
            if generation < self.last_generation:
                raise ValueError("run-control generation rolled back")
            if (
                generation == self.last_generation
                and self.last_record is not None
                and record["action"] != self.last_record["action"]
            ):
                raise ValueError("run-control action changed without a new generation")


__all__ = ["GitHubRunControl", "SCHEMA", "StopRequestedError"]
