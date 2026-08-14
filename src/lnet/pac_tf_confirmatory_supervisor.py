from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TextIO

from .pac_tf_evidence_queue import (
    DEFAULT_ROOT as DEFAULT_SELECTED_EVIDENCE_ROOT,
)
from .pac_tf_evidence_queue import (
    EXPLORATORY_ROOT,
    load_selection_binding,
    validate_selected_evidence_root,
)
from .pac_tf_p1p2_jobs import build_p1p2_jobs
from .pac_tf_p1p2_types import P1P2Config

if TYPE_CHECKING:
    from collections.abc import Mapping

DEFAULT_PROTOCOL = Path(".omx/protocols/pac_tf_confirmatory_20260711.json")
DEFAULT_CAPACITY_ROOT = Path(".omx/results/pac-tf-confirmatory-clean-selection-20260711")
DEFAULT_UNSEEN_ROOT = Path(".omx/results/pac-tf-confirmatory-unseen-20260711")
DEFAULT_EVIDENCE_ROOT = DEFAULT_SELECTED_EVIDENCE_ROOT
DEFAULT_P1P2_ROOT = Path(".omx/results/pac-tf-p1p2-confirmatory-20260711")
DEFAULT_STATE_ROOT = Path(".omx/results/pac-tf-confirmatory-supervisor-20260711")

_WORKER_MODULES = (
    "lnet.pac_recommended_low_data_cli",
    "lnet.pac_tf_evidence_cli",
    "lnet.pac_tf_p1p2_cli",
)


@dataclass(frozen=True, slots=True)
class QueueProgress:
    name: str
    manifest: str
    total: int
    done: int
    failed: int
    running: int
    pending: int

    @property
    def complete(self) -> bool:
        return self.total > 0 and self.done == self.total and self.failed == 0


@dataclass(frozen=True, slots=True)
class Action:
    name: str
    description: str
    commands: tuple[tuple[str, ...], ...]
    requires_gpu: bool = False
    finalize: bool = False


@dataclass(frozen=True, slots=True)
class SupervisorConfig:
    protocol: Path = DEFAULT_PROTOCOL
    capacity_root: Path = DEFAULT_CAPACITY_ROOT
    unseen_root: Path = DEFAULT_UNSEEN_ROOT
    evidence_root: Path = DEFAULT_EVIDENCE_ROOT
    p1p2_root: Path = DEFAULT_P1P2_ROOT
    state_root: Path = DEFAULT_STATE_ROOT
    device: Literal["auto", "cpu", "cuda"] = "cuda"
    workers: int = 8
    total_slots: int = 16
    optimizer_mode: str = "fused"
    poll_seconds: float = 30.0
    min_free_gpu_mib: int = 32_768
    max_gpu_utilization: int = 25
    dry_run: bool = False
    once: bool = False

    def __post_init__(self) -> None:
        if self.workers < 1 or self.total_slots < 1:
            message = "workers and total-slots must be positive"
            raise ValueError(message)
        if self.poll_seconds <= 0.0:
            message = "poll-seconds must be positive"
            raise ValueError(message)
        if self.evidence_root.resolve() == EXPLORATORY_ROOT.resolve():
            message = "confirmatory supervisor refuses the exploratory D64/M16 evidence root"
            raise ValueError(message)


class ConfirmatorySupervisor:
    """Restart-safe state machine for the locked PAC-TF confirmatory program."""

    def __init__(self, config: SupervisorConfig) -> None:
        self.config = config
        self.protocol_bytes = config.protocol.read_bytes()
        self.protocol_sha256 = hashlib.sha256(self.protocol_bytes).hexdigest()
        payload = json.loads(self.protocol_bytes)
        if payload.get("locked_before_final_evaluation") is not True:
            message = "supervisor refused: confirmatory protocol is not locked"
            raise ValueError(message)
        self.protocol = payload
        self.seeds = tuple(_json_int_list(payload, "seeds"))
        self.families = tuple(_json_str_list(payload, "baseline_families"))
        self._lock_handle: TextIO | None = None

    def run(self) -> None:
        self._acquire_lock()
        try:
            while True:
                progress = self._inspect_all()
                self._write_snapshot(progress)
                self._raise_on_failures(progress)
                action = self._next_action(progress)
                if action is None:
                    self._log("supervisor", "complete", "all locked queues are complete")
                    return
                active = active_confirmatory_workers()
                if self.config.dry_run:
                    self._log(
                        action.name,
                        "dry-run",
                        action.description,
                        commands=[shlex.join(command) for command in action.commands],
                        active_processes=active,
                    )
                    return
                if action.requires_gpu and active:
                    self._log(
                        action.name,
                        "waiting",
                        (
                            "existing confirmatory workers are still active; "
                            "not starting another worker"
                        ),
                        active_processes=active,
                    )
                elif action.requires_gpu and not self._gpu_is_safe():
                    self._log(
                        action.name,
                        "waiting",
                        "GPU capacity gate is not yet satisfied",
                    )
                else:
                    self._execute(action)
                    if action.finalize:
                        self._write_completion_marker()
                        self._log(
                            "supervisor",
                            "complete",
                            "all locked queues and final reports are complete",
                        )
                        return
                if self.config.once:
                    return
                time.sleep(self.config.poll_seconds)
        finally:
            self._release_lock()

    def _inspect_all(self) -> dict[str, QueueProgress]:
        capacity_manifest = self.config.capacity_root / "queue_manifest.jsonl"
        self._validate_capacity_manifest(capacity_manifest)
        progress = {
            "capacity": queue_progress(
                "capacity",
                capacity_manifest,
                self.config.capacity_root / "queue_state.jsonl",
                (self.config.capacity_root / "results" / "low_data_recommended_real.csv",),
            )
        }
        if (self.config.evidence_root / "evidence_contract.json").exists():
            self._validate_evidence_manifests()
            for kind in (
                "core_ablation",
                "mechanism_checkpoint",
                "interpretability",
                "sensitivity",
            ):
                progress[f"evidence_{kind}"] = queue_progress(
                    f"evidence_{kind}",
                    self.config.evidence_root / f"{kind}_manifest.jsonl",
                    self.config.evidence_root / "queue_state.jsonl",
                    (self.config.evidence_root / "results" / f"{kind}.csv",),
                )
        unseen_manifest = self.config.unseen_root / "queue_manifest.jsonl"
        if unseen_manifest.exists():
            progress["unseen"] = queue_progress(
                "unseen",
                unseen_manifest,
                self.config.unseen_root / "queue_state.jsonl",
                (self.config.unseen_root / "results" / "low_data_recommended_real.csv",),
            )
        p1p2_manifest = self.config.p1p2_root / "queue_manifest.jsonl"
        if p1p2_manifest.exists():
            progress["p1p2"] = queue_progress(
                "p1p2",
                p1p2_manifest,
                self.config.p1p2_root / "queue_state.jsonl",
                tuple(sorted((self.config.p1p2_root / "results").glob("*.csv"))),
            )
        return progress

    def _next_action(  # noqa: C901, PLR0911, PLR0912 - explicit protocol state machine
        self, progress: dict[str, QueueProgress]
    ) -> Action | None:
        python = sys.executable
        if not progress["capacity"].complete:
            return self._p0_worker_action(
                "capacity_workers",
                self.config.capacity_root,
                preset="stiefel_validation_capacity_selection",
            )
        capacity_selection = (
            self.config.capacity_root / "reports" / "stiefel_validation_capacity_selection.json"
        )
        if not _capacity_selection_is_complete(capacity_selection):
            return Action(
                "capacity_report",
                "write and validate the TRAIN-only capacity-selection report",
                (
                    (
                        python,
                        "-m",
                        "lnet.pac_recommended_low_data_cli",
                        "--stage",
                        "report",
                        "--output-root",
                        str(self.config.capacity_root),
                    ),
                ),
            )
        load_selection_binding(capacity_selection, self.config.protocol)

        if "evidence_core_ablation" not in progress:
            if _has_result_rows(self.config.evidence_root):
                message = (
                    "selected evidence root has results but no valid selected-capacity "
                    "contract; refusing to overwrite provenance"
                )
                raise RuntimeError(message)
            return Action(
                "enqueue_selected_evidence",
                "bind every evidence job to the completed P0 capacity selection",
                (
                    (
                        python,
                        "-m",
                        "lnet.pac_tf_evidence_cli",
                        "--stage",
                        "enqueue",
                        "--output-root",
                        str(self.config.evidence_root),
                        "--protocol",
                        str(self.config.protocol),
                        "--capacity-selection",
                        str(capacity_selection),
                    ),
                ),
            )
        if not progress["evidence_core_ablation"].complete:
            return self._evidence_worker_action("core_ablation")
        if not progress["evidence_mechanism_checkpoint"].complete:
            return self._evidence_worker_action("mechanism_checkpoint")

        unseen_manifest = self.config.unseen_root / "queue_manifest.jsonl"
        selection_path = (
            self.config.unseen_root / "reports" / "confirmatory_baseline_selection.json"
        )
        if not unseen_manifest.exists():
            if _has_result_rows(self.config.unseen_root):
                message = (
                    "unseen queue has result rows but no manifest; refusing to reconstruct "
                    "or overwrite provenance"
                )
                raise RuntimeError(message)
            return Action(
                "enqueue_unseen_validation",
                "enqueue six TRAIN-derived validation trials for every locked family",
                (self._p0_enqueue_command("enqueue-unseen-validation"),),
            )

        collections = manifest_values(unseen_manifest, "evaluation_collection")
        if collections == {"unseen_final_validation"}:
            unseen = progress["unseen"]
            if not unseen.complete:
                return self._p0_worker_action("unseen_validation_workers", self.config.unseen_root)
            if not _baseline_selection_is_complete(
                selection_path,
                protocol_sha256=self.protocol_sha256,
                families=self.families,
            ):
                return Action(
                    "unseen_validation_report",
                    "lock model-specific validation winners before any final TEST access",
                    (
                        (
                            python,
                            "-m",
                            "lnet.pac_recommended_low_data_cli",
                            "--stage",
                            "report",
                            "--output-root",
                            str(self.config.unseen_root),
                        ),
                    ),
                )
            return Action(
                "enqueue_unseen_final",
                "enqueue full-TRAIN refit and one untouched final TEST pass",
                (self._p0_enqueue_command("enqueue-unseen-final"),),
            )
        if collections != {"unseen_final_ucr"}:
            message = f"unexpected unseen manifest collection(s): {sorted(collections)}"
            raise RuntimeError(message)
        _require_baseline_selection(
            selection_path,
            protocol_sha256=self.protocol_sha256,
            families=self.families,
        )
        if not progress["unseen"].complete:
            return self._p0_worker_action("unseen_final_workers", self.config.unseen_root)

        if not progress["evidence_interpretability"].complete:
            self._require_interpretability_checkpoints()
            return self._evidence_worker_action("interpretability")
        if not progress["evidence_sensitivity"].complete:
            return self._evidence_worker_action("sensitivity")

        expected_p1p2 = self._expected_p1p2_rows(selection_path)
        p1p2_manifest = self.config.p1p2_root / "queue_manifest.jsonl"
        if not _manifest_matches(p1p2_manifest, expected_p1p2):
            existing = progress.get("p1p2")
            if existing is not None and (existing.done or existing.failed):
                message = (
                    "P1/P2 manifest differs from the locked protocol/tuning selection and "
                    "already has terminal rows; refusing to overwrite it"
                )
                raise RuntimeError(message)
            return Action(
                "enqueue_p1p2",
                "replace only the unstarted stale manifest with the hard-gated P1/P2 manifest",
                (
                    (
                        python,
                        "-m",
                        "lnet.pac_tf_p1p2_cli",
                        "--stage",
                        "enqueue",
                        "--output-root",
                        str(self.config.p1p2_root),
                        "--protocol-path",
                        str(self.config.protocol),
                        "--selection-path",
                        str(selection_path),
                        "--unseen-root",
                        str(self.config.unseen_root),
                        "--device",
                        self.config.device,
                    ),
                ),
            )
        if "p1p2" not in progress or not progress["p1p2"].complete:
            return Action(
                "p1p2_workers",
                "run the locked low-data, OOD, efficiency, calibration, and error-analysis jobs",
                (
                    (
                        python,
                        "-m",
                        "lnet.pac_tf_p1p2_cli",
                        "--stage",
                        "workers",
                        "--output-root",
                        str(self.config.p1p2_root),
                        "--protocol-path",
                        str(self.config.protocol),
                        "--selection-path",
                        str(selection_path),
                        "--unseen-root",
                        str(self.config.unseen_root),
                        "--device",
                        self.config.device,
                        "--workers",
                        str(self.config.workers),
                        "--total-slots",
                        str(self.config.total_slots),
                    ),
                ),
                requires_gpu=True,
            )

        completion = self.config.state_root / "COMPLETE.json"
        if _completion_matches(completion, self.protocol_sha256):
            return None
        return Action(
            "final_reports",
            "refresh every machine-readable report after all queues are terminal",
            (
                (
                    python,
                    "-m",
                    "lnet.pac_recommended_low_data_cli",
                    "--stage",
                    "report",
                    "--output-root",
                    str(self.config.capacity_root),
                ),
                (
                    python,
                    "-m",
                    "lnet.pac_recommended_low_data_cli",
                    "--stage",
                    "report",
                    "--output-root",
                    str(self.config.unseen_root),
                ),
                (
                    python,
                    "-m",
                    "lnet.pac_tf_evidence_cli",
                    "--stage",
                    "report",
                    "--output-root",
                    str(self.config.evidence_root),
                    "--protocol",
                    str(self.config.protocol),
                ),
                (
                    python,
                    "-m",
                    "lnet.pac_tf_p1p2_cli",
                    "--stage",
                    "report",
                    "--output-root",
                    str(self.config.p1p2_root),
                ),
                (
                    python,
                    "-m",
                    "lnet.pac_tf_confirmatory_report_cli",
                    "--protocol",
                    str(self.config.protocol),
                    "--unseen-root",
                    str(self.config.unseen_root),
                    "--p1p2-root",
                    str(self.config.p1p2_root),
                    "--evidence-root",
                    str(self.config.evidence_root),
                    "--output-root",
                    str(self.config.state_root / "reports"),
                ),
            ),
            finalize=True,
        )

    def _p0_worker_action(self, name: str, root: Path, *, preset: str | None = None) -> Action:
        command = [
            sys.executable,
            "-m",
            "lnet.pac_recommended_low_data_cli",
            "--stage",
            "workers",
            "--output-root",
            str(root),
            "--device",
            self.config.device,
            "--optimizer-mode",
            self.config.optimizer_mode,
            "--workers",
            str(self.config.workers),
            "--total-slots",
            str(self.config.total_slots),
        ]
        if preset is not None:
            command.extend(("--preset", preset))
        command.extend(_repeated_option("--seeds", self.seeds))
        return Action(
            name,
            f"resume {root} without rerunning completed job keys",
            (tuple(command),),
            requires_gpu=True,
        )

    def _p0_enqueue_command(self, stage: str) -> tuple[str, ...]:
        command = [
            sys.executable,
            "-m",
            "lnet.pac_recommended_low_data_cli",
            "--stage",
            stage,
            "--output-root",
            str(self.config.unseen_root),
            "--selection-root",
            str(self.config.capacity_root),
            "--protocol-path",
            str(self.config.protocol),
        ]
        command.extend(_repeated_option("--seeds", self.seeds))
        return tuple(command)

    def _evidence_worker_action(self, kind: str) -> Action:
        return Action(
            f"evidence_{kind}_workers",
            f"resume the locked {kind} evidence queue",
            (
                (
                    sys.executable,
                    "-m",
                    "lnet.pac_tf_evidence_cli",
                    "--stage",
                    "workers",
                    "--kind",
                    kind,
                    "--device",
                    self.config.device,
                    "--workers",
                    str(self.config.workers),
                    "--total-slots",
                    str(self.config.total_slots),
                    "--output-root",
                    str(self.config.evidence_root),
                ),
            ),
            requires_gpu=True,
        )

    def _expected_p1p2_rows(self, selection_path: Path) -> tuple[dict[str, object], ...]:
        config = P1P2Config(
            output_root=self.config.p1p2_root,
            protocol_path=self.config.protocol,
            selection_path=selection_path,
            unseen_root=self.config.unseen_root,
            device=self.config.device,
            workers=self.config.workers,
            total_slots=self.config.total_slots,
        )
        return tuple(asdict(job) for job in build_p1p2_jobs(config))

    def _require_interpretability_checkpoints(self) -> None:
        manifest = self.config.evidence_root / "interpretability_manifest.jsonl"
        missing = [
            key
            for row in read_jsonl(manifest)
            if (key := str(row.get("checkpoint_key", "")))
            and not (self.config.evidence_root / "checkpoints" / f"{key}.pt").is_file()
        ]
        if missing:
            preview = ", ".join(missing[:5])
            message = (
                f"interpretability gate refused: {len(missing)} prerequisite checkpoints "
                f"are missing ({preview})"
            )
            raise RuntimeError(message)

    def _validate_capacity_manifest(self, manifest: Path) -> None:
        rows = read_jsonl(manifest)
        if not rows:
            message = f"capacity manifest is missing or empty: {manifest}"
            raise FileNotFoundError(message)
        if len({str(row.get("key")) for row in rows}) != len(rows):
            message = "capacity manifest contains duplicate job keys"
            raise ValueError(message)
        if any(
            row.get("evaluation_split") != "validation"
            or row.get("data_protocol") != "clean_stratified"
            or row.get("restore_best_validation") is not True
            for row in rows
        ):
            message = "capacity manifest violates the TRAIN-only clean-validation contract"
            raise ValueError(message)

    def _validate_evidence_manifests(self) -> None:
        binding = validate_selected_evidence_root(self.config.evidence_root)
        if binding.protocol_sha256 != self.protocol_sha256:
            message = "selected evidence root is bound to a different locked protocol"
            raise ValueError(message)

    def _raise_on_failures(self, progress: dict[str, QueueProgress]) -> None:
        failures = [row for row in progress.values() if row.failed]
        if not failures:
            return
        summary = ", ".join(f"{row.name}={row.failed}" for row in failures)
        self._log("supervisor", "failed", f"terminal job failures detected: {summary}")
        message = f"confirmatory supervisor stopped on failed jobs: {summary}"
        raise RuntimeError(message)

    def _gpu_is_safe(self) -> bool:
        if self.config.device == "cpu":
            return True
        snapshot = gpu_snapshot()
        if snapshot is None:
            if self.config.device == "cuda":
                message = "CUDA was requested but nvidia-smi capacity data is unavailable"
                raise RuntimeError(message)
            return True
        utilization, free_mib = snapshot
        self._log(
            "gpu_gate",
            "observed",
            "GPU capacity snapshot",
            utilization=utilization,
            free_mib=free_mib,
        )
        return (
            utilization <= self.config.max_gpu_utilization
            and free_mib >= self.config.min_free_gpu_mib
        )

    def _execute(self, action: Action) -> None:
        self._log(action.name, "starting", action.description)
        for command in action.commands:
            rendered = shlex.join(command)
            self._log(action.name, "command", rendered, command=rendered)
            completed = subprocess.run(command, check=False)  # noqa: S603
            if completed.returncode != 0:
                self._log(
                    action.name,
                    "failed",
                    f"command exited with status {completed.returncode}",
                    command=rendered,
                )
                message = f"supervisor command failed ({completed.returncode}): {rendered}"
                raise RuntimeError(message)
        self._log(action.name, "done", action.description)

    def _write_snapshot(self, progress: dict[str, QueueProgress]) -> None:
        payload = {
            "schema_version": "pac_tf_confirmatory_supervisor_state.v1",
            "updated_at": _now(),
            "protocol_sha256": self.protocol_sha256,
            "queues": {name: asdict(row) for name, row in sorted(progress.items())},
        }
        if self.config.dry_run:
            print(json.dumps(payload, indent=2, sort_keys=True), flush=True)  # noqa: T201
            return
        self.config.state_root.mkdir(parents=True, exist_ok=True)
        _atomic_json(self.config.state_root / "STATUS.json", payload)
        self._log(
            "supervisor",
            "poll",
            "queue progress refreshed",
            queues={name: asdict(row) for name, row in sorted(progress.items())},
        )

    def _write_completion_marker(self) -> None:
        payload = {
            "schema_version": "pac_tf_confirmatory_supervisor_complete.v1",
            "completed_at": _now(),
            "protocol_sha256": self.protocol_sha256,
        }
        _atomic_json(self.config.state_root / "COMPLETE.json", payload)

    def _log(self, stage: str, status: str, message: str, **details: object) -> None:
        row = {
            "time": _now(),
            "stage": stage,
            "status": status,
            "message": message,
            **details,
        }
        print(json.dumps(row, sort_keys=True), flush=True)  # noqa: T201
        if self.config.dry_run:
            return
        self.config.state_root.mkdir(parents=True, exist_ok=True)
        with (self.config.state_root / "supervisor.jsonl").open("a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _acquire_lock(self) -> None:
        if self.config.dry_run:
            return
        self.config.state_root.mkdir(parents=True, exist_ok=True)
        handle = (self.config.state_root / "supervisor.lock").open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.close()
            message = "another confirmatory supervisor already holds the durable lock"
            raise RuntimeError(message) from error
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} started={_now()}\n")
        handle.flush()
        self._lock_handle = handle

    def _release_lock(self) -> None:
        handle = self._lock_handle
        if handle is None:
            return
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
        self._lock_handle = None


def read_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    if not path.exists():
        return ()
    return tuple(
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    )


def queue_progress(
    name: str,
    manifest: Path,
    state_path: Path,
    result_paths: tuple[Path, ...],
) -> QueueProgress:
    rows = read_jsonl(manifest)
    keys = tuple(str(row.get("key", "")) for row in rows)
    if not keys or any(not key for key in keys) or len(set(keys)) != len(keys):
        message = f"invalid or empty queue manifest: {manifest}"
        raise ValueError(message)
    latest: dict[str, str] = {}
    for row in read_jsonl(state_path):
        key = str(row.get("key", ""))
        if key in keys:
            latest[key] = str(row.get("status", ""))
    result_status: dict[str, str] = {}
    for path in result_paths:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (row.get("job_key") or row.get("queue_key") or row.get("key") or "").strip()
                status = (row.get("status") or "").strip()
                if (
                    key in keys
                    and status in {"done", "failed"}
                    and (status == "done" or result_status.get(key) != "done")
                ):
                    result_status[key] = status
    status_by_key = {
        key: (
            "done"
            if result_status.get(key) == "done"
            else result_status.get(key, latest.get(key, "pending"))
        )
        for key in keys
    }
    done = sum(status == "done" for status in status_by_key.values())
    failed = sum(status == "failed" for status in status_by_key.values())
    running = sum(status == "running" for status in status_by_key.values())
    pending = len(keys) - done - failed - running
    return QueueProgress(name, str(manifest), len(keys), done, failed, running, pending)


def manifest_values(path: Path, field: str) -> set[str]:
    return {str(row.get(field)) for row in read_jsonl(path) if row.get(field) is not None}


def active_confirmatory_workers() -> list[dict[str, object]]:
    completed = subprocess.run(
        ("/usr/bin/ps", "-eo", "pid=,etimes=,args="),
        check=True,
        capture_output=True,
        text=True,
    )
    active: list[dict[str, object]] = []
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, elapsed_text, command = stripped.split(maxsplit=2)
        pid = int(pid_text)
        if pid == os.getpid() or "--stage workers" not in command:
            continue
        if any(module in command for module in _WORKER_MODULES):
            active.append({"pid": pid, "elapsed_seconds": int(elapsed_text), "command": command})
    return active


def gpu_snapshot() -> tuple[int, int] | None:
    try:
        completed = subprocess.run(
            (
                "/usr/bin/nvidia-smi",
                "--query-gpu=utilization.gpu,memory.free",
                "--format=csv,noheader,nounits",
            ),
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not rows:
        return None
    parsed = [tuple(int(value.strip()) for value in row.split(",")) for row in rows]
    return max(value[0] for value in parsed), min(value[1] for value in parsed)


def _manifest_matches(path: Path, expected: tuple[dict[str, object], ...]) -> bool:
    return read_jsonl(path) == expected


def _capacity_selection_is_complete(path: Path) -> bool:
    if not path.exists():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return (
        payload.get("schema_version") == "pac_validation_capacity_selection.v1"
        and payload.get("status") == "complete"
        and payload.get("official_test_observed") is False
        and isinstance(payload.get("expected_jobs"), int)
        and int(payload.get("expected_jobs", 0)) > 0
        and payload.get("completed_jobs") == payload.get("expected_jobs")
        and isinstance(payload.get("selected_model"), str)
    )


def _baseline_selection_is_complete(
    path: Path, *, protocol_sha256: str, families: tuple[str, ...]
) -> bool:
    try:
        _require_baseline_selection(path, protocol_sha256=protocol_sha256, families=families)
    except (FileNotFoundError, TypeError, ValueError):
        return False
    return True


def _require_baseline_selection(
    path: Path, *, protocol_sha256: str, families: tuple[str, ...]
) -> None:
    if not path.exists():
        message = f"complete baseline-selection artifact is missing: {path}"
        raise FileNotFoundError(message)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "pac_confirmatory_baseline_selection.v1":
        message = "unexpected baseline-selection schema"
        raise ValueError(message)
    if payload.get("status") != "complete":
        message = "baseline selection is not complete"
        raise ValueError(message)
    if payload.get("protocol_sha256") != protocol_sha256:
        message = "baseline selection is not bound to the active locked protocol"
        raise ValueError(message)
    selected = payload.get("selected_trials")
    if not isinstance(selected, dict) or set(selected) != set(families):
        message = "baseline selection does not cover exactly the locked model families"
        raise ValueError(message)


def _has_result_rows(root: Path) -> bool:
    for path in (root / "results").glob("*.csv"):
        with path.open(newline="", encoding="utf-8") as handle:
            if next(csv.DictReader(handle), None) is not None:
                return True
    return False


def _completion_matches(path: Path, protocol_sha256: str) -> bool:
    if not path.exists():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("protocol_sha256") == protocol_sha256


def _json_int_list(payload: dict[str, object], key: str) -> list[int]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        message = f"protocol {key} must be an integer list"
        raise TypeError(message)
    return value


def _json_str_list(payload: dict[str, object], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        message = f"protocol {key} must be a string list"
        raise TypeError(message)
    return value


def _repeated_option(option: str, values: tuple[int, ...]) -> list[str]:
    output: list[str] = []
    for value in values:
        output.extend((option, str(value)))
    return output


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Restart-safe supervisor for the locked PAC-TF confirmatory queues."
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--capacity-root", type=Path, default=DEFAULT_CAPACITY_ROOT)
    parser.add_argument("--unseen-root", type=Path, default=DEFAULT_UNSEEN_ROOT)
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--p1p2-root", type=Path, default=DEFAULT_P1P2_ROOT)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--total-slots", type=int, default=16)
    parser.add_argument("--optimizer-mode", default="fused")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--min-free-gpu-mib", type=int, default=32_768)
    parser.add_argument("--max-gpu-utilization", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    config = SupervisorConfig(
        protocol=arguments.protocol,
        capacity_root=arguments.capacity_root,
        unseen_root=arguments.unseen_root,
        evidence_root=arguments.evidence_root,
        p1p2_root=arguments.p1p2_root,
        state_root=arguments.state_root,
        device=arguments.device,
        workers=arguments.workers,
        total_slots=arguments.total_slots,
        optimizer_mode=arguments.optimizer_mode,
        poll_seconds=arguments.poll_seconds,
        min_free_gpu_mib=arguments.min_free_gpu_mib,
        max_gpu_utilization=arguments.max_gpu_utilization,
        dry_run=arguments.dry_run,
        once=arguments.once,
    )
    ConfirmatorySupervisor(config).run()


if __name__ == "__main__":
    main()
