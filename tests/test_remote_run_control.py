from __future__ import annotations

# ruff: noqa: FBT001, PT011, SLF001, TC003
import json
import subprocess
import time
from pathlib import Path

import pytest

from scripts import remote_run_control as control

TARGET = "a" * 40
REF = "refs/heads/control/campaign"


def _payload(action: str = "run", generation: int | bool = 1) -> dict[str, object]:
    return {
        "schema": control.SCHEMA,
        "campaign_id": "campaign",
        "target_commit": TARGET,
        "generation": generation,
        "action": action,
        "updated_at": "2026-08-21T12:00:00+09:00",
        "reason": "test control update",
    }


def _switch(tmp_path: Path, **kwargs: object) -> control.GitHubRunControl:
    return control.GitHubRunControl(
        repo_url="https://github.com/owner/repo.git",
        ref=REF,
        control_path="h200/control.json",
        campaign_id="campaign",
        target_commit=TARGET,
        repo_root=tmp_path,
        poll_seconds=1,
        timeout_seconds=1,
        max_unreachable_seconds=10,
        **kwargs,
    )


def test_owner_stop_is_latched_and_raised_after_optimizer_update(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = iter((_payload(), _payload("stop", 2)))
    monkeypatch.setattr(control.GitHubRunControl, "_fetch", lambda _self: next(records))
    switch = _switch(tmp_path)
    switch.set_variant("variant")

    assert switch.poll(force=True) == _payload()
    with pytest.raises(control.StopRequestedError) as stopped:
        switch.poll(force=True)
    with pytest.raises(control.StopRequestedError):
        switch.after_optimizer_step()

    assert stopped.value.record["generation"] == 2
    assert switch.stop_requested
    assert switch.active_variant == "variant"


def test_short_network_failure_is_fail_open_then_staleness_stops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    switch = _switch(tmp_path)
    switch.last_success_monotonic = 100.0
    monkeypatch.setattr(
        control.GitHubRunControl,
        "_fetch",
        lambda _self: (_ for _ in ()).throw(OSError("offline")),
    )
    monotonic_values = iter((105.0, 111.0))
    monkeypatch.setattr(time, "monotonic", lambda: next(monotonic_values))

    assert switch.poll(force=True) is None
    with pytest.raises(control.StopRequestedError) as stopped:
        switch.poll(force=True)

    assert stopped.value.record["reason"] == "control_unreachable"


def test_startup_failure_is_closed_without_fabricating_a_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        control.GitHubRunControl,
        "_fetch",
        lambda _self: (_ for _ in ()).throw(OSError("offline")),
    )
    switch = _switch(tmp_path)

    with pytest.raises(RuntimeError, match="unavailable or invalid"):
        switch.require_start(attempts=2, delay_seconds=0)
    assert not switch.stop_requested


@pytest.mark.parametrize(
    ("repo_url", "ref", "path"),
    [
        ("http://github.com/owner/repo.git", REF, "control.json"),
        ("https://example.com/owner/repo.git", REF, "control.json"),
        ("https://user@github.com/owner/repo.git", REF, "control.json"),
        ("https://github.com/owner/repo", REF, "control.json"),
        ("https://github.com/owner/repo.git", "refs/tags/control", "control.json"),
        ("https://github.com/owner/repo.git", "refs/heads/../main", "control.json"),
        ("https://github.com/owner/repo.git", REF, "../control.json"),
    ],
)
def test_transport_identity_is_strict(
    tmp_path: Path,
    repo_url: str,
    ref: str,
    path: str,
) -> None:
    with pytest.raises(ValueError):
        control.GitHubRunControl(
            repo_url=repo_url,
            ref=ref,
            control_path=path,
            campaign_id="campaign",
            target_commit=TARGET,
            repo_root=tmp_path,
        )


def test_payload_rejects_boolean_generation_and_extra_keys(tmp_path: Path) -> None:
    switch = _switch(tmp_path)
    with pytest.raises(ValueError, match="values"):
        switch._validate_payload(_payload(generation=True))
    extra = _payload()
    extra["extra"] = True
    with pytest.raises(ValueError, match="keys"):
        switch._validate_payload(extra)


def test_generation_cannot_roll_back_or_change_action_in_place(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = iter((_payload(generation=2), _payload(generation=1), _payload("stop", 2)))
    monkeypatch.setattr(control.GitHubRunControl, "_fetch", lambda _self: next(records))
    switch = _switch(tmp_path)

    assert switch.poll(force=True) == _payload(generation=2)
    assert switch.poll(force=True) is None
    assert switch.poll(force=True) is None
    assert switch.failures == 2
    assert not switch.stop_requested


def test_previous_stop_requires_a_strictly_newer_run_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        control.GitHubRunControl,
        "_fetch",
        lambda _self: _payload("run", 4),
    )
    stale = _switch(tmp_path, stopped_generation=4)
    with pytest.raises(control.StopRequestedError) as stopped:
        stale.require_start(attempts=1)
    assert stopped.value.record["reason"] == "previous_stop_remains_latched"

    monkeypatch.setattr(
        control.GitHubRunControl,
        "_fetch",
        lambda _self: _payload("run", 5),
    )
    resumed = _switch(tmp_path, stopped_generation=4)
    assert resumed.require_start(attempts=1)["generation"] == 5


def test_git_blob_is_refetched_only_when_remote_sha_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sha = "b" * 40
    calls: list[list[str]] = []

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[object]:
        calls.append(command)
        if command[1] == "ls-remote":
            return subprocess.CompletedProcess(command, 0, f"{sha}\t{REF}\n", "")
        if "show" in command:
            return subprocess.CompletedProcess(command, 0, json.dumps(_payload()).encode(), b"")
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(subprocess, "run", run)
    switch = _switch(tmp_path)

    assert switch.poll(force=True) == _payload()
    assert switch.poll(force=True) == _payload()
    assert sum(len(command) > 3 and command[3] == "fetch" for command in calls) == 1
    assert sum(len(command) > 3 and command[3] == "show" for command in calls) == 1
    assert sum(command[1] == "ls-remote" for command in calls) == 2
