from __future__ import annotations

# ruff: noqa: ANN204, EM101, SLF001, TC003, TRY003
import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import run_a2d_r2k3_stage_allocation_cohort_imagenet100 as cohort
import shared_batch_cohort as shared
import torch
from torch import Tensor, nn


class _OneShotLoader:
    dataset = tuple(range(4))

    def __init__(self) -> None:
        self.iter_calls = 0
        self.batches = [
            (torch.tensor([[1.0], [2.0]]), torch.tensor([0, 1])),
            (torch.tensor([[3.0], [4.0]]), torch.tensor([1, 0])),
        ]

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self):
        self.iter_calls += 1
        if self.iter_calls > 1:
            raise AssertionError("shared loader was iterated more than once")
        return iter(self.batches)


class _Runtime(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.input_pointers: list[int] = []

    def forward(self, inputs: Tensor) -> Tensor:
        self.input_pointers.append(inputs.data_ptr())
        return self.model(inputs)


class _EvaluationRuntime(nn.Module):
    def forward(self, inputs: Tensor) -> tuple[Tensor, ...]:
        logits = torch.cat((inputs, -inputs), dim=-1)
        return logits, logits, logits, logits, inputs


def test_training_iterates_one_loader_and_shares_each_mixed_batch() -> None:
    loader = _OneShotLoader()
    records: dict[str, list[tuple[int, tuple[int, ...], float]]] = {"a": [], "b": []}
    members: list[shared.CohortMember] = []
    model_names: dict[int, str] = {}
    runtimes: list[_Runtime] = []
    for name in records:
        model = nn.Linear(1, 2, bias=False)
        runtime = _Runtime(model)
        runtimes.append(runtime)
        model_names[id(model)] = name
        members.append(
            shared.CohortMember(
                variant=name,
                model=model,
                runtime=runtime,
                optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
            )
        )

    def objective(
        model: nn.Module,
        output: Tensor | tuple[Tensor, ...],
        targets: Tensor,
        permuted_targets: Tensor,
        mixing: float,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        assert isinstance(output, Tensor)
        records[model_names[id(model)]].append(
            (targets.data_ptr(), tuple(permuted_targets.tolist()), mixing)
        )
        return output, output.square().mean(), {}

    stopped_after: list[int] = []
    metrics, steps, _wait = shared.train_epoch(
        members,
        loader,  # type: ignore[arg-type]
        device=torch.device("cpu"),
        mixup_generator=np.random.default_rng(501),
        permutation_generator=torch.Generator().manual_seed(501),
        mixup_alpha=0.8,
        precision="float32",
        channels_last=False,
        objective=objective,
        after_model_batch=lambda *_args: None,
        after_cohort_batch=stopped_after.append,
    )

    assert loader.iter_calls == 1
    assert steps == 2
    assert stopped_after == [1, 2]
    assert runtimes[0].input_pointers == runtimes[1].input_pointers
    assert records["a"] == records["b"]
    assert set(metrics) == {"a", "b"}


def test_evaluation_iterates_validation_loader_once() -> None:
    loader = _OneShotLoader()
    members = [
        shared.CohortMember(
            variant=name,
            model=nn.Linear(1, 1),
            runtime=_EvaluationRuntime(),
            optimizer=torch.optim.SGD(nn.Linear(1, 1).parameters(), lr=0.01),
        )
        for name in ("a", "b")
    ]

    results = shared.evaluate(
        members,
        loader,  # type: ignore[arg-type]
        device=torch.device("cpu"),
        precision="float32",
        channels_last=False,
        finalize=lambda _model, outputs, target, _device: {
            "rows": float(sum(value.shape[0] for value in outputs[0])),
            "labels": float(target.numel()),
        },
    )

    assert loader.iter_calls == 1
    assert results == {
        "a": {"rows": 4.0, "labels": 4.0},
        "b": {"rows": 4.0, "labels": 4.0},
    }


def test_model_failure_aborts_before_later_members_and_cohort_boundary() -> None:
    class FailingRuntime(nn.Module):
        def forward(self, _inputs: Tensor) -> Tensor:
            raise RuntimeError("model failed")

    first_model = nn.Linear(1, 2)
    last_runtime = _Runtime(nn.Linear(1, 2))
    members = [
        shared.CohortMember(
            variant="first",
            model=first_model,
            runtime=_Runtime(first_model),
            optimizer=torch.optim.SGD(first_model.parameters(), lr=0.01),
        ),
        shared.CohortMember(
            variant="failed",
            model=nn.Linear(1, 2),
            runtime=FailingRuntime(),
            optimizer=torch.optim.SGD(nn.Linear(1, 2).parameters(), lr=0.01),
        ),
        shared.CohortMember(
            variant="last",
            model=last_runtime.model,
            runtime=last_runtime,
            optimizer=torch.optim.SGD(last_runtime.model.parameters(), lr=0.01),
        ),
    ]
    boundaries: list[int] = []

    with pytest.raises(RuntimeError, match="model failed"):
        shared.train_epoch(
            members,
            _OneShotLoader(),  # type: ignore[arg-type]
            device=torch.device("cpu"),
            mixup_generator=np.random.default_rng(501),
            permutation_generator=torch.Generator().manual_seed(501),
            mixup_alpha=0.8,
            precision="float32",
            channels_last=False,
            objective=lambda _model, output, *_args: (
                output,
                output.square().mean(),
                {},
            ),
            after_model_batch=lambda *_args: None,
            after_cohort_batch=boundaries.append,
        )

    assert last_runtime.input_pointers == []
    assert boundaries == []


def _member(variant: str) -> cohort.MemberState:
    model = nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _epoch: 1.0)
    return cohort.MemberState(
        variant=variant,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        parameters=2,
    )


def test_corrupt_new_cohort_manifest_falls_back_to_previous_epoch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    members = [_member(variant) for variant in cohort.stage.VARIANTS]
    training_generator = torch.Generator().manual_seed(501)
    mixup_generator = np.random.default_rng(501)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", list)
    contract_sha256 = "a" * 64
    recipe: dict[str, Any] = {"epochs": 2}

    first = cohort._save_checkpoint(
        tmp_path,
        members,
        epoch=1,
        contract_sha256=contract_sha256,
        recipe=recipe,
        training_generator=training_generator,
        mixup_generator=mixup_generator,
        permutation_generator=torch.Generator().manual_seed(501),
        preflight=None,
    )
    second = cohort._save_checkpoint(
        tmp_path,
        members,
        epoch=2,
        contract_sha256=contract_sha256,
        recipe=recipe,
        training_generator=training_generator,
        mixup_generator=mixup_generator,
        permutation_generator=torch.Generator().manual_seed(501),
        preflight=None,
    )
    corrupt = tmp_path / second["files"][cohort.stage.VARIANTS[6]]["path"]
    corrupt.write_bytes(b"corrupt")

    recovered = cohort._latest_manifest(tmp_path, contract_sha256)

    assert recovered is not None
    assert recovered["cohort_token"] == first["cohort_token"]
    assert recovered["epoch"] == 1


def test_checkpoint_retention_keeps_two_complete_epochs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    members = [_member(variant) for variant in cohort.stage.VARIANTS]
    training_generator = torch.Generator().manual_seed(501)
    mixup_generator = np.random.default_rng(501)
    permutation_generator = torch.Generator().manual_seed(501)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", list)
    for epoch in (1, 2, 3):
        cohort._save_checkpoint(
            tmp_path,
            members,
            epoch=epoch,
            contract_sha256="a" * 64,
            recipe={"epochs": 3},
            training_generator=training_generator,
            mixup_generator=mixup_generator,
            permutation_generator=permutation_generator,
            preflight=None,
        )

    manifests = sorted((tmp_path / "checkpoints/shared-cohort/manifests").glob("epoch-*.json"))
    epoch_roots = sorted((tmp_path / "checkpoints/shared-cohort/epochs").glob("epoch-*"))
    assert [path.name for path in manifests] == ["epoch-0002.json", "epoch-0003.json"]
    assert [path.name for path in epoch_roots] == ["epoch-0002", "epoch-0003"]


def test_wandb_finish_failure_does_not_publish_delivery_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FailingRun:
        settings = SimpleNamespace(log_internal=str(tmp_path / "debug-internal.log"))

        def finish(self) -> None:
            raise RuntimeError("finish failed")

    run = FailingRun()
    monkeypatch.setattr(
        cohort.harness,
        "_best_effort_initialize_wandb",
        lambda *_args, **_kwargs: run,
    )
    monkeypatch.setattr(
        cohort.harness,
        "_sync_or_abandon_wandb",
        lambda active, _spool, _delivered: active,
    )
    monkeypatch.setattr(cohort.harness, "_report_wandb_degraded", lambda *_args: None)

    success = cohort._publish_variant_spool(
        tmp_path,
        {},
        "a" * 64,
        variant=cohort.stage.VARIANTS[0],
        parameters=1,
    )

    assert success is False
    assert not cohort._receipt_path(tmp_path, cohort.stage.VARIANTS[0]).exists()


def test_cohort_contract_binds_resolved_loader_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = argparse.Namespace(
        workers=1,
        minimum_cohort_model_images_per_second=800.0,
    )
    monkeypatch.setattr(
        cohort.stage,
        "_contract",
        lambda _args: {
            "recipe": {"cpu_affinity": "0"},
            "runtime": {"hostname": "ephemeral-pod", "torch": "test"},
            "source_sha256": {},
        },
    )
    monkeypatch.setattr(cohort.harness, "_digest", str)
    monkeypatch.setenv("LNET_DATALOADER_WORKERS", "1")
    monkeypatch.setenv("LNET_DATALOADER_PREFETCH_FACTOR", "2")
    monkeypatch.setenv("LNET_CPU_AFFINITY_ACTIVE", "0")

    first = cohort._contract(args)
    monkeypatch.setenv("LNET_DATALOADER_WORKERS", "2")
    second = cohort._contract(args)

    assert first["execution"]["loader_workers"] == 1
    assert first["execution"]["loader_prefetch_factor"] == 2
    assert "cpu_affinity" not in first["execution"]
    assert "cpu_affinity" not in first["recipe"]
    assert "hostname" not in first["runtime"]
    assert first["runtime"]["torch"] == "test"
    assert second["execution"]["loader_workers"] == 2
    assert cohort.harness._contract_sha256(first) != cohort.harness._contract_sha256(second)


def test_cohort_checkpoint_restores_every_member_and_shared_rng(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    members = [_member(variant) for variant in cohort.stage.VARIANTS]
    for index, member in enumerate(members):
        with torch.no_grad():
            member.model.weight.fill_(index + 1)
        member.history = [{"epoch": 1.0}]
        member.global_step = 7
        member.training_seconds = 3.0
    expected = {member.variant: member.model.weight.detach().clone() for member in members}
    training_generator = torch.Generator().manual_seed(501)
    mixup_generator = np.random.default_rng(501)
    permutation_generator = torch.Generator().manual_seed(501)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", list)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda _state: None)
    cohort._save_checkpoint(
        tmp_path,
        members,
        epoch=1,
        contract_sha256="a" * 64,
        recipe={"epochs": 1},
        training_generator=training_generator,
        mixup_generator=mixup_generator,
        permutation_generator=permutation_generator,
        preflight=None,
    )
    restored = [_member(variant) for variant in cohort.stage.VARIANTS]

    epoch, preflight, end_allocated = cohort._restore(
        tmp_path,
        restored,
        contract_sha256="a" * 64,
        training_generator=torch.Generator(),
        mixup_generator=np.random.default_rng(),
        permutation_generator=torch.Generator(),
    )

    assert epoch == 1
    assert preflight is None
    assert end_allocated == 0
    for member in restored:
        assert torch.equal(member.model.weight, expected[member.variant])
        assert member.global_step == 7
        assert member.training_seconds == 3.0
        assert member.history == [{"epoch": 1.0}]
