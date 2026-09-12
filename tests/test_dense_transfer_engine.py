from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dense_transfer.engine import (
    OptimizerPlan,
    TrainProgress,
    UpdateScheduler,
    benchmark_batches,
    checkpoint_payload,
    default_loss,
    load_checkpoint,
    train_batches,
)
import run_dense_transfer as cli
import prepare_dense_transfer_queue as queue


def _loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((output - target) ** 2).mean().float()


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ones(1, 1), torch.zeros(1, 1)


def _scheduler(optimizer: torch.optim.Optimizer) -> UpdateScheduler:
    return UpdateScheduler(optimizer, OptimizerPlan(epochs=1, learning_rate=0.1, weight_decay=0.0, warmup_updates=0), 4)


def test_short_final_accumulation_is_scaled_as_its_actual_window() -> None:
    model = nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = _scheduler(optimizer)
    progress = TrainProgress()
    train_batches(model, [_batch(), _batch()], optimizer, scheduler, torch.device("cpu"), 4, progress=progress, bf16=False, loss_fn=_loss)
    # Two equal microbatch gradients must be averaged over two, not divided by A=4.
    assert model.weight.item() == pytest.approx(0.8)
    assert progress.optimizer_updates == scheduler.update_count == 1


def test_short_last_batch_is_weighted_by_examples_not_microbatch_means() -> None:
    accumulated = nn.Linear(1, 1, bias=False)
    combined = nn.Linear(1, 1, bias=False)
    accumulated.weight.data.fill_(1.0)
    combined.weight.data.fill_(1.0)
    accumulated_optimizer = torch.optim.SGD(accumulated.parameters(), lr=0.1)
    combined_optimizer = torch.optim.SGD(combined.parameters(), lr=0.1)
    train_batches(
        accumulated,
        [(torch.ones(2, 1), torch.zeros(2, 1)), (torch.ones(1, 1), torch.zeros(1, 1))],
        accumulated_optimizer, _scheduler(accumulated_optimizer), torch.device("cpu"), 2,
        physical_batch_size=2, bf16=False, loss_fn=_loss,
    )
    train_batches(
        combined, [(torch.ones(3, 1), torch.zeros(3, 1))], combined_optimizer,
        _scheduler(combined_optimizer), torch.device("cpu"), 1, physical_batch_size=3,
        bf16=False, loss_fn=_loss,
    )
    assert accumulated.weight.item() == pytest.approx(combined.weight.item())


def test_checkpoint_restores_optimizer_scheduler_and_progress(tmp_path: Path) -> None:
    model = nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    scheduler = _scheduler(optimizer)
    progress = TrainProgress()
    train_batches(model, [_batch()] * 4, optimizer, scheduler, torch.device("cpu"), 4, progress=progress, bf16=False, loss_fn=_loss)
    contract = {"fixture": "resume"}
    path = tmp_path / "resume.pt"
    torch.save(checkpoint_payload(model, optimizer, scheduler, progress, contract), path)
    restored = nn.Linear(1, 1, bias=False)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.1)
    restored_scheduler = _scheduler(restored_optimizer)
    restored_progress = load_checkpoint(path, restored, restored_optimizer, restored_scheduler, contract)
    assert restored.weight.item() == pytest.approx(model.weight.item())
    assert restored_progress.optimizer_updates == progress.optimizer_updates
    assert restored_scheduler.update_count == scheduler.update_count


def test_scheduler_counts_optimizer_updates_and_keeps_epoch_drop() -> None:
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    plan = OptimizerPlan(epochs=2, learning_rate=0.1, weight_decay=0.0, warmup_updates=2, drop_epochs=(1,))
    scheduler = UpdateScheduler(optimizer, plan, total_updates=4)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.05)
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
    scheduler.set_epoch(1)
    scheduler.step()
    assert scheduler.update_count == 2
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)


def test_cpu_benchmark_reports_real_update_timing() -> None:
    model = nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    metrics = benchmark_batches(model, [_batch()] * 8, optimizer, _scheduler(optimizer), torch.device("cpu"), 2, steps=3, bf16=False, loss_fn=_loss)
    assert metrics["updates"] == 3
    assert metrics["cold_sec_per_update"] >= 0
    assert metrics["loss_finite"] is True


def test_default_segmentation_loss_all_ignore_is_finite_with_zero_gradients() -> None:
    logits = torch.randn(2, 3, 4, 5, requires_grad=True)
    auxiliary = torch.randn(2, 3, 2, 3, requires_grad=True)
    targets = torch.full((2, 4, 5), 255, dtype=torch.long)

    loss = default_loss({"logits": logits, "aux_logits": auxiliary}, targets)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    loss.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))
    assert torch.equal(auxiliary.grad, torch.zeros_like(auxiliary))


def test_default_segmentation_loss_matches_cross_entropy_when_labels_are_valid() -> None:
    logits = torch.randn(1, 3, 3, 4, requires_grad=True)
    auxiliary = torch.randn(1, 3, 3, 4, requires_grad=True)
    targets = torch.tensor([[[0, 1, 255, 2], [2, 0, 1, 255], [1, 2, 0, 1]]])
    expected_logits = logits.detach().clone().requires_grad_()
    expected_auxiliary = auxiliary.detach().clone().requires_grad_()

    loss = default_loss({"logits": logits, "aux_logits": auxiliary}, targets)
    expected = F.cross_entropy(expected_logits.float(), targets, ignore_index=255)
    expected = expected + 0.4 * F.cross_entropy(expected_auxiliary.float(), targets, ignore_index=255)
    assert loss.item() == pytest.approx(expected.item())
    loss.backward()
    expected.backward()
    assert torch.allclose(logits.grad, expected_logits.grad)
    assert torch.allclose(auxiliary.grad, expected_auxiliary.grad)


class _NaNSegmentationFixture(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.full(
            (images.shape[0], 2, images.shape[-2], images.shape[-1]), float("nan")
        )


def test_nonfinite_logits_are_not_hidden_by_all_ignore_loss() -> None:
    model = _NaNSegmentationFixture()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batch = (torch.ones(1, 3, 2, 2), torch.full((1, 2, 2), 255, dtype=torch.long))
    with pytest.raises(
        FloatingPointError,
        match=r"non-finite loss.*valid_pixels=0 target_pixels=4 logits_finite=False",
    ):
        train_batches(model, [batch], optimizer, _scheduler(optimizer), torch.device("cpu"), 1, bf16=False)


class _ListDetectionFixture(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, images: list[torch.Tensor], targets: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        value = sum((image.mean() for image in images), torch.zeros((), device=self.scale.device))
        target_value = sum((target["value"].float() for target in targets), torch.zeros((), device=self.scale.device))
        return {"loss_detector": (self.scale * value - target_value) ** 2}


def test_coco_list_batches_accumulate_without_tensor_collation() -> None:
    model = _ListDetectionFixture()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batch = ([torch.ones(3, 4, 5)], [{"value": torch.zeros(())}])
    result = train_batches(model, [batch, batch], optimizer, _scheduler(optimizer), torch.device("cpu"), 2, bf16=False)
    assert result["updates"] == 1
    assert model.scale.item() < 1.0


def test_resume_rejects_a_different_runtime_contract(tmp_path: Path) -> None:
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = _scheduler(optimizer)
    path = tmp_path / "resume.pt"
    torch.save(checkpoint_payload(model, optimizer, scheduler, TrainProgress(), {"data": "A"}), path)
    with pytest.raises(ValueError, match="runtime contract"):
        load_checkpoint(path, model, optimizer, scheduler, {"data": "B"})


def _cli_args(tmp_path: Path, mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        task="ade20k", model_key="va_k128", checkpoint=tmp_path / "source.pt", source_root=None,
        data_root=tmp_path, output_root=tmp_path / "out", device="cpu", physical_batch_size=1,
        effective_batch_size=1, workers=0, no_bf16=True, compile_backbone=False,
        checkpoint_backbone_blocks=False, channels_last=False, fused_adamw=False, mode=mode,
        allow_busy=False, steps=1, epochs=None, max_updates=None, resume=None, confirm_training=None,
    )


def _mock_runtime(total_updates: int = 2):
    model = nn.Linear(1, 2, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = UpdateScheduler(optimizer, OptimizerPlan(epochs=0, learning_rate=0.1, weight_decay=0.0, warmup_updates=0), total_updates)
    batch = (torch.ones(1, 1), torch.zeros(1, dtype=torch.long))
    return model, [batch], ["validation"], optimizer, scheduler, {"fixture": "runtime"}, torch.device("cpu")


def test_cli_train_ade_stops_at_scheduler_update_ceiling_and_writes_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _cli_args(tmp_path, "train")
    monkeypatch.setattr(cli, "build_runtime", lambda *_args: _mock_runtime(2))
    monkeypatch.setattr(cli, "require_train_permission", lambda *_args: None)
    monkeypatch.setattr(cli, "evaluate", lambda *_args: {"miou": 0.5})
    monkeypatch.setattr(cli.importlib, "import_module", lambda _name: SimpleNamespace())
    result = cli.run(args)
    assert result["updates"] == 2
    assert (tmp_path / "out" / "status" / "final.json").is_file()


def test_cli_max_updates_is_stopped_not_completed_or_evaluated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _cli_args(tmp_path, "train")
    args.max_updates = 1
    monkeypatch.setattr(cli, "build_runtime", lambda *_args: _mock_runtime(2))
    monkeypatch.setattr(cli, "require_train_permission", lambda *_args: None)
    monkeypatch.setattr(cli, "evaluate", lambda *_args: pytest.fail("partial training must not evaluate as completed"))
    monkeypatch.setattr(cli.importlib, "import_module", lambda _name: SimpleNamespace())
    result = cli.run(args)
    assert result["state"] == "stopped"
    assert result["updates"] == 1


def test_cooperative_stop_waits_for_accumulation_update_and_max_does_not_step() -> None:
    model = nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = _scheduler(optimizer)
    progress = TrainProgress()
    outcome = train_batches(model, [_batch(), _batch()], optimizer, scheduler, torch.device("cpu"), 2, progress=progress, bf16=False, loss_fn=_loss, stop_requested=lambda: True)
    assert outcome["interrupted"] is True
    assert progress.optimizer_updates == 1
    weight = model.weight.detach().clone()
    no_step = train_batches(model, [_batch()], optimizer, scheduler, torch.device("cpu"), 2, progress=progress, max_updates=1, bf16=False, loss_fn=_loss)
    assert no_step["updates"] == 1
    assert torch.equal(model.weight, weight)


def test_cli_evaluate_loads_task_checkpoint_before_evaluation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _cli_args(tmp_path, "evaluate")
    args.resume = tmp_path / "task.pt"
    calls: list[Path] = []
    monkeypatch.setattr(cli, "build_runtime", lambda *_args: _mock_runtime(1))
    monkeypatch.setattr(cli, "load_checkpoint", lambda path, *_args: calls.append(path) or TrainProgress())
    monkeypatch.setattr(cli, "evaluate", lambda *_args: {"miou": 0.7})
    monkeypatch.setattr(cli.importlib, "import_module", lambda _name: SimpleNamespace())
    assert cli.run(args)["miou"] == 0.7
    assert calls == [args.resume]


def test_cli_idle_guard_runs_before_runtime_construction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _cli_args(tmp_path, "smoke")
    monkeypatch.setattr(cli, "require_idle_cuda", lambda *_args: (_ for _ in ()).throw(RuntimeError("busy")))
    monkeypatch.setattr(cli, "build_runtime", lambda *_args: pytest.fail("runtime allocated before idle guard"))
    with pytest.raises(RuntimeError, match="busy"):
        cli.run(args)


def test_epoch_seed_recreates_sampler_order_for_mid_epoch_resume() -> None:
    dataset = TensorDataset(torch.arange(12))
    generator = torch.Generator().manual_seed(501 + 3)
    first = [int(item[0]) for item in DataLoader(dataset, batch_size=1, shuffle=True, generator=generator)]
    # The CLI reseeds with seed + epoch before rebuilding the interrupted epoch.
    generator.manual_seed(501 + 3)
    resumed = [int(item[0]) for item in DataLoader(dataset, batch_size=1, shuffle=True, generator=generator)]
    assert resumed == first
    assert resumed[5:] == first[5:]


def test_queue_has_approved_seeded_lanes_and_task_specific_roots(tmp_path: Path) -> None:
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    for model_key in ("va_k128", "convnextv2_atto", "tinyvim_s"):
        (checkpoints / f"{model_key}_seed501_ep100.pt").write_bytes(b"fixture")
    coco_root, ade_root, tiny_root = tmp_path / "coco", tmp_path / "ADEChallengeData2016", tmp_path / "tinyvim"
    coco_root.mkdir()
    ade_root.mkdir()
    tiny_root.mkdir()
    paths = queue.prepare(tmp_path / "out", checkpoints, coco_root, ade_root, tiny_root)
    payloads = [__import__("json").loads(path.read_text()) for path in paths]
    triples = {(item["source_contract"]["settings"]["task"], item["source_contract"]["settings"]["model_key"], item["downstream_seed"]) for item in payloads}
    assert triples == {
        ("ade20k", "va_k128", 501), ("ade20k", "convnextv2_atto", 501), ("ade20k", "tinyvim_s", 501),
        ("coco", "va_k128", 501), ("coco", "convnextv2_atto", 501), ("coco", "tinyvim_s", 501),
        ("ade20k", "va_k128", 509), ("ade20k", "tinyvim_s", 509),
    }
    assert len({item["id"] for item in payloads}) == len(payloads)
    assert len({item["output_directory"] for item in payloads}) == len(payloads)
    assert all(item["state"] == "DISARMED" for item in payloads)
    assert {item["source_contract"]["settings"]["data_root"] for item in payloads if item["source_contract"]["settings"]["task"] == "coco"} == {str(coco_root.resolve())}
    assert {item["source_contract"]["settings"]["data_root"] for item in payloads if item["source_contract"]["settings"]["task"] == "ade20k"} == {str(ade_root.resolve())}
