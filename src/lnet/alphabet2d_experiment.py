"""Restart-safe synthetic controls for ALPHABET-2D.

The campaign deliberately keeps model selection validation-only.  Test tensors
are generated with the other splits, but are not moved to the accelerator or
evaluated until the best validation checkpoint has been fixed.
"""

# ruff: noqa: EM101, EM102, TRY003
# pyright: reportExplicitAny=false

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from .alphabet2d import (
    Alphabet2D,
    Alphabet2DConfig,
    SpatialLag,
    SpatialWindows,
    spatial_modal_moments,
)
from .alphabet2d_synthetic import (
    Alphabet2DTask,
    EqualPowerPhaseConfig,
    OffAxisSpectralConfig,
    TensorClassificationSplit,
    make_alphabet2d_splits,
    off_axis_oracle_score,
    phase_arrangement_oracle_score,
)
from .pac_campaign_utils import source_file_hashes
from .pac_real2d_math import discrete_pole_real2d
from .pac_recurrence import RecurrenceBackend, recurrence_real2d_directional

ExperimentVariant = Literal[
    "product2d",
    "product_one_bank",
    "energy_only",
    "global_energy_only",
    "global_only",
    "local_covariance_linear",
    "axial2d",
    "raster1d",
]

VARIANTS: Final[tuple[ExperimentVariant, ...]] = (
    "product2d",
    "product_one_bank",
    "energy_only",
    "global_energy_only",
    "global_only",
    "local_covariance_linear",
    "axial2d",
    "raster1d",
)
TASKS: Final[tuple[Alphabet2DTask, ...]] = ("off_axis", "equal_power_phase")
SCHEMA_VERSION: Final = 1


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Shared architecture contract for every learned-pole control."""

    image_size: int = 32
    patch_size: int = 4
    model_dim: int = 32
    modes: int = 8
    depth: int = 2
    mlp_ratio: float = 2.0
    lags: tuple[SpatialLag, ...] = ((1, 0), (0, 1), (1, 1), (1, -1))
    windows: SpatialWindows = "global_2x2"
    recurrence_backend: RecurrenceBackend = "auto"
    layer_scale_init: float = 1.0e-2
    local_covariance_radius: int = 1


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Fixed validation-selected AdamW recipe."""

    train_per_class: int = 512
    validation_per_class: int = 128
    test_per_class: int = 512
    batch_size: int = 64
    max_epochs: int = 100
    patience: int = 15
    learning_rate: float = 3.0e-3
    weight_decay: float = 1.0e-4
    grad_clip_norm: float = 1.0
    throughput_warmup: int = 3
    throughput_repetitions: int = 10


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Complete immutable campaign configuration."""

    tasks: tuple[Alphabet2DTask, ...] = TASKS
    variants: tuple[ExperimentVariant, ...] = VARIANTS
    seeds: tuple[int, ...] = (11, 23, 47, 71, 101)
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    off_axis: OffAxisSpectralConfig = field(default_factory=OffAxisSpectralConfig)
    equal_power_phase: EqualPowerPhaseConfig = field(
        default_factory=EqualPowerPhaseConfig
    )


@dataclass(frozen=True, slots=True)
class ExperimentJob:
    """One independently restartable task/variant/seed fit."""

    task: Alphabet2DTask
    variant: ExperimentVariant
    seed: int

    @property
    def key(self) -> str:
        return f"{self.task}__{self.variant}__seed{self.seed}"


def jobs(config: ExperimentConfig) -> tuple[ExperimentJob, ...]:
    """Return the exact deterministic campaign matrix."""

    return tuple(
        ExperimentJob(task, variant, seed)
        for task in config.tasks
        for variant in config.variants
        for seed in config.seeds
    )


def _complex_multiply(
    left_real: Tensor,
    left_imag: Tensor,
    right_real: Tensor,
    right_imag: Tensor,
) -> tuple[Tensor, Tensor]:
    return (
        left_real * right_real - left_imag * right_imag,
        left_real * right_imag + left_imag * right_real,
    )


def _scan_exact_zoh(
    excitation_real: Tensor,
    excitation_imag: Tensor,
    damping: Tensor,
    frequency: Tensor,
    spacing: float,
    *,
    direction: Literal["forward", "backward"],
    backend: RecurrenceBackend,
) -> tuple[Tensor, Tensor]:
    """Apply the same exact-ZOH pole discretization used by ProductPoleField2D."""

    decay_real, decay_imag, gamma_real, gamma_imag = discrete_pole_real2d(
        damping,
        frequency,
        spacing,
    )
    drive_real, drive_imag = _complex_multiply(
        excitation_real,
        excitation_imag,
        gamma_real,
        gamma_imag,
    )
    expanded_real = decay_real.view(1, 1, -1).expand_as(drive_real)
    expanded_imag = decay_imag.view(1, 1, -1).expand_as(drive_imag)
    return recurrence_real2d_directional(
        expanded_real,
        expanded_imag,
        drive_real,
        drive_imag,
        backend,
        direction,
    )


def _one_dimensional_atlas(modes: int) -> tuple[Tensor, Tensor]:
    level_count = max(1, math.ceil(modes / 2))
    levels = torch.logspace(math.log10(0.5), math.log10(6.0), level_count)
    frequency = 2.0 * math.pi * levels[torch.arange(modes) // 2]
    frequency = frequency * torch.where(
        torch.arange(modes) % 2 == 0,
        torch.ones(modes),
        -torch.ones(modes),
    )
    damping = (frequency.abs() / 3.0).clamp_min(0.5)
    return damping, frequency


class _AuditableDirectionalField(nn.Module):
    """Shared exact-ZOH machinery for axial and raster controls."""

    def __init__(
        self,
        model_dim: int,
        modes: int,
        *,
        backend: RecurrenceBackend,
    ) -> None:
        super().__init__()
        if model_dim < 2 * modes:
            raise ValueError("model_dim must be at least twice the mode count")
        self.model_dim = model_dim
        self.modes = modes
        self.backend: RecurrenceBackend = backend
        self.analysis = nn.Linear(model_dim, 2 * modes, bias=False)
        nn.init.orthogonal_(self.analysis.weight)
        damping, frequency = _one_dimensional_atlas(modes)
        minimum_damping = 1.0e-3
        self.raw_damping = nn.Parameter(torch.log(torch.expm1(damping - minimum_damping)))
        self.register_buffer("base_frequency", frequency)
        self.frequency_offset = nn.Parameter(torch.zeros(modes))
        self.minimum_damping = minimum_damping
        self.frequency_offset_bound = math.pi

    def damping(self) -> Tensor:
        return self.minimum_damping + functional.softplus(self.raw_damping)

    def frequency(self) -> Tensor:
        return self.get_buffer("base_frequency") + self.frequency_offset_bound * torch.tanh(
            self.frequency_offset
        )

    def synthesize(self, states_real: Tensor, states_imag: Tensor) -> Tensor:
        mean_real = states_real.mean(dim=1)
        mean_imag = states_imag.mean(dim=1)
        frame_real, frame_imag = self.analysis.weight.chunk(2, dim=0)
        return torch.matmul(mean_real, frame_real) + torch.matmul(mean_imag, frame_imag)


class AxialPoleField2D(_AuditableDirectionalField):
    """Four-way horizontal/vertical exact-ZOH axial pole control."""

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        excitation_real, excitation_imag = self.analysis(inputs).chunk(2, dim=-1)
        batch, height, width, modes = excitation_real.shape
        horizontal_real = excitation_real.reshape(batch * height, width, modes)
        horizontal_imag = excitation_imag.reshape(batch * height, width, modes)
        horizontal = [
            _scan_exact_zoh(
                horizontal_real,
                horizontal_imag,
                self.damping(),
                self.frequency(),
                1.0 / width,
                direction=direction,
                backend=self.backend,
            )
            for direction in ("forward", "backward")
        ]
        vertical_real = excitation_real.permute(0, 2, 1, 3).reshape(
            batch * width,
            height,
            modes,
        )
        vertical_imag = excitation_imag.permute(0, 2, 1, 3).reshape(
            batch * width,
            height,
            modes,
        )
        vertical = [
            _scan_exact_zoh(
                vertical_real,
                vertical_imag,
                self.damping(),
                self.frequency(),
                1.0 / height,
                direction=direction,
                backend=self.backend,
            )
            for direction in ("forward", "backward")
        ]
        real_states = [
            state[0].reshape(batch, height, width, modes) for state in horizontal
        ]
        imag_states = [
            state[1].reshape(batch, height, width, modes) for state in horizontal
        ]
        real_states.extend(
            state[0]
            .reshape(batch, width, height, modes)
            .permute(0, 2, 1, 3)
            .contiguous()
            for state in vertical
        )
        imag_states.extend(
            state[1]
            .reshape(batch, width, height, modes)
            .permute(0, 2, 1, 3)
            .contiguous()
            for state in vertical
        )
        return torch.stack(real_states, dim=1), torch.stack(imag_states, dim=1)


class RasterPoleField1D(_AuditableDirectionalField):
    """Forward/backward exact-ZOH pole control over rasterized patch tokens."""

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        excitation_real, excitation_imag = self.analysis(inputs).chunk(2, dim=-1)
        batch, height, width, modes = excitation_real.shape
        flat_real = excitation_real.reshape(batch, height * width, modes)
        flat_imag = excitation_imag.reshape(batch, height * width, modes)
        states = [
            _scan_exact_zoh(
                flat_real,
                flat_imag,
                self.damping(),
                self.frequency(),
                1.0 / (height * width),
                direction=direction,
                backend=self.backend,
            )
            for direction in ("forward", "backward")
        ]
        return (
            torch.stack(
                [state[0].reshape(batch, height, width, modes) for state in states],
                dim=1,
            ),
            torch.stack(
                [state[1].reshape(batch, height, width, modes) for state in states],
                dim=1,
            ),
        )


class _DirectionalBlock(nn.Module):
    def __init__(
        self,
        field: _AuditableDirectionalField,
        *,
        model_dim: int,
        mlp_ratio: float,
        layer_scale_init: float,
    ) -> None:
        super().__init__()
        hidden_dim = max(model_dim, round(model_dim * mlp_ratio))
        self.local = nn.Conv2d(model_dim, model_dim, 3, padding=1, groups=model_dim)
        self.norm = nn.RMSNorm(model_dim)
        self.field = field
        self.pole_scale = nn.Parameter(torch.full((model_dim,), layer_scale_init))
        self.direct_scale = nn.Parameter(torch.zeros(model_dim))
        self.mlp_norm = nn.RMSNorm(model_dim)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, model_dim),
        )
        self.mlp_scale = nn.Parameter(torch.full((model_dim,), layer_scale_init))

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        local = self.local(inputs.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        normalized = self.norm(functional.silu(local))
        states_real, states_imag = self.field(normalized)
        pole_update = self.field.synthesize(states_real, states_imag)
        updated = inputs + self.pole_scale * (
            pole_update + self.direct_scale * normalized
        )
        output = updated + self.mlp_scale * self.mlp(self.mlp_norm(updated))
        return output, states_real, states_imag


class _DirectionalClassifier(nn.Module):
    """Architecture-matched two-bank classifier for axial/raster controls."""

    def __init__(self, config: ModelConfig, variant: Literal["axial2d", "raster1d"]) -> None:
        super().__init__()
        field_class = AxialPoleField2D if variant == "axial2d" else RasterPoleField1D
        quadrant_count = 4 if variant == "axial2d" else 2
        self.config = config
        self.patch_embed = nn.Conv2d(
            1,
            config.model_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.blocks = nn.ModuleList(
            [
                _DirectionalBlock(
                    field_class(
                        config.model_dim,
                        config.modes,
                        backend=config.recurrence_backend,
                    ),
                    model_dim=config.model_dim,
                    mlp_ratio=config.mlp_ratio,
                    layer_scale_init=config.layer_scale_init,
                )
                for _ in range(config.depth)
            ]
        )
        self.reader_local = nn.Conv2d(
            config.model_dim,
            config.model_dim,
            3,
            padding=1,
            groups=config.model_dim,
        )
        self.reader_norm = nn.RMSNorm(config.model_dim)
        self.reader = field_class(
            config.model_dim,
            config.modes,
            backend=config.recurrence_backend,
        )
        window_count = 1 if config.windows == "global" else 5
        per_bank = (
            window_count
            * quadrant_count
            * config.modes
            * (1 + 2 * len(config.lags))
        )
        self.classifier = nn.Linear(2 * per_bank, 2)

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.patch_embed(inputs).permute(0, 2, 3, 1)
        direct_real: Tensor | None = None
        direct_imag: Tensor | None = None
        for raw_block in self.blocks:
            block = cast("_DirectionalBlock", raw_block)
            features, direct_real, direct_imag = block(features)
        if direct_real is None or direct_imag is None:
            raise RuntimeError("directional classifier requires at least one block")
        reader_input = self.reader_local(features.permute(0, 3, 1, 2)).permute(
            0,
            2,
            3,
            1,
        )
        reader_input = self.reader_norm(functional.silu(reader_input))
        reader_real, reader_imag = self.reader(reader_input)
        direct = spatial_modal_moments(
            direct_real,
            direct_imag,
            lags=self.config.lags,
            windows=self.config.windows,
        )
        reader = spatial_modal_moments(
            reader_real,
            reader_imag,
            lags=self.config.lags,
            windows=self.config.windows,
        )
        return self.classifier(torch.cat((direct, reader), dim=-1))


def _readout_indices(config: ModelConfig, variant: ExperimentVariant) -> Tensor:
    coordinates = 1 + 2 * len(config.lags)
    per_window = 4 * config.modes * coordinates
    window_count = 1 if config.windows == "global" else 5
    per_bank = window_count * per_window
    if variant == "product2d":
        return torch.arange(2 * per_bank)
    if variant == "product_one_bank":
        return torch.arange(per_bank)
    if variant == "global_only":
        return torch.cat((torch.arange(per_window), per_bank + torch.arange(per_window)))
    if variant not in {"energy_only", "global_energy_only"}:
        raise ValueError(f"variant has no product descriptor selector: {variant}")
    selected_windows = range(window_count) if variant == "energy_only" else range(1)
    indices = [
        bank * per_bank + window * per_window + mode * coordinates
        for bank in range(2)
        for window in selected_windows
        for mode in range(4 * config.modes)
    ]
    return torch.tensor(indices, dtype=torch.long)


class _ProductReadoutClassifier(nn.Module):
    """Existing Alphabet2D backbone with an explicitly selected affine readout."""

    def __init__(self, config: ModelConfig, variant: ExperimentVariant) -> None:
        super().__init__()
        backbone_config = Alphabet2DConfig(
            input_channels=1,
            output_dim=2,
            image_size=config.image_size,
            patch_size=config.patch_size,
            model_dim=config.model_dim,
            modes=config.modes,
            depth=config.depth,
            mlp_ratio=config.mlp_ratio,
            lags=config.lags,
            windows=config.windows,
            recurrence_backend=config.recurrence_backend,
            layer_scale_init=config.layer_scale_init,
        )
        self.backbone = Alphabet2D(backbone_config)
        indices = _readout_indices(config, variant)
        self.register_buffer("readout_indices", indices)
        self.classifier = nn.Linear(indices.numel(), 2)

    def forward(self, inputs: Tensor) -> Tensor:
        descriptor, _ = self.backbone.forward_features(inputs)
        return self.classifier(
            descriptor.index_select(-1, self.get_buffer("readout_indices"))
        )


def _local_covariance_lags(radius: int) -> tuple[SpatialLag, ...]:
    if radius < 0:
        raise ValueError("local covariance radius must be non-negative")
    return tuple(
        (dx, dy)
        for dy in range(radius + 1)
        for dx in range(-radius, radius + 1)
        if dy > 0 or dx > 0
    )


class _LocalCovarianceLinear(nn.Module):
    """Raw mean/covariance coordinates followed by exactly one affine head."""

    def __init__(self, radius: int) -> None:
        super().__init__()
        self.lags = _local_covariance_lags(radius)
        self.classifier = nn.Linear(2 + len(self.lags), 2)

    def forward(self, inputs: Tensor) -> Tensor:
        centered = inputs - inputs.mean(dim=(-2, -1), keepdim=True)
        coordinates = [
            inputs.mean(dim=(-3, -2, -1)),
            centered.square().mean(dim=(-3, -2, -1)),
        ]
        height, width = inputs.shape[-2:]
        for delta_x, delta_y in self.lags:
            current_y = slice(delta_y, height)
            previous_y = slice(0, height - delta_y)
            if delta_x >= 0:
                current_x = slice(delta_x, width)
                previous_x = slice(0, width - delta_x)
            else:
                current_x = slice(0, width + delta_x)
                previous_x = slice(-delta_x, width)
            covariance = (
                centered[..., current_y, current_x]
                * centered[..., previous_y, previous_x]
            ).mean(dim=(-3, -2, -1))
            coordinates.append(covariance)
        return self.classifier(torch.stack(coordinates, dim=-1))


def build_model(variant: ExperimentVariant, config: ModelConfig) -> nn.Module:
    """Build one exact named ablation."""

    if variant in {
        "product2d",
        "product_one_bank",
        "energy_only",
        "global_energy_only",
        "global_only",
    }:
        return _ProductReadoutClassifier(config, variant)
    if variant in {"axial2d", "raster1d"}:
        directional_variant = cast("Literal['axial2d', 'raster1d']", variant)
        return _DirectionalClassifier(config, directional_variant)
    if variant == "local_covariance_linear":
        return _LocalCovarianceLinear(config.local_covariance_radius)
    raise ValueError(f"unknown experiment variant: {variant}")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _source_hashes() -> dict[str, str]:
    return source_file_hashes(
        ("alphabet2d.py", "alphabet2d_synthetic.py", "alphabet2d_experiment.py"),
        project_root=Path(__file__).parent,
    )


def _runtime_provenance() -> dict[str, object]:
    cuda_available = torch.cuda.is_available()
    return {
        "hostname": platform.node(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_device": torch.cuda.get_device_name(0) if cuda_available else None,
        "cudnn": torch.backends.cudnn.version(),
        "source_sha256": _source_hashes(),
    }


def _config_payload(config: ExperimentConfig) -> dict[str, Any]:
    return asdict(config)


def _contract_digest(config: ExperimentConfig) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "config": _config_payload(config),
        "source_sha256": _source_hashes(),
    }
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_torch_save(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    torch.save(payload, temporary)
    temporary.replace(path)


def initialize_campaign(root: Path, config: ExperimentConfig) -> dict[str, Any]:
    """Write or validate the immutable campaign contract."""

    if not config.tasks or not config.variants or not config.seeds:
        raise ValueError("tasks, variants, and seeds must all be non-empty")
    unknown_variants = set(config.variants) - set(VARIANTS)
    unknown_tasks = set(config.tasks) - set(TASKS)
    if unknown_variants or unknown_tasks:
        message = (
            f"unknown campaign entries: tasks={sorted(unknown_tasks)},"
            f" variants={sorted(unknown_variants)}"
        )
        raise ValueError(message)
    digest = _contract_digest(config)
    raw_payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_digest": digest,
        "config": _config_payload(config),
        "runtime": _runtime_provenance(),
        "jobs": [asdict(job) | {"key": job.key} for job in jobs(config)],
        "selection_rule": "maximum validation accuracy; test accessed only after selection",
        "restart_safe": True,
    }
    payload = cast("dict[str, Any]", json.loads(_canonical_json(raw_payload)))
    contract_path = root / "contract.json"
    if contract_path.exists():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError("immutable ALPHABET-2D campaign contract differs")
    else:
        _atomic_json(contract_path, payload)
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    return payload


def _device(value: str) -> torch.device:
    active = torch.device(value)
    if active.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return active


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _batches(
    split: TensorClassificationSplit,
    batch_size: int,
    *,
    shuffle_seed: int | None,
) -> tuple[tuple[Tensor, Tensor], ...]:
    count = split.targets.numel()
    if shuffle_seed is None:
        order = torch.arange(count)
    else:
        generator = torch.Generator(device="cpu").manual_seed(shuffle_seed)
        order = torch.randperm(count, generator=generator)
    return tuple(
        (split.inputs[indices], split.targets[indices])
        for indices in order.split(batch_size)
    )


def _evaluate(
    model: nn.Module,
    split: TensorClassificationSplit,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    correct = 0
    count = 0
    loss_sum = 0.0
    with torch.inference_mode():
        for cpu_inputs, cpu_targets in _batches(split, batch_size, shuffle_seed=None):
            inputs = cpu_inputs.to(device)
            targets = cpu_targets.to(device)
            logits = model(inputs)
            loss_sum += float(functional.cross_entropy(logits, targets, reduction="sum"))
            correct += int((logits.argmax(dim=-1) == targets).sum())
            count += targets.numel()
    return correct / count, loss_sum / count


def _oracle_metrics(
    job: ExperimentJob,
    split: TensorClassificationSplit,
    config: ExperimentConfig,
) -> dict[str, float]:
    if job.task == "off_axis":
        scores = off_axis_oracle_score(split.inputs, config.off_axis)
    else:
        scores = phase_arrangement_oracle_score(split.inputs, config.equal_power_phase)
    predictions = (scores > 0).long()
    return {"test_accuracy": float((predictions == split.targets).float().mean())}


def _throughput(
    model: nn.Module,
    inputs: Tensor,
    *,
    warmup: int,
    repetitions: int,
    device: torch.device,
) -> float:
    model.eval()
    active = inputs.to(device)
    with torch.inference_mode():
        for _ in range(warmup):
            model(active)
        _synchronize(device)
        started = time.perf_counter()
        for _ in range(repetitions):
            model(active)
        _synchronize(device)
    elapsed = time.perf_counter() - started
    return active.shape[0] * repetitions / max(elapsed, 1.0e-12)


def _validate_existing_result(
    path: Path,
    job: ExperimentJob,
    digest: str,
) -> dict[str, Any]:
    result = cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))
    if (
        result.get("status") != "done"
        or result.get("job_key") != job.key
        or result.get("contract_digest") != digest
    ):
        raise RuntimeError(f"invalid immutable result for {job.key}")
    return result


def run_job(  # noqa: C901, PLR0915
    root: Path,
    job: ExperimentJob,
    config: ExperimentConfig,
    *,
    device: str = "cuda",
) -> dict[str, Any]:
    """Fit or resume one job and atomically publish its final JSON result."""

    initialize_campaign(root, config)
    if job not in jobs(config):
        raise ValueError(f"job is outside the immutable campaign: {job}")
    digest = _contract_digest(config)
    result_path = root / "results" / f"{job.key}.json"
    if result_path.exists():
        return _validate_existing_result(result_path, job, digest)

    active_device = _device(device)
    torch.manual_seed(job.seed)
    if active_device.type == "cuda":
        torch.cuda.manual_seed_all(job.seed)
    model = build_model(job.variant, config.model).to(active_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    splits = make_alphabet2d_splits(
        job.task,
        train_per_class=config.training.train_per_class,
        validation_per_class=config.training.validation_per_class,
        test_per_class=config.training.test_per_class,
        seed=job.seed,
        off_axis_config=config.off_axis,
        phase_config=config.equal_power_phase,
    )
    checkpoint_path = root / "checkpoints" / f"{job.key}.pt"
    start_epoch = 0
    best_epoch = -1
    best_accuracy = -math.inf
    stale_epochs = 0
    best_state: dict[str, Tensor] | None = None
    history: list[dict[str, float | int]] = []
    finite_gradients = True
    train_examples = 0
    train_seconds = 0.0
    if checkpoint_path.exists():
        checkpoint = cast(
            "dict[str, Any]",
            torch.load(checkpoint_path, map_location="cpu", weights_only=True),
        )
        if (
            checkpoint.get("job_key") != job.key
            or checkpoint.get("contract_digest") != digest
        ):
            raise RuntimeError(f"checkpoint contract differs for {job.key}")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["next_epoch"])
        best_epoch = int(checkpoint["best_epoch"])
        best_accuracy = float(checkpoint["best_accuracy"])
        stale_epochs = int(checkpoint["stale_epochs"])
        best_state = cast("dict[str, Tensor] | None", checkpoint["best_state"])
        history = cast("list[dict[str, float | int]]", checkpoint["history"])
        finite_gradients = bool(checkpoint["finite_gradients"])
        train_examples = int(checkpoint["train_examples"])
        train_seconds = float(checkpoint["train_seconds"])

    for epoch in range(start_epoch, config.training.max_epochs):
        model.train()
        started = time.perf_counter()
        epoch_examples = 0
        for cpu_inputs, cpu_targets in _batches(
            splits.train,
            config.training.batch_size,
            shuffle_seed=job.seed * 1_000_003 + epoch,
        ):
            inputs = cpu_inputs.to(active_device)
            targets = cpu_targets.to(active_device)
            optimizer.zero_grad(set_to_none=True)
            loss = functional.cross_entropy(model(inputs), targets)
            loss.backward()
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.grad is not None
            ]
            step_is_finite = bool(gradients) and all(
                bool(torch.isfinite(gradient).all()) for gradient in gradients
            )
            finite_gradients = finite_gradients and step_is_finite
            if not step_is_finite:
                raise FloatingPointError(f"non-finite gradient in {job.key}, epoch {epoch}")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.training.grad_clip_norm,
            )
            optimizer.step()
            epoch_examples += targets.numel()
        _synchronize(active_device)
        elapsed = time.perf_counter() - started
        train_examples += epoch_examples
        train_seconds += elapsed
        validation_accuracy, validation_loss = _evaluate(
            model,
            splits.validation,
            batch_size=config.training.batch_size,
            device=active_device,
        )
        improved = validation_accuracy > best_accuracy
        if improved:
            best_accuracy = validation_accuracy
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1
        history.append(
            {
                "epoch": epoch,
                "validation_accuracy": validation_accuracy,
                "validation_loss": validation_loss,
                "training_examples_per_second": epoch_examples / max(elapsed, 1.0e-12),
            }
        )
        _atomic_torch_save(
            checkpoint_path,
            {
                "schema_version": SCHEMA_VERSION,
                "job_key": job.key,
                "contract_digest": digest,
                "next_epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_epoch": best_epoch,
                "best_accuracy": best_accuracy,
                "stale_epochs": stale_epochs,
                "best_state": best_state,
                "history": history,
                "finite_gradients": finite_gradients,
                "train_examples": train_examples,
                "train_seconds": train_seconds,
            },
        )
        if stale_epochs >= config.training.patience:
            break

    if best_state is None:
        raise RuntimeError(f"no validation checkpoint was produced for {job.key}")
    model.load_state_dict(best_state)
    test_accuracy, test_loss = _evaluate(
        model,
        splits.test,
        batch_size=config.training.batch_size,
        device=active_device,
    )
    throughput_inputs = splits.test.inputs[: min(config.training.batch_size, 64)]
    inference_throughput = _throughput(
        model,
        throughput_inputs,
        warmup=config.training.throughput_warmup,
        repetitions=config.training.throughput_repetitions,
        device=active_device,
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "done",
        "job_key": job.key,
        "job": asdict(job),
        "contract_digest": digest,
        "exact_config": _config_payload(config),
        "selection": {
            "criterion": "validation_accuracy",
            "best_epoch": best_epoch,
            "best_validation_accuracy": best_accuracy,
            "epochs_completed": len(history),
            "test_used_for_selection": False,
        },
        "test": {"accuracy": test_accuracy, "cross_entropy": test_loss},
        "oracle": _oracle_metrics(job, splits.test, config),
        "parameters": {
            "total": total_parameters,
            "trainable": trainable_parameters,
        },
        "throughput": {
            "training_examples_per_second": train_examples / max(train_seconds, 1.0e-12),
            "inference_examples_per_second": inference_throughput,
            "inference_batch_size": int(throughput_inputs.shape[0]),
            "device": str(active_device),
        },
        "finite_gradients": finite_gradients,
        "history": history,
    }
    _atomic_json(result_path, result)
    return result


def summarize(root: Path, config: ExperimentConfig) -> dict[str, Any]:
    """Aggregate only a complete, contract-valid result matrix."""

    digest = _contract_digest(config)
    rows = [
        _validate_existing_result(root / "results" / f"{job.key}.json", job, digest)
        for job in jobs(config)
    ]
    groups: dict[str, dict[str, float | int]] = {}
    for task in config.tasks:
        for variant in config.variants:
            selected = [
                row
                for row in rows
                if row["job"]["task"] == task and row["job"]["variant"] == variant
            ]
            accuracies = [float(row["test"]["accuracy"]) for row in selected]
            groups[f"{task}__{variant}"] = {
                "runs": len(selected),
                "mean_test_accuracy": sum(accuracies) / len(accuracies),
                "min_test_accuracy": min(accuracies),
                "max_test_accuracy": max(accuracies),
            }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "contract_digest": digest,
        "jobs": len(rows),
        "groups": groups,
    }
    _atomic_json(root / "summary.json", summary)
    return summary


def run_campaign(
    root: Path,
    config: ExperimentConfig,
    *,
    device: str = "cuda",
    max_jobs: int | None = None,
) -> dict[str, Any]:
    """Run unfinished jobs sequentially; completed immutable results are skipped."""

    initialize_campaign(root, config)
    processed = 0
    for job in jobs(config):
        result_path = root / "results" / f"{job.key}.json"
        if result_path.exists():
            _validate_existing_result(result_path, job, _contract_digest(config))
            continue
        run_job(root, job, config, device=device)
        processed += 1
        if max_jobs is not None and processed >= max_jobs:
            break
    finished = sum(
        (root / "results" / f"{job.key}.json").exists()
        for job in jobs(config)
    )
    if finished == len(jobs(config)):
        return summarize(root, config)
    return {
        "jobs_processed": processed,
        "jobs_finished": finished,
        "jobs_total": len(jobs(config)),
    }


__all__ = [
    "TASKS",
    "VARIANTS",
    "AxialPoleField2D",
    "ExperimentConfig",
    "ExperimentJob",
    "ExperimentVariant",
    "ModelConfig",
    "RasterPoleField1D",
    "TrainingConfig",
    "build_model",
    "initialize_campaign",
    "jobs",
    "run_campaign",
    "run_job",
    "summarize",
]
