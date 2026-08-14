"""Clean circular energy, modulus-cascade, and spatial-window S-B controls."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class ShapeGateConfig:
    """Frozen S-B transform contract."""

    modes: int = 25
    pole_radius: float = 0.90

    def validate(self) -> None:
        side = round(math.sqrt(self.modes))
        if side * side != self.modes:
            message = "modes must form a square 2D atlas"
            raise ValueError(message)
        if not 0.0 < self.pole_radius < 1.0:
            message = "pole_radius must lie in (0, 1)"
            raise ValueError(message)


@dataclass(frozen=True, slots=True)
class TextonArrangementConfig:
    """Central-cluster versus quadrant-distributed texton task."""

    height: int = 64
    width: int = 64
    texton_scale: float = 2.8
    carrier_frequency: float = math.pi / 2.5
    noise_standard_deviation: float = 0.0
    jitter_pixels: int = 2

    def validate(self) -> None:
        if min(self.height, self.width) < 32:
            message = "texton task requires at least 32x32 fields"
            raise ValueError(message)
        if self.texton_scale <= 0.0 or self.noise_standard_deviation < 0.0:
            message = "texton scale must be positive and noise nonnegative"
            raise ValueError(message)
        if self.jitter_pixels < 0:
            message = "jitter must be nonnegative"
            raise ValueError(message)


@dataclass(frozen=True, slots=True)
class HomometricTextonConfig:
    """Non-congruent homometric point patterns convolved with one compact texton."""

    height: int = 60
    width: int = 60
    lattice_period: int = 12
    texton_radius: int = 1
    carrier_frequency: float = math.pi / 2.5
    jitter_pixels: int = 2

    def validate(self) -> None:
        if self.lattice_period != 12:
            message = "the preregistered homometric pair uses period 12"
            raise ValueError(message)
        if self.width % self.lattice_period:
            message = "width must be divisible by the homometric lattice period"
            raise ValueError(message)
        if self.height < 16 or self.width < 24:
            message = "homometric texton fields are too small"
            raise ValueError(message)
        spacing = self.width // self.lattice_period
        if self.texton_radius < 0 or 2 * self.texton_radius + 1 > spacing:
            message = "compact textons must not overlap on the homometric lattice"
            raise ValueError(message)
        if self.jitter_pixels < 0:
            message = "jitter must be nonnegative"
            raise ValueError(message)


@dataclass(frozen=True, slots=True)
class PairedInputAudit:
    """Numerical equality audit for paired nuisance controls."""

    maximum_power_absolute_error: float
    relative_power_l2_error: float
    maximum_histogram_absolute_error: float


@dataclass(frozen=True, slots=True)
class PairAuditThresholds:
    """Explicit validity thresholds for an immutable S-B campaign."""

    maximum_power_absolute_error: float = 2.0e-5
    relative_power_l2_error: float = 2.0e-6
    maximum_histogram_absolute_error: float = 2.0e-6


def _frequency_atlas(config: ShapeGateConfig, device: torch.device) -> tuple[Tensor, Tensor]:
    config.validate()
    side = round(math.sqrt(config.modes))
    values = torch.linspace(
        -3.0 * math.pi / 4.0,
        3.0 * math.pi / 4.0,
        side,
        device=device,
    )
    grid_y, grid_x = torch.meshgrid(values, values, indexing="ij")
    return grid_x.flatten(), grid_y.flatten()


def _transfer_1d(
    grid: Tensor,
    frequency: Tensor,
    radius: float,
) -> Tensor:
    pole = torch.polar(torch.full_like(frequency, radius), frequency)
    phase = torch.polar(torch.ones_like(grid), -grid)
    transfer = (1.0 - radius) / (1.0 - pole[:, None] * phase[None, :])
    normalization = transfer.abs().square().mean(dim=-1, keepdim=True).sqrt()
    return transfer / normalization.clamp_min(torch.finfo(grid.dtype).eps)


def _product_transfer(
    height: int,
    width: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    config: ShapeGateConfig,
) -> Tensor:
    frequency_x, frequency_y = _frequency_atlas(config, device)
    grid_x = torch.fft.fftfreq(
        width,
        device=device,
        dtype=dtype,
    ) * (2.0 * math.pi)
    grid_y = torch.fft.fftfreq(
        height,
        device=device,
        dtype=dtype,
    ) * (2.0 * math.pi)
    transfer_x = _transfer_1d(grid_x, frequency_x, config.pole_radius)
    transfer_y = _transfer_1d(grid_y, frequency_y, config.pole_radius)
    return transfer_y[:, :, None] * transfer_x[:, None, :]


def circular_product_responses(
    inputs: Tensor,
    config: ShapeGateConfig,
) -> Tensor:
    """Apply a periodic product-Poisson bank and return complex response maps."""
    if inputs.ndim != 4 or inputs.shape[1] != 1:
        message = "shape-gate inputs must have shape [B,1,H,W]"
        raise ValueError(message)
    transfer = _product_transfer(
        inputs.shape[-2],
        inputs.shape[-1],
        dtype=inputs.dtype,
        device=inputs.device,
        config=config,
    )
    coefficients = torch.fft.fft2(inputs[:, 0], norm="ortho")
    return torch.fft.ifft2(
        coefficients[:, None] * transfer[None],
        norm="ortho",
    )


def circular_energy_features(inputs: Tensor, config: ShapeGateConfig) -> Tensor:
    """Return first-bank global energies, which depend only on Fourier power."""
    response = circular_product_responses(inputs, config)
    return response.abs().square().mean(dim=(-2, -1)).log1p()


def modulus_cascade_maps(inputs: Tensor, config: ShapeGateConfig) -> Tensor:
    """Return second-bank maps after the sole explicit modulus nonlinearity."""
    first = circular_product_responses(inputs, config)
    envelope = first.abs()
    envelope = envelope - envelope.mean(dim=(-2, -1), keepdim=True)
    transfer = _product_transfer(
        envelope.shape[-2],
        envelope.shape[-1],
        dtype=envelope.dtype,
        device=envelope.device,
        config=config,
    )
    coefficients = torch.fft.fft2(envelope, norm="ortho")
    return torch.fft.ifft2(coefficients * transfer[None], norm="ortho")


def cascade_global_features(inputs: Tensor, config: ShapeGateConfig) -> Tensor:
    """Return translation-invariant second-bank global energies."""
    second = modulus_cascade_maps(inputs, config)
    return second.abs().square().mean(dim=(-2, -1)).log1p()


def linear_cascade_features(inputs: Tensor, config: ShapeGateConfig) -> Tensor:
    """Return a no-modulus linear-to-linear control that remains power-only."""
    first = circular_product_responses(inputs, config)
    transfer = _product_transfer(
        first.shape[-2],
        first.shape[-1],
        dtype=first.real.dtype,
        device=first.device,
        config=config,
    )
    combined = torch.fft.ifft2(
        torch.fft.fft2(first, norm="ortho") * transfer[None],
        norm="ortho",
    )
    return combined.abs().square().mean(dim=(-2, -1)).log1p()


def _dct_low_frequency(envelope: Tensor) -> Tensor:
    height, width = envelope.shape[-2:]
    y = torch.arange(height, dtype=envelope.dtype, device=envelope.device)
    x = torch.arange(width, dtype=envelope.dtype, device=envelope.device)
    basis_y = torch.stack(
        (
            torch.ones_like(y),
            torch.cos(math.pi * (y + 0.5) / height),
        )
    )
    basis_x = torch.stack(
        (
            torch.ones_like(x),
            torch.cos(math.pi * (x + 0.5) / width),
        )
    )
    basis = torch.einsum("ay,bx->abyx", basis_y, basis_x).reshape(4, height, width)
    basis = basis / basis.square().sum(dim=(-2, -1), keepdim=True).sqrt()
    return torch.einsum("bhw,khw->bk", envelope, basis)


def cascade_window_features(inputs: Tensor, config: ShapeGateConfig) -> Tensor:
    """Return 2x2 second-bank energies plus four low-frequency DCT moments."""
    second = modulus_cascade_maps(inputs, config)
    energy = second[:, :4].abs().square()
    middle_y = energy.shape[-2] // 2
    middle_x = energy.shape[-1] // 2
    quadrants = (
        energy[..., :middle_y, :middle_x],
        energy[..., :middle_y, middle_x:],
        energy[..., middle_y:, :middle_x],
        energy[..., middle_y:, middle_x:],
    )
    window_energy = torch.cat(
        [quadrant.mean(dim=(-2, -1)).log1p() for quadrant in quadrants],
        dim=-1,
    )
    envelope = second.abs().mean(dim=1)
    dct = _dct_low_frequency(envelope)
    return torch.cat((window_energy, dct), dim=-1)


def cascade_path_energy_maps(inputs: Tensor, config: ShapeGateConfig) -> Tensor:
    """Return the common ``|z2|^2`` maps used by every v3 cascade readout."""
    return modulus_cascade_maps(inputs, config).abs().square()


def _shared_path_indices(path_count: int, available_paths: int, device: torch.device) -> Tensor:
    if path_count < 4 or path_count > available_paths or path_count % 4:
        message = "shared path count must be a multiple of four within the bank"
        raise ValueError(message)
    return torch.arange(path_count, device=device)


def shared_path_global_features(
    inputs: Tensor,
    config: ShapeGateConfig,
    *,
    path_count: int = 20,
) -> Tensor:
    """Return one global ``|z2|^2`` coordinate for each fixed cascade path."""
    energy = cascade_path_energy_maps(inputs, config)
    indices = _shared_path_indices(path_count, energy.shape[1], energy.device)
    selected = energy.index_select(1, indices)
    return selected.mean(dim=(-2, -1))


def quadrant_dct_basis(
    height: int,
    width: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Return four orthonormal, piecewise-constant 2x2 DCT basis maps."""
    if height % 2 or width % 2:
        message = "2x2 spatial coordinates require even map dimensions"
        raise ValueError(message)
    signs = torch.tensor(
        (
            (1.0, 1.0, 1.0, 1.0),
            (1.0, -1.0, 1.0, -1.0),
            (1.0, 1.0, -1.0, -1.0),
            (1.0, -1.0, -1.0, 1.0),
        ),
        dtype=dtype,
        device=device,
    )
    basis = torch.empty(4, height, width, dtype=dtype, device=device)
    middle_y = height // 2
    middle_x = width // 2
    slices = (
        (slice(0, middle_y), slice(0, middle_x)),
        (slice(0, middle_y), slice(middle_x, width)),
        (slice(middle_y, height), slice(0, middle_x)),
        (slice(middle_y, height), slice(middle_x, width)),
    )
    for quadrant, (rows, columns) in enumerate(slices):
        basis[:, rows, columns] = signs[:, quadrant, None, None]
    return basis / math.sqrt(height * width)


def shared_path_window_features(
    inputs: Tensor,
    config: ShapeGateConfig,
    *,
    path_count: int = 20,
) -> Tensor:
    """Return equal-budget spatial coordinates from the same v3 cascade paths.

    Each path contributes exactly one coordinate.  The assigned 2x2 DCT basis
    cycles through DC, horizontal, vertical, and checkerboard, so this readout
    has the same width and sees the same paths as ``shared_path_global_features``.
    """
    energy = cascade_path_energy_maps(inputs, config)
    indices = _shared_path_indices(path_count, energy.shape[1], energy.device)
    selected = energy.index_select(1, indices)
    basis = quadrant_dct_basis(
        selected.shape[-2],
        selected.shape[-1],
        dtype=selected.dtype,
        device=selected.device,
    )
    assigned_basis = basis[
        torch.arange(path_count, device=selected.device) % basis.shape[0]
    ]
    return torch.einsum("bphw,phw->bp", selected, assigned_basis)


def _texton_template(
    config: TextonArrangementConfig,
    centers: tuple[tuple[float, float], ...],
) -> Tensor:
    y = torch.arange(config.height, dtype=torch.float64)
    x = torch.arange(config.width, dtype=torch.float64)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    template = torch.zeros(config.height, config.width, dtype=torch.float64)
    for center_y, center_x in centers:
        offset_y = grid_y - center_y
        offset_x = grid_x - center_x
        radius = offset_x.square() + offset_y.square()
        carrier = torch.cos(config.carrier_frequency * (offset_x + 0.35 * offset_y))
        template += torch.exp(-radius / (2.0 * config.texton_scale**2)) * carrier
    return template


def make_texton_arrangement_dataset(
    samples_per_class: int,
    *,
    seed: int,
    config: TextonArrangementConfig | None = None,
) -> tuple[Tensor, Tensor]:
    """Generate central-cluster and quadrant-distributed copies of one texton."""
    active = config or TextonArrangementConfig()
    active.validate()
    if samples_per_class < 1:
        message = "samples_per_class must be positive"
        raise ValueError(message)
    center_y = active.height / 2.0
    center_x = active.width / 2.0
    cluster_offset = 3.5
    signs = ((-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0))
    central = tuple(
        (
            center_y + sign_y * cluster_offset,
            center_x + sign_x * cluster_offset,
        )
        for sign_y, sign_x in signs
    )
    central_template = _texton_template(active, central)
    templates = torch.stack(
        (
            central_template,
            torch.roll(
                central_template,
                shifts=(-active.height // 4, -active.width // 4),
                dims=(-2, -1),
            ),
        )
    )
    templates = templates / templates.square().mean(dim=(-2, -1), keepdim=True).sqrt()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    gain = torch.exp(
        0.15
        * torch.randn(samples_per_class, 1, 1, generator=generator, dtype=torch.float64)
    )
    examples = [[], []]
    for index in range(samples_per_class):
        shift_y = 0
        shift_x = 0
        if active.jitter_pixels:
            shift_y = int(
                torch.randint(
                    -active.jitter_pixels,
                    active.jitter_pixels + 1,
                    (),
                    generator=generator,
                )
            )
            shift_x = int(
                torch.randint(
                    -active.jitter_pixels,
                    active.jitter_pixels + 1,
                    (),
                    generator=generator,
                )
            )
        for class_id, template in enumerate(templates):
            example = gain[index] * template
            if active.noise_standard_deviation:
                example = example + active.noise_standard_deviation * torch.randn(
                    active.height,
                    active.width,
                    generator=generator,
                    dtype=torch.float64,
                )
            example = torch.roll(example, (shift_y, shift_x), dims=(-2, -1))
            examples[class_id].append(example)
    stacked = [torch.stack(class_examples)[:, None] for class_examples in examples]
    targets = [
        torch.full((samples_per_class,), class_id, dtype=torch.long)
        for class_id in range(2)
    ]
    inputs = torch.cat(stacked).float()
    labels = torch.cat(targets)
    order = torch.randperm(labels.numel(), generator=generator)
    return inputs[order], labels[order]


def homometric_point_sets() -> tuple[tuple[tuple[int, int], ...], ...]:
    """Return a fixed non-dihedrally-congruent homometric pair on ``Z_12``."""
    first = ((0, value) for value in (0, 1, 4, 6))
    second = ((0, value) for value in (0, 1, 3, 7))
    return tuple(first), tuple(second)


def toroidal_difference_multiset(
    points: tuple[tuple[int, int], ...],
    *,
    height: int,
    width: int,
) -> tuple[tuple[int, int], ...]:
    """Return the sorted directed difference multiset on a finite torus."""
    return tuple(
        sorted(
            ((ay - by) % height, (ax - bx) % width)
            for (ay, ax), (by, bx) in product(points, repeat=2)
        )
    )


def _dihedral_transforms(
    points: tuple[tuple[int, int], ...],
    *,
    height: int,
    width: int,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    if height != width:
        matrices = (
            (1, 0, 0, 1),
            (-1, 0, 0, 1),
            (1, 0, 0, -1),
            (-1, 0, 0, -1),
        )
    else:
        matrices = (
            (1, 0, 0, 1),
            (0, -1, 1, 0),
            (-1, 0, 0, -1),
            (0, 1, -1, 0),
            (1, 0, 0, -1),
            (-1, 0, 0, 1),
            (0, 1, 1, 0),
            (0, -1, -1, 0),
        )
    return tuple(
        tuple(
            sorted(
                (
                    (a * y + b * x) % height,
                    (c * y + d * x) % width,
                )
                for y, x in points
            )
        )
        for a, b, c, d in matrices
    )


def toroidally_congruent(
    first: tuple[tuple[int, int], ...],
    second: tuple[tuple[int, int], ...],
    *,
    height: int,
    width: int,
) -> bool:
    """Return whether point sets agree under a toroidal translation and D4 map."""
    target = tuple(sorted(second))
    for transformed in _dihedral_transforms(first, height=height, width=width):
        anchor_y, anchor_x = transformed[0]
        for target_y, target_x in target:
            shift_y = target_y - anchor_y
            shift_x = target_x - anchor_x
            shifted = tuple(
                sorted(
                    ((y + shift_y) % height, (x + shift_x) % width)
                    for y, x in transformed
                )
            )
            if shifted == target:
                return True
    return False


def _compact_texton(config: HomometricTextonConfig) -> Tensor:
    radius = config.texton_radius
    offsets = torch.arange(-radius, radius + 1, dtype=torch.float64)
    grid_y, grid_x = torch.meshgrid(offsets, offsets, indexing="ij")
    scale = max(float(radius), 1.0)
    envelope = torch.exp(-(grid_x.square() + grid_y.square()) / (2.0 * scale**2))
    carrier = torch.cos(config.carrier_frequency * (grid_x + 0.35 * grid_y))
    patch = envelope * carrier
    kernel = torch.zeros(config.height, config.width, dtype=torch.float64)
    for row in range(patch.shape[0]):
        for column in range(patch.shape[1]):
            kernel[(row - radius) % config.height, (column - radius) % config.width] = (
                patch[row, column]
            )
    return kernel


def _embedded_homometric_points(
    config: HomometricTextonConfig,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    abstract = homometric_point_sets()
    spacing = config.width // config.lattice_period
    center_y = config.height // 2
    return tuple(
        tuple((center_y, (index * spacing) % config.width) for _, index in point_set)
        for point_set in abstract
    )


def make_homometric_texton_arrangement_pairs(
    samples: int,
    *,
    seed: int,
    config: HomometricTextonConfig | None = None,
) -> Tensor:
    """Return paired non-congruent texton arrangements before any shuffle."""
    active = config or HomometricTextonConfig()
    active.validate()
    if samples < 1:
        message = "samples must be positive"
        raise ValueError(message)
    point_sets = _embedded_homometric_points(active)
    if toroidally_congruent(
        point_sets[0],
        point_sets[1],
        height=active.height,
        width=active.width,
    ):
        message = "homometric point sets unexpectedly became congruent"
        raise RuntimeError(message)
    impulses = torch.zeros(2, active.height, active.width, dtype=torch.float64)
    for class_id, points in enumerate(point_sets):
        for row, column in points:
            impulses[class_id, row, column] = 1.0
    kernel = _compact_texton(active)
    templates = torch.fft.ifft2(
        torch.fft.fft2(impulses, norm="ortho")
        * torch.fft.fft2(kernel, norm="ortho"),
        norm="ortho",
    ).real
    templates = templates / templates.square().mean(dim=(-2, -1), keepdim=True).sqrt()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    pairs = []
    for _ in range(samples):
        gain = torch.exp(0.15 * torch.randn((), generator=generator, dtype=torch.float64))
        shift_y = shift_x = 0
        if active.jitter_pixels:
            shift_y = int(
                torch.randint(
                    -active.jitter_pixels,
                    active.jitter_pixels + 1,
                    (),
                    generator=generator,
                )
            )
            shift_x = int(
                torch.randint(
                    -active.jitter_pixels,
                    active.jitter_pixels + 1,
                    (),
                    generator=generator,
                )
            )
        pair = torch.roll(gain * templates, (shift_y, shift_x), dims=(-2, -1))
        pairs.append(pair[:, None])
    return torch.stack(pairs).float()


def audit_paired_inputs(pairs: Tensor) -> PairedInputAudit:
    """Audit power and histogram equality on ``[pair, class, channel, H, W]``."""
    if pairs.ndim != 5 or pairs.shape[1] != 2:
        message = "paired inputs must have shape [N,2,C,H,W]"
        raise ValueError(message)
    power = torch.fft.fft2(pairs.double(), norm="ortho").abs().square()
    power_difference = power[:, 0] - power[:, 1]
    denominator = power[:, 0].square().sum().sqrt().clamp_min(
        torch.finfo(torch.float64).eps
    )
    histograms = pairs.double().flatten(2).sort(dim=-1).values
    return PairedInputAudit(
        maximum_power_absolute_error=float(power_difference.abs().amax()),
        relative_power_l2_error=float(power_difference.square().sum().sqrt() / denominator),
        maximum_histogram_absolute_error=float(
            (histograms[:, 0] - histograms[:, 1]).abs().amax()
        ),
    )


def paired_audit_is_valid(
    audit: PairedInputAudit,
    thresholds: PairAuditThresholds,
) -> bool:
    """Apply every preregistered paired-input validity threshold."""
    return (
        audit.maximum_power_absolute_error
        <= thresholds.maximum_power_absolute_error
        and audit.relative_power_l2_error <= thresholds.relative_power_l2_error
        and audit.maximum_histogram_absolute_error
        <= thresholds.maximum_histogram_absolute_error
    )


def make_homometric_texton_arrangement_dataset(
    samples_per_class: int,
    *,
    seed: int,
    config: HomometricTextonConfig | None = None,
) -> tuple[Tensor, Tensor]:
    """Shuffle an already-audited paired homometric texton dataset."""
    pairs = make_homometric_texton_arrangement_pairs(
        samples_per_class,
        seed=seed,
        config=config,
    )
    audit = audit_paired_inputs(pairs)
    if not paired_audit_is_valid(audit, PairAuditThresholds()):
        message = f"homometric pair audit failed before shuffle: {audit}"
        raise RuntimeError(message)
    inputs = pairs.permute(1, 0, 2, 3, 4).reshape(
        2 * samples_per_class,
        *pairs.shape[2:],
    )
    targets = torch.arange(2, dtype=torch.long).repeat_interleave(samples_per_class)
    generator = torch.Generator(device="cpu").manual_seed(seed + 97_003)
    order = torch.randperm(targets.numel(), generator=generator)
    return inputs[order], targets[order]


__all__ = [
    "HomometricTextonConfig",
    "PairAuditThresholds",
    "PairedInputAudit",
    "ShapeGateConfig",
    "TextonArrangementConfig",
    "audit_paired_inputs",
    "cascade_global_features",
    "cascade_path_energy_maps",
    "cascade_window_features",
    "circular_energy_features",
    "circular_product_responses",
    "homometric_point_sets",
    "linear_cascade_features",
    "make_homometric_texton_arrangement_dataset",
    "make_homometric_texton_arrangement_pairs",
    "make_texton_arrangement_dataset",
    "modulus_cascade_maps",
    "paired_audit_is_valid",
    "quadrant_dct_basis",
    "shared_path_global_features",
    "shared_path_window_features",
    "toroidal_difference_multiset",
    "toroidally_congruent",
]
