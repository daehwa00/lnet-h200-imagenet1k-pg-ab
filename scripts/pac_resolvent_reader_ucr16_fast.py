"""UCR-16 screen for lag reader versus lag-free resolvent reader."""

# pyright: reportPrivateUsage=false
# ruff: noqa: SLF001

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from lnet.pac_final_validation import UCR_SECONDS
from lnet.pac_metrics import count_parameters
from optimization.learned_two_tap_resolvent_reader import (
    LagReaderALPHABET,
    ResolventReaderALPHABET,
)
from scripts import pac_pole_attention_ucr16_fast as base

ROOT = Path(".omx/results/pac-resolvent-reader-ucr16-fast-20260720")


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    paths = (
        Path("optimization/learned_two_tap_alphabet.py"),
        Path("optimization/learned_two_tap_resolvent_reader.py"),
        Path("optimization/masked_modal_moments.py"),
        Path("scripts/pac_pole_attention_ucr16_fast.py"),
        Path("scripts/pac_resolvent_reader_ucr16_fast.py"),
    )
    return {str(path): hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths}


def _design() -> dict[str, object]:
    baseline = LagReaderALPHABET(1, base.MODEL_DIM, base.MODES, 5)
    candidate = ResolventReaderALPHABET(1, base.MODEL_DIM, base.MODES, 5)
    baseline_parameters = count_parameters(baseline)
    candidate_parameters = count_parameters(candidate)
    return {
        "schema": "pac_resolvent_reader_ucr16_fast_contract.v1",
        "purpose": "validation-only replacement screen for the fixed reader lag-1/lag-4 moments",
        "official_test_accessed": False,
        "datasets": list(base.DATASETS),
        "excluded_slow_datasets": ["FordA", "FordB"],
        "seeds": list(base.SEEDS),
        "variants": {
            "lag_reader": "fixed reader lag-1/lag-4 moments",
            "pole_attention": "lag-free resolvent reader (legacy harness candidate key)",
        },
        "model_dim": base.MODEL_DIM,
        "modes": base.MODES,
        "epochs": base.EPOCHS,
        "batch_size": base.BATCH_SIZE,
        "learning_rate": base.LEARNING_RATE,
        "weight_decay": base.WEIGHT_DECAY,
        "grad_clip_norm": base.GRAD_CLIP_NORM,
        "hyperparameter_tuning": False,
        "controlled_difference": (
            "the learned-two-tap writer, reader local lift, reader poles, pooled real stream, "
            "D, M, writer moments, and classifier width are fixed; only reader lag moments "
            "are replaced by two same-pole resolvent recurrences and zero-initialized "
            "residual self-attention across the M mode tokens"
        ),
        "resolvent_contract": {
            "auxiliary_states": "p'=lambda*p+alpha*z; q'=lambda*q+alpha*p",
            "discretization": "exact ZOH using the reader pole transition and alpha-scaled gain",
            "tokens": "[log1p(E), Re NCorr(z,p), Im NCorr(z,p), Re NCorr(z,q), Im NCorr(z,q)]",
            "mode_mixing": "Y=X+gamma*MHA_mode(X), one head, gamma initialized to zero",
            "reader_descriptor_width": "5M",
            "last_state_query": False,
            "fixed_reader_lags": False,
            "complexity": "linear in sequence length before O(M^2) mode mixing",
        },
        "parameter_control": {
            "lag_reader": baseline_parameters,
            "resolvent_reader": candidate_parameters,
            "candidate_minus_lag": candidate_parameters - baseline_parameters,
            "expected_candidate_minus_lag": 121,
        },
        "source_sha256": _source_hashes(),
    }


def design_sha256() -> str:
    encoded = json.dumps(_design(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def jobs() -> list[base.Job]:
    digest = design_sha256()
    return [
        base.Job(
            key=f"resolvent_reader_ucr16_fast:{dataset}:{variant}:seed{seed}",
            dataset=dataset,
            variant=variant,
            split_seed=seed,
            train_seed=seed,
            model_dim=base.MODEL_DIM,
            modes=base.MODES,
            heads=1,
            epochs=base.EPOCHS,
            batch_size=base.BATCH_SIZE,
            learning_rate=base.LEARNING_RATE,
            weight_decay=base.WEIGHT_DECAY,
            grad_clip_norm=base.GRAD_CLIP_NORM,
            evaluation_split="validation",
            estimated_seconds=UCR_SECONDS[dataset],
            design_sha256=digest,
        )
        for dataset in base.DATASETS
        for seed in base.SEEDS
        for variant in base.VARIANTS
    ]


def _build_model(job: base.Job, input_dim: int, output_dim: int):  # noqa: ANN202
    if job.variant == "lag_reader":
        return LagReaderALPHABET(input_dim, job.model_dim, job.modes, output_dim)
    return ResolventReaderALPHABET(input_dim, job.model_dim, job.modes, output_dim)


def main() -> None:
    base.ROOT = ROOT
    base._design = _design
    base.design_sha256 = design_sha256
    base.jobs = jobs
    base._build_model = _build_model
    base.main()


if __name__ == "__main__":
    main()
