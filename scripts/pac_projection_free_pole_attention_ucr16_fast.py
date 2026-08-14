"""UCR-16 screen for lag reader versus projection-free pole attention."""

# pyright: reportPrivateUsage=false
# ruff: noqa: SLF001

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from lnet.pac_final_validation import UCR_SECONDS
from lnet.pac_metrics import count_parameters
from optimization.learned_two_tap_pole_attention import (
    LagReaderALPHABET,
    ProjectionFreePoleAttentionALPHABET,
)
from scripts import pac_pole_attention_ucr16_fast as base

ROOT = Path(".omx/results/pac-projection-free-pole-attention-ucr16-fast-20260720")


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    paths = (
        Path("optimization/learned_two_tap_alphabet.py"),
        Path("optimization/learned_two_tap_pole_attention.py"),
        Path("optimization/masked_modal_moments.py"),
        Path("scripts/pac_pole_attention_ucr16_fast.py"),
        Path("scripts/pac_projection_free_pole_attention_ucr16_fast.py"),
    )
    return {str(path): hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths}


def _design() -> dict[str, object]:
    baseline = LagReaderALPHABET(1, base.MODEL_DIM, base.MODES, 5)
    candidate = ProjectionFreePoleAttentionALPHABET(
        1,
        base.MODEL_DIM,
        base.MODES,
        5,
        heads=base.HEADS,
    )
    baseline_parameters = count_parameters(baseline)
    candidate_parameters = count_parameters(candidate)
    return {
        "schema": "pac_projection_free_pole_attention_ucr16_fast_contract.v1",
        "purpose": "validation-only capacity-controlled replacement screen for lag-1/lag-4",
        "official_test_accessed": False,
        "datasets": list(base.DATASETS),
        "excluded_slow_datasets": ["FordA", "FordB"],
        "seeds": list(base.SEEDS),
        "variants": list(base.VARIANTS),
        "model_dim": base.MODEL_DIM,
        "modes": base.MODES,
        "attention_heads": base.HEADS,
        "epochs": base.EPOCHS,
        "batch_size": base.BATCH_SIZE,
        "learning_rate": base.LEARNING_RATE,
        "weight_decay": base.WEIGHT_DECAY,
        "grad_clip_norm": base.GRAD_CLIP_NORM,
        "hyperparameter_tuning": False,
        "controlled_difference": (
            "the learned-two-tap writer, reader local lift, exact-pole scan, writer moments, "
            "pooled real stream, D, M, classifier width, and unprojected normalized reader "
            "values are fixed; lag-1/lag-4 coherences are replaced only by two bounded "
            "per-mode Hermitian attention maps"
        ),
        "attention_contract": {
            "query": "last valid reader complex state",
            "similarity": (
                "FP32 per-mode normalized Hermitian products; all real and imaginary "
                "relative-phase coordinates feed every bounded head projection"
            ),
            "logits": (
                "bounded learned real/imaginary mixture, positive temperature, "
                "bounded relative-time bias"
            ),
            "values": (
                "the same RMS-normalized D-dimensional reader feature is convex-pooled "
                "by each head; no learned value projection"
            ),
            "value_projection": "absent",
            "all_padding": "zero descriptor",
            "complexity": "linear in sequence length",
            "reader_descriptor_width": "energy M plus H*D = 5M at D=32,M=16,H=2",
        },
        "parameter_control": {
            "lag_reader": baseline_parameters,
            "projection_free_attention": candidate_parameters,
            "candidate_minus_lag": candidate_parameters - baseline_parameters,
            "expected_candidate_minus_lag": 68,
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
            key=f"projection_free_pole_attention_ucr16_fast:{dataset}:{variant}:seed{seed}",
            dataset=dataset,
            variant=variant,
            split_seed=seed,
            train_seed=seed,
            model_dim=base.MODEL_DIM,
            modes=base.MODES,
            heads=base.HEADS,
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
    return ProjectionFreePoleAttentionALPHABET(
        input_dim,
        job.model_dim,
        job.modes,
        output_dim,
        heads=job.heads,
    )


def main() -> None:
    base.ROOT = ROOT
    base._design = _design
    base.design_sha256 = design_sha256
    base.jobs = jobs
    base._build_model = _build_model
    base.main()


if __name__ == "__main__":
    main()
