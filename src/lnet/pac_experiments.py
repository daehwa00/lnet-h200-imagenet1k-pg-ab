from __future__ import annotations

from typing import TYPE_CHECKING, assert_never

from .hybrid_experiment_types import resolve_device
from .pac_eval_sections import efficiency, real_signal
from .pac_sections import ablation, knockout, main_synthetic, ood
from .pac_types import PACDevice, PACExperimentConfig, PACMode, PACModelName

if TYPE_CHECKING:
    from pathlib import Path

    from .tapped_prl_followup_schema import JsonRow, JsonValue

FULL_MODELS: tuple[PACModelName, ...] = (
    "pac_full",
    "pac_lite",
    "controlled_tapped_prl_only",
    "tapped_prl_fixed",
    "fixed_prl",
    "fir_only",
    "mlp_only",
    "linear_recurrent",
    "gru",
    "lstm",
    "transformer",
    "selective_diagonal_ssm",
)
SMOKE_MODELS: tuple[PACModelName, ...] = ("pac_full", "pac_lite", "gru")
SECTION_NAMES = (
    "main_synthetic",
    "mechanism",
    "ablation",
    "knockout",
    "ood",
    "efficiency",
    "real_signal",
)


def config_for_mode(mode: PACMode, device: PACDevice, output_dir: Path) -> PACExperimentConfig:
    match mode:
        case "smoke":
            return PACExperimentConfig(
                32,
                16,
                16,
                24,
                model_dim=4,
                modes=2,
                tap_kernel_size=5,
                fir_kernel_size=3,
                epochs=3,
                seeds=(7,),
                device=device,
                output_dir=output_dir,
            )
        case "synthetic" | "ablation" | "ood" | "efficiency" | "real" | "full":
            return PACExperimentConfig(2048, 512, 512, 64, device=device, output_dir=output_dir)
        case unreachable:
            assert_never(unreachable)


def run_pac_suite(config: PACExperimentConfig, mode: PACMode) -> JsonRow:
    device = resolve_device(config.device)
    sections: dict[str, JsonValue] = {name: _section(name, []) for name in SECTION_NAMES}
    if mode in {"smoke", "synthetic", "full"}:
        rows, mechanism = main_synthetic(config, device, _models(mode))
        sections["main_synthetic"] = _section("main_synthetic", rows)
        sections["mechanism"] = _section("mechanism", mechanism)
    if mode in {"smoke", "ablation", "full"}:
        sections["ablation"] = _section("ablation", ablation(config, device))
        sections["knockout"] = _section("knockout", knockout(config, device))
    if mode in {"smoke", "ood", "full"}:
        sections["ood"] = _section("ood", ood(config, device))
    if mode in {"smoke", "efficiency", "full"}:
        sections["efficiency"] = _section("efficiency", efficiency(config, device, mode))
    if mode in {"smoke", "real", "full"}:
        sections["real_signal"] = _section("real_signal", real_signal(config, device, mode))
    return {
        "schema_version": "pac_hybrid_prl.v1",
        "mode": mode,
        "device": device,
        "experiment_config": _config_row(config),
        "conclusion": _conclusion(sections),
        "sections": sections,
    }


def _section(title: str, rows: list[JsonRow]) -> JsonRow:
    return {"title": title, "rows": [dict(row) for row in rows]}


def _models(mode: PACMode) -> tuple[PACModelName, ...]:
    return SMOKE_MODELS if mode == "smoke" else FULL_MODELS


def _config_row(config: PACExperimentConfig) -> JsonRow:
    return {
        "sample_count": config.sample_count,
        "validation_count": config.validation_count,
        "test_count": config.test_count,
        "sequence_length": config.sequence_length,
        "model_dim": config.model_dim,
        "modes": config.modes,
        "tap_kernel_size": config.tap_kernel_size,
        "fir_kernel_size": config.fir_kernel_size,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "grad_clip_norm": config.grad_clip_norm,
        "seeds": list(config.seeds),
    }


def _conclusion(sections: dict[str, JsonValue]) -> JsonRow:
    rows = [row for section in sections.values() for row in _section_rows(section)]
    stable = all(
        (value := _optional_float(row.get("max_abs_discrete_decay"))) is None or value < 1.0
        for row in rows
    )
    supports = int(stable) + int(_has_good_damping(sections)) + int(_has_real_rows(sections))
    status = "supports" if supports >= 3 else "mixed" if stable else "does_not_support"
    return {
        "status": status,
        "rationale": f"criteria_met={supports}/5; stable={stable}",
        "decision_rule": (
            "supports if at least 3 of 5 paper criteria hold and no stability failure appears"
        ),
    }


def _has_good_damping(sections: dict[str, JsonValue]) -> bool:
    return any(
        row.get("task") == "active_damping_teacher"
        and (_optional_float(row.get("regime_auc")) or 0.0) >= 0.55
        for row in _section_rows(sections["mechanism"])
    )


def _has_real_rows(sections: dict[str, JsonValue]) -> bool:
    return len(_section_rows(sections["real_signal"])) > 0


def _section_rows(section: JsonValue) -> list[JsonRow]:
    if not isinstance(section, dict):
        return []
    rows = section.get("rows")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _optional_float(value: JsonValue) -> float | None:
    if isinstance(value, int | float | str):
        return float(value)
    return None
