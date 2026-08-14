from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from .pac_metrics import count_parameters
from .pac_model import PACHybridPRLBlock
from .pac_overnight_io import write_csv_rows
from .pac_overnight_models import build_overnight_classifier

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from .pac_types import PACBranchName, PACExperimentConfig
    from .tapped_prl_followup_schema import JsonValue


def run_param_audit(
    output_root: Path,
    config: PACExperimentConfig,
    models: Iterable[str],
) -> str:
    rows: list[dict[str, JsonValue]] = []
    for model_name in models:
        model = build_overnight_classifier(model_name, config, class_count=3)
        active = _active_branches(model)
        rows.extend(_named_param_rows(model_name, model, active))
        rows.extend(_branch_total_rows(model_name, model, active))
        rows.append(_forward_norm_row(model_name, model, config))
    write_csv_rows(output_root / "results" / "param_count_audit.csv", rows)
    status = _param_status(rows)
    report_path = output_root / "reports" / "overnight_param_audit.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(f"# Parameter Audit\n\nparam_count_status: {status}\n", encoding="utf-8")
    return status


def _named_param_rows(
    model_name: str,
    model: nn.Module,
    active: tuple[str, ...],
) -> list[dict[str, JsonValue]]:
    rows: list[dict[str, JsonValue]] = []
    for name, parameter in model.named_parameters():
        rows.append(
            {
                "row_type": "parameter",
                "model": model_name,
                "variant": model_name,
                "branch_prl_enabled": "prl" in active,
                "branch_fir_enabled": "fir" in active,
                "branch_mlp_enabled": "mlp" in active,
                "damping_control_enabled": _damping_enabled(model),
                "parameter_name": name,
                "shape": "x".join(str(size) for size in parameter.shape),
                "numel": parameter.numel(),
                "requires_grad": parameter.requires_grad,
                "branch": _branch_from_name(name),
            }
        )
    return rows


def _branch_total_rows(
    model_name: str,
    model: nn.Module,
    active: tuple[str, ...],
) -> list[dict[str, JsonValue]]:
    totals: dict[str, int] = {}
    trainable: dict[str, int] = {}
    for name, parameter in model.named_parameters():
        branch = _branch_from_name(name)
        totals[branch] = totals.get(branch, 0) + parameter.numel()
        if parameter.requires_grad:
            trainable[branch] = trainable.get(branch, 0) + parameter.numel()
    return [
        {
            "row_type": "branch_total",
            "model": model_name,
            "variant": model_name,
            "branch": branch,
            "numel": total,
            "trainable_numel": trainable.get(branch, 0),
            "active_branch": branch in active,
            "total_params": sum(totals.values()),
            "trainable_params": count_parameters(model),
        }
        for branch, total in sorted(totals.items())
    ]


def _forward_norm_row(
    model_name: str,
    model: nn.Module,
    config: PACExperimentConfig,
) -> dict[str, JsonValue]:
    inputs = torch.randn(2, min(16, config.sequence_length), 1)
    encoder = _pac_encoder(model)
    if encoder is None:
        return {"row_type": "forward_norm", "model": model_name}
    with torch.no_grad():
        projected = encoder.input_projection(inputs)
        branch_outputs = encoder.branch_outputs(projected)
        fused = _fused_norm_tensor(encoder, projected, branch_outputs)
    return {
        "row_type": "forward_norm",
        "model": model_name,
        "mean_norm_prl_output": _active_norm(branch_outputs.get("prl")),
        "mean_norm_fir_output": _active_norm(branch_outputs.get("fir")),
        "mean_norm_mlp_output": _active_norm(branch_outputs.get("mlp")),
        "mean_norm_fused_output": float(fused.norm(dim=-1).mean().item()),
    }


def _fused_norm_tensor(
    encoder: PACHybridPRLBlock,
    projected: torch.Tensor,
    branch_outputs: dict[PACBranchName, torch.Tensor],
) -> torch.Tensor:
    fused = torch.zeros_like(projected)
    scale = encoder.branch_scales.to(device=projected.device, dtype=projected.dtype)
    for index, branch in enumerate(("prl", "fir", "mlp")):
        output = branch_outputs.get(branch)
        if output is not None:
            fused = fused + output * scale[index]
    return fused


def _active_norm(tensor: torch.Tensor | None) -> float | None:
    if tensor is None:
        return None
    return float(tensor.norm(dim=-1).mean().item())


def _active_branches(model: nn.Module) -> tuple[str, ...]:
    encoder = _pac_encoder(model)
    if encoder is None:
        return ()
    return tuple(encoder.active_branches)


def _pac_encoder(model: nn.Module) -> PACHybridPRLBlock | None:
    encoder = getattr(model, "encoder", None)
    return encoder if isinstance(encoder, PACHybridPRLBlock) else None


def _damping_enabled(model: nn.Module) -> bool:
    encoder = _pac_encoder(model)
    if encoder is None or encoder.prl_branch is None:
        return False
    return encoder.prl_branch.damping_control_range > 0.0


def _branch_from_name(name: str) -> str:
    branch_markers = (
        ("prl_branch", "prl"),
        ("damping_control", "damping_control"),
        ("fir_", "fir"),
        ("mlp_branch", "mlp"),
        ("classifier", "classifier"),
        ("input_projection", "input_projection"),
    )
    for marker, branch in branch_markers:
        if marker in name:
            return branch
    return "other"


def _param_status(rows: list[dict[str, JsonValue]]) -> str:
    full_active = _value(rows, "pac_full", "forward_norm", "mean_norm_mlp_output")
    lite_active = _value(rows, "pac_lite", "forward_norm", "mean_norm_mlp_output")
    full_trainable = _branch_value(rows, "pac_full", "mlp")
    lite_trainable = _branch_value(rows, "pac_lite", "mlp")
    if (
        full_active is not None
        and lite_active is None
        and full_trainable > 0
        and lite_trainable > 0
    ):
        return "logging_bug"
    if full_active is not None and lite_active is None:
        return "verified"
    return "architecture_bug"


def _value(
    rows: list[dict[str, JsonValue]], model: str, row_type: str, key: str
) -> JsonValue | None:
    for row in rows:
        if row.get("model") == model and row.get("row_type") == row_type:
            return row.get(key)
    return None


def _branch_value(rows: list[dict[str, JsonValue]], model: str, branch: str) -> int:
    for row in rows:
        if (
            row.get("model") == model
            and row.get("row_type") == "branch_total"
            and row.get("branch") == branch
        ):
            value = row.get("trainable_numel")
            return int(value) if isinstance(value, int) else 0
    return 0
