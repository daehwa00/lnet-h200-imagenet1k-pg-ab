from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from math import cos, log, pi, sin
from time import perf_counter
from typing import TYPE_CHECKING, Final, assert_never

if TYPE_CHECKING:
    from pathlib import Path

    from .benchmarks import BenchmarkSummary

from .advanced_experiments import train_regression_model
from .hybrid_experiment_types import (
    HybridExperimentConfig,
    make_task,
    resolve_device,
    training_config,
)
from .hybrid_experiments import run_hybrid_experiment_suite
from .hybrid_metrics import count_trainable_parameters
from .tapped_prl_experiment_lane_a import (
    LaneAExperimentConfig,
    branch_ablation_specs,
    fusion_variant_specs,
    instantiate_lane_a_model,
    k_sweep_specs,
    mode_sweep_specs,
    tap_parameterization_specs,
)
from .tapped_prl_experiment_reports import write_report_artifacts
from .tapped_prl_experiment_schema import (
    CheckpointSchemaError,
    ExperimentRow,
    FairnessMetadata,
    GateSelectionRecord,
    HypothesisCheck,
    ParameterMatchMetadata,
    StageCheckpoint,
    StageConfig,
    StageName,
    TeacherMetadata,
    WarningEntry,
    load_selection_candidates,
    write_checkpoint,
)

DEFAULT_STAGE3_RULE_ID: Final[str] = "default_stage3_gate_v1"
FULL_STAGE3_TRAINED_GATE: Final[str] = "softmax"
DEFAULT_GATE_SELECTION_TIE_BREAKS: Final[tuple[str, ...]] = (
    "relative_param_error",
    "elapsed",
    "gate_precedence",
)
GATE_PRECEDENCE: Final[tuple[str, ...]] = (
    "softmax",
    "temperature_softmax_tau_0_5",
    "temperature_softmax_tau_2_0",
    "fixed_learned_scalar",
    "fixed_equal",
    "no_gate_sum",
)


def run_stage(
    stage: StageName,
    output_dir: Path,
    *,
    smoke: bool,
    gate_choice: str | None = None,
    gate_selection_checkpoint: Path | None = None,
) -> StageCheckpoint:
    checkpoint = (
        _build_smoke_checkpoint(
            stage,
            gate_choice=gate_choice,
            gate_selection_checkpoint=gate_selection_checkpoint,
        )
        if smoke
        else _build_full_checkpoint(
            stage,
            gate_choice=gate_choice,
            gate_selection_checkpoint=gate_selection_checkpoint,
        )
    )
    path = output_dir / f"tapped-prl-{stage}-results.json"
    written_checkpoint = _with_artifact(checkpoint, str(path))
    write_checkpoint(path, written_checkpoint)
    return written_checkpoint


def run_all(output_dir: Path, *, smoke: bool) -> tuple[StageCheckpoint, ...]:
    if smoke:
        stage1 = run_stage("stage1", output_dir=output_dir, smoke=True)
        stage2 = run_stage("stage2", output_dir=output_dir, smoke=True)
        stage1_path = output_dir / "tapped-prl-stage1-results.json"
        stage3 = run_stage(
            "stage3",
            output_dir=output_dir,
            smoke=True,
            gate_selection_checkpoint=stage1_path,
        )
    else:
        stage1, stage2, stage3 = _build_full_checkpoints()
        stage1_path = output_dir / "tapped-prl-stage1-results.json"
        stage2_path = output_dir / "tapped-prl-stage2-results.json"
        stage1 = _with_artifact(stage1, str(stage1_path))
        stage2 = _with_artifact(stage2, str(stage2_path))
        write_checkpoint(stage1_path, stage1)
        write_checkpoint(stage2_path, stage2)
        stage3 = _with_gate_selection(
            stage3,
            selected_gate=FULL_STAGE3_TRAINED_GATE,
            selection_paths=(str(stage1_path),),
            selection_row_ids=_stage1_rows_for_gate(stage1, FULL_STAGE3_TRAINED_GATE),
            rule_id="actual_trained_gate",
            override_reason=(
                "Full stage3 preserves the gate variant used during learned evaluation."
            ),
        )
    checkpoints = (stage1, stage2, stage3)
    report_artifacts = write_report_artifacts(output_dir, checkpoints)
    stage3_path = output_dir / "tapped-prl-stage3-results.json"
    written_stage3 = _with_artifacts(stage3, (str(stage3_path), *report_artifacts))
    write_checkpoint(stage3_path, written_stage3)
    return (stage1, stage2, written_stage3)


def _build_smoke_checkpoint(
    stage: StageName,
    *,
    gate_choice: str | None,
    gate_selection_checkpoint: Path | None,
) -> StageCheckpoint:
    match stage:
        case "stage1":
            return _build_stage1_checkpoint(smoke=True)
        case "stage2":
            return _build_stage2_checkpoint(smoke=True)
        case "stage3":
            return _build_stage3_checkpoint(
                smoke=True,
                gate_choice=gate_choice,
                gate_selection_checkpoint=gate_selection_checkpoint,
            )
        case unreachable:
            assert_never(unreachable)


def _build_full_checkpoint(
    stage: StageName,
    *,
    gate_choice: str | None,
    gate_selection_checkpoint: Path | None,
) -> StageCheckpoint:
    match stage:
        case "stage1":
            stage1, _, _ = _build_full_checkpoints()
            return stage1
        case "stage2":
            _, stage2, _ = _build_full_checkpoints()
            return stage2
        case "stage3":
            if gate_choice is None and gate_selection_checkpoint is None:
                message = "stage3 requires --gate-choice or --gate-selection-checkpoint"
                raise CheckpointSchemaError(message)
            selected_gate, selection_paths, selection_row_ids, rule_id, override_reason = (
                _resolve_gate_selection(gate_choice, gate_selection_checkpoint)
            )
            _require_full_stage3_gate(selected_gate)
            _, _, stage3 = _build_full_checkpoints()
            return _with_gate_selection(
                stage3,
                selected_gate=selected_gate,
                selection_paths=selection_paths,
                selection_row_ids=selection_row_ids,
                rule_id=rule_id,
                override_reason=override_reason,
            )
        case unreachable:
            assert_never(unreachable)


def _build_full_checkpoints() -> tuple[StageCheckpoint, StageCheckpoint, StageCheckpoint]:
    config = _full_experiment_config()
    stage1 = _build_full_stage1_checkpoint(config)
    summary_started_at = perf_counter()
    summary = run_hybrid_experiment_suite(config)
    summary_elapsed = perf_counter() - summary_started_at
    amortized_elapsed = _summary_amortized_elapsed(summary, summary_elapsed)
    return (
        stage1,
        _build_full_stage2_checkpoint(config, summary, amortized_elapsed),
        _build_full_stage3_checkpoint(config, summary, amortized_elapsed),
    )


def _build_full_stage1_checkpoint(config: HybridExperimentConfig) -> StageCheckpoint:
    stage_config = _default_config(
        stage="stage1",
        smoke=False,
        comparison_groups=(
            "stage1/branch_ablation/modal_teacher/components",
            "stage1/k_sweep/modal_teacher/k_values",
            "stage1/m_sweep/modal_teacher/mode_values",
            "stage1/tap_parameterization/modal_teacher/tap_values",
            "stage1/gate_variant/modal_teacher/baselines",
        ),
    )
    lane_config = LaneAExperimentConfig(
        raw_input_dim=config.raw_input_dim,
        output_dim=config.output_dim,
        model_dim=config.model_dim,
        modes=config.modes,
        fir_kernel_size=config.fir_kernel_size,
        prl_tap_kernel_size=config.prl_tap_kernel_size or config.fir_kernel_size,
        prl_tap_kernel_sizes=(1, 2, 4, 8),
        mode_values=(1, 2, 4, 6),
        branch_sets=config.branch_sets,
        tap_parameterizations=("shared_scalar", "tap_specific_reader", "normalized_taps"),
    )
    task = make_task("modal", config)
    training = training_config(config, resolve_device(config.device))
    specs = (
        *branch_ablation_specs(lane_config, teacher_label=task.label),
        *k_sweep_specs(lane_config, teacher_label=task.label),
        *mode_sweep_specs(lane_config, teacher_label=task.label),
        *tap_parameterization_specs(lane_config, teacher_label=task.label),
        *fusion_variant_specs(lane_config, teacher_label=task.label),
    )
    rows: list[ExperimentRow] = []
    for index, spec in enumerate(specs):
        model = instantiate_lane_a_model(spec, lane_config)
        started_at = perf_counter()
        outcome = train_regression_model(model, task.task, training)
        comparison_group = spec.comparison_group
        gate_selection_scope = spec.gate_selection_scope
        if spec.comparison_type == "gate_variant":
            comparison_group = "stage1/gate_variant/modal_teacher/baselines"
            gate_selection_scope = "gatectx/modal_teacher/baselines"
        rows.append(
            _make_row(
                stage_config,
                f"stage1-full-{index}-{spec.comparison_type}-{spec.model_label}",
                spec.comparison_type,
                comparison_group,
                gate_selection_scope,
                spec.model_label,
                spec.gate_variant_label,
                count_trainable_parameters(model),
                _finite_loss(outcome.validation_loss),
                perf_counter() - started_at,
                spec.isolated_vs_joint,
                spec.tap_parameterization,
                spec.prl_tap_kernel_size,
                spec.fir_kernel_size,
                spec.mode_count,
                teacher_label=task.label,
                metadata_status="proxy_only",
            ),
        )
    return StageCheckpoint(
        schema_version="1.0",
        stage="stage1",
        created_at=_timestamp(),
        config=stage_config,
        rows=tuple(rows),
        hypothesis_checks=(
            HypothesisCheck(
                "stage1_actual_lane_a_execution",
                "supports",
                tuple(row.row_id for row in rows[: min(5, len(rows))]),
                "Full mode trains Lane A model specs instead of serializing fixed scaffold rows.",
                tuple(sorted({row.comparison_group for row in rows})),
            ),
        ),
        artifacts=(),
        warnings=(),
        gate_selection_by_comparison_group=(),
    )


def _build_full_stage2_checkpoint(
    config: HybridExperimentConfig,
    summary: BenchmarkSummary,
    amortized_elapsed: float,
) -> StageCheckpoint:
    stage_config = _default_config(
        stage="stage2",
        smoke=False,
        comparison_groups=(
            "stage2/strict_delay/delay_teacher/family",
            "stage2/delayed_modal/delayed_modal_teacher/family",
        ),
    )
    rows: list[ExperimentRow] = []
    for index, row in enumerate(_section_rows(summary, "Strict Delay Family")):
        delay_steps = int(row[0])
        kernel_size = int(row[1])
        rows.append(
            _make_row(
                stage_config,
                f"stage2-full-strict-delay-{index}",
                "strict_delay",
                f"stage2/strict_delay/delay_teacher/delay_T_{delay_steps}",
                f"gatectx/delay_teacher/delay_T_{delay_steps}",
                "hybrid_prl_fir_mlp",
                "softmax",
                int(row[2]),
                float(row[3]),
                amortized_elapsed,
                "joint",
                "shared_scalar",
                kernel_size,
                kernel_size,
                config.modes,
                teacher_label=f"strict_delay_{delay_steps}",
                true_delay=delay_steps,
                metadata_status=row[5],
            ),
        )
    delayed_params = _full_hybrid_param_count(summary)
    for index, row in enumerate(_section_rows(summary, "Delayed Modal Teacher")):
        teacher_label = row[0]
        true_delay = int(row[1])
        teacher_metadata = _delayed_modal_teacher_metadata(
            teacher_label=teacher_label,
            true_delay=true_delay,
            target_horizon=config.sequence_length,
        )
        tap_peak_index = _optional_int_string(row[2])
        tap_mass_near_true_delay = _optional_float_string(row[3])
        tap_peak_error = _optional_int_string(row[4])
        mean_pole_error = _optional_float_string(row[5])
        max_pole_error = _optional_float_string(row[6])
        rows.extend(
            (
                _make_row(
                    stage_config,
                    f"stage2-full-delayed-modal-{index}",
                    "delayed_modal",
                    f"stage2/delayed_modal/{teacher_label}/pole_tap",
                    f"gatectx/{teacher_label}/pole_tap",
                    "hybrid_prl_fir_mlp",
                    "softmax",
                    delayed_params,
                    float(row[7]),
                    amortized_elapsed,
                    "joint",
                    "shared_scalar",
                    config.prl_tap_kernel_size or config.fir_kernel_size,
                    config.fir_kernel_size,
                    config.modes,
                    teacher_metadata=teacher_metadata,
                    teacher_label=teacher_label,
                    true_delay=true_delay,
                    metadata_status="full_ground_truth",
                    tap_peak_index=tap_peak_index,
                    tap_mass_near_true_delay=tap_mass_near_true_delay,
                    tap_peak_error=tap_peak_error,
                    mean_pole_error=mean_pole_error,
                    max_pole_error=max_pole_error,
                ),
                _make_row(
                    stage_config,
                    f"stage2-full-interpretability-{index}",
                    "interpretability",
                    f"stage2/interpretability/{teacher_label}/pole_tap",
                    f"gatectx/{teacher_label}/pole_tap",
                    "pole_tap_diagnostic",
                    "softmax",
                    delayed_params,
                    float(row[7]),
                    amortized_elapsed,
                    "joint",
                    "shared_scalar",
                    config.prl_tap_kernel_size or config.fir_kernel_size,
                    config.fir_kernel_size,
                    config.modes,
                    teacher_metadata=teacher_metadata,
                    teacher_label=teacher_label,
                    true_delay=true_delay,
                    metadata_status="full_ground_truth",
                    tap_peak_index=tap_peak_index,
                    tap_mass_near_true_delay=tap_mass_near_true_delay,
                    tap_peak_error=tap_peak_error,
                    mean_pole_error=mean_pole_error,
                    max_pole_error=max_pole_error,
                ),
            ),
        )
    return StageCheckpoint(
        schema_version="1.0",
        stage="stage2",
        created_at=_timestamp(),
        config=stage_config,
        rows=tuple(rows),
        hypothesis_checks=(
            HypothesisCheck(
                "stage2_actual_delay_and_modal_execution",
                "supports",
                tuple(row.row_id for row in rows[: min(5, len(rows))]),
                "Full mode records strict-delay and delayed-modal rows from learned runs.",
                tuple(sorted({row.comparison_group for row in rows})),
            ),
        ),
        artifacts=(),
        warnings=_amortized_elapsed_warnings(tuple(row.row_id for row in rows), "stage2"),
        gate_selection_by_comparison_group=(),
    )


def _build_full_stage3_checkpoint(
    config: HybridExperimentConfig,
    summary: BenchmarkSummary,
    amortized_elapsed: float,
) -> StageCheckpoint:
    stage_config = _default_config(
        stage="stage3",
        smoke=False,
        comparison_groups=("stage3/parameter_matched/all/baselines",),
    )
    source_rows = _section_rows(summary, "Parameter-Matched Baselines")
    target_params_by_teacher = {
        row[0]: int(row[2]) for row in source_rows if row[1] == "Hybrid Modal PRL"
    }
    rows = tuple(
        _make_row(
            stage_config,
            f"stage3-full-parameter-match-{index}",
            "parameter_matched",
            f"stage3/parameter_matched/{row[0]}/baselines",
            "gatectx/modal_teacher/baselines",
            row[1],
            "softmax",
            int(row[2]),
            float(row[3]),
            amortized_elapsed,
            "joint",
            "shared_scalar",
            config.prl_tap_kernel_size or config.fir_kernel_size,
            config.fir_kernel_size,
            config.modes,
            teacher_label=row[0],
            metadata_status="proxy_only",
            target_params=target_params_by_teacher.get(row[0]),
            relative_param_error=_relative_param_error(
                int(row[2]),
                target_params_by_teacher.get(row[0]),
            ),
            matched_param_candidate=row[1],
        )
        for index, row in enumerate(source_rows)
    )
    return StageCheckpoint(
        schema_version="1.0",
        stage="stage3",
        created_at=_timestamp(),
        config=stage_config,
        rows=rows,
        hypothesis_checks=(
            HypothesisCheck(
                "stage3_actual_parameter_matched_execution",
                "supports",
                tuple(row.row_id for row in rows[: min(5, len(rows))]),
                "Full mode records parameter-matched baseline rows from learned runs.",
                tuple(sorted({row.comparison_group for row in rows})),
            ),
        ),
        artifacts=(),
        warnings=_amortized_elapsed_warnings(tuple(row.row_id for row in rows), "stage3"),
        gate_selection_by_comparison_group=(),
    )


def _full_experiment_config() -> HybridExperimentConfig:
    return HybridExperimentConfig(
        epochs=20,
        device="auto",
        run_parameter_match=True,
        run_delay_sweep=True,
        run_gate_diagnostics=True,
    )


def _section_rows(summary: BenchmarkSummary, title: str) -> tuple[tuple[str, ...], ...]:
    for section in summary.sections:
        if section.title == title:
            return tuple(tuple(str(cell) for cell in row) for row in section.rows)
    message = f"missing benchmark section: {title}"
    raise CheckpointSchemaError(message)


def _full_hybrid_param_count(summary: BenchmarkSummary) -> int:
    for row in _section_rows(summary, "Parameter-Matched Baselines"):
        if row[1] == "Hybrid Modal PRL":
            return int(row[2])
    message = "missing Hybrid Modal PRL parameter row"
    raise CheckpointSchemaError(message)


def _summary_amortized_elapsed(summary: BenchmarkSummary, summary_elapsed: float) -> float:
    row_count = (
        len(_section_rows(summary, "Strict Delay Family"))
        + (2 * len(_section_rows(summary, "Delayed Modal Teacher")))
        + len(_section_rows(summary, "Parameter-Matched Baselines"))
    )
    return summary_elapsed / max(row_count, 1)


def _stage1_rows_for_gate(checkpoint: StageCheckpoint, gate_variant: str) -> tuple[str, ...]:
    return tuple(
        row.row_id
        for row in checkpoint.rows
        if row.comparison_type == "gate_variant" and row.gate_variant == gate_variant
    )


def _require_full_stage3_gate(selected_gate: str) -> None:
    if selected_gate == FULL_STAGE3_TRAINED_GATE:
        return
    message = (
        "stage3 full checkpoints can only use the gate variant that was actually trained "
        f"({FULL_STAGE3_TRAINED_GATE}); rerun full evaluation with a different training "
        "gate before requesting that label"
    )
    raise CheckpointSchemaError(message)


def _delayed_modal_teacher_metadata(
    *,
    teacher_label: str,
    true_delay: int,
    target_horizon: int,
) -> TeacherMetadata:
    if teacher_label in {"delayed_exp_smoke", "delayed_exponential_4"}:
        discrete_pole = 0.85
        return TeacherMetadata(
            teacher_kind=teacher_label,
            true_delay=true_delay,
            discrete_pole_real=discrete_pole,
            discrete_pole_imag=0.0,
            continuous_pole_real=log(discrete_pole),
            continuous_pole_imag=0.0,
            damping_radius=discrete_pole,
            angular_frequency=0.0,
            target_horizon=target_horizon,
            metadata_status="full_ground_truth",
        )
    if teacher_label == "delayed_oscillatory_4":
        radius = 0.90
        omega = pi / 4.0
        return TeacherMetadata(
            teacher_kind=teacher_label,
            true_delay=true_delay,
            discrete_pole_real=radius * cos(omega),
            discrete_pole_imag=radius * sin(omega),
            continuous_pole_real=log(radius),
            continuous_pole_imag=omega,
            damping_radius=radius,
            angular_frequency=omega,
            target_horizon=target_horizon,
            metadata_status="full_ground_truth",
        )
    message = f"unknown delayed-modal teacher label: {teacher_label}"
    raise CheckpointSchemaError(message)


def _finite_loss(value: float | None) -> float:
    if value is None:
        return float("inf")
    return value


def _relative_param_error(params: int, target_params: int | None) -> float | None:
    if target_params is None:
        return None
    return abs(params - target_params) / max(target_params, 1)


def _with_gate_selection(
    checkpoint: StageCheckpoint,
    *,
    selected_gate: str,
    selection_paths: tuple[str, ...],
    selection_row_ids: tuple[str, ...],
    rule_id: str,
    override_reason: str | None,
) -> StageCheckpoint:
    rows = tuple(replace(row, gate_variant=selected_gate) for row in checkpoint.rows)
    comparison_groups = tuple(sorted({row.comparison_group for row in rows}))
    selection_metric, tie_breaks = _gate_selection_metadata(rule_id)
    records = tuple(
        GateSelectionRecord(
            comparison_group=comparison_group,
            gate_selection_scope="gatectx/modal_teacher/baselines",
            selected_gate_variant=selected_gate,
            selection_rule_id=rule_id,
            selection_checkpoint_paths=selection_paths,
            selection_source_row_ids=selection_row_ids,
            selection_metric=selection_metric,
            tie_breaks=tie_breaks,
            override_reason=override_reason,
        )
        for comparison_group in comparison_groups
    )
    return replace(checkpoint, rows=rows, gate_selection_by_comparison_group=records)


def _gate_selection_metadata(rule_id: str) -> tuple[str, tuple[str, ...]]:
    if rule_id == DEFAULT_STAGE3_RULE_ID:
        return "validation_loss", DEFAULT_GATE_SELECTION_TIE_BREAKS
    if rule_id == "actual_trained_gate":
        return "actual_trained_gate", ()
    if rule_id == "explicit_gate_choice":
        return "explicit_user_choice", ()
    return "unspecified", ()


def _build_stage1_checkpoint(*, smoke: bool) -> StageCheckpoint:
    config = _default_config(
        stage="stage1",
        smoke=smoke,
        comparison_groups=(
            "stage1/branch_ablation/modal_teacher/components",
            "stage1/k_sweep/modal_teacher/k_values",
            "stage1/m_sweep/modal_teacher/mode_values",
            "stage1/tap_parameterization/modal_teacher/tap_values",
            "stage1/gate_variant/modal_teacher/baselines",
        ),
    )
    rows = (
        _make_row(
            config,
            "branch-ablation-1",
            "branch_ablation",
            "stage1/branch_ablation/modal_teacher/components",
            "gatectx/modal_teacher/components",
            "hybrid_tapped_prl_fir_mlp",
            "no_gate_sum",
            128,
            0.210,
            0.310,
            "joint",
            "dense",
            8,
            3,
            4,
        ),
        _make_row(
            config,
            "k-sweep-1",
            "k_sweep",
            "stage1/k_sweep/modal_teacher/k_values",
            "gatectx/modal_teacher/k_values",
            "tapped_prl",
            "no_gate_sum",
            96,
            0.180,
            0.280,
            "isolated",
            "dense",
            8,
            3,
            4,
        ),
        _make_row(
            config,
            "m-sweep-1",
            "m_sweep",
            "stage1/m_sweep/modal_teacher/mode_values",
            "gatectx/modal_teacher/mode_values",
            "tapped_prl",
            "no_gate_sum",
            88,
            0.175,
            0.270,
            "isolated",
            "dense",
            4,
            3,
            6,
        ),
        _make_row(
            config,
            "tap-param-1",
            "tap_parameterization",
            "stage1/tap_parameterization/modal_teacher/tap_values",
            "gatectx/modal_teacher/tap_values",
            "tapped_prl",
            "fixed_equal",
            88,
            0.165,
            0.260,
            "isolated",
            "learned",
            8,
            3,
            4,
        ),
        _make_row(
            config,
            "gate-softmax-1",
            "gate_variant",
            "stage1/gate_variant/modal_teacher/baselines",
            "gatectx/modal_teacher/baselines",
            "hybrid_tapped_prl_fir_mlp",
            "softmax",
            132,
            0.140,
            0.250,
            "joint",
            "dense",
            8,
            3,
            4,
        ),
        _make_row(
            config,
            "gate-sum-1",
            "gate_variant",
            "stage1/gate_variant/modal_teacher/baselines",
            "gatectx/modal_teacher/baselines",
            "hybrid_tapped_prl_fir_mlp",
            "no_gate_sum",
            132,
            0.190,
            0.240,
            "joint",
            "dense",
            8,
            3,
            4,
        ),
    )
    return StageCheckpoint(
        schema_version="1.0",
        stage="stage1",
        created_at=_timestamp(),
        config=config,
        rows=rows,
        hypothesis_checks=(
            HypothesisCheck(
                "tapped_prl_delay_advantage",
                "supports",
                ("k-sweep-1", "tap-param-1"),
                "Smoke rows preserve isolated tap-length evidence without FIR confounding.",
                ("stage1/k_sweep/modal_teacher/k_values",),
            ),
        ),
        artifacts=(),
        warnings=_scaffold_warnings(
            (
                "branch-ablation-1",
                "k-sweep-1",
                "m-sweep-1",
                "tap-param-1",
                "gate-softmax-1",
                "gate-sum-1",
            ),
            "stage1",
        ),
        gate_selection_by_comparison_group=(),
    )


def _build_stage2_checkpoint(*, smoke: bool) -> StageCheckpoint:
    config = _default_config(
        stage="stage2",
        smoke=smoke,
        comparison_groups=(
            "stage2/strict_delay/delay_teacher/delay_T_4",
            "stage2/delayed_modal/delayed_exp_smoke/delayed_modal",
            "stage2/interpretability/delayed_exp_smoke/pole_tap",
        ),
    )
    rows = (
        _make_row(
            config,
            "strict-delay-1",
            "strict_delay",
            "stage2/strict_delay/delay_teacher/delay_T_4",
            "gatectx/delay_teacher/delay_T_4",
            "tapped_prl",
            "softmax",
            110,
            0.155,
            0.260,
            "isolated",
            "dense",
            8,
            3,
            4,
            teacher_label="delay_teacher",
            true_delay=4,
            metadata_status="full_ground_truth",
        ),
        _make_row(
            config,
            "delayed-modal-1",
            "delayed_modal",
            "stage2/delayed_modal/delayed_exp_smoke/delayed_modal",
            "gatectx/delayed_exp_smoke/modal",
            "tapped_prl",
            "softmax",
            114,
            0.145,
            0.265,
            "isolated",
            "dense",
            8,
            3,
            4,
            teacher_label="delayed_exp_smoke",
            true_delay=4,
            metadata_status="full_ground_truth",
        ),
        _make_row(
            config,
            "interpretability-1",
            "interpretability",
            "stage2/interpretability/delayed_exp_smoke/pole_tap",
            "gatectx/delayed_exp_smoke/pole_tap",
            "tapped_prl",
            "softmax",
            114,
            0.149,
            0.266,
            "isolated",
            "dense",
            8,
            3,
            4,
            teacher_label="delayed_exp_smoke",
            true_delay=4,
            metadata_status="full_ground_truth",
        ),
    )
    return StageCheckpoint(
        schema_version="1.0",
        stage="stage2",
        created_at=_timestamp(),
        config=config,
        rows=rows,
        hypothesis_checks=(
            HypothesisCheck(
                "delayed_modal_recovery",
                "mixed",
                ("delayed-modal-1", "interpretability-1"),
                "The smoke surface carries teacher delay and pole metadata into diagnostic rows.",
                ("stage2/delayed_modal/delayed_exp_smoke/delayed_modal",),
            ),
        ),
        artifacts=(),
        warnings=_scaffold_warnings(
            (
                "strict-delay-1",
                "delayed-modal-1",
                "interpretability-1",
            ),
            "stage2",
        ),
        gate_selection_by_comparison_group=(),
    )


def _build_stage3_checkpoint(
    *, smoke: bool, gate_choice: str | None, gate_selection_checkpoint: Path | None
) -> StageCheckpoint:
    if gate_choice is None and gate_selection_checkpoint is None:
        message = "stage3 requires --gate-choice or --gate-selection-checkpoint"
        raise CheckpointSchemaError(message)
    selected_gate, selection_paths, selection_row_ids, rule_id, override_reason = (
        _resolve_gate_selection(gate_choice, gate_selection_checkpoint)
    )
    config = _default_config(
        stage="stage3",
        smoke=smoke,
        comparison_groups=("stage3/parameter_matched/modal_teacher/baselines",),
    )
    rows = (
        _make_row(
            config,
            "parameter-match-1",
            "parameter_matched",
            "stage3/parameter_matched/modal_teacher/baselines",
            "gatectx/modal_teacher/baselines",
            "tapped_prl",
            selected_gate,
            132,
            0.138,
            0.290,
            "joint",
            "dense",
            8,
            3,
            4,
            target_params=132,
            relative_param_error=0.0,
            matched_param_candidate="reference_tapped_prl",
        ),
        _make_row(
            config,
            "parameter-match-2",
            "parameter_matched",
            "stage3/parameter_matched/modal_teacher/baselines",
            "gatectx/modal_teacher/baselines",
            "causal_transformer",
            selected_gate,
            126,
            0.142,
            0.315,
            "joint",
            "dense",
            8,
            3,
            4,
            target_params=132,
            relative_param_error=0.045,
            matched_param_candidate="transformer_dim_6",
        ),
    )
    selection_metric, tie_breaks = _gate_selection_metadata(rule_id)
    return StageCheckpoint(
        schema_version="1.0",
        stage="stage3",
        created_at=_timestamp(),
        config=config,
        rows=rows,
        hypothesis_checks=(
            HypothesisCheck(
                "parameter_efficiency_against_baselines",
                "supports",
                ("parameter-match-1", "parameter-match-2"),
                (
                    "The staged checkpoint records matched-parameter comparison rows "
                    "and elapsed-time evidence."
                ),
                ("stage3/parameter_matched/modal_teacher/baselines",),
            ),
        ),
        artifacts=(),
        warnings=_scaffold_warnings(("parameter-match-1", "parameter-match-2"), "stage3"),
        gate_selection_by_comparison_group=(
            GateSelectionRecord(
                comparison_group="stage3/parameter_matched/modal_teacher/baselines",
                gate_selection_scope="gatectx/modal_teacher/baselines",
                selected_gate_variant=selected_gate,
                selection_rule_id=rule_id,
                selection_checkpoint_paths=selection_paths,
                selection_source_row_ids=selection_row_ids,
                selection_metric=selection_metric,
                tie_breaks=tie_breaks,
                override_reason=override_reason,
            ),
        ),
    )


def _resolve_gate_selection(
    gate_choice: str | None, gate_selection_checkpoint: Path | None
) -> tuple[str, tuple[str, ...], tuple[str, ...], str, str | None]:
    if gate_choice is not None:
        return gate_choice, (), (), "explicit_gate_choice", "user provided gate choice"
    if gate_selection_checkpoint is None:
        message = "stage3 gate selection resolution requires a source"
        raise CheckpointSchemaError(message)
    source_stage, candidates = load_selection_candidates(gate_selection_checkpoint)
    if source_stage != "stage1":
        message = "gate selection checkpoints must come from stage1"
        raise CheckpointSchemaError(message)
    admissible = tuple(
        candidate
        for candidate in candidates
        if candidate.comparison_type == "gate_variant"
        and candidate.gate_selection_scope == "gatectx/modal_teacher/baselines"
        and candidate.availability_status == "available"
        and candidate.fairness_exception is None
        and candidate.gate_variant in GATE_PRECEDENCE
    )
    if not admissible:
        message = "no admissible stage1 gate rows were available for stage3"
        raise CheckpointSchemaError(message)
    ordered = sorted(
        admissible,
        key=lambda candidate: (
            candidate.validation_loss,
            candidate.relative_param_error if candidate.relative_param_error is not None else 1.0,
            candidate.elapsed,
            GATE_PRECEDENCE.index(candidate.gate_variant),
        ),
    )
    best = ordered[0]
    return (
        best.gate_variant,
        (str(gate_selection_checkpoint),),
        (best.row_id,),
        DEFAULT_STAGE3_RULE_ID,
        None,
    )


def _default_config(
    *, stage: StageName, smoke: bool, comparison_groups: tuple[str, ...]
) -> StageConfig:
    return StageConfig(
        stage=stage,
        mode="smoke" if smoke else "full",
        smoke=smoke,
        seeds=(7,),
        split_id="smoke-split" if smoke else "full-split",
        train_size=16 if smoke else 128,
        validation_size=8 if smoke else 32,
        sequence_length=8 if smoke else 40,
        batch_size=16 if smoke else 128,
        optimizer_family="adamw",
        learning_rate=5.0e-2,
        epoch_budget=1 if smoke else 20,
        early_stopping_rule="min_validation_loss",
        device="cpu",
        comparison_groups=comparison_groups,
    )


def _make_row(
    config: StageConfig,
    row_id: str,
    comparison_type: str,
    comparison_group: str,
    gate_selection_scope: str,
    model_label: str,
    gate_variant: str,
    params: int,
    validation_loss: float,
    elapsed: float,
    isolated_vs_joint: str,
    tap_parameterization: str,
    prl_tap_kernel_size: int,
    fir_kernel_size: int,
    mode_count: int,
    *,
    teacher_label: str = "modal_teacher",
    true_delay: int | None = None,
    metadata_status: str = "proxy_only",
    teacher_metadata: TeacherMetadata | None = None,
    target_params: int | None = None,
    relative_param_error: float | None = None,
    matched_param_candidate: str | None = None,
    tap_peak_index: int | None = None,
    tap_mass_near_true_delay: float | None = None,
    tap_peak_error: int | None = None,
    mean_pole_error: float | None = None,
    max_pole_error: float | None = None,
) -> ExperimentRow:
    teacher = teacher_metadata or TeacherMetadata(
        teacher_kind=teacher_label,
        true_delay=true_delay,
        discrete_pole_real=None,
        discrete_pole_imag=None,
        continuous_pole_real=None,
        continuous_pole_imag=None,
        damping_radius=None,
        angular_frequency=None,
        target_horizon=prl_tap_kernel_size,
        metadata_status=metadata_status,
    )
    fairness = FairnessMetadata(
        split_id=config.split_id,
        seed=config.seeds[0],
        optimizer_family=config.optimizer_family,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        epoch_budget=config.epoch_budget,
        early_stopping_rule=config.early_stopping_rule,
        shared_seed_group=f"{config.stage}-seed-group",
        shared_split_group=f"{config.stage}-split-group",
        fairness_exception=None,
    )
    parameter_match = ParameterMatchMetadata(
        target_model_label="tapped_prl" if target_params is not None else None,
        target_params=target_params,
        selected_params=params if target_params is not None else None,
        relative_param_error=relative_param_error,
        candidate_grid_id="baseline_grid_v1" if target_params is not None else None,
        selected_candidate=matched_param_candidate,
        target_context_horizon=prl_tap_kernel_size if target_params is not None else None,
        selected_context_horizon=prl_tap_kernel_size if target_params is not None else None,
        parameter_tolerance=0.2 if target_params is not None else None,
        parameter_constraint_satisfied=(relative_param_error or 0.0) <= 0.2
        if target_params is not None
        else None,
        horizon_constraint_satisfied=True if target_params is not None else None,
        mismatch_reason=None,
    )
    return ExperimentRow(
        row_id=row_id,
        stage=config.stage,
        comparison_group=comparison_group,
        gate_selection_scope=gate_selection_scope,
        model_label=model_label,
        teacher_label=teacher_label,
        teacher_metadata=teacher,
        comparison_type=comparison_type,
        isolated_vs_joint=isolated_vs_joint,
        gate_variant=gate_variant,
        tap_parameterization=tap_parameterization,
        prl_tap_kernel_size=prl_tap_kernel_size,
        fir_kernel_size=fir_kernel_size,
        mode_count=mode_count,
        params=params,
        target_params=target_params,
        relative_param_error=relative_param_error,
        matched_param_candidate=matched_param_candidate,
        context_horizon=prl_tap_kernel_size,
        target_context_horizon=prl_tap_kernel_size if target_params is not None else None,
        validation_loss=validation_loss,
        elapsed=elapsed,
        seed=config.seeds[0],
        split_id=config.split_id,
        optimizer_family=config.optimizer_family,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        epoch_budget=config.epoch_budget,
        early_stopping_rule=config.early_stopping_rule,
        availability_status="available",
        unavailable_reason=None,
        fairness_exception=None,
        fairness_metadata=fairness,
        parameter_match_metadata=parameter_match,
        tap_peak_index=tap_peak_index,
        tap_mass_near_true_delay=tap_mass_near_true_delay,
        tap_peak_error=tap_peak_error,
        mean_pole_error=mean_pole_error,
        max_pole_error=max_pole_error,
    )


def _optional_int_string(value: str) -> int | None:
    if value == "n/a":
        return None
    return int(value)


def _optional_float_string(value: str) -> float | None:
    if value == "n/a":
        return None
    return float(value)


def _scaffold_warnings(row_ids: tuple[str, ...], stage: StageName) -> tuple[WarningEntry, ...]:
    return tuple(
        WarningEntry(
            warning_id=f"{stage}_{row_id}_scaffold_proxy",
            warning_severity="warning",
            message=(
                "Staged checkpoint row is deterministic scaffold/proxy evidence; "
                "use hybrid-experiment-report.md for learned validation results."
            ),
            row_id=row_id,
            comparison_group=None,
        )
        for row_id in row_ids
    )


def _amortized_elapsed_warnings(
    row_ids: tuple[str, ...], stage: StageName
) -> tuple[WarningEntry, ...]:
    return tuple(
        WarningEntry(
            warning_id=f"{stage}_{row_id}_amortized_elapsed",
            warning_severity="info",
            message=(
                "Elapsed is amortized from the shared full hybrid suite runtime; "
                "upstream benchmark rows do not expose per-row timing."
            ),
            row_id=row_id,
            comparison_group=None,
        )
        for row_id in row_ids
    )


def _with_artifact(checkpoint: StageCheckpoint, artifact: str) -> StageCheckpoint:
    return replace(checkpoint, artifacts=(artifact,))


def _with_artifacts(checkpoint: StageCheckpoint, artifacts: tuple[str, ...]) -> StageCheckpoint:
    return replace(checkpoint, artifacts=artifacts)


def _timestamp() -> str:
    return datetime.now(tz=UTC).isoformat()
