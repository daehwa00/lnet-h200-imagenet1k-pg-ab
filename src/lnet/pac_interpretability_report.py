from __future__ import annotations

import os
from operator import itemgetter
from statistics import mean
from typing import TYPE_CHECKING, Final

os.environ.setdefault("MPLBACKEND", "Agg")

from matplotlib import pyplot as plt

from .pac_overnight_io import read_csv

if TYPE_CHECKING:
    from pathlib import Path

RESULT_FILES: Final[dict[str, str]] = {
    "synthetic_performance": "synthetic_performance.csv",
    "pole_recovery": "pole_recovery.csv",
    "tap_recovery": "tap_recovery.csv",
    "damping_alignment": "damping_alignment.csv",
    "mode_knockout": "mode_knockout.csv",
    "impulse_response_nmse": "impulse_response_nmse.csv",
    "real_modal_class_stats": "real_modal_class_stats.csv",
    "hermitian_attribution": "hermitian_attribution.csv",
    "real_performance": "real_performance.csv",
}


def write_interpretability_report(root: Path) -> Path:
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _bar(
        figures / "pole_map_frequency_error.png",
        _mean_by(
            read_csv(root / "results" / RESULT_FILES["pole_recovery"]),
            "task",
            "frequency_abs_error",
        ),
        "Pole frequency absolute error (lower is better)",
    )
    _bar(
        figures / "tap_profiles_delay_mass.png",
        _mean_by(
            read_csv(root / "results" / RESULT_FILES["tap_recovery"]),
            "task",
            "tap_mass_near_true_delay",
        ),
        "Tap mass near true delay (higher is better)",
    )
    _bar(
        figures / "damping_regime_auc.png",
        _mean_by(
            read_csv(root / "results" / RESULT_FILES["damping_alignment"]), "task", "auc_mean"
        ),
        "Damping regime AUC (higher is better)",
    )
    _bar(
        figures / "mode_knockout_relative_delta.png",
        _top_by(read_csv(root / "results" / RESULT_FILES["mode_knockout"]), "relative_delta", 12),
        "Mode knockout relative loss increase",
    )
    _bar(
        figures / "impulse_response_nmse.png",
        _mean_by(
            read_csv(root / "results" / RESULT_FILES["impulse_response_nmse"]),
            "task",
            "impulse_response_nmse",
        ),
        "Impulse response NMSE (lower is better)",
    )
    _bar(
        figures / "real_class_mode_energy.png",
        _mean_by(
            read_csv(root / "results" / RESULT_FILES["real_modal_class_stats"]),
            "dataset_or_task",
            "mean_modal_energy",
        ),
        "Class-wise modal energy",
    )
    report = root / "reports" / "interpretability_summary.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_report_text(root), encoding="utf-8")
    _update_integrated_report(root, report)
    return report


def _report_text(root: Path) -> str:
    synthetic = read_csv(root / "results" / RESULT_FILES["synthetic_performance"])
    real = read_csv(root / "results" / RESULT_FILES["real_performance"])
    damping = read_csv(root / "results" / RESULT_FILES["damping_alignment"])
    knockout = read_csv(root / "results" / RESULT_FILES["mode_knockout"])
    return "\n".join(
        (
            "# PAC Mechanistic Interpretability Evidence",
            "",
            _task_explanation(),
            "",
            "## Coverage",
            "",
            f"- synthetic rows: {len(synthetic)}",
            f"- real rows: {len(real)}",
            f"- damping diagnostic rows: {len(damping)}",
            f"- mode knockout rows: {len(knockout)}",
            "",
            "## Figures",
            "",
            (
                "- `figures/pole_map_frequency_error.png`: teacher frequency와 learned pole "
                "frequency의 오차입니다."
            ),
            (
                "- `figures/tap_profiles_delay_mass.png`: learned tap이 true delay 근처에 "
                "모이는지 봅니다."
            ),
            (
                "- `figures/damping_regime_auc.png`: effective damping이 fast/slow regime을 "
                "분리하는지 봅니다."
            ),
            (
                "- `figures/mode_knockout_relative_delta.png`: mode 제거가 손실을 얼마나 "
                "증가시키는지 봅니다."
            ),
            (
                "- `figures/real_class_mode_energy.png`: real dataset class별 modal energy "
                "차이를 봅니다."
            ),
            "",
            "## Conservative Claim",
            "",
            (
                "PAC는 aligned modal/delay/damping task에서 해석 가능한 modal-control 구조를 "
                "제공하는지 검증합니다. FIR negative control이나 PAC가 약한 real dataset에서 "
                "해석 신호가 약하면, 이는 negative control로 보고합니다."
            ),
            "",
        )
    )


def _task_explanation() -> str:
    return (
        "이 보고서는 PAC가 학습한 pole, tap, damping, modal state가 실제 teacher 또는 "
        "class 구조와 정렬되는지 확인합니다. 즉 단순히 test loss가 낮은지뿐 아니라, "
        "모델 내부의 해석 가능한 변수들이 우리가 설계한 dynamical mechanism을 "
        "따라가는지 확인합니다.\n\n"
        "`modal_teacher`, `delayed_exponential`, `delayed_oscillatory`, "
        "`multi_mode_delayed_resonance`는 pole 기반 teacher입니다. 입력 sequence "
        "`x[1:N]`를 받아 출력 sequence `y[1:N]`를 만들며, teacher는 대략 "
        "`y_n = (h * x)_n`, `h_n = sum_m r_m exp(p_m n)` 형태입니다. "
        "`modal_teacher`는 즉시 modal response를 만들고, delayed 계열은 몇 step 뒤에 "
        "response가 시작됩니다. 따라서 이 family에서는 learned pole의 damping/frequency와 "
        "learned tap의 delay 위치가 의미 있는지 봅니다.\n\n"
        "`active_damping_teacher`, `context_damped_exponential`, "
        "`delayed_context_damped_exponential`은 context에 따라 damping regime이 바뀌는 "
        "teacher입니다. 같은 modal memory라도 어떤 구간에서는 빨리 사라지고, 어떤 "
        "구간에서는 오래 남습니다. 따라서 effective damping이 teacher의 fast/slow "
        "regime과 상관되는지, AUC/correlation/Cohen's d로 확인합니다.\n\n"
        "`random_fir_teacher`는 negative control입니다. 이 task는 finite impulse response, "
        "즉 짧은 convolution으로 만들어지므로 clean pole recovery가 나오는 것이 오히려 "
        "이상합니다. 여기서는 PAC가 모든 문제에 억지로 pole 해석을 붙이는지 확인합니다.\n\n"
        "이번 synthetic 비교는 real classification 추천 모델을 그대로 넣은 것이 아닙니다. "
        "classification 모델은 sequence 전체를 class label 하나로 보내는 head가 붙어 "
        "있지만, 여기서는 `x[1:N] -> y[1:N]` sequence regression이 목표입니다. "
        "그래서 같은 PAC/Hermitian 아이디어를 sequence-regression 출력에 맞춘 대응 "
        "모델로 사용합니다."
    )


def _bar(path: Path, values: dict[str, float], title: str) -> None:
    if not values:
        return
    labels = list(values)
    heights = [values[label] for label in labels]
    plt.figure(figsize=(max(6, len(labels) * 0.55), 3.6))
    plt.bar(range(len(labels)), heights)
    plt.xticks(range(len(labels)), labels, rotation=35, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _mean_by(rows: list[dict[str, str]], key: str, value_key: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        label = row.get(key)
        value = _float(row.get(value_key))
        if label is not None and value is not None:
            grouped.setdefault(label, []).append(value)
    return {label: mean(values) for label, values in grouped.items() if values}


def _top_by(rows: list[dict[str, str]], value_key: str, limit: int) -> dict[str, float]:
    values: list[tuple[str, float]] = []
    for row in rows:
        value = _float(row.get(value_key))
        if value is not None:
            block = row.get("block_index", "?")
            mode = row.get("mode", "?")
            label = f"{row.get('task', 'task')}:b{block}m{mode}"
            values.append((label, value))
    values.sort(key=itemgetter(1), reverse=True)
    return dict(values[:limit])


def _float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _update_integrated_report(root: Path, report: Path) -> None:
    integrated = (
        root.parent / "integrated-final-20260709" / "reports" / "pac_integrated_paper_report.md"
    )
    if not integrated.exists():
        return
    marker = "## Mechanistic Interpretability Evidence"
    text = integrated.read_text(encoding="utf-8")
    if marker in text:
        return
    rel = report.relative_to(root.parent)
    text += (
        "\n"
        f"{marker}\n\n"
        "PAC의 pole, tap, damping control, mode knockout, class-wise Hermitian/modal statistics를 "
        "별도 큐에서 검증합니다. 이 섹션은 성능 순위가 아니라 모델 내부 구조가 task mechanism과 "
        "정렬되는지를 보기 위한 evidence입니다.\n\n"
        f"- 상세 보고서: `{rel}`\n"
        f"- 산출물 root: `{root.relative_to(root.parent)}`\n"
    )
    integrated.write_text(text, encoding="utf-8")
