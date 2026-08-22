"""Static four-path vertical product scan with selectable output epilogues.

The forward kernel deliberately does not materialize the four full-resolution
vertical product states.  Non-terminal stages write direction-aligned coarse
endpoints, while terminal stages skip those stores entirely.  Both epilogues
accumulate exact full-grid directional energies into the descriptor buffer.
"""

from __future__ import annotations

# pyright: reportArgumentType=false, reportAssignmentType=false, reportCallIssue=false
# pyright: reportGeneralTypeIssues=false, reportMissingParameterType=false
# pyright: reportPrivateUsage=false, reportUnknownLambdaType=false
from typing import Protocol, cast

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton
from triton.language.extra import libdevice

from .pac_kernel_launch_config import (
    LaunchGeometry,
    LaunchScope,
    autotuned,
    make_launch_scope,
    register_default,
)
from .pac_product_scan_contracts import (
    DEFAULT_EPSILON,
    ComplexField,
    FusedOutputs,
    ProductGainNormalization,
    gain_kind,
    gain_normalization,
    supports_pac_triton_product_scan_coarse4,
    supports_pac_triton_product_scan_descriptor4,
    validate_global_inverse_gain,
    validate_product_scan,
)
from .pac_product_scan_normalization import static_product_scan_auxiliary
from .pac_product_scan_reference import (
    _product_scan_coarse4_from_tables_reference,
    _product_scan_descriptor4_from_tables_reference,
    _product_scan_full16_from_tables_reference,
    product_scan_coarse4_reference,
    product_scan_descriptor4_reference,
    product_scan_full16_reference,
    raw_product_descriptor_reference,
)
from .pac_triton_grouped_path_cffn import (
    d4_grouped_cell_path_collapse_reference,
    d4_grouped_cell_path_swiglu_reference,
)

FORWARD_LAUNCH_NAME = "product_scan_coarse4_forward"
FULL16_FORWARD_LAUNCH_NAME = "product_scan_full16_forward"
PATH_COLLAPSE_FORWARD_LAUNCH_NAME = "product_scan_path_collapse_forward"
DESCRIPTOR_FORWARD_LAUNCH_NAME = "product_scan_descriptor4_forward"
FINALIZE_LAUNCH_NAME = "product_scan_coarse4_descriptor_finalize"
BACKWARD_LAUNCH_NAME = "product_scan_coarse4_backward"
FULL16_BACKWARD_LAUNCH_NAME = "product_scan_full16_backward"
PATH_COLLAPSE_BACKWARD_LAUNCH_NAME = "product_scan_path_collapse_backward"
_PATH_COUNT = 4
_PACKED_PATH_INPUTS = 8
_PACKED_PATH_OUTPUTS = 2
_PATH_DOT_INPUTS = 16
_PATH_DOT_OUTPUTS = 16
_PATHS_TL = tl.constexpr(_PATH_COUNT)
_PACKED_PATH_INPUTS_TL = tl.constexpr(_PACKED_PATH_INPUTS)
_PACKED_PATH_OUTPUTS_TL = tl.constexpr(_PACKED_PATH_OUTPUTS)
_PATH_DOT_INPUTS_TL = tl.constexpr(_PATH_DOT_INPUTS)
_PATH_DOT_OUTPUTS_TL = tl.constexpr(_PATH_DOT_OUTPUTS)
PathCollapseGradients = tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
]
DESCRIPTOR_BACKWARD_LAUNCH_NAME = "product_scan_descriptor4_backward"

_SCAN_LAUNCH_CANDIDATES = tuple(
    LaunchGeometry.build(
        num_warps=warps,
        blocks={"BLOCK_LINES": block_lines, "BLOCK_MODES": modes},
    )
    for block_lines, modes, warps in (
        *((1, modes, warps) for warps in (2, 4, 8) for modes in (4, 8, 16, 32)),
        (2, 4, 4),
        (2, 4, 8),
        (2, 8, 4),
        (2, 8, 8),
        (4, 4, 4),
        (4, 4, 8),
    )
)
_PATH_SCAN_LAUNCH_CANDIDATES = tuple(
    LaunchGeometry.build(
        num_warps=warps,
        blocks={"BLOCK_LINES": 1, "BLOCK_MODES": modes},
    )
    for modes in (4, 8, 16, 32, 64)
    for warps in (4, 8)
)
_FINALIZE_LAUNCH_CANDIDATES = tuple(
    LaunchGeometry.build(num_warps=warps, blocks={"BLOCK_MODES": modes})
    for warps in (2, 4)
    for modes in (16, 32, 64)
)
_SCAN_DEFAULT = LaunchGeometry.build(
    num_warps=4,
    blocks={"BLOCK_LINES": 1, "BLOCK_MODES": 8},
)
_FINALIZE_DEFAULT = LaunchGeometry.build(num_warps=4, blocks={"BLOCK_MODES": 32})

register_default(
    FORWARD_LAUNCH_NAME,
    _SCAN_DEFAULT,
    candidates=_SCAN_LAUNCH_CANDIDATES,
)
register_default(
    FULL16_FORWARD_LAUNCH_NAME,
    _SCAN_DEFAULT,
    candidates=_SCAN_LAUNCH_CANDIDATES,
)
register_default(
    PATH_COLLAPSE_FORWARD_LAUNCH_NAME,
    _PATH_SCAN_LAUNCH_CANDIDATES[0],
    candidates=_PATH_SCAN_LAUNCH_CANDIDATES,
)
register_default(
    DESCRIPTOR_FORWARD_LAUNCH_NAME,
    _SCAN_DEFAULT,
    candidates=_SCAN_LAUNCH_CANDIDATES,
)
register_default(
    BACKWARD_LAUNCH_NAME,
    _SCAN_DEFAULT,
    candidates=_SCAN_LAUNCH_CANDIDATES,
)
register_default(
    FULL16_BACKWARD_LAUNCH_NAME,
    _SCAN_DEFAULT,
    candidates=_SCAN_LAUNCH_CANDIDATES,
)
register_default(
    PATH_COLLAPSE_BACKWARD_LAUNCH_NAME,
    _PATH_SCAN_LAUNCH_CANDIDATES[0],
    candidates=_PATH_SCAN_LAUNCH_CANDIDATES,
)
register_default(
    DESCRIPTOR_BACKWARD_LAUNCH_NAME,
    _SCAN_DEFAULT,
    candidates=_SCAN_LAUNCH_CANDIDATES,
)
register_default(
    FINALIZE_LAUNCH_NAME,
    _FINALIZE_DEFAULT,
    candidates=_FINALIZE_LAUNCH_CANDIDATES,
)


class _AutogradContext(Protocol):
    saved_tensors: tuple[Tensor, ...]
    epsilon: float
    gain_kind: int

    def save_for_backward(self, *tensors: Tensor) -> None: ...


def _scan_launch_scope(
    kernel: object,
    source: Tensor,
    *,
    gain_kind: int,
    full_coarse: bool = False,
    path_collapse: bool = False,
) -> LaunchScope:
    batch, height, width, modes = source.shape
    return make_launch_scope(
        kernel,
        source,
        shape={
            "batch": batch,
            "height": height,
            "width": width,
            "modes": modes,
            "gain_kind": gain_kind,
            "full_coarse": full_coarse,
            "path_collapse": path_collapse,
        },
    )


@triton.jit
def _compose_complex_pair(
    left_ar,
    left_ai,
    left_a_r,
    left_a_i,
    left_b_r,
    left_b_i,
    right_ar,
    right_ai,
    right_a_r,
    right_a_i,
    right_b_r,
    right_b_i,
):
    composed_ar = right_ar * left_ar - right_ai * left_ai
    composed_ai = right_ai * left_ar + right_ar * left_ai
    composed_a_r = right_ar * left_a_r - right_ai * left_a_i + right_a_r
    composed_a_i = right_ai * left_a_r + right_ar * left_a_i + right_a_i
    composed_b_r = right_ar * left_b_r - right_ai * left_b_i + right_b_r
    composed_b_i = right_ai * left_b_r + right_ar * left_b_i + right_b_i
    return (
        composed_ar,
        composed_ai,
        composed_a_r,
        composed_a_i,
        composed_b_r,
        composed_b_i,
    )


@triton.jit
def _product_scan_coarse4_associative_forward_kernel(
    decay_real,
    decay_imag,
    gamma_real,
    gamma_imag,
    source_real_a,
    source_imag_a,
    source_real_b,
    source_imag_b,
    variance_x,
    variance_y,
    global_inverse_gain,
    coarse_real,
    coarse_imag,
    path_input_weight,
    path_input_bias,
    path_output_weight,
    path_output_bias,
    collapsed_real,
    collapsed_imag,
    descriptor_energy,
    height: tl.constexpr,
    width: int,
    line_count: int,
    modes: int,
    packed_path_hidden: int,
    epsilon: tl.constexpr,
    gain_kind: tl.constexpr,
    EMIT_COARSE: tl.constexpr,
    FULL_COARSE: tl.constexpr,
    COLLAPSE_PATHS: tl.constexpr,
    PATH_SWIGLU: tl.constexpr,
    SWIGLU_HIDDEN: tl.constexpr,
    EMIT_DESCRIPTOR: tl.constexpr,
    BLOCK_HEIGHT: tl.constexpr,
    BLOCK_LINES: tl.constexpr,
    BLOCK_MODES: tl.constexpr,
    BLOCK_PATH_HIDDEN: tl.constexpr,
    BLOCK_SWIGLU_VALUE: tl.constexpr,
) -> None:
    batch_size = line_count // width
    line_group = tl.program_id(0)
    if BLOCK_LINES == 1:
        batch = line_group // width
        x_group = line_group - batch * width
    else:
        # Grouped programs are interleaved across batches.  Adjacent CTAs then
        # update different descriptor rows instead of contending on one batch.
        x_group = line_group // batch_size
        batch = line_group - x_group * batch_size
    x = x_group * BLOCK_LINES + tl.arange(0, BLOCK_LINES)[None, :, None]
    y = tl.arange(0, BLOCK_HEIGHT)[:, None, None]
    mode_vector = tl.program_id(1) * BLOCK_MODES + tl.arange(0, BLOCK_MODES)
    mode = mode_vector[None, None, :]
    valid_line = x < width
    active = valid_line & (y < height) & (mode < modes)
    valid_mode = valid_line & (mode < modes)
    offset = ((batch * height + y) * width + x) * modes + mode

    ar = tl.load(decay_real + mode, mask=mode < modes, other=0.0).to(tl.float32)
    ai = tl.load(decay_imag + mode, mask=mode < modes, other=0.0).to(tl.float32)
    gr = tl.load(gamma_real + mode, mask=mode < modes, other=0.0).to(tl.float32)
    gi = tl.load(gamma_imag + mode, mask=mode < modes, other=0.0).to(tl.float32)
    sar = tl.load(source_real_a + offset, mask=active, other=0.0).to(tl.float32)
    sai = tl.load(source_imag_a + offset, mask=active, other=0.0).to(tl.float32)
    sbr = tl.load(source_real_b + offset, mask=active, other=0.0).to(tl.float32)
    sbi = tl.load(source_imag_b + offset, mask=active, other=0.0).to(tl.float32)
    scan_ar = tl.where(active, ar, 1.0)
    scan_ai = tl.where(active, ai, 0.0)
    positive = tl.associative_scan(
        (
            scan_ar,
            scan_ai,
            gr * sar - gi * sai,
            gr * sai + gi * sar,
            gr * sbr - gi * sbi,
            gr * sbi + gi * sbr,
        ),
        axis=0,
        combine_fn=_compose_complex_pair,
    )
    negative = tl.associative_scan(
        (
            scan_ar,
            -scan_ai,
            gr * sar + gi * sai,
            gr * sai - gi * sar,
            gr * sbr + gi * sbi,
            gr * sbi - gi * sbr,
        ),
        axis=0,
        combine_fn=_compose_complex_pair,
        reverse=True,
    )
    par, pai, pbr, pbi = positive[2], positive[3], positive[4], positive[5]
    nar, nai, nbr, nbi = negative[2], negative[3], negative[4], negative[5]
    if gain_kind == 1:
        inverse_pa = tl.load(global_inverse_gain + mode, mask=valid_mode, other=0.0)
        inverse_pb = inverse_pa
        inverse_na = inverse_pa
        inverse_nb = inverse_pa
    else:
        x_offset = x * modes + mode
        y_offset = y * modes + mode
        vx_positive = tl.load(variance_x + x_offset, mask=valid_mode, other=0.0)
        vx_negative = tl.load(
            variance_x + width * modes + x_offset,
            mask=valid_mode,
            other=0.0,
        )
        vy_positive = tl.load(variance_y + y_offset, mask=active, other=0.0)
        vy_negative = tl.load(
            variance_y + height * modes + y_offset,
            mask=active,
            other=0.0,
        )
        inverse_pa = tl.rsqrt(tl.maximum(vx_positive * vy_positive, epsilon))
        inverse_pb = tl.rsqrt(tl.maximum(vx_negative * vy_positive, epsilon))
        inverse_na = tl.rsqrt(tl.maximum(vx_positive * vy_negative, epsilon))
        inverse_nb = tl.rsqrt(tl.maximum(vx_negative * vy_negative, epsilon))
    # Zero padded scan lanes once at the normalization boundary.  Keeping the
    # padded recurrence state alive and masking every downstream consumer costs
    # registers, while its zero variance also magnifies the state by rsqrt(eps).
    r0, i0 = tl.where(active, par * inverse_pa, 0.0), tl.where(active, pai * inverse_pa, 0.0)
    r1, i1 = tl.where(active, pbr * inverse_pb, 0.0), tl.where(active, pbi * inverse_pb, 0.0)
    r2, i2 = tl.where(active, nar * inverse_na, 0.0), tl.where(active, nai * inverse_na, 0.0)
    r3, i3 = tl.where(active, nbr * inverse_nb, 0.0), tl.where(active, nbi * inverse_nb, 0.0)

    if COLLAPSE_PATHS:
        r0_flat = tl.reshape(
            tl.permute(r0, (2, 0, 1)),
            (BLOCK_MODES, BLOCK_HEIGHT * BLOCK_LINES),
        )
        r1_flat = tl.reshape(
            tl.permute(r1, (2, 0, 1)),
            (BLOCK_MODES, BLOCK_HEIGHT * BLOCK_LINES),
        )
        r2_flat = tl.reshape(
            tl.permute(r2, (2, 0, 1)),
            (BLOCK_MODES, BLOCK_HEIGHT * BLOCK_LINES),
        )
        r3_flat = tl.reshape(
            tl.permute(r3, (2, 0, 1)),
            (BLOCK_MODES, BLOCK_HEIGHT * BLOCK_LINES),
        )
        i0_flat = tl.reshape(
            tl.permute(i0, (2, 0, 1)),
            (BLOCK_MODES, BLOCK_HEIGHT * BLOCK_LINES),
        )
        i1_flat = tl.reshape(
            tl.permute(i1, (2, 0, 1)),
            (BLOCK_MODES, BLOCK_HEIGHT * BLOCK_LINES),
        )
        i2_flat = tl.reshape(
            tl.permute(i2, (2, 0, 1)),
            (BLOCK_MODES, BLOCK_HEIGHT * BLOCK_LINES),
        )
        i3_flat = tl.reshape(
            tl.permute(i3, (2, 0, 1)),
            (BLOCK_MODES, BLOCK_HEIGHT * BLOCK_LINES),
        )
        path_coordinate = tl.arange(0, _PATH_DOT_INPUTS_TL)
        path = path_coordinate % _PATHS_TL
        active_path = tl.where(
            path[None, None, :] == 0,
            r0_flat[:, :, None],
            tl.where(
                path[None, None, :] == 1,
                r1_flat[:, :, None],
                tl.where(
                    path[None, None, :] == 2,
                    r2_flat[:, :, None],
                    r3_flat[:, :, None],
                ),
            ),
        )
        active_path_imag = tl.where(
            path[None, None, :] == 0,
            i0_flat[:, :, None],
            tl.where(
                path[None, None, :] == 1,
                i1_flat[:, :, None],
                tl.where(
                    path[None, None, :] == 2,
                    i2_flat[:, :, None],
                    i3_flat[:, :, None],
                ),
            ),
        )
        packed_path = tl.where(
            path_coordinate[None, None, :] < _PATHS_TL,
            active_path,
            active_path_imag,
        ).to(tl.bfloat16)
        hidden_coordinate = tl.arange(0, BLOCK_PATH_HIDDEN)
        input_weight_offset = (
            mode_vector[:, None, None] * packed_path_hidden + hidden_coordinate[None, None, :]
        ) * _PACKED_PATH_INPUTS_TL + path_coordinate[None, :, None]
        active_input_weight = tl.load(
            path_input_weight + input_weight_offset,
            mask=(mode_vector[:, None, None] < modes)
            & (hidden_coordinate[None, None, :] < packed_path_hidden)
            & (path_coordinate[None, :, None] < _PACKED_PATH_INPUTS_TL),
            other=0.0,
        ).to(tl.bfloat16)
        preactivation = tl.dot(packed_path, active_input_weight)
        active_input_bias = tl.load(
            path_input_bias
            + mode_vector[:, None] * packed_path_hidden
            + hidden_coordinate[None, :],
            mask=(mode_vector[:, None] < modes) & (hidden_coordinate[None, :] < packed_path_hidden),
            other=0.0,
        ).to(tl.bfloat16)
        preactivation = (preactivation + active_input_bias[:, None, :]).to(tl.bfloat16)
        output_coordinate = tl.arange(0, _PATH_DOT_OUTPUTS_TL)
        active_output_bias = tl.load(
            path_output_bias
            + mode_vector[:, None] * _PACKED_PATH_OUTPUTS_TL
            + output_coordinate[None, :],
            mask=(mode_vector[:, None] < modes)
            & (output_coordinate[None, :] < _PACKED_PATH_OUTPUTS_TL),
            other=0.0,
        ).to(tl.bfloat16)
        if PATH_SWIGLU:
            value_coordinate = tl.arange(0, BLOCK_SWIGLU_VALUE)
            value_indices = tl.broadcast_to(
                value_coordinate[None, None, :],
                (
                    BLOCK_MODES,
                    BLOCK_HEIGHT * BLOCK_LINES,
                    BLOCK_SWIGLU_VALUE,
                ),
            )
            gate_indices = tl.broadcast_to(
                (2 * SWIGLU_HIDDEN + value_coordinate % SWIGLU_HIDDEN)[None, None, :],
                (
                    BLOCK_MODES,
                    BLOCK_HEIGHT * BLOCK_LINES,
                    BLOCK_SWIGLU_VALUE,
                ),
            )
            value = tl.gather(preactivation, value_indices, axis=2).to(tl.bfloat16)
            gate_logits = tl.gather(preactivation, gate_indices, axis=2).to(tl.float32)
            gate = (gate_logits * tl.sigmoid(gate_logits)).to(tl.bfloat16)
            gated_value = tl.where(
                value_coordinate[None, None, :] < 2 * SWIGLU_HIDDEN,
                value * gate,
                0.0,
            ).to(tl.bfloat16)
            output_weight_offset = (
                mode_vector[:, None, None] * _PACKED_PATH_OUTPUTS_TL
                + output_coordinate[None, None, :]
            ) * (2 * SWIGLU_HIDDEN) + value_coordinate[None, :, None]
            active_output_weight = tl.load(
                path_output_weight + output_weight_offset,
                mask=(mode_vector[:, None, None] < modes)
                & (value_coordinate[None, :, None] < 2 * SWIGLU_HIDDEN)
                & (output_coordinate[None, None, :] < _PACKED_PATH_OUTPUTS_TL),
                other=0.0,
            ).to(tl.bfloat16)
            correction = tl.dot(gated_value, active_output_weight)
            path_output = (correction + active_output_bias[:, None, :]).to(tl.bfloat16)
        else:
            hidden_float = preactivation.to(tl.float32)
            hidden = (hidden_float * tl.sigmoid(hidden_float)).to(tl.bfloat16)
            output_weight_offset = (
                mode_vector[:, None, None] * _PACKED_PATH_OUTPUTS_TL
                + output_coordinate[None, None, :]
            ) * packed_path_hidden + hidden_coordinate[None, :, None]
            active_output_weight = tl.load(
                path_output_weight + output_weight_offset,
                mask=(mode_vector[:, None, None] < modes)
                & (hidden_coordinate[None, :, None] < packed_path_hidden)
                & (output_coordinate[None, None, :] < _PACKED_PATH_OUTPUTS_TL),
                other=0.0,
            ).to(tl.bfloat16)
            path_output = tl.dot(hidden, active_output_weight)
            path_output = (path_output + active_output_bias[:, None, :]).to(tl.bfloat16)
        physical_offset = tl.reshape(
            tl.permute(offset, (2, 0, 1)),
            (BLOCK_MODES, BLOCK_HEIGHT * BLOCK_LINES),
        )
        physical_active = tl.reshape(
            tl.permute(active, (2, 0, 1)),
            (BLOCK_MODES, BLOCK_HEIGHT * BLOCK_LINES),
        )
        collapsed_real_value = tl.sum(
            tl.where(output_coordinate[None, None, :] == 0, path_output, 0.0),
            axis=2,
        )
        collapsed_imag_value = tl.sum(
            tl.where(output_coordinate[None, None, :] == 1, path_output, 0.0),
            axis=2,
        )
        tl.store(
            collapsed_real + physical_offset,
            collapsed_real_value,
            mask=physical_active,
        )
        tl.store(
            collapsed_imag + physical_offset,
            collapsed_imag_value,
            mask=physical_active,
        )

    if EMIT_COARSE:
        cell = (batch * (height // 2) + y // 2) * (width // 2) + x // 2
        if FULL_COARSE:
            parity_x = x & 1
            parity_y = y & 1
            local0 = 2 * parity_x + parity_y
            local1 = 2 * (1 - parity_x) + parity_y
            local2 = 2 * parity_x + (1 - parity_y)
            local3 = 2 * (1 - parity_x) + (1 - parity_y)
            base0 = ((cell * 4) * 4 + local0) * modes + mode
            base1 = ((cell * 4 + 1) * 4 + local1) * modes + mode
            base2 = ((cell * 4 + 2) * 4 + local2) * modes + mode
            base3 = ((cell * 4 + 3) * 4 + local3) * modes + mode
            tl.store(coarse_real + base0, r0, mask=active)
            tl.store(coarse_imag + base0, i0, mask=active)
            tl.store(coarse_real + base1, r1, mask=active)
            tl.store(coarse_imag + base1, i1, mask=active)
            tl.store(coarse_real + base2, r2, mask=active)
            tl.store(coarse_imag + base2, i2, mask=active)
            tl.store(coarse_real + base3, r3, mask=active)
            tl.store(coarse_imag + base3, i3, mask=active)
        else:
            coarse_base = (cell * 4) * modes + mode
            positive_endpoint = active & ((y & 1) == 1)
            negative_endpoint = active & ((y & 1) == 0)
            tl.store(
                coarse_real + coarse_base,
                r0,
                mask=positive_endpoint & ((x & 1) == 1),
            )
            tl.store(
                coarse_imag + coarse_base,
                i0,
                mask=positive_endpoint & ((x & 1) == 1),
            )
            tl.store(
                coarse_real + coarse_base + modes,
                r1,
                mask=positive_endpoint & ((x & 1) == 0),
            )
            tl.store(
                coarse_imag + coarse_base + modes,
                i1,
                mask=positive_endpoint & ((x & 1) == 0),
            )
            tl.store(
                coarse_real + coarse_base + 2 * modes,
                r2,
                mask=negative_endpoint & ((x & 1) == 1),
            )
            tl.store(
                coarse_imag + coarse_base + 2 * modes,
                i2,
                mask=negative_endpoint & ((x & 1) == 1),
            )
            tl.store(
                coarse_real + coarse_base + 3 * modes,
                r3,
                mask=negative_endpoint & ((x & 1) == 0),
            )
            tl.store(
                coarse_imag + coarse_base + 3 * modes,
                i3,
                mask=negative_endpoint & ((x & 1) == 0),
            )

    raw0 = tl.sum(tl.sum(r0 * r0 + i0 * i0, axis=0), axis=0)
    raw1 = tl.sum(tl.sum(r1 * r1 + i1 * i1, axis=0), axis=0)
    raw2 = tl.sum(tl.sum(r2 * r2 + i2 * i2, axis=0), axis=0)
    raw3 = tl.sum(tl.sum(r3 * r3 + i3 * i3, axis=0), axis=0)
    descriptor_mode = batch * (4 * modes) + mode_vector
    valid_descriptor_mode = mode_vector < modes
    tl.atomic_add(
        descriptor_energy + descriptor_mode,
        raw0,
        mask=valid_descriptor_mode & EMIT_DESCRIPTOR,
    )
    tl.atomic_add(
        descriptor_energy + descriptor_mode + modes,
        raw1,
        mask=valid_descriptor_mode & EMIT_DESCRIPTOR,
    )
    tl.atomic_add(
        descriptor_energy + descriptor_mode + 2 * modes,
        raw2,
        mask=valid_descriptor_mode & EMIT_DESCRIPTOR,
    )
    tl.atomic_add(
        descriptor_energy + descriptor_mode + 3 * modes,
        raw3,
        mask=valid_descriptor_mode & EMIT_DESCRIPTOR,
    )


@triton.jit
def _finalize_descriptor_kernel(
    descriptor_energy,
    descriptor,
    spatial_size: int,
    modes: int,
    BLOCK_MODES: tl.constexpr,
) -> None:
    batch = tl.program_id(0)
    mode = tl.program_id(1) * BLOCK_MODES + tl.arange(0, BLOCK_MODES)
    valid_mode = mode < modes
    base = batch * (4 * modes) + mode
    inverse_spatial = 1.0 / spatial_size
    for component in tl.static_range(4):
        offset = base + component * modes
        energy = tl.load(descriptor_energy + offset, mask=valid_mode, other=0.0)
        tl.store(
            descriptor + offset,
            libdevice.log1p(energy * inverse_spatial),
            mask=valid_mode,
        )


@triton.jit
def _unpack_path_gradient(
    gradient,
    coordinate,
    index: tl.constexpr,
    BLOCK_HEIGHT: tl.constexpr,
    BLOCK_LINES: tl.constexpr,
    BLOCK_MODES: tl.constexpr,
):
    values = tl.sum(
        tl.where(coordinate[None, None, :] == index, gradient, 0.0),
        axis=2,
    )
    return tl.permute(
        tl.reshape(values, (BLOCK_MODES, BLOCK_HEIGHT, BLOCK_LINES)),
        (1, 2, 0),
    )


@triton.jit
def _product_scan_coarse4_associative_backward_kernel(  # noqa: PLR0912
    decay_real,
    decay_imag,
    gamma_real,
    gamma_imag,
    source_real_a,
    source_imag_a,
    source_real_b,
    source_imag_b,
    variance_x,
    variance_y,
    global_inverse_gain,
    grad_coarse_real,
    grad_coarse_imag,
    grad_collapsed_real,
    grad_collapsed_imag,
    path_input_weight,
    path_input_bias,
    path_output_weight,
    _path_output_bias,
    grad_path_input_weight,
    grad_path_input_bias,
    grad_path_output_weight,
    grad_path_output_bias,
    descriptor_gradient_factor,
    grad_decay_real,
    grad_decay_imag,
    grad_gamma_real,
    grad_gamma_imag,
    grad_source_real_a,
    grad_source_imag_a,
    grad_source_real_b,
    grad_source_imag_b,
    height: tl.constexpr,
    width: int,
    line_count: int,
    modes: int,
    packed_path_hidden: int,
    epsilon: tl.constexpr,
    gain_kind: tl.constexpr,
    HAS_COARSE_GRAD: tl.constexpr,
    FULL_COARSE_GRAD: tl.constexpr,
    COLLAPSE_PATHS_GRAD: tl.constexpr,
    PATH_SWIGLU: tl.constexpr,
    SWIGLU_HIDDEN: tl.constexpr,
    HAS_DESCRIPTOR_GRAD: tl.constexpr,
    BLOCK_HEIGHT: tl.constexpr,
    BLOCK_LINES: tl.constexpr,
    BLOCK_MODES: tl.constexpr,
    BLOCK_PATH_HIDDEN: tl.constexpr,
    BLOCK_SWIGLU_VALUE: tl.constexpr,
) -> None:
    batch_size = line_count // width
    line_group = tl.program_id(0)
    if BLOCK_LINES == 1:
        batch = line_group // width
        x_group = line_group - batch * width
    else:
        # Spread coefficient atomics across batches while each grouped CTA
        # accumulates several neighboring x-lines before issuing its update.
        x_group = line_group // batch_size
        batch = line_group - x_group * batch_size
    x = x_group * BLOCK_LINES + tl.arange(0, BLOCK_LINES)[None, :, None]
    y = tl.arange(0, BLOCK_HEIGHT)[:, None, None]
    mode_vector = tl.program_id(1) * BLOCK_MODES + tl.arange(0, BLOCK_MODES)
    mode = mode_vector[None, None, :]
    valid_line = x < width
    active = valid_line & (y < height) & (mode < modes)
    valid_mode = valid_line & (mode < modes)
    offset = ((batch * height + y) * width + x) * modes + mode
    ar = tl.load(decay_real + mode, mask=valid_mode, other=0.0).to(tl.float32)
    ai = tl.load(decay_imag + mode, mask=valid_mode, other=0.0).to(tl.float32)
    gr = tl.load(gamma_real + mode, mask=valid_mode, other=0.0).to(tl.float32)
    gi = tl.load(gamma_imag + mode, mask=valid_mode, other=0.0).to(tl.float32)
    sar = tl.load(source_real_a + offset, mask=active, other=0.0).to(tl.float32)
    sai = tl.load(source_imag_a + offset, mask=active, other=0.0).to(tl.float32)
    sbr = tl.load(source_real_b + offset, mask=active, other=0.0).to(tl.float32)
    sbi = tl.load(source_imag_b + offset, mask=active, other=0.0).to(tl.float32)
    scan_ar = tl.where(active, ar, 1.0)
    scan_ai = tl.where(active, ai, 0.0)
    positive = tl.associative_scan(
        (
            scan_ar,
            scan_ai,
            gr * sar - gi * sai,
            gr * sai + gi * sar,
            gr * sbr - gi * sbi,
            gr * sbi + gi * sbr,
        ),
        axis=0,
        combine_fn=_compose_complex_pair,
    )
    negative = tl.associative_scan(
        (
            scan_ar,
            -scan_ai,
            gr * sar + gi * sai,
            gr * sai - gi * sar,
            gr * sbr + gi * sbi,
            gr * sbi - gi * sbr,
        ),
        axis=0,
        combine_fn=_compose_complex_pair,
        reverse=True,
    )
    decay_norm = ar * ar + ai * ai
    par, pai, pbr, pbi = positive[2], positive[3], positive[4], positive[5]
    nar, nai, nbr, nbi = negative[2], negative[3], negative[4], negative[5]
    if gain_kind == 1:
        inverse_pa = tl.load(global_inverse_gain + mode, mask=valid_mode, other=0.0)
        inverse_pb = inverse_pa
        inverse_na = inverse_pa
        inverse_nb = inverse_pa
    else:
        x_offset = x * modes + mode
        y_offset = y * modes + mode
        vx_positive = tl.load(variance_x + x_offset, mask=valid_mode, other=0.0)
        vx_negative = tl.load(
            variance_x + width * modes + x_offset,
            mask=valid_mode,
            other=0.0,
        )
        vy_positive = tl.load(variance_y + y_offset, mask=active, other=0.0)
        vy_negative = tl.load(
            variance_y + height * modes + y_offset,
            mask=active,
            other=0.0,
        )
        inverse_pa = tl.rsqrt(tl.maximum(vx_positive * vy_positive, epsilon))
        inverse_pb = tl.rsqrt(tl.maximum(vx_negative * vy_positive, epsilon))
        inverse_na = tl.rsqrt(tl.maximum(vx_positive * vy_negative, epsilon))
        inverse_nb = tl.rsqrt(tl.maximum(vx_negative * vy_negative, epsilon))
    r0, i0 = tl.where(active, par * inverse_pa, 0.0), tl.where(active, pai * inverse_pa, 0.0)
    r1, i1 = tl.where(active, pbr * inverse_pb, 0.0), tl.where(active, pbi * inverse_pb, 0.0)
    r2, i2 = tl.where(active, nar * inverse_na, 0.0), tl.where(active, nai * inverse_na, 0.0)
    r3, i3 = tl.where(active, nbr * inverse_nb, 0.0), tl.where(active, nbi * inverse_nb, 0.0)
    descriptor_base = batch * (4 * modes) + mode
    raw_factor0 = tl.load(
        descriptor_gradient_factor + descriptor_base,
        mask=valid_mode & HAS_DESCRIPTOR_GRAD,
        other=0.0,
    )
    raw_factor1 = tl.load(
        descriptor_gradient_factor + descriptor_base + modes,
        mask=valid_mode & HAS_DESCRIPTOR_GRAD,
        other=0.0,
    )
    raw_factor2 = tl.load(
        descriptor_gradient_factor + descriptor_base + 2 * modes,
        mask=valid_mode & HAS_DESCRIPTOR_GRAD,
        other=0.0,
    )
    raw_factor3 = tl.load(
        descriptor_gradient_factor + descriptor_base + 3 * modes,
        mask=valid_mode & HAS_DESCRIPTOR_GRAD,
        other=0.0,
    )
    grad_r0 = 2.0 * raw_factor0 * r0 * inverse_pa
    grad_i0 = 2.0 * raw_factor0 * i0 * inverse_pa
    grad_r1 = 2.0 * raw_factor1 * r1 * inverse_pb
    grad_i1 = 2.0 * raw_factor1 * i1 * inverse_pb
    grad_r2 = 2.0 * raw_factor2 * r2 * inverse_na
    grad_i2 = 2.0 * raw_factor2 * i2 * inverse_na
    grad_r3 = 2.0 * raw_factor3 * r3 * inverse_nb
    grad_i3 = 2.0 * raw_factor3 * i3 * inverse_nb

    if COLLAPSE_PATHS_GRAD:
        r0_flat = tl.reshape(
            tl.permute(r0, (2, 0, 1)),
            (BLOCK_MODES, BLOCK_HEIGHT * BLOCK_LINES),
        )
        r1_flat = tl.reshape(
            tl.permute(r1, (2, 0, 1)),
            (BLOCK_MODES, BLOCK_HEIGHT * BLOCK_LINES),
        )
        r2_flat = tl.reshape(
            tl.permute(r2, (2, 0, 1)),
            (BLOCK_MODES, BLOCK_HEIGHT * BLOCK_LINES),
        )
        r3_flat = tl.reshape(
            tl.permute(r3, (2, 0, 1)),
            (BLOCK_MODES, BLOCK_HEIGHT * BLOCK_LINES),
        )
        i0_flat = tl.reshape(
            tl.permute(i0, (2, 0, 1)),
            (BLOCK_MODES, BLOCK_HEIGHT * BLOCK_LINES),
        )
        i1_flat = tl.reshape(
            tl.permute(i1, (2, 0, 1)),
            (BLOCK_MODES, BLOCK_HEIGHT * BLOCK_LINES),
        )
        i2_flat = tl.reshape(
            tl.permute(i2, (2, 0, 1)),
            (BLOCK_MODES, BLOCK_HEIGHT * BLOCK_LINES),
        )
        i3_flat = tl.reshape(
            tl.permute(i3, (2, 0, 1)),
            (BLOCK_MODES, BLOCK_HEIGHT * BLOCK_LINES),
        )
        path_coordinate = tl.arange(0, _PATH_DOT_INPUTS_TL)
        path = path_coordinate % _PATHS_TL
        active_path_real = tl.where(
            path[None, None, :] == 0,
            r0_flat[:, :, None],
            tl.where(
                path[None, None, :] == 1,
                r1_flat[:, :, None],
                tl.where(
                    path[None, None, :] == 2,
                    r2_flat[:, :, None],
                    r3_flat[:, :, None],
                ),
            ),
        )
        active_path_imag = tl.where(
            path[None, None, :] == 0,
            i0_flat[:, :, None],
            tl.where(
                path[None, None, :] == 1,
                i1_flat[:, :, None],
                tl.where(
                    path[None, None, :] == 2,
                    i2_flat[:, :, None],
                    i3_flat[:, :, None],
                ),
            ),
        )
        packed_path = tl.where(
            path_coordinate[None, None, :] < _PATHS_TL,
            active_path_real,
            active_path_imag,
        ).to(tl.bfloat16)
        hidden_coordinate = tl.arange(0, BLOCK_PATH_HIDDEN)
        input_weight_offset = (
            mode_vector[:, None, None] * packed_path_hidden + hidden_coordinate[None, None, :]
        ) * _PACKED_PATH_INPUTS_TL + path_coordinate[None, :, None]
        active_input_weight = tl.load(
            path_input_weight + input_weight_offset,
            mask=(mode_vector[:, None, None] < modes)
            & (hidden_coordinate[None, None, :] < packed_path_hidden)
            & (path_coordinate[None, :, None] < _PACKED_PATH_INPUTS_TL),
            other=0.0,
        ).to(tl.bfloat16)
        preactivation = tl.dot(packed_path, active_input_weight)
        active_input_bias = tl.load(
            path_input_bias
            + mode_vector[:, None] * packed_path_hidden
            + hidden_coordinate[None, :],
            mask=(mode_vector[:, None] < modes) & (hidden_coordinate[None, :] < packed_path_hidden),
            other=0.0,
        ).to(tl.bfloat16)
        preactivation = (preactivation + active_input_bias[:, None, :]).to(tl.bfloat16)
        output_coordinate = tl.arange(0, _PATH_DOT_OUTPUTS_TL)
        physical_offset = tl.reshape(
            tl.permute(offset, (2, 0, 1)),
            (BLOCK_MODES, BLOCK_HEIGHT * BLOCK_LINES),
        )
        physical_active = tl.reshape(
            tl.permute(active, (2, 0, 1)),
            (BLOCK_MODES, BLOCK_HEIGHT * BLOCK_LINES),
        )
        active_grad_real = tl.load(
            grad_collapsed_real + physical_offset,
            mask=physical_active,
            other=0.0,
        ).to(tl.bfloat16)
        active_grad_imag = tl.load(
            grad_collapsed_imag + physical_offset,
            mask=physical_active,
            other=0.0,
        ).to(tl.bfloat16)
        active_grad_output = tl.where(
            output_coordinate[None, None, :] == 0,
            active_grad_real[:, :, None],
            tl.where(
                output_coordinate[None, None, :] == 1,
                active_grad_imag[:, :, None],
                0.0,
            ),
        ).to(tl.bfloat16)
        if PATH_SWIGLU:
            value_coordinate = tl.arange(0, BLOCK_SWIGLU_VALUE)
            value_indices = tl.broadcast_to(
                value_coordinate[None, None, :],
                (
                    BLOCK_MODES,
                    BLOCK_HEIGHT * BLOCK_LINES,
                    BLOCK_SWIGLU_VALUE,
                ),
            )
            gate_indices = tl.broadcast_to(
                (2 * SWIGLU_HIDDEN + value_coordinate % SWIGLU_HIDDEN)[None, None, :],
                (
                    BLOCK_MODES,
                    BLOCK_HEIGHT * BLOCK_LINES,
                    BLOCK_SWIGLU_VALUE,
                ),
            )
            value = tl.gather(preactivation, value_indices, axis=2).to(tl.bfloat16)
            gate_logits = tl.gather(preactivation, gate_indices, axis=2).to(tl.float32)
            gate_sigmoid = tl.sigmoid(gate_logits)
            gate = (gate_logits * gate_sigmoid).to(tl.bfloat16)
            gated_value = tl.where(
                value_coordinate[None, None, :] < 2 * SWIGLU_HIDDEN,
                value * gate,
                0.0,
            ).to(tl.bfloat16)
            output_weight_offset = (
                mode_vector[:, None, None] * _PACKED_PATH_OUTPUTS_TL
                + output_coordinate[None, :, None]
            ) * (2 * SWIGLU_HIDDEN) + value_coordinate[None, None, :]
            active_output_weight = tl.load(
                path_output_weight + output_weight_offset,
                mask=(mode_vector[:, None, None] < modes)
                & (output_coordinate[None, :, None] < _PACKED_PATH_OUTPUTS_TL)
                & (value_coordinate[None, None, :] < 2 * SWIGLU_HIDDEN),
                other=0.0,
            ).to(tl.bfloat16)
            grad_gated_value = tl.dot(active_grad_output, active_output_weight).to(tl.float32)
            active_value = value.to(tl.float32)
            active_gate = gate.to(tl.float32)
            grad_value = grad_gated_value * active_gate
            gate_derivative = gate_sigmoid * (1.0 + gate_logits * (1.0 - gate_sigmoid))
            grad_gate_contribution = grad_gated_value * active_value * gate_derivative
            value_match = (
                hidden_coordinate[None, None, None, :] == value_coordinate[None, None, :, None]
            )
            value_joint_gradient = tl.sum(
                tl.where(value_match, grad_value[:, :, :, None], 0.0),
                axis=2,
            )
            gate_match = (
                hidden_coordinate[None, None, None, :]
                == (2 * SWIGLU_HIDDEN + value_coordinate % SWIGLU_HIDDEN)[None, None, :, None]
            )
            gate_joint_gradient = tl.sum(
                tl.where(gate_match, grad_gate_contribution[:, :, :, None], 0.0),
                axis=2,
            )
            grad_preactivation = (value_joint_gradient + gate_joint_gradient).to(tl.bfloat16)
            if BLOCK_HEIGHT * BLOCK_LINES >= 16:
                path_output_weight_gradient = tl.dot(
                    tl.permute(active_grad_output, (0, 2, 1)),
                    gated_value,
                )
            else:
                path_output_weight_gradient = tl.sum(
                    active_grad_output[:, :, :, None] * gated_value[:, :, None, :],
                    axis=1,
                )
            output_weight_mask = (
                (mode_vector[:, None, None] < modes)
                & (output_coordinate[None, :, None] < _PACKED_PATH_OUTPUTS_TL)
                & (value_coordinate[None, None, :] < 2 * SWIGLU_HIDDEN)
            )
        else:
            preactivation_float = preactivation.to(tl.float32)
            sigmoid = tl.sigmoid(preactivation_float)
            hidden = (preactivation_float * sigmoid).to(tl.bfloat16)
            output_weight_offset = (
                mode_vector[:, None, None] * _PACKED_PATH_OUTPUTS_TL
                + output_coordinate[None, :, None]
            ) * packed_path_hidden + hidden_coordinate[None, None, :]
            active_output_weight = tl.load(
                path_output_weight + output_weight_offset,
                mask=(mode_vector[:, None, None] < modes)
                & (output_coordinate[None, :, None] < _PACKED_PATH_OUTPUTS_TL)
                & (hidden_coordinate[None, None, :] < packed_path_hidden),
                other=0.0,
            ).to(tl.bfloat16)
            grad_hidden = tl.dot(active_grad_output, active_output_weight).to(tl.float32)
            grad_preactivation = (
                grad_hidden * sigmoid * (1.0 + preactivation_float * (1.0 - sigmoid))
            ).to(tl.bfloat16)
            if BLOCK_HEIGHT * BLOCK_LINES >= 16:
                path_output_weight_gradient = tl.dot(
                    tl.permute(active_grad_output, (0, 2, 1)),
                    hidden,
                )
            else:
                path_output_weight_gradient = tl.sum(
                    active_grad_output[:, :, :, None] * hidden[:, :, None, :],
                    axis=1,
                )
            output_weight_mask = (
                (mode_vector[:, None, None] < modes)
                & (output_coordinate[None, :, None] < _PACKED_PATH_OUTPUTS_TL)
                & (hidden_coordinate[None, None, :] < packed_path_hidden)
            )
        tl.atomic_add(
            grad_path_output_weight + output_weight_offset,
            path_output_weight_gradient,
            mask=output_weight_mask,
        )
        tl.atomic_add(
            grad_path_output_bias
            + mode_vector[:, None] * _PACKED_PATH_OUTPUTS_TL
            + output_coordinate[None, :],
            tl.sum(active_grad_output.to(tl.float32), axis=1),
            mask=(mode_vector[:, None] < modes)
            & (output_coordinate[None, :] < _PACKED_PATH_OUTPUTS_TL),
        )
        if BLOCK_HEIGHT * BLOCK_LINES >= 16:
            path_input_weight_gradient = tl.dot(
                tl.permute(grad_preactivation, (0, 2, 1)),
                packed_path,
            )
        else:
            path_input_weight_gradient = tl.sum(
                grad_preactivation[:, :, :, None] * packed_path[:, :, None, :],
                axis=1,
            )
        path_input_weight_gradient_offset = (
            mode_vector[:, None, None] * packed_path_hidden + hidden_coordinate[None, :, None]
        ) * _PACKED_PATH_INPUTS_TL + path_coordinate[None, None, :]
        tl.atomic_add(
            grad_path_input_weight + path_input_weight_gradient_offset,
            path_input_weight_gradient,
            mask=(mode_vector[:, None, None] < modes)
            & (hidden_coordinate[None, :, None] < packed_path_hidden)
            & (path_coordinate[None, None, :] < _PACKED_PATH_INPUTS_TL),
        )
        tl.atomic_add(
            grad_path_input_bias
            + mode_vector[:, None] * packed_path_hidden
            + hidden_coordinate[None, :],
            tl.sum(grad_preactivation.to(tl.float32), axis=1),
            mask=(mode_vector[:, None] < modes) & (hidden_coordinate[None, :] < packed_path_hidden),
        )
        grad_packed_path = tl.dot(
            grad_preactivation,
            tl.permute(active_input_weight, (0, 2, 1)),
        ).to(tl.float32)

        grad_r0 += (
            _unpack_path_gradient(
                grad_packed_path,
                path_coordinate,
                0,
                BLOCK_HEIGHT,
                BLOCK_LINES,
                BLOCK_MODES,
            )
            * inverse_pa
        )
        grad_r1 += (
            _unpack_path_gradient(
                grad_packed_path,
                path_coordinate,
                1,
                BLOCK_HEIGHT,
                BLOCK_LINES,
                BLOCK_MODES,
            )
            * inverse_pb
        )
        grad_r2 += (
            _unpack_path_gradient(
                grad_packed_path,
                path_coordinate,
                2,
                BLOCK_HEIGHT,
                BLOCK_LINES,
                BLOCK_MODES,
            )
            * inverse_na
        )
        grad_r3 += (
            _unpack_path_gradient(
                grad_packed_path,
                path_coordinate,
                3,
                BLOCK_HEIGHT,
                BLOCK_LINES,
                BLOCK_MODES,
            )
            * inverse_nb
        )
        grad_i0 += (
            _unpack_path_gradient(
                grad_packed_path,
                path_coordinate,
                4,
                BLOCK_HEIGHT,
                BLOCK_LINES,
                BLOCK_MODES,
            )
            * inverse_pa
        )
        grad_i1 += (
            _unpack_path_gradient(
                grad_packed_path,
                path_coordinate,
                5,
                BLOCK_HEIGHT,
                BLOCK_LINES,
                BLOCK_MODES,
            )
            * inverse_pb
        )
        grad_i2 += (
            _unpack_path_gradient(
                grad_packed_path,
                path_coordinate,
                6,
                BLOCK_HEIGHT,
                BLOCK_LINES,
                BLOCK_MODES,
            )
            * inverse_na
        )
        grad_i3 += (
            _unpack_path_gradient(
                grad_packed_path,
                path_coordinate,
                7,
                BLOCK_HEIGHT,
                BLOCK_LINES,
                BLOCK_MODES,
            )
            * inverse_nb
        )

    if HAS_COARSE_GRAD:
        cell = (batch * (height // 2) + y // 2) * (width // 2) + x // 2
        if FULL_COARSE_GRAD:
            parity_x = x & 1
            parity_y = y & 1
            local0 = 2 * parity_x + parity_y
            local1 = 2 * (1 - parity_x) + parity_y
            local2 = 2 * parity_x + (1 - parity_y)
            local3 = 2 * (1 - parity_x) + (1 - parity_y)
            base0 = ((cell * 4) * 4 + local0) * modes + mode
            base1 = ((cell * 4 + 1) * 4 + local1) * modes + mode
            base2 = ((cell * 4 + 2) * 4 + local2) * modes + mode
            base3 = ((cell * 4 + 3) * 4 + local3) * modes + mode
            grad_r0 += tl.load(grad_coarse_real + base0, mask=active, other=0.0) * inverse_pa
            grad_i0 += tl.load(grad_coarse_imag + base0, mask=active, other=0.0) * inverse_pa
            grad_r1 += tl.load(grad_coarse_real + base1, mask=active, other=0.0) * inverse_pb
            grad_i1 += tl.load(grad_coarse_imag + base1, mask=active, other=0.0) * inverse_pb
            grad_r2 += tl.load(grad_coarse_real + base2, mask=active, other=0.0) * inverse_na
            grad_i2 += tl.load(grad_coarse_imag + base2, mask=active, other=0.0) * inverse_na
            grad_r3 += tl.load(grad_coarse_real + base3, mask=active, other=0.0) * inverse_nb
            grad_i3 += tl.load(grad_coarse_imag + base3, mask=active, other=0.0) * inverse_nb
        else:
            coarse_base = (cell * 4) * modes + mode
            positive_endpoint = active & ((y & 1) == 1)
            negative_endpoint = active & ((y & 1) == 0)
            mask0 = positive_endpoint & ((x & 1) == 1)
            mask1 = positive_endpoint & ((x & 1) == 0)
            mask2 = negative_endpoint & ((x & 1) == 1)
            mask3 = negative_endpoint & ((x & 1) == 0)
            grad_r0 += tl.load(grad_coarse_real + coarse_base, mask=mask0, other=0.0) * inverse_pa
            grad_i0 += tl.load(grad_coarse_imag + coarse_base, mask=mask0, other=0.0) * inverse_pa
            grad_r1 += (
                tl.load(grad_coarse_real + coarse_base + modes, mask=mask1, other=0.0) * inverse_pb
            )
            grad_i1 += (
                tl.load(grad_coarse_imag + coarse_base + modes, mask=mask1, other=0.0) * inverse_pb
            )
            grad_r2 += (
                tl.load(grad_coarse_real + coarse_base + 2 * modes, mask=mask2, other=0.0)
                * inverse_na
            )
            grad_i2 += (
                tl.load(grad_coarse_imag + coarse_base + 2 * modes, mask=mask2, other=0.0)
                * inverse_na
            )
            grad_r3 += (
                tl.load(grad_coarse_real + coarse_base + 3 * modes, mask=mask3, other=0.0)
                * inverse_nb
            )
            grad_i3 += (
                tl.load(grad_coarse_imag + coarse_base + 3 * modes, mask=mask3, other=0.0)
                * inverse_nb
            )
    positive_adjoint = tl.associative_scan(
        (
            tl.where(active, ar, 1.0),
            tl.where(active, -ai, 0.0),
            grad_r0,
            grad_i0,
            grad_r1,
            grad_i1,
        ),
        axis=0,
        combine_fn=_compose_complex_pair,
        reverse=True,
    )
    negative_adjoint = tl.associative_scan(
        (
            tl.where(active, ar, 1.0),
            tl.where(active, ai, 0.0),
            grad_r2,
            grad_i2,
            grad_r3,
            grad_i3,
        ),
        axis=0,
        combine_fn=_compose_complex_pair,
    )
    l0r, l0i = positive_adjoint[2], positive_adjoint[3]
    l1r, l1i = positive_adjoint[4], positive_adjoint[5]
    l2r, l2i = negative_adjoint[2], negative_adjoint[3]
    l3r, l3i = negative_adjoint[4], negative_adjoint[5]
    tl.store(
        grad_source_real_a + offset,
        gr * l0r + gi * l0i + gr * l2r - gi * l2i,
        mask=active,
    )
    tl.store(
        grad_source_imag_a + offset,
        -gi * l0r + gr * l0i + gi * l2r + gr * l2i,
        mask=active,
    )
    tl.store(
        grad_source_real_b + offset,
        gr * l1r + gi * l1i + gr * l3r - gi * l3i,
        mask=active,
    )
    tl.store(
        grad_source_imag_b + offset,
        -gi * l1r + gr * l1i + gi * l3r + gr * l3i,
        mask=active,
    )

    inverse_decay = 1.0 / decay_norm
    positive_drive_ar, positive_drive_ai = gr * sar - gi * sai, gr * sai + gi * sar
    positive_drive_br, positive_drive_bi = gr * sbr - gi * sbi, gr * sbi + gi * sbr
    previous_0r = (ar * (par - positive_drive_ar) + ai * (pai - positive_drive_ai)) * inverse_decay
    previous_0i = (ar * (pai - positive_drive_ai) - ai * (par - positive_drive_ar)) * inverse_decay
    previous_1r = (ar * (pbr - positive_drive_br) + ai * (pbi - positive_drive_bi)) * inverse_decay
    previous_1i = (ar * (pbi - positive_drive_bi) - ai * (pbr - positive_drive_br)) * inverse_decay
    negative_drive_ar, negative_drive_ai = gr * sar + gi * sai, gr * sai - gi * sar
    negative_drive_br, negative_drive_bi = gr * sbr + gi * sbi, gr * sbi - gi * sbr
    previous_2r = (ar * (nar - negative_drive_ar) - ai * (nai - negative_drive_ai)) * inverse_decay
    previous_2i = (ai * (nar - negative_drive_ar) + ar * (nai - negative_drive_ai)) * inverse_decay
    previous_3r = (ar * (nbr - negative_drive_br) - ai * (nbi - negative_drive_bi)) * inverse_decay
    previous_3i = (ai * (nbr - negative_drive_br) + ar * (nbi - negative_drive_bi)) * inverse_decay
    grad_dr = tl.sum(
        tl.sum(
            tl.where(
                active,
                l0r * previous_0r
                + l0i * previous_0i
                + l1r * previous_1r
                + l1i * previous_1i
                + l2r * previous_2r
                + l2i * previous_2i
                + l3r * previous_3r
                + l3i * previous_3i,
                0.0,
            ),
            axis=0,
        ),
        axis=0,
    )
    grad_di = tl.sum(
        tl.sum(
            tl.where(
                active,
                -l0r * previous_0i
                + l0i * previous_0r
                - l1r * previous_1i
                + l1i * previous_1r
                + l2r * previous_2i
                - l2i * previous_2r
                + l3r * previous_3i
                - l3i * previous_3r,
                0.0,
            ),
            axis=0,
        ),
        axis=0,
    )
    grad_gr = tl.sum(
        tl.sum(
            tl.where(
                active,
                l0r * sar
                + l0i * sai
                + l1r * sbr
                + l1i * sbi
                + l2r * sar
                + l2i * sai
                + l3r * sbr
                + l3i * sbi,
                0.0,
            ),
            axis=0,
        ),
        axis=0,
    )
    grad_gi = tl.sum(
        tl.sum(
            tl.where(
                active,
                -l0r * sai
                + l0i * sar
                - l1r * sbi
                + l1i * sbr
                + l2r * sai
                - l2i * sar
                + l3r * sbi
                - l3i * sbr,
                0.0,
            ),
            axis=0,
        ),
        axis=0,
    )
    coefficient_mode = mode_vector
    valid_coefficient_mode = mode_vector < modes
    tl.atomic_add(
        grad_decay_real + coefficient_mode,
        grad_dr,
        mask=valid_coefficient_mode,
    )
    tl.atomic_add(
        grad_decay_imag + coefficient_mode,
        grad_di,
        mask=valid_coefficient_mode,
    )
    tl.atomic_add(
        grad_gamma_real + coefficient_mode,
        grad_gr,
        mask=valid_coefficient_mode,
    )
    tl.atomic_add(
        grad_gamma_imag + coefficient_mode,
        grad_gi,
        mask=valid_coefficient_mode,
    )


def _launch_product_scan4_backward(  # noqa: C901, PLR0912
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    source_real_a: Tensor,
    source_imag_a: Tensor,
    source_real_b: Tensor,
    source_imag_b: Tensor,
    variance_x: Tensor,
    variance_y: Tensor,
    global_inverse_gain: Tensor,
    grad_coarse_real: Tensor,
    grad_coarse_imag: Tensor,
    descriptor_gradient_factor: Tensor,
    epsilon: float,
    gain_kind: int,
    *,
    emit_coarse: bool,
    full_coarse: bool = False,
    path_collapse: tuple[Tensor, Tensor, Tensor, Tensor] | None = None,
    path_swiglu: bool = False,
    has_descriptor_grad: bool = True,
    launch_name: str,
    source_gradient_buffers: tuple[Tensor, Tensor, Tensor, Tensor] | None = None,
) -> tuple[Tensor, ...]:
    pole = (decay_real, decay_imag, gamma_real, gamma_imag)
    source_a = (source_real_a, source_imag_a)
    source_b = (source_real_b, source_imag_b)
    validate_product_scan(
        *pole,
        *source_a,
        *source_b,
        variance_x,
        variance_y,
        epsilon,
        gain_kind,
        emit_coarse=emit_coarse,
    )
    validate_global_inverse_gain(
        global_inverse_gain,
        modes=source_real_a.shape[-1],
        gain_kind=gain_kind,
        reference=source_real_a,
    )
    if not decay_real.is_cuda:
        raise RuntimeError("the fused product adjoint kernel requires CUDA")
    batch, height, width, modes = source_real_a.shape
    if emit_coarse:
        expected_shape = (
            (batch, height // 2, width // 2, 4, 4, modes)
            if full_coarse
            else (batch, height // 2, width // 2, 4, modes)
        )
        if grad_coarse_real.shape != expected_shape:
            raise ValueError("fused product adjoint received an invalid coarse gradient")
        if grad_coarse_imag.shape != grad_coarse_real.shape:
            raise ValueError("fused product adjoint coarse gradients must match")
    if path_collapse is None:
        empty = source_real_a.new_empty((0,))
        grad_collapsed_real = empty
        grad_collapsed_imag = empty
        path_tensors = (empty, empty, empty, empty)
        path_gradients = tuple(torch.empty_like(value) for value in path_tensors)
        packed_path_hidden = 1
        swiglu_hidden = 0
        swiglu_hidden = 0
    else:
        expected_collapsed_shape = (batch, height, width, 1, modes)
        if grad_coarse_real.shape != expected_collapsed_shape:
            raise ValueError("fused scan-path adjoint received an invalid output gradient")
        if grad_coarse_imag.shape != grad_coarse_real.shape:
            raise ValueError("fused scan-path output gradients must match")
        grad_collapsed_real = grad_coarse_real.contiguous()
        grad_collapsed_imag = grad_coarse_imag.contiguous()
        path_tensors = tuple(value.contiguous() for value in path_collapse)
        path_gradients = tuple(
            torch.zeros_like(value, memory_format=torch.contiguous_format) for value in path_tensors
        )
        packed_path_hidden = path_tensors[0].shape[1]
        swiglu_hidden = packed_path_hidden // 3 if path_swiglu else 0
    if has_descriptor_grad:
        if descriptor_gradient_factor.shape != (batch, 4 * modes):
            raise ValueError("fused product adjoint received an invalid descriptor-gradient factor")
    elif descriptor_gradient_factor.numel() != 0:
        raise ValueError("descriptor-free product adjoint requires an empty gradient factor")
    coefficients = tuple(value.contiguous() for value in pole)
    sources = tuple(value.contiguous() for value in (*source_a, *source_b))
    coefficient_gradients = tuple(
        torch.zeros_like(value, memory_format=torch.contiguous_format) for value in coefficients
    )
    if source_gradient_buffers is None:
        source_gradients = tuple(
            torch.empty_like(value, memory_format=torch.contiguous_format) for value in sources
        )
    else:
        if any(
            gradient.shape != source.shape
            or gradient.dtype != source.dtype
            or gradient.device != source.device
            or not gradient.is_contiguous()
            for gradient, source in zip(source_gradient_buffers, sources, strict=True)
        ):
            raise ValueError("fused product adjoint reuse buffers must match contiguous sources")
        source_gradients = source_gradient_buffers
    line_count = batch * width
    backward_kernel = autotuned(
        _product_scan_coarse4_associative_backward_kernel,
        launch_name,
        key=(
            "height",
            "width",
            "line_count",
            "modes",
            "gain_kind",
        ),
        reset_to_zero=(
            "grad_decay_real",
            "grad_decay_imag",
            "grad_gamma_real",
            "grad_gamma_imag",
            *(
                (
                    "grad_path_input_weight",
                    "grad_path_input_bias",
                    "grad_path_output_weight",
                    "grad_path_output_bias",
                )
                if path_collapse is not None
                else ()
            ),
        ),
        scope=_scan_launch_scope(
            _product_scan_coarse4_associative_backward_kernel,
            source_real_a,
            gain_kind=gain_kind,
            full_coarse=full_coarse,
            path_collapse=path_collapse is not None,
        ),
    )
    wrap_triton(backward_kernel)[
        lambda metadata: (
            batch * triton.cdiv(width, metadata["BLOCK_LINES"]),
            triton.cdiv(modes, metadata["BLOCK_MODES"]),
        )
    ](
        *coefficients,
        *sources,
        variance_x.contiguous(),
        variance_y.contiguous(),
        global_inverse_gain.contiguous(),
        grad_coarse_real.contiguous(),
        grad_coarse_imag.contiguous(),
        grad_collapsed_real,
        grad_collapsed_imag,
        *path_tensors,
        *path_gradients,
        descriptor_gradient_factor.contiguous(),
        *coefficient_gradients,
        *source_gradients,
        height,
        width,
        line_count,
        modes,
        packed_path_hidden,
        epsilon=epsilon,
        gain_kind=gain_kind,
        HAS_COARSE_GRAD=emit_coarse,
        FULL_COARSE_GRAD=full_coarse,
        COLLAPSE_PATHS_GRAD=path_collapse is not None,
        PATH_SWIGLU=path_swiglu,
        SWIGLU_HIDDEN=swiglu_hidden,
        HAS_DESCRIPTOR_GRAD=has_descriptor_grad,
        BLOCK_HEIGHT=triton.next_power_of_2(height),
        BLOCK_PATH_HIDDEN=max(16, triton.next_power_of_2(packed_path_hidden)),
        BLOCK_SWIGLU_VALUE=(
            max(16, triton.next_power_of_2(2 * swiglu_hidden)) if path_swiglu else 16
        ),
    )
    base_gradients = (*coefficient_gradients, *source_gradients)
    if path_collapse is None:
        return base_gradients
    return *base_gradients, *path_gradients


@triton_op("lnet::pac_product_scan_coarse4_backward", mutates_args={})
def _product_scan_coarse4_backward_op(
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    source_real_a: Tensor,
    source_imag_a: Tensor,
    source_real_b: Tensor,
    source_imag_b: Tensor,
    variance_x: Tensor,
    variance_y: Tensor,
    global_inverse_gain: Tensor,
    grad_coarse_real: Tensor,
    grad_coarse_imag: Tensor,
    descriptor_gradient_factor: Tensor,
    epsilon: float,
    gain_kind: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    return cast(
        "tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]",
        _launch_product_scan4_backward(
            decay_real,
            decay_imag,
            gamma_real,
            gamma_imag,
            source_real_a,
            source_imag_a,
            source_real_b,
            source_imag_b,
            variance_x,
            variance_y,
            global_inverse_gain,
            grad_coarse_real,
            grad_coarse_imag,
            descriptor_gradient_factor,
            epsilon,
            gain_kind,
            emit_coarse=True,
            full_coarse=False,
            launch_name=BACKWARD_LAUNCH_NAME,
        ),
    )


@triton_op("lnet::pac_product_scan_full16_backward", mutates_args={})
def _product_scan_full16_backward_op(
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    source_real_a: Tensor,
    source_imag_a: Tensor,
    source_real_b: Tensor,
    source_imag_b: Tensor,
    variance_x: Tensor,
    variance_y: Tensor,
    global_inverse_gain: Tensor,
    grad_full_real: Tensor,
    grad_full_imag: Tensor,
    descriptor_gradient_factor: Tensor,
    epsilon: float,
    gain_kind: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    return cast(
        "tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]",
        _launch_product_scan4_backward(
            decay_real,
            decay_imag,
            gamma_real,
            gamma_imag,
            source_real_a,
            source_imag_a,
            source_real_b,
            source_imag_b,
            variance_x,
            variance_y,
            global_inverse_gain,
            grad_full_real,
            grad_full_imag,
            descriptor_gradient_factor,
            epsilon,
            gain_kind,
            emit_coarse=True,
            full_coarse=True,
            launch_name=FULL16_BACKWARD_LAUNCH_NAME,
        ),
    )


@triton_op("lnet::pac_product_scan_path_collapse_backward", mutates_args={})
def _product_scan_path_collapse_backward_op(  # pyright: ignore[reportUnusedFunction]
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    source_real_a: Tensor,
    source_imag_a: Tensor,
    source_real_b: Tensor,
    source_imag_b: Tensor,
    variance_x: Tensor,
    variance_y: Tensor,
    global_inverse_gain: Tensor,
    path_input_weight: Tensor,
    path_input_bias: Tensor,
    path_output_weight: Tensor,
    path_output_bias: Tensor,
    grad_collapsed_real: Tensor,
    grad_collapsed_imag: Tensor,
    descriptor_gradient_factor: Tensor,
    epsilon: float,
    gain_kind: int,
    path_swiglu: bool,
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
]:
    return cast(
        "PathCollapseGradients",
        _launch_product_scan4_backward(
            decay_real,
            decay_imag,
            gamma_real,
            gamma_imag,
            source_real_a,
            source_imag_a,
            source_real_b,
            source_imag_b,
            variance_x,
            variance_y,
            global_inverse_gain,
            grad_collapsed_real,
            grad_collapsed_imag,
            descriptor_gradient_factor,
            epsilon,
            gain_kind,
            emit_coarse=False,
            path_collapse=(
                path_input_weight,
                path_input_bias,
                path_output_weight,
                path_output_bias,
            ),
            path_swiglu=path_swiglu,
            has_descriptor_grad=False,
            launch_name=PATH_COLLAPSE_BACKWARD_LAUNCH_NAME,
        ),
    )


@triton_op("lnet::pac_product_scan_descriptor4_backward", mutates_args={})
def _product_scan_descriptor4_backward_op(
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    source_real_a: Tensor,
    source_imag_a: Tensor,
    source_real_b: Tensor,
    source_imag_b: Tensor,
    variance_x: Tensor,
    variance_y: Tensor,
    global_inverse_gain: Tensor,
    descriptor_gradient_factor: Tensor,
    epsilon: float,
    gain_kind: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    empty = source_real_a.new_empty((0,))
    return cast(
        "tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]",
        _launch_product_scan4_backward(
            decay_real,
            decay_imag,
            gamma_real,
            gamma_imag,
            source_real_a,
            source_imag_a,
            source_real_b,
            source_imag_b,
            variance_x,
            variance_y,
            global_inverse_gain,
            empty,
            empty,
            descriptor_gradient_factor,
            epsilon,
            gain_kind,
            emit_coarse=False,
            full_coarse=False,
            launch_name=DESCRIPTOR_BACKWARD_LAUNCH_NAME,
        ),
    )


def _launch_product_scan4_forward(
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    source_real_a: Tensor,
    source_imag_a: Tensor,
    source_real_b: Tensor,
    source_imag_b: Tensor,
    variance_x: Tensor,
    variance_y: Tensor,
    global_inverse_gain: Tensor,
    epsilon: float,
    gain_kind: int,
    *,
    emit_coarse: bool,
    full_coarse: bool = False,
    path_collapse: tuple[Tensor, Tensor, Tensor, Tensor] | None = None,
    path_swiglu: bool = False,
    emit_descriptor: bool = True,
    launch_name: str,
) -> FusedOutputs:
    pole = (decay_real, decay_imag, gamma_real, gamma_imag)
    source_a = (source_real_a, source_imag_a)
    source_b = (source_real_b, source_imag_b)
    validate_product_scan(
        *pole,
        *source_a,
        *source_b,
        variance_x,
        variance_y,
        epsilon,
        gain_kind,
        emit_coarse=emit_coarse,
    )
    validate_global_inverse_gain(
        global_inverse_gain,
        modes=source_real_a.shape[-1],
        gain_kind=gain_kind,
        reference=source_real_a,
    )
    sources = tuple(value.contiguous() for value in (*source_a, *source_b))
    coefficients = tuple(value.contiguous() for value in pole)
    batch, height, width, modes = source_real_a.shape
    coarse_shape = (
        (
            (batch, height // 2, width // 2, 4, 4, modes)
            if full_coarse
            else (batch, height // 2, width // 2, 4, modes)
        )
        if emit_coarse
        else (0,)
    )
    coarse_real = torch.empty(coarse_shape, dtype=source_real_a.dtype, device=source_real_a.device)
    coarse_imag = torch.empty_like(coarse_real)
    if path_collapse is None:
        empty = torch.empty((0,), dtype=torch.float32, device=source_real_a.device)
        path_tensors = (empty, empty, empty, empty)
        collapsed_real = torch.empty((0,), dtype=source_real_a.dtype, device=source_real_a.device)
        collapsed_imag = torch.empty_like(collapsed_real)
        packed_path_hidden = 1
        swiglu_hidden = 0
    else:
        path_input_weight, path_input_bias, path_output_weight, path_output_bias = (
            value.contiguous() for value in path_collapse
        )
        swiglu_hidden = path_input_weight.shape[1] // 3 if path_swiglu else 0
        expected_output_hidden = 2 * swiglu_hidden if path_swiglu else path_input_weight.shape[1]
        if (
            path_input_weight.ndim != 3
            or path_input_weight.shape[0] != modes
            or path_input_weight.shape[2] != _PACKED_PATH_INPUTS
            or path_input_bias.shape != path_input_weight.shape[:2]
            or (path_swiglu and (path_input_weight.shape[1] < 3 or path_input_weight.shape[1] % 3))
            or path_output_weight.shape != (modes, _PACKED_PATH_OUTPUTS, expected_output_hidden)
            or path_output_bias.shape != (modes, _PACKED_PATH_OUTPUTS)
        ):
            raise ValueError("scan-path collapse parameters have incompatible shapes")
        path_tensors = (
            path_input_weight,
            path_input_bias,
            path_output_weight,
            path_output_bias,
        )
        if any(
            value.device != source_real_a.device
            or value.dtype != torch.float32
            or not value.is_contiguous()
            for value in path_tensors
        ):
            raise TypeError("scan-path collapse parameters must be contiguous FP32 CUDA tensors")
        collapsed_shape = (batch, height, width, 1, modes)
        collapsed_real = torch.empty(
            collapsed_shape,
            dtype=source_real_a.dtype,
            device=source_real_a.device,
        )
        collapsed_imag = torch.empty_like(collapsed_real)
        packed_path_hidden = path_input_weight.shape[1]
    if emit_descriptor:
        descriptor_energy = torch.zeros(
            (batch, 4 * modes),
            dtype=torch.float32,
            device=source_real_a.device,
        )
        descriptor = torch.empty_like(descriptor_energy)
    else:
        descriptor_energy = torch.empty((0,), dtype=torch.float32, device=source_real_a.device)
        descriptor = torch.empty_like(descriptor_energy)
    forward_kernel = autotuned(
        _product_scan_coarse4_associative_forward_kernel,
        launch_name,
        key=(
            "height",
            "width",
            "line_count",
            "modes",
            "gain_kind",
        ),
        reset_to_zero=("descriptor_energy",),
        scope=_scan_launch_scope(
            _product_scan_coarse4_associative_forward_kernel,
            source_real_a,
            gain_kind=gain_kind,
            full_coarse=full_coarse,
            path_collapse=path_collapse is not None,
        ),
    )
    line_count = batch * width
    wrap_triton(forward_kernel)[
        lambda metadata: (
            batch * triton.cdiv(width, metadata["BLOCK_LINES"]),
            triton.cdiv(modes, metadata["BLOCK_MODES"]),
        )
    ](
        *coefficients,
        *sources,
        variance_x.contiguous(),
        variance_y.contiguous(),
        global_inverse_gain.contiguous(),
        coarse_real,
        coarse_imag,
        *path_tensors,
        collapsed_real,
        collapsed_imag,
        descriptor_energy,
        height,
        width,
        line_count,
        modes,
        packed_path_hidden,
        epsilon=epsilon,
        gain_kind=gain_kind,
        EMIT_COARSE=emit_coarse,
        FULL_COARSE=full_coarse,
        COLLAPSE_PATHS=path_collapse is not None,
        PATH_SWIGLU=path_swiglu,
        SWIGLU_HIDDEN=swiglu_hidden,
        EMIT_DESCRIPTOR=emit_descriptor,
        BLOCK_HEIGHT=triton.next_power_of_2(height),
        BLOCK_PATH_HIDDEN=max(16, triton.next_power_of_2(packed_path_hidden)),
        BLOCK_SWIGLU_VALUE=(
            max(16, triton.next_power_of_2(2 * swiglu_hidden)) if path_swiglu else 16
        ),
    )
    if emit_descriptor:
        finalize_kernel = autotuned(
            _finalize_descriptor_kernel,
            FINALIZE_LAUNCH_NAME,
            key=("spatial_size", "modes"),
            scope=make_launch_scope(
                _finalize_descriptor_kernel,
                descriptor_energy,
                shape={
                    "batch": batch,
                    "height": height,
                    "width": width,
                    "modes": modes,
                },
            ),
        )
        wrap_triton(finalize_kernel)[
            lambda metadata: (batch, triton.cdiv(modes, metadata["BLOCK_MODES"]))
        ](
            descriptor_energy,
            descriptor,
            height * width,
            modes,
        )
    if path_collapse is not None:
        return collapsed_real, collapsed_imag, descriptor
    return coarse_real, coarse_imag, descriptor


@triton_op("lnet::pac_product_scan_coarse4", mutates_args={})
def _product_scan_coarse4_op(
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    source_real_a: Tensor,
    source_imag_a: Tensor,
    source_real_b: Tensor,
    source_imag_b: Tensor,
    variance_x: Tensor,
    variance_y: Tensor,
    global_inverse_gain: Tensor,
    epsilon: float,
    gain_kind: int,
) -> FusedOutputs:
    pole = (decay_real, decay_imag, gamma_real, gamma_imag)
    source_a = (source_real_a, source_imag_a)
    source_b = (source_real_b, source_imag_b)
    if not source_real_a.is_cuda:
        return _product_scan_coarse4_from_tables_reference(
            pole,
            source_a,
            source_b,
            variance_x,
            variance_y,
            epsilon=epsilon,
            gain_normalization=gain_normalization(gain_kind),
        )
    return _launch_product_scan4_forward(
        *pole,
        *source_a,
        *source_b,
        variance_x,
        variance_y,
        global_inverse_gain,
        epsilon,
        gain_kind,
        emit_coarse=True,
        full_coarse=False,
        launch_name=FORWARD_LAUNCH_NAME,
    )


@triton_op("lnet::pac_product_scan_full16", mutates_args={})
def _product_scan_full16_op(
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    source_real_a: Tensor,
    source_imag_a: Tensor,
    source_real_b: Tensor,
    source_imag_b: Tensor,
    variance_x: Tensor,
    variance_y: Tensor,
    global_inverse_gain: Tensor,
    epsilon: float,
    gain_kind: int,
) -> FusedOutputs:
    pole = (decay_real, decay_imag, gamma_real, gamma_imag)
    source_a = (source_real_a, source_imag_a)
    source_b = (source_real_b, source_imag_b)
    if not source_real_a.is_cuda:
        return _product_scan_full16_from_tables_reference(
            pole,
            source_a,
            source_b,
            variance_x,
            variance_y,
            epsilon=epsilon,
            gain_normalization=gain_normalization(gain_kind),
        )
    return _launch_product_scan4_forward(
        *pole,
        *source_a,
        *source_b,
        variance_x,
        variance_y,
        global_inverse_gain,
        epsilon,
        gain_kind,
        emit_coarse=True,
        full_coarse=True,
        launch_name=FULL16_FORWARD_LAUNCH_NAME,
    )


@triton_op("lnet::pac_product_scan_path_collapse", mutates_args={})
def _product_scan_path_collapse_op(  # pyright: ignore[reportUnusedFunction]
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    source_real_a: Tensor,
    source_imag_a: Tensor,
    source_real_b: Tensor,
    source_imag_b: Tensor,
    variance_x: Tensor,
    variance_y: Tensor,
    global_inverse_gain: Tensor,
    path_input_weight: Tensor,
    path_input_bias: Tensor,
    path_output_weight: Tensor,
    path_output_bias: Tensor,
    epsilon: float,
    gain_kind: int,
    path_swiglu: bool,
) -> FusedOutputs:
    pole = (decay_real, decay_imag, gamma_real, gamma_imag)
    source_a = (source_real_a, source_imag_a)
    source_b = (source_real_b, source_imag_b)
    path = (path_input_weight, path_input_bias, path_output_weight, path_output_bias)
    if not source_real_a.is_cuda:
        full_real, full_imag, descriptor = _product_scan_full16_from_tables_reference(
            pole,
            source_a,
            source_b,
            variance_x,
            variance_y,
            epsilon=epsilon,
            gain_normalization=gain_normalization(gain_kind),
        )
        collapse_reference = (
            d4_grouped_cell_path_swiglu_reference
            if path_swiglu
            else d4_grouped_cell_path_collapse_reference
        )
        collapsed = collapse_reference(full_real, full_imag, *path)
        return collapsed[0], collapsed[1], descriptor
    return _launch_product_scan4_forward(
        *pole,
        *source_a,
        *source_b,
        variance_x,
        variance_y,
        global_inverse_gain,
        epsilon,
        gain_kind,
        emit_coarse=False,
        path_collapse=path,
        path_swiglu=path_swiglu,
        emit_descriptor=False,
        launch_name=PATH_COLLAPSE_FORWARD_LAUNCH_NAME,
    )


@triton_op("lnet::pac_product_scan_descriptor4", mutates_args={})
def _product_scan_descriptor4_op(
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    source_real_a: Tensor,
    source_imag_a: Tensor,
    source_real_b: Tensor,
    source_imag_b: Tensor,
    variance_x: Tensor,
    variance_y: Tensor,
    global_inverse_gain: Tensor,
    epsilon: float,
    gain_kind: int,
) -> Tensor:
    pole = (decay_real, decay_imag, gamma_real, gamma_imag)
    source_a = (source_real_a, source_imag_a)
    source_b = (source_real_b, source_imag_b)
    if not source_real_a.is_cuda:
        return _product_scan_descriptor4_from_tables_reference(
            pole,
            source_a,
            source_b,
            variance_x,
            variance_y,
            epsilon=epsilon,
            gain_normalization=gain_normalization(gain_kind),
        )
    return _launch_product_scan4_forward(
        *pole,
        *source_a,
        *source_b,
        variance_x,
        variance_y,
        global_inverse_gain,
        epsilon,
        gain_kind,
        emit_coarse=False,
        full_coarse=False,
        launch_name=DESCRIPTOR_FORWARD_LAUNCH_NAME,
    )[2]


def _setup_coarse_context(
    ctx: _AutogradContext,
    inputs: tuple[Tensor, ...],
    output: FusedOutputs,
) -> None:
    *tensors, epsilon, gain_kind = inputs
    ctx.epsilon = float(epsilon)
    ctx.gain_kind = int(gain_kind)
    ctx.save_for_backward(*tensors, output[2])


def _coarse_backward_impl(
    ctx: _AutogradContext,
    grad_coarse_real: Tensor | None,
    grad_coarse_imag: Tensor | None,
    grad_descriptor: Tensor | None,
    *,
    full_coarse: bool,
) -> tuple[Tensor | None, ...]:
    (
        decay_real,
        decay_imag,
        gamma_real,
        gamma_imag,
        source_real_a,
        source_imag_a,
        source_real_b,
        source_imag_b,
        variance_x,
        variance_y,
        global_inverse_gain,
        descriptor,
    ) = ctx.saved_tensors
    batch, height, width, modes = source_real_a.shape
    coarse_shape = (
        (batch, height // 2, width // 2, 4, 4, modes)
        if full_coarse
        else (batch, height // 2, width // 2, 4, modes)
    )
    if grad_coarse_real is None:
        grad_coarse_real = source_real_a.new_zeros(coarse_shape)
    if grad_coarse_imag is None:
        grad_coarse_imag = source_real_a.new_zeros(coarse_shape)
    if grad_descriptor is None:
        grad_descriptor = torch.zeros_like(descriptor)
    if not source_real_a.is_cuda:
        differentiable = tuple(
            value.detach().requires_grad_()
            for value in (
                decay_real,
                decay_imag,
                gamma_real,
                gamma_imag,
                source_real_a,
                source_imag_a,
                source_real_b,
                source_imag_b,
            )
        )
        with torch.enable_grad():
            reference = (
                _product_scan_full16_from_tables_reference
                if full_coarse
                else _product_scan_coarse4_from_tables_reference
            )
            outputs = reference(
                cast("tuple[Tensor, Tensor, Tensor, Tensor]", differentiable[:4]),
                cast("ComplexField", differentiable[4:6]),
                cast("ComplexField", differentiable[6:8]),
                variance_x,
                variance_y,
                epsilon=ctx.epsilon,
                gain_normalization=gain_normalization(ctx.gain_kind),
            )
            gradients = torch.autograd.grad(
                outputs,
                differentiable,
                (grad_coarse_real, grad_coarse_imag, grad_descriptor),
            )
        return *gradients, None, None, None, None, None
    # The factor is constant across every x-line.  Materialize it once per
    # batch/mode so the scan kernel does not repeat four exponentials for each
    # of the ``width`` programs; AOTInductor can fuse this expression into the
    # descriptor-gradient producer.
    descriptor_gradient_factor = (
        grad_descriptor * torch.exp(-descriptor) * (1.0 / (height * width))
    ).contiguous()
    backward = (
        _product_scan_full16_backward_op if full_coarse else _product_scan_coarse4_backward_op
    )
    gradients = backward(
        decay_real,
        decay_imag,
        gamma_real,
        gamma_imag,
        source_real_a,
        source_imag_a,
        source_real_b,
        source_imag_b,
        variance_x,
        variance_y,
        global_inverse_gain,
        grad_coarse_real.contiguous(),
        grad_coarse_imag.contiguous(),
        descriptor_gradient_factor,
        ctx.epsilon,
        ctx.gain_kind,
    )
    return *gradients, None, None, None, None, None


def _coarse_backward(
    ctx: _AutogradContext,
    grad_coarse_real: Tensor | None,
    grad_coarse_imag: Tensor | None,
    grad_descriptor: Tensor | None,
) -> tuple[Tensor | None, ...]:
    return _coarse_backward_impl(
        ctx,
        grad_coarse_real,
        grad_coarse_imag,
        grad_descriptor,
        full_coarse=False,
    )


def _full16_backward(
    ctx: _AutogradContext,
    grad_full_real: Tensor | None,
    grad_full_imag: Tensor | None,
    grad_descriptor: Tensor | None,
) -> tuple[Tensor | None, ...]:
    return _coarse_backward_impl(
        ctx,
        grad_full_real,
        grad_full_imag,
        grad_descriptor,
        full_coarse=True,
    )


torch.library.register_autograd(
    "lnet::pac_product_scan_coarse4",
    _coarse_backward,
    setup_context=_setup_coarse_context,
)
torch.library.register_autograd(
    "lnet::pac_product_scan_full16",
    _full16_backward,
    setup_context=_setup_coarse_context,
)


def _setup_descriptor_context(
    ctx: _AutogradContext,
    inputs: tuple[Tensor, ...],
    output: Tensor,
) -> None:
    *tensors, epsilon, gain_kind = inputs
    ctx.epsilon = float(epsilon)
    ctx.gain_kind = int(gain_kind)
    ctx.save_for_backward(*tensors, output)


def _descriptor_backward(
    ctx: _AutogradContext,
    grad_descriptor: Tensor | None,
) -> tuple[Tensor | None, ...]:
    (
        decay_real,
        decay_imag,
        gamma_real,
        gamma_imag,
        source_real_a,
        source_imag_a,
        source_real_b,
        source_imag_b,
        variance_x,
        variance_y,
        global_inverse_gain,
        descriptor,
    ) = ctx.saved_tensors
    if grad_descriptor is None:
        grad_descriptor = torch.zeros_like(descriptor)
    if not source_real_a.is_cuda:
        differentiable = tuple(
            value.detach().requires_grad_()
            for value in (
                decay_real,
                decay_imag,
                gamma_real,
                gamma_imag,
                source_real_a,
                source_imag_a,
                source_real_b,
                source_imag_b,
            )
        )
        with torch.enable_grad():
            output = _product_scan_descriptor4_from_tables_reference(
                cast("tuple[Tensor, Tensor, Tensor, Tensor]", differentiable[:4]),
                cast("ComplexField", differentiable[4:6]),
                cast("ComplexField", differentiable[6:8]),
                variance_x,
                variance_y,
                epsilon=ctx.epsilon,
                gain_normalization=gain_normalization(ctx.gain_kind),
            )
            gradients = torch.autograd.grad(output, differentiable, grad_descriptor)
        return *gradients, None, None, None, None, None
    height, width = source_real_a.shape[1:3]
    descriptor_gradient_factor = (
        grad_descriptor * torch.exp(-descriptor) * (1.0 / (height * width))
    ).contiguous()
    gradients = _product_scan_descriptor4_backward_op(
        decay_real,
        decay_imag,
        gamma_real,
        gamma_imag,
        source_real_a,
        source_imag_a,
        source_real_b,
        source_imag_b,
        variance_x,
        variance_y,
        global_inverse_gain,
        descriptor_gradient_factor,
        ctx.epsilon,
        ctx.gain_kind,
    )
    return *gradients, None, None, None, None, None


torch.library.register_autograd(
    "lnet::pac_product_scan_descriptor4",
    _descriptor_backward,
    setup_context=_setup_descriptor_context,
)


def pac_triton_product_scan_coarse4(
    pole_x: tuple[Tensor, Tensor, Tensor, Tensor],
    pole_y: tuple[Tensor, Tensor, Tensor, Tensor],
    source_a: ComplexField,
    source_b: ComplexField,
    *,
    epsilon: float = DEFAULT_EPSILON,
    gain_normalization: ProductGainNormalization = "pointwise",
) -> FusedOutputs:
    """Return coarse product paths and an exact full-grid product-Q descriptor."""
    gain_kind_value = gain_kind(gain_normalization)
    variance_x, variance_y, global_inverse_gain = static_product_scan_auxiliary(
        pole_x,
        pole_y,
        source_a[0],
        epsilon=epsilon,
        gain_kind=gain_kind_value,
    )
    return _product_scan_coarse4_op(
        *pole_y,
        *source_a,
        *source_b,
        variance_x,
        variance_y,
        global_inverse_gain,
        epsilon,
        gain_kind_value,
    )


def pac_triton_product_scan_full16(
    pole_x: tuple[Tensor, Tensor, Tensor, Tensor],
    pole_y: tuple[Tensor, Tensor, Tensor, Tensor],
    source_a: ComplexField,
    source_b: ComplexField,
    *,
    epsilon: float = DEFAULT_EPSILON,
    gain_normalization: ProductGainNormalization = "pointwise",
) -> FusedOutputs:
    """Return all direction-relative 2x2 product states and the descriptor."""
    gain_kind_value = gain_kind(gain_normalization)
    variance_x, variance_y, global_inverse_gain = static_product_scan_auxiliary(
        pole_x,
        pole_y,
        source_a[0],
        epsilon=epsilon,
        gain_kind=gain_kind_value,
    )
    return _product_scan_full16_op(
        *pole_y,
        *source_a,
        *source_b,
        variance_x,
        variance_y,
        global_inverse_gain,
        epsilon,
        gain_kind_value,
    )


def pac_triton_product_scan_descriptor4(
    pole_x: tuple[Tensor, Tensor, Tensor, Tensor],
    pole_y: tuple[Tensor, Tensor, Tensor, Tensor],
    source_a: ComplexField,
    source_b: ComplexField,
    *,
    epsilon: float = DEFAULT_EPSILON,
    gain_normalization: ProductGainNormalization = "pointwise",
) -> Tensor:
    """Return an exact full-grid product-Q descriptor without coarse states."""
    gain_kind_value = gain_kind(gain_normalization)
    variance_x, variance_y, global_inverse_gain = static_product_scan_auxiliary(
        pole_x,
        pole_y,
        source_a[0],
        epsilon=epsilon,
        gain_kind=gain_kind_value,
    )
    return _product_scan_descriptor4_op(
        *pole_y,
        *source_a,
        *source_b,
        variance_x,
        variance_y,
        global_inverse_gain,
        epsilon,
        gain_kind_value,
    )


__all__ = [
    "DEFAULT_EPSILON",
    "ProductGainNormalization",
    "pac_triton_product_scan_coarse4",
    "pac_triton_product_scan_descriptor4",
    "pac_triton_product_scan_full16",
    "product_scan_coarse4_reference",
    "product_scan_descriptor4_reference",
    "product_scan_full16_reference",
    "raw_product_descriptor_reference",
    "supports_pac_triton_product_scan_coarse4",
    "supports_pac_triton_product_scan_descriptor4",
]
