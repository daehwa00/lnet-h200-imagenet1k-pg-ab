"""Spatial transitions produce spatial states only, never global descriptors."""
from typing import List
import torch
from torch import Tensor
from torch.library import triton_op
from .pac_kernel_launch_config import register_default, registered_default, registered_candidates
from .pac_product_scan_contracts import DEFAULT_EPSILON, gain_kind
from .pac_product_scan_normalization import static_product_scan_auxiliary
from .pac_triton_bidirectional_product_scan import pac_triton_bidirectional_product_scan
from . import pac_triton_product_scan_coarse4 as kernels


for suffix in ('coarse4', 'full16'):
    for direction in ('forward', 'backward'):
        original=f'product_scan_{suffix}_{direction}'
        register_default(f'{original}_spatial',registered_default(original),
                         candidates=registered_candidates(original))


@triton_op('lnet::spatial_product_forward',mutates_args={})
def _forward(values: List[Tensor], epsilon: float, kind: int, full: bool) -> tuple[Tensor,Tensor]:
    output=kernels._launch_product_scan4_forward(*values,epsilon,kind,
        emit_coarse=True,full_coarse=full,emit_descriptor=False,
        launch_name=f'product_scan_{"full16" if full else "coarse4"}_forward_spatial')
    return output[0],output[1]


@triton_op('lnet::spatial_product_backward',mutates_args={})
def _backward_op(values: List[Tensor], gr: Tensor, gi: Tensor,
                 epsilon: float, kind: int, full: bool) -> List[Tensor]:
    empty=values[4].new_empty((0,))
    gradients=kernels._launch_product_scan4_backward(*values,gr.contiguous(),gi.contiguous(),
        empty,epsilon,kind,emit_coarse=True,full_coarse=full,has_descriptor_grad=False,
        launch_name=f'product_scan_{"full16" if full else "coarse4"}_backward_spatial')
    return list(gradients)


def _setup(ctx, inputs, output):
    values,ctx.epsilon,ctx.kind,ctx.full=inputs
    ctx.save_for_backward(*values)


def _backward(ctx,gr,gi):
    values=list(ctx.saved_tensors)
    b,h,w,p=values[4].shape
    shape=(b,h//2,w//2,4,4,p) if ctx.full else (b,h//2,w//2,4,p)
    if gr is None:gr=values[4].new_zeros(shape)
    if gi is None:gi=values[4].new_zeros(shape)
    grads=_backward_op(values,gr,gi,ctx.epsilon,ctx.kind,ctx.full)
    return grads+[None,None,None],None,None,None


torch.library.register_autograd('lnet::spatial_product_forward',_backward,setup_context=_setup)


def spatial_product_scan(pole_x,pole_y,source,*,epilogue,gain_normalization,
                         memory_policy='retain',epsilon=DEFAULT_EPSILON):
    if epilogue not in ('coarse','full16'):
        raise ValueError('Spatial-only execution requires a spatial epilogue')
    if memory_policy not in ('retain','recompute'):raise ValueError('Unknown memory policy')
    if memory_policy=='recompute' and torch.is_grad_enabled():
        from torch.utils.checkpoint import checkpoint
        def replay(*values):
            return spatial_product_scan(tuple(values[:4]),tuple(values[4:8]),tuple(values[8:]),
                epilogue=epilogue,gain_normalization=gain_normalization,
                memory_policy='retain',epsilon=epsilon)
        return checkpoint(replay,*pole_x,*pole_y,*source,use_reentrant=False)
    horizontal=pac_triton_bidirectional_product_scan(pole_x,source)
    kind=gain_kind(gain_normalization)
    aux=static_product_scan_auxiliary(pole_x,pole_y,source[0],epsilon=epsilon,gain_kind=kind)
    if not source[0].is_cuda:
        from .pac_product_scan_reference import (_product_scans_from_tables_reference,
                                                _product_coarse4_reference,_product_full16_reference)
        scans=_product_scans_from_tables_reference(pole_y,horizontal[:2],horizontal[2:],aux[0])
        select=_product_full16_reference if epilogue=='full16' else _product_coarse4_reference
        return select(scans,epsilon=epsilon,gain_normalization=gain_normalization)
    return _forward(list(pole_y)+list(horizontal)+list(aux),epsilon,kind,epilogue=='full16')
