"""Fused sequential vertical recurrence; unchanged spatial extent and FP32 state."""
import torch
import triton
import triton.language as tl


@triton.jit
def _forward(AR, AI, GR, GI, XR, XI, XV, YR, YI, YV,
             H: tl.constexpr, W: tl.constexpr, P: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = lane < N
    mode = lane % P
    batch = lane // (W * P)
    column_mode = lane % (W * P)
    ar = tl.load(AR + mode, valid, 0)
    ai = tl.load(AI + mode, valid, 0)
    gr = tl.load(GR + mode, valid, 0)
    gi = tl.load(GI + mode, valid, 0)
    sr = tl.full((BLOCK,), 0, tl.float32)
    si = tl.full((BLOCK,), 0, tl.float32)
    variance = tl.full((BLOCK,), 0, tl.float32)
    vd = ar * ar + ai * ai
    vg = gr * gr + gi * gi
    for row in range(H):
        offset = batch * H * W * P + row * W * P + column_mode
        xr = tl.load(XR + offset, valid, 0)
        xi = tl.load(XI + offset, valid, 0)
        xv = tl.load(XV + offset, valid, 0)
        nr = ((ar * sr - ai * si) + gr * xr) - gi * xi
        ni = ((ar * si + ai * sr) + gr * xi) + gi * xr
        variance = vd * variance + vg * xv
        tl.store(YR + offset, nr, valid)
        tl.store(YI + offset, ni, valid)
        tl.store(YV + offset, variance, valid)
        sr, si = nr, ni


@triton.jit
def _backward(AR, AI, GR, GI, XR, XI, YR, YI, DYR, DYI, DXR, DXI, PARTIAL,
              H: tl.constexpr, W: tl.constexpr, P: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = lane < N
    mode = lane % P
    batch = lane // (W * P)
    column_mode = lane % (W * P)
    ar = tl.load(AR + mode, valid, 0)
    ai = tl.load(AI + mode, valid, 0)
    gr = tl.load(GR + mode, valid, 0)
    gi = tl.load(GI + mode, valid, 0)
    qr = tl.full((BLOCK,), 0, tl.float32)
    qi = tl.full((BLOCK,), 0, tl.float32)
    dar = tl.full((BLOCK,), 0, tl.float32)
    dai = tl.full((BLOCK,), 0, tl.float32)
    dgr = tl.full((BLOCK,), 0, tl.float32)
    dgi = tl.full((BLOCK,), 0, tl.float32)
    for reverse_row in range(H):
        row = H - 1 - reverse_row
        offset = batch * H * W * P + row * W * P + column_mode
        qr = qr + tl.load(DYR + offset, valid, 0)
        qi = qi + tl.load(DYI + offset, valid, 0)
        xr = tl.load(XR + offset, valid, 0)
        xi = tl.load(XI + offset, valid, 0)
        pr = tl.load(YR + offset - W * P, valid & (row > 0), 0)
        pi = tl.load(YI + offset - W * P, valid & (row > 0), 0)
        dar = dar + (qr * pr + qi * pi)
        dai = dai + (qi * pr - qr * pi)
        dgr = dgr + (qr * xr + qi * xi)
        dgi = dgi + (qi * xr - qr * xi)
        tl.store(DXR + offset, gr * qr + gi * qi, valid)
        tl.store(DXI + offset, gr * qi - gi * qr, valid)
        nr = ar * qr + ai * qi
        ni = ar * qi - ai * qr
        qr, qi = nr, ni
    tl.store(PARTIAL + lane, dar, valid)
    tl.store(PARTIAL + N + lane, dai, valid)
    tl.store(PARTIAL + 2 * N + lane, dgr, valid)
    tl.store(PARTIAL + 3 * N + lane, dgi, valid)


class _Vertical(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ar, ai, gr, gi, xr, xi, xv):
        b, h, w, p = xr.shape
        yr, yi, yv = (torch.empty_like(xr) for _ in range(3))
        n = b * w * p
        _forward[(triton.cdiv(n, 128),)](ar, ai, gr, gi, xr, xi, xv, yr, yi, yv,
            h, w, p, n, 128, num_warps=4, enable_fp_fusion=False)
        ctx.save_for_backward(ar, ai, gr, gi, xr, xi, yr, yi)
        ctx.mark_non_differentiable(yv)
        return yr, yi, yv

    @staticmethod
    def backward(ctx, dyr, dyi, _dyv):
        ar, ai, gr, gi, xr, xi, yr, yi = ctx.saved_tensors
        b, h, w, p = xr.shape
        dyr = torch.zeros_like(yr) if dyr is None else dyr.contiguous()
        dyi = torch.zeros_like(yi) if dyi is None else dyi.contiguous()
        dxr, dxi = torch.empty_like(xr), torch.empty_like(xi)
        partial = torch.empty((4, b, w, p), device=xr.device, dtype=torch.float32)
        n = b * w * p
        _backward[(triton.cdiv(n, 128),)](ar, ai, gr, gi, xr, xi, yr, yi, dyr, dyi,
            dxr, dxi, partial, h, w, p, n, 128, num_warps=4, enable_fp_fusion=False)
        grads = partial.sum(dim=(1, 2))
        return grads[0], grads[1], grads[2], grads[3], dxr, dxi, None


def vertical_product_scan(pole, source, source_variance, *, reverse):
    xr, xi = source
    dtype = xr.dtype
    ar, ai, gr, gi = (v.reshape(-1).float().contiguous() for v in pole)
    xr, xi, xv = xr.float(), xi.float(), source_variance.detach().float()
    if reverse:
        xr, xi, xv = xr.flip(1), xi.flip(1), xv.flip(1)
        ai, gi = -ai, -gi
    yr, yi, yv = _Vertical.apply(ar, ai, gr, gi, xr.contiguous(), xi.contiguous(), xv.contiguous())
    if reverse:
        yr, yi, yv = yr.flip(1), yi.flip(1), yv.flip(1)
    return yr.to(dtype), yi.to(dtype), yv
