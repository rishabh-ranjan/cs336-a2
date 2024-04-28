import torch
import triton
import triton.language as tl


def bw_g(grad_out, x, g):
    eps = 1e-5
    rms = torch.sqrt((x**2).mean(-1, keepdim=True) + eps)
    return (grad_out * x / rms).view(-1, x.size(-1)).sum(0)


def bw_x(grad_out, x, g):
    eps = 1e-5
    h = x.size(-1)
    rsq = (x**2).mean(-1) + eps
    r = rsq.sqrt()
    neq = -(x[..., :, None] @ x[..., None, :]) / h
    diag = torch.zeros(h, h).fill_diagonal_(1)
    nr = neq + rsq[..., None, None] * diag
    rcu = r * rsq
    m = nr * g[..., :, None] / rcu[..., None, None]
    return (grad_out[..., None, :] @ m)[..., 0, :]


class RMSNormPyTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)
        eps = 1e-5
        rms = ((x**2).mean(-1, keepdim=True) + eps).sqrt()
        return x / rms * w

    @staticmethod
    def backward(ctx, grad_out):
        x, w = ctx.saved_tensors
        return bw_x(grad_out, x, w), bw_g(grad_out, x, w)


@triton.jit
def _rms_norm_fwd(
    out_ptr: tl.pointer_type,
    x_ptr: tl.pointer_type,
    w_ptr: tl.pointer_type,
    h: tl.uint32,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start_ptr = x_ptr + row_idx * h
    offsets = tl.arange(0, BLOCK_SIZE)
    x_ptrs = row_start_ptr + offsets
    w_ptrs = w_ptr + offsets
    mask = offsets < h
    row = tl.load(x_ptrs, mask=mask, other=0.0)
    w = tl.load(w_ptrs, mask=mask, other=0.0)
    eps = 1e-5
    rms = tl.sqrt(tl.sum(row * row) / h + eps)
    out = row / rms * w
    out_ptrs = out_ptr + row_idx * h + offsets
    tl.store(out_ptrs, out, mask=mask)


class RMSNormTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)

        h = x.shape[-1]

        assert w.dim() == 1 and w.size(0) == h, "dimension mismatch"
        assert x.is_cuda and w.is_cuda, "expected CUDA tensors"
        assert (
            x.is_contiguous()
        ), "our pointer arithmetic will assume contiguous tensors"

        ctx.BLOCK_SIZE = triton.next_power_of_2(h)
        out = torch.empty_like(x)

        grid = (x.view(-1, h).size(0),)
        _rms_norm_fwd[grid](out, x, w, h, ctx.BLOCK_SIZE)

        return out
