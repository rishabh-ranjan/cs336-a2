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


@triton.jit
def _rms_norm_bwd(
    grad_x_ptr: tl.pointer_type,
    part_grad_g_ptr: tl.pointer_type,
    x_ptr: tl.pointer_type,
    g_ptr: tl.pointer_type,
    grad_out_ptr: tl.pointer_type,
    h: tl.uint32,
    BLOCK_SIZE: tl.constexpr,
):
    eps = 1e-5

    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < h

    x_ptrs = x_ptr + row_idx * h + offsets
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    g_ptrs = g_ptr + offsets
    g = tl.load(g_ptrs, mask=mask, other=0.0)

    grad_out_ptrs = grad_out_ptr + row_idx * h + offsets
    grad_out = tl.load(grad_out_ptrs, mask=mask, other=0.0)

    rsq = tl.sum(x * x) / h + eps
    r = tl.sqrt(rsq)
    rcu = r * rsq

    part_grad_g = grad_out * x / r
    part_grad_g_ptrs = part_grad_g_ptr + row_idx * h + offsets
    tl.store(part_grad_g_ptrs, part_grad_g, mask=mask)

    s = tl.sum(grad_out * x * g)
    grad_x = -(x * s) / (h * rcu)
    grad_x += grad_out * g / r
    grad_x_ptrs = grad_x_ptr + row_idx * h + offsets
    tl.store(grad_x_ptrs, grad_x, mask=mask)


class RMSNormTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, g):
        ctx.save_for_backward(x, g)

        h = x.size(-1)

        assert g.dim() == 1 and g.size(0) == h, "dimension mismatch"
        assert x.is_cuda and g.is_cuda, "expected CUDA tensors"
        assert (
            x.is_contiguous()
        ), "our pointer arithmetic will assume contiguous tensors"

        ctx.BLOCK_SIZE = triton.next_power_of_2(h)
        out = torch.empty_like(x)

        grid = (x.view(-1, h).size(0),)
        _rms_norm_fwd[grid](out, x, g, h, ctx.BLOCK_SIZE)

        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, g = ctx.saved_tensors

        h = x.size(-1)

        assert grad_out.is_cuda, "expected CUDA tensors"
        assert (
            grad_out.is_contiguous()
        ), "our pointer arithmetic will assume contiguous tensors"

        grad_x = torch.empty_like(x)
        part_grad_g = torch.empty_like(x)

        grid = (x.view(-1, h).size(0),)
        _rms_norm_bwd[grid](grad_x, part_grad_g, x, g, grad_out, h, ctx.BLOCK_SIZE)

        grad_g = part_grad_g.view(-1, h).sum(0)

        return grad_x, grad_g


class TritonRMSNorm(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.g = torch.nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        return RMSNormTriton.apply(x, self.g)
