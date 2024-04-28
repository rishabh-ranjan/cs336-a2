import torch

class RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)
        eps = 1e-5
        rms = ((x ** 2).mean(-1, keepdim=True) + eps).sqrt()
        return x / rms * w
