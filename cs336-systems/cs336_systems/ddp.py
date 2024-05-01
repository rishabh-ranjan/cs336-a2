import torch
from torch import nn
import torch.distributed as dist


class DDP(nn.Module):
    def __init__(self, module: nn.Module, bucket_size_mb: float | None = None):
        super().__init__()
        self.module = module

        requires_grad: dict[str, bool] = {}
        for param_name, param in self.module.named_parameters():
            requires_grad[param_name] = param.requires_grad

        module.requires_grad_(False)

        handles = []
        for param in self.parameters():
            handle = dist.broadcast(param, 0, async_op=True)
            handles.append(handle)

        for handle in handles:
            handle.wait()

        for param_name, param in self.module.named_parameters():
            param.requires_grad_(requires_grad[param_name])

        self.handles = []
        world_size = dist.get_world_size()

        def hook(param):
            param.grad /= world_size
            handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
            self.handles.append(handle)

        for param in self.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(hook)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()

        self.handles = []
