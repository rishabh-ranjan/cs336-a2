import torch
from torch import nn
import torch.distributed as dist


class DDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

        handles = []
        for param in self.parameters():
            handle = dist.broadcast(param.data, 0, async_op=True)
            handles.append(handle)

        for handle in handles:
            handle.wait()

        self.handles = []

        def ddp_hook(param):
            handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
            self.handles.append(handle)

        for param in self.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(ddp_hook)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()

        self.handles.clear()

        world_size = dist.get_world_size()
        for param in self.parameters():
            if param.requires_grad:
                param.grad /= world_size
