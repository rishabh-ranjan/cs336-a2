import torch
from torch import nn
import torch.distributed as dist


class DDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

        self.grad_params = []
        for param in module.parameters():
            if param.requires_grad:
                self.grad_params.append(param)

        # for param in self.grad_params:
        #     param.requires_grad = False

        # handles = []
        # for param in self.grad_params:
        #     handle = dist.broadcast(param.data, 0, async_op=True)
        #     handles.append(handle)

        # for handle in handles:
        #     handle.wait()

        # for param in self.grad_params:
        #     dist.broadcast(param.data, 0)

        # for param in self.grad_params:
        #     param.requires_grad = True

        # self.handles = []

        # def ddp_hook(param):
        #     print("ddp_hook")
        #     handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
        #     self.handles.append(handle)

        # for param in self.grad_params:
        #     param.register_post_accumulate_grad_hook(ddp_hook)

        print("DDP initialized")

    def forward(self, *args, **kwargs):
        print("DDP forward")
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        print("finish_gradient_synchronization")
        for handle in self.handles:
            handle.wait()

        self.handles.clear()

        world_size = dist.get_world_size()
        for param in self.grad_params:
            param.grad /= world_size
