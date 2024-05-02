import torch
from torch import optim
import torch.distributed as dist


class ShardedOptimizer(optim.Optimizer):
    def __init__(self, params, optimizer_cls, **kwargs):
        self.rank_wise_params = [[] for _ in range(dist.get_world_size())]
        self.rank_wise_param_size = torch.zeros(dist.get_world_size())
        self.local_param_groups = []
        super().__init__(params, {})
        self.optimizer = optimizer_cls(self.local_param_groups, **kwargs)

    def step(self, closure=None, **kwargs):
        self.optimizer.step(closure, **kwargs)

        rank_wise_flat = []
        handles = []
        for rank, params in enumerate(self.rank_wise_params):
            flat = torch._utils._flatten_dense_tensors(params)
            rank_wise_flat.append(flat)
            handle = dist.broadcast(flat, rank, async_op=True)
            handles.append(handle)

        for handle in handles:
            handle.wait()

        for rank, params in enumerate(self.rank_wise_params):
            for flat, params in zip(rank_wise_flat, self.rank_wise_params):
                unflats = torch._utils._unflatten_dense_tensors(flat, params)
                for param, unflat in zip(params, unflats):
                    requires_grad = param.requires_grad
                    param.requires_grad_(False)
                    param.copy_(unflat)
                    param.requires_grad_(requires_grad)

    def add_param_group(self, param_group):
        global_params = param_group["params"]

        local_params = []
        for param in global_params:
            param_rank = self.rank_wise_param_size.argmin()
            param_size = param.numel() * param.element_size()
            self.rank_wise_params[param_rank].append(param)
            self.rank_wise_param_size[param_rank] += param_size
            if param_rank == dist.get_rank():
                local_params.append(param)

        param_group["params"] = local_params
        self.local_param_groups.append(param_group)
