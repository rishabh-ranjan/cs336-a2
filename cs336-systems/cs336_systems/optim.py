from copy import deepcopy

import torch
from torch import optim
import torch.distributed as dist


class ShardedOptimizer(optim.Optimizer):
    def __init__(self, params, optimizer_cls, **kwargs):
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        self.rank_wise_params = [[] for _ in range(self.world_size)]
        self.rank_wise_param_size = torch.zeros(self.world_size, dtype=torch.int64)
        self.param_groups = []
        super().__init__(params, {})
        self.optimizer = optimizer_cls(self.param_groups, **kwargs)

    def step(self, closure=None, **kwargs):
        self.optimizer.step(closure, **kwargs)

        handles = []
        for rank, params in enumerate(self.rank_wise_params):
            for param in params:
                handle = dist.broadcast(param.detach(), rank, async_op=True)
                handles.append(handle)

        for handle in handles:
            handle.wait()

    def add_param_group(self, param_group):
        local_params = []
        for param in param_group["params"]:
            param_rank = self.rank_wise_param_size.argmin()
            param_size = param.numel() * param.element_size()
            self.rank_wise_params[param_rank].append(param)
            self.rank_wise_param_size[param_rank] += param_size
            if param_rank == self.rank:
                local_params.append(param)

        param_group["params"] = local_params
        self.param_groups.append(param_group)
