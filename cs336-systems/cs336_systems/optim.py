import torch
from torch import optim
import torch.distributed as dist


class ShardedOptimizer(optim.Optimizer):
    def __init__(self, params, optimizer_cls, **kwargs):
        super().__init__(params, {})

        self.rank_wise_param_size = torch.zeros(dist.get_world_size())

        self.optimizer = optimizer_cls([], **kwargs)

    def step(self, closure, **kwargs):
        return self.optimizer.step(closure, **kwargs)

    def add_param_group(self, param_group):
        global_params = param_group["params"]

        local_params = []
        for param in global_params:
            param_rank = rank_wise_param_size.argmin()
            param_size = param.numel() * param.element_size()
            rank_wise_param_size[param_rank] += param_size
            if param_rank == dist.get_rank():
                local_params.append(param)

        param_group["params"] = local_params

        return self.optimizer.add_param_group(param_group)
