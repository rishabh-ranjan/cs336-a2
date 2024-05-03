from copy import deepcopy

import torch
from torch import optim
import torch.distributed as dist


class ShardedOptimizer(optim.Optimizer):
    def __init__(self, params, optimizer_cls, **kwargs):
        # self.rank_wise_params = [[] for _ in range(dist.get_world_size())]
        # self.rank_wise_param_size = torch.zeros(
        #     dist.get_world_size(), dtype=torch.int64
        # )
        self.local_param_groups = []
        params = list(params)
        super().__init__(params, {})
        # self.optimizer = optimizer_cls(self.local_param_groups, **kwargs)
        self.optimizer = optimizer_cls(params, **kwargs)

    def step(self, closure=None, **kwargs):
        print(f"step")
        self.optimizer.step(closure, **kwargs)

        # handles = []
        # for rank, params in enumerate(self.rank_wise_params):
        #     for param in params:
        #         handle = dist.broadcast(param.detach(), rank, async_op=True)
        #         handles.append(handle)

        # for handle in handles:
        #     handle.wait()

    def add_param_group(self, param_group):
        self.local_param_groups.append(param_group)
        pass
        # print(f"param_group has {len(param_group['params'])} parameters")
        # local_params = []
        # for param in param_group["params"]:
        #     param_rank = self.rank_wise_param_size.argmin()
        #     param_size = param.numel() * param.element_size()
        #     self.rank_wise_params[param_rank].append(param)
        #     self.rank_wise_param_size[param_rank] += param_size
        #     if param_rank == dist.get_rank():
        #         local_params.append(param)
        # print(f"local_params has {len(local_params)} parameters")

        # param_group["params"] = local_params
        # self.local_param_groups.append(param_group)
        # print(
        #     f"self.local_param_groups[-1]['params'] has {len(self.local_param_groups[-1]['params'])} parameters"
        # )
