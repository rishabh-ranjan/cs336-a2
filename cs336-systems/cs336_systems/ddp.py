import torch
from torch import nn
import torch.distributed as dist


class DDP(nn.Module):
    def __init__(self, module: nn.Module, bucket_size_mb: float = 0, naive=False):
        super().__init__()
        self.module = module
        self.max_bucket_size = bucket_size_mb * 1e6

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

        self.bucket: list[torch.Tensor] = []
        self.bucket_size = 0

        self.flats: list[torch.Tensor] = []
        self.buckets: list[list[torch.Tensor]] = []

        def naive_hook(param):
            handles = []
            for param in self.module.parameters():
                param.grad /= dist.get_world_size()
                handle = dist.all_reduce(
                    param.grad, op=dist.ReduceOp.SUM, async_op=True
                )
                handles.append(handle)

            for handle in handles:
                handle.wait()

        def hook(param):
            t = param.grad
            t_size = t.numel() * t.element_size()
            if self.bucket_size + t_size > self.max_bucket_size:
                if self.bucket_size > 0:
                    self.flush()
            self.bucket.append(t)
            self.bucket_size += t_size

        if naive:
            for param in self.parameters():
                if param.requires_grad:
                    param.register_post_accumulate_grad_hook(naive_hook)
                    break
        else:
            for param in self.parameters():
                if param.requires_grad:
                    param.register_post_accumulate_grad_hook(hook)

    def flush(self):
        # for t in self.bucket:
        #     t /= dist.get_world_size()
        #     handle = dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)
        #     self.handles.append(handle)

        flat = torch._utils._flatten_dense_tensors(self.bucket)
        flat /= dist.get_world_size()
        handle = dist.all_reduce(flat, op=dist.ReduceOp.SUM, async_op=True)
        self.handles.append(handle)

        self.flats.append(flat)
        self.buckets.append(self.bucket)

        self.bucket = []
        self.bucket_size = 0

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        if self.bucket_size > 0:
            self.flush()

        for handle in self.handles:
            handle.wait()

        self.handles = []

        for flat, bucket in zip(self.flats, self.buckets):
            sync_bucket = torch._utils._unflatten_dense_tensors(flat, bucket)
            for sync_t, t in zip(sync_bucket, bucket):
                t.copy_(sync_t)

        self.flats = []
        self.buckets = []
