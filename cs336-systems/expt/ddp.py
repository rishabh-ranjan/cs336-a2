"""Problem (naive_ddp)"""

import argparse
import os
import time

import torch
import torch.distributed as dist

torch.manual_seed(0)

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

import expt

CONFIG = {
    "small": {
        "d_model": 768,
        "d_ff": 3072,
        "num_layers": 12,
        "num_heads": 12,
    },
    "medium": {
        "d_model": 1024,
        "d_ff": 4096,
        "num_layers": 24,
        "num_heads": 16,
    },
    "large": {
        "d_model": 1280,
        "d_ff": 5120,
        "num_layers": 36,
        "num_heads": 20,
    },
    "xl": {
        "d_model": 1600,
        "d_ff": 6400,
        "num_layers": 48,
        "num_heads": 25,
    },
    "2.7b": {
        "d_model": 2560,
        "d_ff": 10240,
        "num_layers": 32,
        "num_heads": 32,
    },
}


@expt.run
def main(store, **kwargs):
    if kwargs["ddp"]:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])

        dist.init_process_group(
            backend=kwargs["backend"],
            rank=rank,
            world_size=world_size,
        )
    else:
        rank = 0
        local_rank = 0
        world_size = 1

    store["info"] = {
        **store["info"],
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
    }

    if kwargs["device"] == "cuda":
        device = f"cuda:{local_rank}"
    elif kwargs["device"] == "cpu":
        device = "cpu"

    if kwargs["lm_size"]:
        kwargs.update(CONFIG[kwargs["lm_size"]])

    net = BasicsTransformerLM(
        vocab_size=kwargs["vocab_size"],
        context_length=kwargs["context_length"],
        d_model=kwargs["d_model"],
        d_ff=kwargs["d_ff"],
        num_layers=kwargs["num_layers"],
        num_heads=kwargs["num_heads"],
        attn_pdrop=kwargs["attn_pdrop"],
        residual_pdrop=kwargs["residual_pdrop"],
    )
    net = net.to(device)

    # broadcast init params
    if kwargs["ddp"]:
        handles = []
        with torch.no_grad():
            for param in net.parameters():
                handle = dist.broadcast(param, 0, async_op=True)
                handles.append(handle)

        for handle in handles:
            handle.wait()

    opt = AdamW(net.parameters())

    assert kwargs["batch_size"] % world_size == 0
    local_batch_size = kwargs["batch_size"] // world_size

    step_times = []
    comm_times = []
    for _ in range(kwargs["num_train_steps"]):
        torch.cuda.synchronize()
        step_tic = time.perf_counter()

        net.train()
        batch = torch.randint(
            kwargs["vocab_size"], (local_batch_size, kwargs["context_length"] + 1)
        )
        batch = batch.to(device)
        x = batch[:, :-1]
        y = batch[:, 1:]

        logit = net(x)
        loss = cross_entropy(logit, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()

        # all reduce grads
        if kwargs["ddp"]:
            torch.cuda.synchronize()
            comm_tic = time.perf_counter()

            handles = []
            for param in net.parameters():
                handle = dist.all_reduce(
                    param.grad, op=dist.ReduceOp.AVG, async_op=True
                )
                handles.append(handle)

            for handle in handles:
                handle.wait()

            torch.cuda.synchronize()
            comm_toc = time.perf_counter()
            comm_times.append(comm_toc - comm_tic)

        opt.step()

        torch.cuda.synchronize()
        step_toc = time.perf_counter()
        step_times.append(step_toc - step_tic)

    store["step_times"] = step_times
    store["comm_times"] = comm_times

    if kwargs["store_net"]:
        store["net"] = net.state_dict()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs_dir", type=str, default="/dfs/scratch1/ranjanr/runs/cs336/2024-04-28"
    )
    parser.add_argument("--dev", type=int, default=1)

    parser.add_argument("--lm_size", type=str, default="small")
    parser.add_argument("--vocab_size", type=int, default=10_000)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--d_model", type=int)
    parser.add_argument("--d_ff", type=int)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--num_heads", type=int)
    parser.add_argument("--attn_pdrop", type=float, default=0.0)
    parser.add_argument("--residual_pdrop", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_train_steps", type=int, default=10)
    parser.add_argument("--backend", type=str, default="nccl")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ddp", type=int, default=1)
    parser.add_argument("--store_net", type=int, default=0)

    kwargs = parser.parse_args().__dict__
    main(**kwargs)
