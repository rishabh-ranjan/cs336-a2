"""Problem (distributed_communication_single_node)"""

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import expt


def setup(rank, **kwargs):
    os.environ["MASTER_ADDR"] = kwargs["master_addr"]
    os.environ["MASTER_PORT"] = kwargs["master_port"]
    dist.init_process_group(
        kwargs["backend"], rank=rank, world_size=kwargs["world_size"]
    )


def worker(rank, kwargs):
    setup(rank, **kwargs)
    device = f"cuda:{rank}" if kwargs["device"] == "cuda" else "cpu"
    data = torch.randn(kwargs["num_elements"]).to(device)

    for _ in range(kwargs["warmup_steps"]):
        dist.all_reduce(data)

    tic = time.perf_counter()
    dist.all_reduce(data)
    toc = time.perf_counter()

    buf = torch.tensor(toc - tic).to(device)
    dist.reduce(buf, 0)
    if rank == 0:
        t = buf.item() / kwargs["world_size"]
        print(f"time (s): {t}")
        kwargs["store"]["time"] = {"time (s)": t}


@expt.run
def main(**kwargs):
    mp.spawn(worker, (kwargs,), nprocs=kwargs["world_size"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs_dir", type=str, default="runs/2024-04-25")
    parser.add_argument("--dev", type=int, default=1)
    parser.add_argument("--master_addr", type=str, default="localhost")
    parser.add_argument("--master_port", type=str, default="12355")
    parser.add_argument("--backend", type=str, default="nccl")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_elements", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--local_world_size", type=int, default=2)
    parser.add_argument("--node_rank", type=int, default=0)
    parser.add_argument("--num_nodes", type=int, default=2)
    kwargs = parser.parse_args().__dict__
    main(**kwargs)
