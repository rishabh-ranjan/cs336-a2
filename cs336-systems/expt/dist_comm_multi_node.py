"""Problem (distributed_communication_multi_node)"""

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import expt


@expt.run
def main(**kwargs):
    store = kwargs["store"]

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    store["info"] = {
        **store["info"],
        "rank": rank,
        "world_size": world_size,
    }

    dist.init_process_group(
        backend=kwargs["backend"],
        rank=rank,
        world_size=world_size,
    )

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
        t = buf.item() / world_size
        store["info"] = {
            **store["info"],
            "time (s)": t,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs_dir", type=str, default="runs/2024-04-26")
    parser.add_argument("--dev", type=int, default=1)
    parser.add_argument("--backend", type=str, default="nccl")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_elements", type=int, default=1024)
    parser.add_argument("--warmup_steps", type=int, default=5)
    kwargs = parser.parse_args().__dict__
    main(**kwargs)
