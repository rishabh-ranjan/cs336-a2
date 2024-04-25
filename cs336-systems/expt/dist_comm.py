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
    dist.all_reduce(data)


@expt.run
def main(**kwargs):
    tic = time.perf_counter_ns()
    mp.spawn(worker, (kwargs,), nprocs=kwargs["world_size"])
    torch.cuda.synchronize()
    toc = time.perf_counter_ns()
    return {"time (s)": (toc - tic) / 1e9}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=str, default="runs/2024-04-24")
    parser.add_argument("--dev", type=int, default=1)
    parser.add_argument("--master_addr", type=str, default="localhost")
    parser.add_argument("--master_port", type=str, default="12355")
    parser.add_argument("--backend", type=str, default="gloo")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_elements", type=int, default=4)
    parser.add_argument("--world_size", type=int, default=2)
    kwargs = parser.parse_args().__dict__
    main(**kwargs)
