import subprocess
from types import SimpleNamespace

import torch
import torch.distributed.run
from tqdm.auto import tqdm

import dist_comm_multi_node

if __name__ == "__main__":
    with tqdm(desc="expt", total=63) as pbar:
        for backend, device in [
            ("gloo", "cpu"),
            ("gloo", "cuda"),
            ("nccl", "cuda"),
        ]:
            for num_elements in [
                128_000,
                250_000,
                2_500_000,
                12_500_000,
                25_000_000,
                125_000_000,
                250_000_000,
            ]:
                for num_procs_per_node in [
                    1,
                    2,
                    3,
                ]:
                    subprocess.run(
                        [
                            f"torchrun",
                            f"--nnodes=2",
                            f"--nproc_per_node={num_procs_per_node}",
                            f"--max_restarts=0",
                            f"--rdzv_id=0",
                            f"--rdzv_backend=c10d",
                            f"--rdzv_endpoint=turing4.stanford.edu:29400",
                            f"-m",
                            f"dist_comm_multi_node",
                            f"--runs_dir=runs/2024-04-26",
                            f"--dev=1",
                            f"--backend={backend}",
                            f"--device={device}",
                            f"--num_elements={num_elements}",
                            f"--warmup_steps=5",
                        ]
                    )
                    pbar.update(1)
                    print()
