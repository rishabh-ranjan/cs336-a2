import argparse
import time

import expt
import torch
from torch import nn

from cs336_basics.model import RMSNorm


@expt.run
def main(store, args):
    x = torch.randn(args.batch_size, args.d_model)

    if args.norm == "rms":
        norm = RMSNorm(args.d_model)
    elif args.norm == "layer":
        norm = nn.LayerNorm(args.d_model)

    with torch.no_grad():
        for _ in range(args.warmup_steps):
            y = norm(x)
        torch.cuda.synchronize()

        times = []
        for _ in range(args.benchmark_steps):
            torch.cuda.synchronize()
            tic = time.perf_counter()
            y = norm(x)
            torch.cuda.synchronize()
            toc = time.perf_counter()
            times.append(toc - tic)

    store["time (s)"] = torch.tensor(times)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs_dir", type=str, default="/dfs/scratch1/ranjanr/runs/cs336/2024-05-04"
    )
    parser.add_argument("--dev", type=int, default=1)

    parser.add_argument("--batch_size", type=int, default=50_000)
    parser.add_argument("--d_model", type=int, default=1_024)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--benchmark_steps", type=int, default=1000)
    parser.add_argument("--norm", type=str, default="rms")

    args = parser.parse_args()
    main(args)
