"""Problem (benchmarking_script)"""

import argparse
import time

import torch

torch.manual_seed(0)

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy


def main(args):
    # setup
    device = torch.device("cuda")

    net = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        attn_pdrop=args.attn_pdrop,
        residual_pdrop=args.residual_pdrop,
    )
    net.to(device)

    batch = torch.randint(args.vocab_size, (args.batch_size, args.context_length + 1))
    batch = batch.to(device)
    x = batch[:, :-1]
    y = batch[:, 1:]

    # warmup
    for _ in range(args.warmup_steps):
        logit = net(x)
        loss = cross_entropy(logit, y)
        loss.backward()
    torch.cuda.synchronize()

    # benchmark
    fw_times = []
    bw_times = []
    for _ in range(args.benchmark_steps):
        tic = time.perf_counter()

        logit = net(x)
        loss = cross_entropy(logit, y)

        torch.cuda.synchronize()
        toc = time.perf_counter()
        fw_times.append(toc - tic)

        tic = time.perf_counter()

        loss.backward()

        torch.cuda.synchronize()
        toc = time.perf_counter()
        bw_times.append(toc - tic)

    fw_time = torch.tensor(fw_times)
    bw_time = torch.tensor(bw_times)

    print(f"=== Forward ===")
    print(f"All:\t{fw_time}")
    print(f"Mean:\t{fw_time.mean():.2e}")
    print(f"Std:\t{fw_time.std():.2e}")

    print(f"=== Backward ===")
    print(f"All:\t{bw_time}")
    print(f"Mean:\t{bw_time.mean():.2e}")
    print(f"Std:\t{bw_time.std():.2e}")

    return fw_time, bw_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab_size", type=int, default=10_000)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=2560)
    parser.add_argument("--num_layers", type=int, default=32)
    parser.add_argument("--num_heads", type=int, default=32)
    parser.add_argument("--d_ff", type=int, default=10240)
    parser.add_argument("--attn_pdrop", type=float, default=0.0)
    parser.add_argument("--residual_pdrop", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--warmup_steps", type=int, default=1)
    parser.add_argument("--benchmark_steps", type=int, default=5)

    args = parser.parse_args()
    main(args)
