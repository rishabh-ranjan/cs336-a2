import argparse
import time

import expt
import torch
from torch import nn

import cs336_basics.model
from cs336_basics.model import BasicsTransformerLM


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
def main(store, args):
    device = "cuda"

    config = CONFIG[args.lm_size]

    cs336_basics.model.global_norm = args.norm
    net = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        d_model=config["d_model"],
        d_ff=config["d_ff"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        context_length=args.context_length,
        attn_pdrop=args.attn_pdrop,
        residual_pdrop=args.residual_pdrop,
    )
    net = net.to(device)

    batch = torch.randint(args.vocab_size, (args.batch_size, args.context_length))
    batch = batch.to(device)

    net.eval()
    with torch.no_grad():

        for _ in range(args.warmup_steps):
            logit = net(batch)
        torch.cuda.synchronize()

        times = []
        for _ in range(args.benchmark_steps):
            torch.cuda.synchronize()
            tic = time.perf_counter()
            logit = net(batch)
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

    parser.add_argument("--lm_size", type=str, default="small")
    parser.add_argument("--vocab_size", type=int, default=10_000)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--attn_pdrop", type=float, default=0.1)
    parser.add_argument("--residual_pdrop", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--benchmark_steps", type=int, default=5)
    parser.add_argument("--norm", type=str, default="rms")

    args = parser.parse_args()
    main(args)
