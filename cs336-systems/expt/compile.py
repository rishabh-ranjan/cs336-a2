import argparse
import time

import expt
import torch

torch.manual_seed(42)

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

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
    config = CONFIG[args.lm_size]

    # setup
    device = torch.device("cuda")

    net = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        attn_pdrop=args.attn_pdrop,
        residual_pdrop=args.residual_pdrop,
    )
    net.to(device)
    if args.compile:
        net = torch.compile(net)

    opt = AdamW(net.parameters())

    batch = torch.randint(args.vocab_size, (args.batch_size, args.context_length + 1))
    batch = batch.to(device)
    x = batch[:, :-1]
    y = batch[:, 1:]

    # warmup
    for _ in range(args.warmup_steps):
        logit = net(x)
        loss = cross_entropy(logit, y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    # benchmark
    fw_times = []
    times = []
    for _ in range(args.benchmark_steps):
        torch.cuda.synchronize()
        tic = time.perf_counter()

        logit = net(x)
        loss = cross_entropy(logit, y)

        torch.cuda.synchronize()
        fw_toc = time.perf_counter()
        fw_times.append(fw_toc - tic)

        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        toc = time.perf_counter()
        times.append(toc - tic)

    store["fw_time"] = torch.tensor(fw_times)
    store["time"] = torch.tensor(times)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs_dir", type=str, default="/dfs/scratch1/ranjanr/runs/cs336/2024-05-03"
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
    parser.add_argument("--compile", type=int, default=0)
    args = parser.parse_args()
    main(args)
