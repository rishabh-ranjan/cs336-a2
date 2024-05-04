"""Problem (function_call_table)"""

import argparse

import torch
from torch.profiler import profile, record_function, ProfilerActivity

torch.manual_seed(0)

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


def main(args):
    if args.lm_size:
        args.__dict__.update(CONFIG[args.lm_size])

    # setup
    device = torch.device("cuda")

    if args.profile_memory:
        torch.cuda.memory._record_memory_history(max_entries=1000000)

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

    opt = AdamW(net.parameters())

    def run_step():
        batch = torch.randint(
            args.vocab_size, (args.batch_size, args.context_length + 1)
        )
        batch = batch.to(device)
        x = batch[:, :-1]
        y = batch[:, 1:]

        if args.inference_only:
            net.eval()
            with torch.no_grad():
                with record_function("forward_pass"):
                    logit = net(x)

        else:
            net.train()
            # torch.cuda.synchronize()

            with record_function("forward_pass"):
                logit = net(x)
                # torch.cuda.synchronize()

            with record_function("backward_pass"):
                loss = cross_entropy(logit, y)
                loss.backward()
                # torch.cuda.synchronize()

            with record_function("optimizer"):
                opt.step()
                opt.zero_grad(set_to_none=True)
                # torch.cuda.synchronize()

    # warmup
    for _ in range(args.warmup_steps):
        run_step()
    torch.cuda.synchronize()

    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=0, warmup=0, active=1, repeat=args.benchmark_steps
        ),
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
        record_shapes=args.record_shapes,
        profile_memory=args.profile_memory,
        with_stack=args.with_stack,
    ) as prof:
        for _ in range(args.benchmark_steps):
            run_step()
            prof.step()

    if args.profile_memory:
        prof.export_memory_timeline(
            f"out/timeline__inference_only={args.inference_only}.html"
        )

    if args.with_stack:
        prof.export_stacks("out/lm_profiler_stacks.txt", "self_cpu_time_total")

    print(
        prof.key_averages(group_by_input_shape=args.record_shapes).table(
            sort_by=args.sort_by, row_limit=50
        )
    )

    if args.profile_memory:
        torch.cuda.memory._dump_snapshot(
            f"out/memory_snapshot__inference_only={args.inference_only}.pickle"
        )
        torch.cuda.memory._record_memory_history(enabled=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lm_size", type=str, default="small")
    parser.add_argument("--vocab_size", type=int, default=10_000)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=2_560)
    parser.add_argument("--num_layers", type=int, default=32)
    parser.add_argument("--num_heads", type=int, default=32)
    parser.add_argument("--d_ff", type=int, default=10_240)
    parser.add_argument("--attn_pdrop", type=float, default=0.0)
    parser.add_argument("--residual_pdrop", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--benchmark_steps", type=int, default=3)
    parser.add_argument("--inference_only", type=int, default=0)
    parser.add_argument("--record_shapes", type=int, default=1)
    parser.add_argument("--profile_memory", type=int, default=1)
    parser.add_argument("--with_stack", type=int, default=1)
    parser.add_argument("--sort_by", type=str, default="self_cpu_time_total")

    args = parser.parse_args()
    main(args)
