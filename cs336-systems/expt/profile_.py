"""Problem (function_call_table)"""

import argparse

import torch
from torch.profiler import profile, record_function, ProfilerActivity

torch.manual_seed(0)

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW


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

    opt = AdamW(net.parameters())

    def run_step():
        if args.inference_only:
            net.eval()
            with torch.no_grad():
                with record_function("forward_pass"):
                    logit = net(x)

        else:
            net.train()
            torch.cuda.synchronize()

            with record_function("forward_pass"):
                logit = net(x)
                torch.cuda.synchronize()

            with record_function("backward_pass"):
                loss = cross_entropy(logit, y)
                loss.backward()
                torch.cuda.synchronize()

            with record_function("optimizer"):
                opt.step()
                opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize()

    # warmup
    for _ in range(args.warmup_steps):
        run_step()
    torch.cuda.synchronize()

    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
        record_shapes=args.record_shapes,
        profile_memory=args.profile_memory,
        with_stack=args.with_stack,
    ) as prof:
        for _ in range(args.benchmark_steps):
            run_step()
            prof.step()

    if args.with_stack:
        prof.export_stacks("out/lm_profiler_stacks.txt", "self_cuda_time_total")

    print(
        prof.key_averages(group_by_input_shape=args.record_shapes).table(
            sort_by="cpu_time_total", row_limit=50
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab_size", type=int, default=10_000)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=1_600)
    parser.add_argument("--num_layers", type=int, default=48)
    parser.add_argument("--num_heads", type=int, default=25)
    parser.add_argument("--d_ff", type=int, default=6_400)
    parser.add_argument("--attn_pdrop", type=float, default=0.0)
    parser.add_argument("--residual_pdrop", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--warmup_steps", type=int, default=1)
    parser.add_argument("--benchmark_steps", type=int, default=5)
    parser.add_argument(
        "--inference_only", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--record_shapes", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--profile_memory", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--with_stack", action=argparse.BooleanOptionalAction, default=True
    )

    args = parser.parse_args()
    main(args)
