import argparse
import os

import expt
import torch
import torch.distributed as dist

torch.manual_seed(42)

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.optim import ShardedOptimizer


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
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    if args.lm_size:
        args.__dict__.update(CONFIG[args.lm_size])
        store["info"] = {
            **store["info"],
            **CONFIG[args.lm_size],
        }

    if rank == 0:
        if args.shard_optimizer:
            print(" with optimizer state sharding ")
            print("-------------------------------")
        else:
            print(" without optimizer state sharding ")
            print("----------------------------------")

    # setup
    device = torch.device(f"cuda:{local_rank}")

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

    if rank == 0:
        print("at model initialization:")
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device) / 2**30
        torch.cuda.reset_peak_memory_stats(device)
        mem = torch.cuda.memory_allocated(device) / 2**30
        param_mem = mem
        print()
        print(f"     peak memory usage = {peak:.3f} GiB")
        print(f"  current memory usage = {mem:.3f} GiB")
        print()
        print(f"  === peak memory components ===")
        print(f"    model parameters = {param_mem:.3f} GiB")
        print(f"         activations = 0 GiB")
        print(f"     optimizer state = 0 GiB")
        print()

    if args.shard_optimizer:
        opt = ShardedOptimizer(net.parameters(), AdamW)
    else:
        opt = AdamW(net.parameters())

    for step in range(1):
        # if rank == 0:
        #     print(f"--- step {step+1} ---")
        #     print()
        batch = torch.randint(
            args.vocab_size, (args.batch_size, args.context_length + 1)
        )
        batch = batch.to(device)
        x = batch[:, :-1]
        y = batch[:, 1:]

        net.train()
        logit = net(x)
        loss = cross_entropy(logit, y)
        loss.backward()

        if rank == 0:
            print("before optimizer step:")
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device) / 2**30
            torch.cuda.reset_peak_memory_stats(device)
            pre_mem = torch.cuda.memory_allocated(device) / 2**30
            activ_mem = peak - param_mem
            if step == 0:
                opt_mem = 0
            print()
            print(f"  new peak memory usage = {peak:.3f} GiB")
            print(f"   current memory usage = {pre_mem:.3f} GiB")
            print()
            print(f"  === peak memory components ===")
            print(f"    model parameters = {param_mem:.3f} GiB")
            print(f"         activations = {activ_mem:.3f} GiB")
            print(f"     optimizer state = {opt_mem} GiB")
            print()

        opt.step()

        if rank == 0:
            print("after optimizer step:")
            torch.cuda.synchronize(device)
            peak = torch.cuda.memory_allocated(device) / 2**30
            post_mem = torch.cuda.memory_allocated(device) / 2**30
            opt_mem = post_mem - pre_mem
            print()
            print(f"  new peak memory usage = {peak:.3f} GiB")
            print(f"   current memory usage = {post_mem:.3f} GiB")
            print()
            print(f"  === peak memory components ===")
            print(f"    model parameters = {param_mem:.3f} GiB")
            print(f"         activations = 0 GiB")
            print(f"     optimizer state = {opt_mem:.3f} GiB")

        opt.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs_dir", type=str, default="/dfs/scratch1/ranjanr/runs/cs336/2024-05-04"
    )
    parser.add_argument("--dev", type=int, default=1)

    parser.add_argument("--lm_size", type=str, default="xl")
    parser.add_argument("--vocab_size", type=int, default=10_000)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--attn_pdrop", type=float, default=0.1)
    parser.add_argument("--residual_pdrop", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--shard_optimizer", type=int, default=0)
    args = parser.parse_args()
    main(args)
