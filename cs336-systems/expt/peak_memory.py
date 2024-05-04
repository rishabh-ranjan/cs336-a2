import argparse

import expt
import torch
import wandb

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
    if args.lm_size:
        args.__dict__.update(CONFIG[args.lm_size])
        store["info"] = {
            **store["info"],
            **CONFIG[args.lm_size],
        }

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

    param_memory = torch.cuda.memory_allocated(device)

    opt = AdamW(net.parameters())

    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp))

    for step in range(args.num_steps):
        batch = torch.randint(
            args.vocab_size, (args.batch_size, args.context_length + 1)
        )
        batch = batch.to(device)
        x = batch[:, :-1]
        y = batch[:, 1:]

        if args.inference_only:
            net.eval()
            with torch.no_grad():
                logit = net(x)
            if step == 0:
                activation_memory = (
                    torch.cuda.max_memory_allocated(device) - param_memory
                )
        else:
            net.train()
            with torch.cuda.amp.autocast(enabled=bool(args.amp)):
                logit = net(x)
                loss = cross_entropy(logit, y)
            if step == 0:
                activation_memory = (
                    torch.cuda.max_memory_allocated(device) - param_memory
                )
            # loss.backward()
            # opt.step()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

    peak_memory = torch.cuda.max_memory_allocated(device)
    store["info"] = {
        **store["info"],
        "param memory (B)": param_memory,
        "activation memory (B)": activation_memory,
        "peak memory (B)": peak_memory,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs_dir", type=str, default="/dfs/scratch1/ranjanr/runs/cs336/2024-05-04"
    )
    parser.add_argument("--dev", type=int, default=1)

    parser.add_argument("--lm_size", type=str, default="2.7b")
    parser.add_argument("--vocab_size", type=int, default=10_000)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--attn_pdrop", type=float, default=0.1)
    parser.add_argument("--residual_pdrop", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--inference_only", type=int, default=0)
    parser.add_argument("--num_steps", type=int, default=2)
    parser.add_argument("--amp", type=int, default=1)
    args = parser.parse_args()
    main(args)
