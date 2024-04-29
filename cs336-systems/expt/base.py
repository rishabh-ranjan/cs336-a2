"""Problem (naive_ddp)"""

import argparse

import torch

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


def main(**kwargs):
    device = torch.device("cuda")

    if kwargs["lm_size"]:
        kwargs.update(CONFIG[kwargs["lm_size"]])

    net = BasicsTransformerLM(
        vocab_size=kwargs["vocab_size"],
        context_length=kwargs["context_length"],
        d_model=kwargs["d_model"],
        d_ff=kwargs["d_ff"],
        num_layers=kwargs["num_layers"],
        num_heads=kwargs["num_heads"],
        attn_pdrop=kwargs["attn_pdrop"],
        residual_pdrop=kwargs["residual_pdrop"],
    )
    net = net.to(device)

    opt = AdamW(net.parameters())

    for _ in range(kwargs["num_train_steps"]):
        net.train()
        batch = torch.randint(
            kwargs["vocab_size"], (kwargs["batch_size"], kwargs["context_length"] + 1)
        )
        batch = batch.to(device)
        x = batch[:, :-1]
        y = batch[:, 1:]

        logit = net(x)
        loss = cross_entropy(logit, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lm_size", type=str, default="small")
    parser.add_argument("--vocab_size", type=int, default=10_000)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--d_model", type=int)
    parser.add_argument("--d_ff", type=int)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--num_heads", type=int)
    parser.add_argument("--attn_pdrop", type=float, default=0.0)
    parser.add_argument("--residual_pdrop", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_train_steps", type=int, default=10)

    kwargs = parser.parse_args().__dict__
    main(**kwargs)
