import functools
from pathlib import Path
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from tqdm.auto import tqdm
import wandb


sns.set_theme(context="paper", style="whitegrid", palette="colorblind", font_scale=0.75)
plt.rcParams["figure.dpi"] = 157
plt.rcParams["xtick.major.size"] = 0
plt.rcParams["ytick.major.size"] = 0
LINE = 5.5


def save_fig(fig, save_key):
    path = f"fig/{save_key}.pdf"
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.01)
    print(f"saved at {path}")


class PathDict(dict):
    def __init__(self, store_dir, device="cpu", **kwargs):
        super().__init__(**kwargs)
        self.store_dir = store_dir
        self.device = device

        Path(store_dir).mkdir(parents=True, exist_ok=True)

    def __setitem__(self, key, val):
        path = f"{self.store_dir}/{key}.pt"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(val, path)

    def __getitem__(self, key):
        path = f"{self.store_dir}/{key}.pt"
        val = torch.load(path, map_location=self.device)
        return val


def run(fn):
    @functools.wraps(fn)
    def wrap(**kwargs):
        print(f"{kwargs=}")
        frozen_kwargs = kwargs.copy()

        timestamp = str(time.time_ns())
        kwargs["timestamp"] = timestamp

        store_dir = Path(kwargs["runs_dir"]) / timestamp
        store = PathDict(store_dir)
        store["info"] = kwargs

        wandb_run = None
        wandb_project = kwargs.get("wandb_project", None)
        if wandb_project:
            wandb_name = kwargs.get("wandb_name", None)
            if wandb_name:
                wandb_name = f"{wandb_name}-{timestamp[:3]}"
            wandb_run = wandb.init(
                project=wandb_project,
                name=wandb_name,
                config={**kwargs},
            )
            print(f"{wandb_run.name=}")
            kwargs["wandb_run"] = wandb_run

        kwargs["store"] = store

        fn(**kwargs)

        store["info"] = {
            **store["info"],
            "done": True,
        }
        diff = {
            k: v
            for k, v in store["info"].items()
            if k not in frozen_kwargs or v != frozen_kwargs[k]
        }
        print(f"info/kwargs={diff}")

    return wrap


def scan(runs_dir):
    runs_dir = Path(runs_dir)
    out = []
    for store_dir in tqdm(sorted(runs_dir.glob("*"))):
        store = PathDict(store_dir)
        info = store["info"]
        if info.get("done", False):
            print(f"!rm -r {store_dir}  # not done")
            continue
        if info.get("dev", True):
            print(f"!rm -r {store_dir}  # dev")
            continue
        out.append(info)
    out = pd.DataFrame(out).set_index("timestamp")
    return out
