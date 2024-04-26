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
        if not super().__contains__(key):
            path = f"{self.store_dir}/{key}.pt"
            val = torch.load(path, map_location=self.device)
            super().__setitem__(key, val)

        return super().__getitem__(key)

    def __contains__(self, key):
        return Path(f"{self.store_dir}/{key}.pt").exists()


def run(fn):
    @functools.wraps(fn)
    def wrap(**kwargs):
        timestamp = str(time.time_ns())
        kwargs["timestamp"] = timestamp
        print(f"{kwargs=}")

        store_dir = Path(kwargs["runs_dir"]) / timestamp
        store = PathDict(store_dir)
        store["kwargs"] = kwargs

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
        store["done"] = True

    return wrap


def scan(runs_dir, store_keys=["kwargs"]):
    runs_dir = Path(runs_dir)
    raw = []
    for store_dir in tqdm(sorted(runs_dir.glob("*"))):
        store = PathDict(store_dir)
        if "done" not in store:
            print(f"!rm -r {store_dir}  # not done")
            continue
        if store["kwargs"]["dev"]:
            print(f"!rm -r {store_dir}  # dev")
            continue
        rec = {}
        for store_key in store_keys:
            for k, v in store[store_key].items():
                rec[f"{store_key}/{k}"] = v
        raw.append(rec)
    return pd.DataFrame(raw).set_index("kwargs/timestamp")
