import functools
from pathlib import Path
import time

import pandas as pd
import torch
from tqdm.auto import tqdm
import wandb


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
        if key not in self:
            path = f"{self.store_dir}/{key}.pt"
            val = torch.load(path, map_location=self.device)
            super().__setitem__(key, val)

        return super().__getitem__(key)


def run(fn):
    @functools.wraps(fn)
    def wrap(**kwargs):
        ts = str(time.time_ns())
        store_dir = Path(kwargs["run_dir"]) / ts
        kwargs["store_dir"] = str(store_dir)
        print(f"{kwargs=}")

        store = PathDict(store_dir)
        store["kwargs"] = kwargs

        wandb_run = None
        wandb_project = kwargs.get("wandb_project", None)
        if wandb_project:
            wandb_name = kwargs.get("wandb_name", None)
            if wandb_name:
                wandb_name = f"{wandb_name}-{ts[:3]}"
            wandb_run = wandb.init(
                project=wandb_project,
                name=wandb_name,
                config={**kwargs},
            )
            print(f"{wandb_run.name=}")
            kwargs["wandb_run"] = wandb_run

        kwargs["store"] = store
        out = fn(**kwargs)
        print(f"{out=}")
        store["out"] = out
        store["done"] = True

    return wrap


def scan(run_dir="../runs/latest"):
    run_dir = Path(run_dir)
    store = PathDict(run_dir)
    raw = []
    for path in tqdm(sorted(run_dir.iterdir())):
        if not (path / "done").exists():
            print(f"!rm -r {path}")
            continue
        rec = {"ts": path.name}
        rec.update(store["args"])
        raw.append(rec)
    return pd.DataFrame(raw)
