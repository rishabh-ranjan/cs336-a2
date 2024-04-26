from tqdm.auto import tqdm

import dist_comm

if __name__ == "__main__":
    with tqdm(desc="expt", total=63) as pbar:
        for backend, device in [
            ("gloo", "cpu"),
            ("gloo", "cuda"),
            ("nccl", "cuda"),
        ]:
            for num_elements in [
                128_000,
                250_000,
                2_500_000,
                12_500_000,
                25_000_000,
                125_000_000,
                250_000_000,
            ]:
                for world_size in [
                    2,
                    4,
                    6,
                ]:
                    dist_comm.main(
                        runs_dir="runs/2024-04-25",
                        dev=0,
                        master_addr="localhost",
                        master_port="12355",
                        backend=backend,
                        device=device,
                        num_elements=num_elements,
                        world_size=world_size,
                        warmup_steps=5,
                    )
                    pbar.update(1)
                    print()
