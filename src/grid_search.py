"""
Mini grid search — à implémenter.

Doit exposer un main() exécutable via :
    python -m src.grid_search --config configs/config.yaml

Exigences minimales :
- lire la section 'hparams' de la config
- lancer plusieurs runs en variant les hyperparamètres
- journaliser les hparams et résultats de chaque run (ex: TensorBoard HParams ou équivalent)
"""

"""
Grid search script.
"""
import argparse
import yaml
import itertools
import copy
import os
from torch.utils.tensorboard import SummaryWriter

from .train import train_loop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        base_config = yaml.safe_load(f)

    hparams = base_config.get("hparams", {})

    if not hparams:
        raise ValueError("No 'hparams' section found in config.yaml")

    keys, values = zip(*hparams.items())

    for idx, combo in enumerate(itertools.product(*values)):
        cfg = copy.deepcopy(base_config)

        print(f"\n Running grid search config #{idx}: {dict(zip(keys, combo))}")

        # Apply hyperparameters
        for k, v in zip(keys, combo):
            if k == "lr":
                cfg["train"]["optimizer"]["lr"] = v
            elif k == "batch_size":
                cfg["train"]["batch_size"] = v
            elif k in ["embed_dim", "num_filters"]:
                cfg["model"][k] = v
            else:  
                # Generic override for any custom hyperparams
                cfg["train"][k] = v

        run_dir = f"runs/gs_run_{idx}"
        art_dir = f"artifacts/gs_run_{idx}"
        cfg["paths"]["runs_dir"] = run_dir
        cfg["paths"]["artifacts_dir"] = art_dir

        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(art_dir, exist_ok=True)

        writer = SummaryWriter(run_dir)
        writer.add_hparams(
            {k: v for k, v in zip(keys, combo)},
            {}
        )

        # Train & collect results
        result = train_loop(cfg, mode="gs")

        # result must be like {"train_acc":..., "val_acc":..., "loss":...}
        if isinstance(result, dict):
            writer.add_hparams(
                {k: v for k, v in zip(keys, combo)},
                result
            )

        writer.close()

    print("\n Grid search completed. View results with:")
    print("tensorboard --logdir=runs")


if __name__ == "__main__":
    main()
