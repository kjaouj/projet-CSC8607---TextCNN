"""
Recherche de taux d'apprentissage (LR finder) — à implémenter.

Doit exposer un main() exécutable via :
    python -m src.lr_finder --config configs/config.yaml

Exigences minimales :
- produire un log/trace permettant de visualiser (lr, loss) dans TensorBoard ou équivalent.
"""

import argparse
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from .data_loading import get_dataloaders
from .model import build_model
from .utils import get_device


def lr_finder(config_path):
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = get_device(config)
    print(f"Using device: {device}")

    # Load data
    train_loader, _, _, meta = get_dataloaders(config)

    # Build model
    model = build_model(config, meta).to(device)
    model.train()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)

    lr = 1e-7
    max_lr = 1
    mult = (max_lr / lr) ** (1 / len(train_loader))

    writer = SummaryWriter("runs/lr_finder")
    print("Starting LR Finder...")

    for batch_idx, batch in enumerate(train_loader):

        # Support both (X, y) and (X, y, metadata)
        if len(batch) == 2:
            X, y = batch
        elif len(batch) == 3:
            X, y, _ = batch
        else:
            raise ValueError(f"Unexpected batch format: {len(batch)} values")

        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)

        if torch.isnan(loss):
            print(" Loss diverged — stopping LR Finder.")
            break

        loss.backward()
        optimizer.step()

        # Log scalars
        writer.add_scalar("lr_finder/loss", loss.item(), batch_idx)
        writer.add_scalar("lr_finder/lr", lr, batch_idx)

        # Increase LR
        lr *= mult
        for g in optimizer.param_groups:
            g["lr"] = lr

        if lr > max_lr:
            print("Reached maximum LR")
            break

    writer.close()
    print("LR Finder finished. Run:  tensorboard --logdir=runs")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    lr_finder(args.config)


if __name__ == "__main__":
    main()

