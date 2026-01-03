"""
Entraînement principal (à implémenter par l'étudiant·e).

Doit exposer un main() exécutable via :
    python -m src.train --config configs/config.yaml [--seed 42]

Exigences minimales :
- lire la config YAML
- respecter les chemins 'runs/' et 'artifacts/' définis dans la config
- journaliser les scalars 'train/loss' et 'val/loss' (et au moins une métrique de classification si applicable)
- supporter le flag --overfit_small (si True, sur-apprendre sur un très petit échantillon)
"""

"""
Training script.
"""
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from .data_loading import get_dataloaders
from .model import build_model
from .utils import set_seed, get_device, save_config_snapshot


def train_loop(config, mode="train"):

    # -----------------------------
    # Setup
    # -----------------------------
    set_seed(config["train"]["seed"])
    device = get_device(config)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(device)}")

    train_loader, val_loader, test_loader, meta = get_dataloaders(config)

    num_classes = meta["num_classes"]
    print(f"Number of classes: {num_classes}")

    model = build_model(config, meta).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["train"]["optimizer"]["lr"],
        weight_decay=config["train"]["optimizer"]["weight_decay"]
    )

    writer = SummaryWriter(config["paths"]["runs_dir"])
    save_config_snapshot(config, config["paths"]["runs_dir"])

    best_val_acc = 0.0
    epochs_no_improve = 0

    # Prefix for TensorBoard
    # If mode is 'final', we use 'Final' to create distinct graphs
    pxt = "Final" if mode == "final" else "GS"

    # -----------------------------
    # Training
    # -----------------------------
    for epoch in range(config["train"]["epochs"]):

        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            X, y = batch

            # cast des labels
            y = y.long()

            # sécurité (évite CUDA assert)
            if y.max() >= num_classes or y.min() < 0:
                print(f"Batch skipped (invalid label value: {y.min()}–{y.max()})")
                continue

            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (preds.argmax(1) == y).sum().item()
            total += y.size(0)

        train_loss = total_loss / max(1, len(train_loader))
        train_acc = correct / max(1, total)

        # -----------------------------
        # Validation
        # -----------------------------
        model.eval()
        val_loss_total = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                X, y = batch
                y = y.long()

                if y.max() >= num_classes or y.min() < 0:
                    continue

                X, y = X.to(device), y.to(device)
                preds = model(X)
                val_loss_total += criterion(preds, y).item()
                correct += (preds.argmax(1) == y).sum().item()
                total += y.size(0)

        val_loss = val_loss_total / max(1, len(val_loader))
        val_acc = correct / max(1, total)

        # -----------------------------
        # Log
        # -----------------------------
        writer.add_scalar(f"{pxt}/Loss_Train", train_loss, epoch)
        writer.add_scalar(f"{pxt}/Accuracy_Train", train_acc, epoch)
        writer.add_scalar(f"{pxt}/Loss_Val", val_loss, epoch)
        writer.add_scalar(f"{pxt}/Accuracy_Val", val_acc, epoch)

        print(
            f"Epoch {epoch+1}/{config['train']['epochs']} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
        )

        # -----------------------------
        # Save best
        # -----------------------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{config['paths']['artifacts_dir']}/best.ckpt")
            print("Saved new best checkpoint!")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= 5: # Patience of 5
                print(f"Early stopping triggered after {epoch+1} epochs!")
                break

    writer.close()

    return {"val_acc": best_val_acc, "val_loss": val_loss}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overfit_small", action="store_true", help="Overfit on a small sample (1000 ex)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # CLI flag overrides config file
    if args.overfit_small:
        config["train"]["overfit_small"] = True

    mode = "final" if not config["train"].get("overfit_small", False) else "overfit"
    train_loop(config, mode=mode)


if __name__ == "__main__":
    main()
