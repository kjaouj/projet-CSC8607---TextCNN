"""
Évaluation — à implémenter.

Doit exposer un main() exécutable via :
    python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best.ckpt

Exigences minimales :
- charger le modèle et le checkpoint
- calculer et afficher/consigner les métriques de test
"""

"""
Evaluation script.
"""
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import numpy as np

from src.data_loading import get_dataloaders
from src.model import build_model
from src.utils import load_config, get_device


def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    return np.array(all_labels), np.array(all_preds)


def print_metrics(y_true, y_pred):
    print("\n==================== Evaluation Metrics ====================")

    # Find unique sorted labels
    labels = sorted(list(set(y_true)))
    num_classes = len(labels)

    # Default AG_NEWS class names
    agnews_names = ["World", "Sports", "Business", "Sci/Tech"]

    # Trim names to match real labels
    class_names = agnews_names[:num_classes]

    # Print which classes we are evaluating
    print(f"\nDetected {num_classes} classes: {labels}")
    print(f"Using class names: {class_names}\n")

    # Basic metrics
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}\n")

    print("Precision / Recall / F1:")
    print("Macro Precision:", precision_score(y_true, y_pred, average="macro", labels=labels))
    print("Macro Recall   :", recall_score(y_true, y_pred, average="macro", labels=labels))
    print("Macro F1-Score :", f1_score(y_true, y_pred, average="macro", labels=labels))

    print("\nWeighted Precision:", precision_score(y_true, y_pred, average="weighted", labels=labels))
    print("Weighted Recall   :", recall_score(y_true, y_pred, average="weighted", labels=labels))
    print("Weighted F1-Score :", f1_score(y_true, y_pred, average="weighted", labels=labels))

    print("\n==================== Classification Report ====================\n")
    print(classification_report(
        y_true, y_pred,
        target_names=class_names,
        labels=labels
    ))

    print("==================== Confusion Matrix ====================\n")
    print(confusion_matrix(y_true, y_pred, labels=labels))

    print("\n============================================================\n")



def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Device
    device = get_device(config)
    print(f"Using device: {device}")

    # Load dataloaders (regenerates vocab & tokenizer)
    train_loader, val_loader, test_loader, meta = get_dataloaders(config)

    # Build model
    model = build_model(config, meta).to(device)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt)

    # Evaluate
    y_true, y_pred = evaluate_model(model, test_loader, device)

    # Class names from AG_NEWS
    class_names = ["World", "Sports", "Business", "Sci/Tech"]

    # Print metrics
    print_metrics(y_true, y_pred)


if __name__ == "__main__":
    main()


