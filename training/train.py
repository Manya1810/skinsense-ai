"""SkinSense AI - Training Pipeline for Dermatology Models
Supports:
- 4 Candidate Architectures: EfficientNet-B0, MobileNetV2, ResNet50, ViT-B/16
- Patient-aware leakage-free train/validation/test split
- AdamW optimizer + Cosine Annealing learning rate schedule
- CrossEntropyLoss with optional inverse-frequency class-weighting
- Macro F1 validation tracking and checkpoint saving
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.dataset import (
    CLASS_TO_IDX,
    HAM10000_CLASSES,
    DermatologyDataset,
    compute_class_weights,
    get_transforms,
    split_metadata_without_leakage,
)
from training.models import SUPPORTED_MODELS, build_model


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Runs one training epoch and returns (mean_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _ in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, Dict[str, float]]:
    """Evaluates on validation split, computing Loss, Accuracy, and Macro F1."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    total = len(all_labels)
    val_loss = running_loss / max(total, 1)
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean() if total > 0 else 0.0

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    per_class_recalls = recall_score(all_labels, all_preds, average=None, labels=list(range(len(HAM10000_CLASSES))), zero_division=0)
    per_class_dict = {
        HAM10000_CLASSES[i]: float(per_class_recalls[i])
        for i in range(len(HAM10000_CLASSES))
    }

    return val_loss, accuracy, float(macro_f1), per_class_dict


def run_training(
    data_dir: str,
    model_name: str = "efficientnet_b0",
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    use_class_weights: bool = True,
    pretrained: bool = True,
    output_dir: str = "models/checkpoints",
    seed: int = 42,
) -> Dict:
    """Executes full training pipeline, saving the best validation checkpoint."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n==========================================")
    print(f"SkinSense AI Training Pipeline")
    print(f"Model: {model_name} | Device: {device} | Pretrained: {pretrained}")
    print(f"Data Dir: {data_dir} | Epochs: {epochs} | Batch Size: {batch_size}")
    print(f"==========================================\n")

    # Load dataset metadata
    data_path = Path(data_dir)
    csv_file = data_path / "HAM10000_metadata.csv"
    if not csv_file.exists():
        # Fallback search for any csv in data_dir
        csv_candidates = list(data_path.glob("*.csv"))
        if not csv_candidates:
            raise FileNotFoundError(f"No metadata CSV found in '{data_dir}'.")
        csv_file = csv_candidates[0]

    df = pd.read_csv(csv_file)
    if "image_path" not in df.columns:
        # Resolve images from images folder
        images_dir = data_path / "images"
        if not images_dir.exists():
            images_dir = data_path
        df["image_path"] = df["image_id"].apply(
            lambda x: str((images_dir / f"{x}.jpg").resolve())
            if (images_dir / f"{x}.jpg").exists()
            else str((images_dir / f"{x}.png").resolve())
        )

    # Patient-aware split (no leakage)
    train_df, val_df, test_df = split_metadata_without_leakage(df, seed=seed)
    print(f"Dataset split without leakage:")
    print(f" - Train images: {len(train_df)}")
    print(f" - Validation images: {len(val_df)}")
    print(f" - Held-out Test images: {len(test_df)} (frozen until final evaluation)\n")

    # Save test split for untouched final evaluation
    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(out_dir_path / f"{model_name}_heldout_test_split.csv", index=False)

    train_ds = DermatologyDataset(train_df, transform=get_transforms("train"))
    val_ds = DermatologyDataset(val_df, transform=get_transforms("val"))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Initialize model
    model, meta = build_model(
        model_name=model_name,
        num_classes=len(HAM10000_CLASSES),
        pretrained=pretrained,
        device=device,
    )

    # Loss criterion
    if use_class_weights and len(train_df) > 0:
        class_weights = compute_class_weights(train_df).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Using inverse-frequency class-weighted CrossEntropyLoss.")
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_macro_f1 = -1.0
    best_checkpoint_file = out_dir_path / f"{model_name}_best.pt"
    history = []

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_macro_f1, val_recalls = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        duration = time.time() - epoch_start
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({duration:.1f}s) | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} Macro F1: {val_macro_f1:.3f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_macro_f1,
            "val_recalls": val_recalls,
        })

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            torch.save(
                {
                    "epoch": epoch,
                    "model_name": model_name,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_macro_f1": val_macro_f1,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "classes": HAM10000_CLASSES,
                    "meta": meta,
                },
                best_checkpoint_file,
            )
            print(f" -> Saved new best checkpoint (Macro F1: {val_macro_f1:.4f}) to '{best_checkpoint_file}'")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s. Best Val Macro F1: {best_val_macro_f1:.4f}")

    # Save history
    history_file = out_dir_path / f"{model_name}_history.json"
    with open(history_file, "w") as f:
        json.dump(
            {
                "model_name": model_name,
                "epochs": epochs,
                "best_val_macro_f1": best_val_macro_f1,
                "history": history,
            },
            f,
            indent=2,
        )

    return {
        "model_name": model_name,
        "checkpoint_path": str(best_checkpoint_file),
        "best_val_macro_f1": best_val_macro_f1,
        "total_time_seconds": total_time,
        "history": history,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SkinSense AI - Dermatology Model Training")
    parser.add_argument("--data-dir", type=str, default="data/ham10000_sample", help="Path to dataset directory")
    parser.add_argument("--model", type=str, default="efficientnet_b0", choices=SUPPORTED_MODELS, help="Candidate architecture")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--no-class-weights", action="store_true", help="Disable inverse-frequency class weights")
    parser.add_argument("--scratch", action="store_true", help="Train from scratch without ImageNet pretrained weights")
    parser.add_argument("--output-dir", type=str, default="models/checkpoints", help="Directory to save checkpoints")

    args = parser.parse_args()
    run_training(
        data_dir=args.data_dir,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_class_weights=not args.no_class_weights,
        pretrained=not args.scratch,
        output_dir=args.output_dir,
    )
