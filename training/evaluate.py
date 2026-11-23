"""SkinSense AI - Evaluation & Benchmarking Pipeline
Evaluates candidate models on the held-out test split. Computes:
- Overall Accuracy
- Per-class Precision, Recall (Sensitivity), and F1
- Macro F1 (unweighted average across all classes)
- Confusion Matrix
- Expected Calibration Error (ECE)
- CPU Inference Latency (ms per image)
- Ensemble comparison (probability averaging)
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from training.dataset import (
    CLASS_FULL_NAMES,
    HAM10000_CLASSES,
    DermatologyDataset,
    get_transforms,
)
from training.models import SUPPORTED_MODELS, build_model


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE).
    
    Measures the difference between predicted confidence and observed empirical accuracy.
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def benchmark_latency(model: torch.nn.Module, device: torch.device, repetitions: int = 30) -> float:
    """Measures single-image inference latency on target device in milliseconds."""
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)

    times = []
    with torch.no_grad():
        for _ in range(repetitions):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            times.append((time.perf_counter() - t0) * 1000.0)

    return float(np.median(times))


def evaluate_model_on_test_set(
    checkpoint_path: str,
    test_csv: str,
    device: Optional[torch.device] = None,
) -> Dict:
    """Evaluates a single model checkpoint on the held-out test split."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = ckpt.get("model_name", "efficientnet_b0")

    model, meta = build_model(
        model_name=model_name,
        num_classes=len(HAM10000_CLASSES),
        pretrained=False,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    model.eval()

    test_df = pd.read_csv(test_csv)
    test_ds = DermatologyDataset(test_df, transform=get_transforms("test"))
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for image, label, _ in test_loader:
            image = image.to(device)
            logits = model(image)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            pred = int(np.argmax(probs))

            all_probs.append(probs)
            all_preds.append(pred)
            all_labels.append(label.item())

    all_probs_np = np.array(all_probs)
    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels)

    # Metrics
    total = len(all_labels_np)
    accuracy = float((all_preds_np == all_labels_np).mean()) if total > 0 else 0.0
    macro_f1 = float(f1_score(all_labels_np, all_preds_np, average="macro", zero_division=0))
    ece = compute_ece(all_probs_np, all_labels_np)
    latency_ms = benchmark_latency(model, device)
    cm = confusion_matrix(all_labels_np, all_preds_np, labels=list(range(len(HAM10000_CLASSES))))

    report_dict = classification_report(
        all_labels_np,
        all_preds_np,
        labels=list(range(len(HAM10000_CLASSES))),
        target_names=HAM10000_CLASSES,
        output_dict=True,
        zero_division=0,
    )

    per_class_summary = {}
    for cls in HAM10000_CLASSES:
        cls_metrics = report_dict.get(cls, {})
        per_class_summary[cls] = {
            "full_name": CLASS_FULL_NAMES[cls],
            "precision": float(cls_metrics.get("precision", 0.0)),
            "recall_sensitivity": float(cls_metrics.get("recall", 0.0)),
            "f1_score": float(cls_metrics.get("f1-score", 0.0)),
            "support": int(cls_metrics.get("support", 0)),
        }

    results = {
        "model_name": model_name,
        "checkpoint_path": str(checkpoint_path),
        "test_samples": total,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "expected_calibration_error_ece": ece,
        "cpu_latency_ms": latency_ms,
        "confusion_matrix": cm.tolist(),
        "classes": HAM10000_CLASSES,
        "per_class_metrics": per_class_summary,
    }

    return results


def evaluate_ensemble(
    checkpoint_paths: List[str],
    test_csv: str,
    device: Optional[torch.device] = None,
) -> Dict:
    """Evaluates an ensemble of models using simple probability averaging."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models_loaded = []
    model_names = []
    for cp in checkpoint_paths:
        ckpt = torch.load(cp, map_location=device, weights_only=False)
        m_name = ckpt.get("model_name", "efficientnet_b0")
        model, _ = build_model(
            model_name=m_name,
            num_classes=len(HAM10000_CLASSES),
            pretrained=False,
            checkpoint_path=cp,
            device=device,
        )
        model.eval()
        models_loaded.append(model)
        model_names.append(m_name)

    test_df = pd.read_csv(test_csv)
    test_ds = DermatologyDataset(test_df, transform=get_transforms("test"))
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    ensemble_probs = []
    all_labels = []

    with torch.no_grad():
        for image, label, _ in test_loader:
            image = image.to(device)
            sample_probs = []
            for model in models_loaded:
                logits = model(image)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                sample_probs.append(probs)

            # Average probabilities across candidate models
            avg_prob = np.mean(sample_probs, axis=0)
            ensemble_probs.append(avg_prob)
            all_labels.append(label.item())

    ensemble_probs_np = np.array(ensemble_probs)
    ensemble_preds_np = np.argmax(ensemble_probs_np, axis=1)
    all_labels_np = np.array(all_labels)

    accuracy = float((ensemble_preds_np == all_labels_np).mean())
    macro_f1 = float(f1_score(all_labels_np, ensemble_preds_np, average="macro", zero_division=0))
    ece = compute_ece(ensemble_probs_np, all_labels_np)

    return {
        "ensemble_models": model_names,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "ece": ece,
        "total_test_samples": len(all_labels_np),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SkinSense AI - Model Evaluation & Calibration")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint .pt file")
    parser.add_argument("--test-csv", type=str, required=True, help="Path to held-out test split CSV")
    parser.add_argument("--output", type=str, default=None, help="Path to save evaluation JSON results")

    args = parser.parse_args()
    res = evaluate_model_on_test_set(args.checkpoint, args.test_csv)
    print("\n" + "=" * 50)
    print(f"Evaluation Results for: {res['model_name']}")
    print(f"Accuracy: {res['accuracy']:.4f} | Macro F1: {res['macro_f1']:.4f}")
    print(f"ECE (Calibration): {res['expected_calibration_error_ece']:.4f}")
    print(f"Latency: {res['cpu_latency_ms']:.2f} ms / image on CPU")
    print("=" * 50)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(res, f, indent=2)
        print(f"Saved results to {args.output}")
