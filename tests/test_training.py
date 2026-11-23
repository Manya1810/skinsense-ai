"""SkinSense AI - Automated Unit Tests for Dataset & Model Factory
Verifies:
- Patient-aware leakage-free splitting (images with same lesion_id stay together)
- Model building for all 4 candidate architectures (EfficientNet-B0, MobileNetV2, ResNet50, ViT-B/16)
- Synthetic dataset generation
"""

from pathlib import Path
import pandas as pd
import pytest
import torch

from training.dataset import (
    HAM10000_CLASSES,
    compute_class_weights,
    split_metadata_without_leakage,
)
from training.generate_dummy_data import create_demo_dataset
from training.models import SUPPORTED_MODELS, build_model


def test_synthetic_data_generation(tmp_path):
    dataset_path = create_demo_dataset(output_dir=str(tmp_path / "dummy_data"), samples_per_class=2)
    csv_file = dataset_path / "HAM10000_metadata.csv"
    assert csv_file.exists()
    df = pd.read_csv(csv_file)
    assert len(df) == 14  # 7 classes * 2 samples
    assert set(df["dx"].unique()) == set(HAM10000_CLASSES)


def test_leakage_free_split():
    # Construct dataframe with repeated lesion_ids to test grouping
    records = []
    for i in range(10):
        lesion = f"LESION_{i}"
        # 3 images per lesion
        for j in range(3):
            records.append({
                "lesion_id": lesion,
                "image_id": f"IMG_{i}_{j}",
                "dx": HAM10000_CLASSES[i % len(HAM10000_CLASSES)],
            })
    df = pd.DataFrame(records)

    train_df, val_df, test_df = split_metadata_without_leakage(
        df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42
    )

    train_lesions = set(train_df["lesion_id"])
    val_lesions = set(val_df["lesion_id"])
    test_lesions = set(test_df["lesion_id"])

    # Verify ZERO overlap between splits
    assert len(train_lesions.intersection(val_lesions)) == 0, "Leakage detected between train and val!"
    assert len(train_lesions.intersection(test_lesions)) == 0, "Leakage detected between train and test!"
    assert len(val_lesions.intersection(test_lesions)) == 0, "Leakage detected between val and test!"


def test_model_building_all_architectures():
    for m_name in SUPPORTED_MODELS:
        model, meta = build_model(model_name=m_name, num_classes=7, pretrained=False)
        assert meta["model_key"] == m_name
        assert meta["num_classes"] == 7
        assert meta["total_parameters"] > 1_000_000

        # Run forward pass on dummy tensor
        dummy = torch.randn(1, 3, 224, 224)
        out = model(dummy)
        assert out.shape == (1, 7)


if __name__ == "__main__":
    pytest.main(["-v", __file__])

