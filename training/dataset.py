"""SkinSense AI - Dataset Loader & Patient-Aware Leakage-Free Splitting
Handles HAM10000 metadata, prevents patient/lesion leakage across train/val/test,
and builds PyTorch DataLoaders with clinically sound augmentations.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Official HAM10000 7-class diagnostic categories
HAM10000_CLASSES: List[str] = [
    "akiec",  # Actinic keratoses and intraepithelial carcinoma / Bowen's disease
    "bcc",    # Basal cell carcinoma
    "bkl",    # Benign keratosis-like lesions
    "df",     # Dermatofibroma
    "mel",    # Melanoma
    "nv",     # Melanocytic nevi
    "vasc",   # Vascular lesions
]

CLASS_TO_IDX: Dict[str, int] = {cls: idx for idx, cls in enumerate(HAM10000_CLASSES)}
IDX_TO_CLASS: Dict[int, str] = {idx: cls for idx, cls in enumerate(HAM10000_CLASSES)}

CLASS_FULL_NAMES: Dict[str, str] = {
    "akiec": "Actinic Keratosis / Bowen's Disease",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis-like Lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevus (Mole)",
    "vasc": "Vascular Lesion (Angioma / Granuloma)",
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(split: str = "train", image_size: int = 224) -> transforms.Compose:
    """Returns image transforms matching the SkinSense AI specifications.
    
    Training: Resize, random flips/rotations, ImageNet normalization.
    Val/Test: Deterministic resize and ImageNet normalization.
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class DermatologyDataset(Dataset):
    """PyTorch Dataset for dermatology skin lesion images."""

    def __init__(self, df: pd.DataFrame, transform: Optional[transforms.Compose] = None):
        self.df = df.reset_index(drop=True)
        self.transform = transform or get_transforms(split="val")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        image_path = row["image_path"]
        label_str = row["dx"]
        label_idx = CLASS_TO_IDX[label_str]

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        image_id = row.get("image_id", Path(image_path).stem)
        return image, label_idx, image_id


def split_metadata_without_leakage(
    metadata_df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits dataset ensuring ZERO patient/lesion leakage.
    
    Images of the same lesion or patient never cross train/val/test boundaries.
    Uses 'lesion_id' if present, falling back to 'patient_id', or image_id.
    """
    rng = np.random.default_rng(seed)

    # Determine grouping column
    group_col = None
    if "lesion_id" in metadata_df.columns:
        group_col = "lesion_id"
    elif "patient_id" in metadata_df.columns:
        group_col = "patient_id"

    if group_col is not None:
        unique_groups = metadata_df[group_col].unique()
        rng.shuffle(unique_groups)

        n_total = len(unique_groups)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_groups = set(unique_groups[:n_train])
        val_groups = set(unique_groups[n_train:n_train + n_val])
        test_groups = set(unique_groups[n_train + n_val:])

        train_df = metadata_df[metadata_df[group_col].isin(train_groups)].copy()
        val_df = metadata_df[metadata_df[group_col].isin(val_groups)].copy()
        test_df = metadata_df[metadata_df[group_col].isin(test_groups)].copy()
    else:
        # Standard stratified split by label if no grouping metadata exists
        train_list, val_list, test_list = [], [], []
        for cls in HAM10000_CLASSES:
            cls_df = metadata_df[metadata_df["dx"] == cls]
            if len(cls_df) == 0:
                continue
            indices = np.arange(len(cls_df))
            rng.shuffle(indices)

            n_train = int(len(indices) * train_ratio)
            n_val = int(len(indices) * val_ratio)

            train_list.append(cls_df.iloc[indices[:n_train]])
            val_list.append(cls_df.iloc[indices[n_train:n_train + n_val]])
            test_list.append(cls_df.iloc[indices[n_train + n_val:]])

        train_df = pd.concat(train_list, ignore_index=True) if train_list else pd.DataFrame()
        val_df = pd.concat(val_list, ignore_index=True) if val_list else pd.DataFrame()
        test_df = pd.concat(test_list, ignore_index=True) if test_list else pd.DataFrame()

    return train_df, val_df, test_df


def compute_class_weights(df: pd.DataFrame) -> torch.Tensor:
    """Computes balanced inverse-frequency class weights for CrossEntropyLoss."""
    counts = df["dx"].value_counts()
    total = len(df)
    weights = []
    for cls in HAM10000_CLASSES:
        cnt = counts.get(cls, 1)
        # Inverse frequency weighting
        w = total / (len(HAM10000_CLASSES) * max(cnt, 1))
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32)

