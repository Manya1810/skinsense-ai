"""SkinSense AI - 4 Candidate Model Architectures & Factory
Implements:
1. EfficientNet-B0 (Efficient CNN baseline)
2. MobileNetV2 (Lightweight low-resource / CPU baseline)
3. ResNet50 (Residual connection baseline)
4. ViT-B/16 (Vision Transformer architecture)

Supports both Transfer Learning (fine-tuning from ImageNet pretrained weights)
and training from scratch, with custom classification heads for HAM10000.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torchvision import models

SUPPORTED_MODELS = ["efficientnet_b0", "mobilenet_v2", "resnet50", "vit_b_16"]

MODEL_METADATA: Dict[str, Dict[str, Any]] = {
    "efficientnet_b0": {
        "display_name": "EfficientNet-B0",
        "family": "Compound Scaled CNN",
        "paper": "Tan & Le (ICML 2019): Rethinking Model Scaling for Convolutional Neural Networks",
        "strengths": "State-of-the-art parameter efficiency, high Macro F1 per FLOP, robust mobile deployment.",
        "input_size": 224,
    },
    "mobilenet_v2": {
        "display_name": "MobileNetV2",
        "family": "Inverted Residual CNN",
        "paper": "Sandler et al. (CVPR 2018): Inverted Residuals and Linear Bottlenecks",
        "strengths": "Ultra-low memory footprint, fast inference on consumer laptops/CPUs, low latency.",
        "input_size": 224,
    },
    "resnet50": {
        "display_name": "ResNet50",
        "family": "Deep Residual Network",
        "paper": "He et al. (CVPR 2016): Deep Residual Learning for Image Recognition",
        "strengths": "Proven dermoscopy benchmark, deep feature extraction, skip-connections prevent vanishing gradient.",
        "input_size": 224,
    },
    "vit_b_16": {
        "display_name": "Vision Transformer (ViT-B/16)",
        "family": "Self-Attention Transformer",
        "paper": "Dosovitskiy et al. (ICLR 2021): An Image is Worth 16x16 Words",
        "strengths": "Global self-attention over 16x16 image patches, captures long-range spatial lesion context.",
        "input_size": 224,
    },
}


def build_model(
    model_name: str,
    num_classes: int = 7,
    pretrained: bool = True,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Builds and initializes one of the 4 candidate models with a modified classification head."""
    model_name_clean = model_name.lower().replace("-", "_")

    if model_name_clean == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )

    elif model_name_clean == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes),
        )

    elif model_name_clean == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_classes),
        )

    elif model_name_clean == "vit_b_16":
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = models.vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(in_features, num_classes),
        )

    else:
        raise ValueError(
            f"Unsupported model '{model_name}'. Choose from: {SUPPORTED_MODELS}"
        )

    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)

    if device:
        model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    meta = {
        **MODEL_METADATA.get(model_name_clean, {}),
        "model_key": model_name_clean,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "is_pretrained": pretrained,
        "num_classes": num_classes,
    }

    return model, meta
