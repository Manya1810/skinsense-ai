# SkinSense AI — Model Cards & Architectural Comparative Analysis

## 1. Candidate Architectures & Design Rationale

| Model Architecture | Parameter Count | Architectural Family | Core Rationale for Selection | Primary Metric Focus |
|---|---|---|---|---|
| **EfficientNet-B0** | ~4.0M | Compound Scaled CNN | Optimal parameter-to-performance efficiency via compound depth/width/resolution scaling. | Macro F1, high class-wise recall, low memory footprint. |
| **MobileNetV2** | ~2.2M | Inverted Residual CNN | Lightweight inverted bottleneck architecture designed for resource-constrained edge / CPU inference. | Inference speed, ultra-low latency (<30ms on CPU), mobile readiness. |
| **ResNet50** | ~23.5M | Deep Residual Network | Classical deep baseline with identity shortcut connections; proven dermoscopy benchmark in ISIC challenges. | High feature capacity, robust gradient propagation across layers. |
| **ViT-B/16** | ~85.8M | Vision Transformer | Self-attention over non-overlapping 16×16 image patches without inductive convolutional bias. | Global spatial context capture, comparison of CNN vs Transformer representations. |

---

## 2. Training Objective & Hyperparameter Specifications

- **Input Dimension:** `3 × 224 × 224` pixels (RGB).
- **Loss Function:** CrossEntropyLoss with inverse-frequency class-weighting to combat 67% `nv` class imbalance:
  $$w_c = \frac{N_{\text{total}}}{C \times N_c}$$
- **Optimizer:** AdamW (`weight_decay = 1e-2`).
- **Initial Learning Rate:** `1e-4` with Cosine Annealing learning rate schedule (`T_max = epochs`).
- **Batch Size:** `8` (optimized for laptop CPU / entry-level GPU memory constraints).
- **Epochs:** `5–10` (smoke-test/rapid evaluation), `20–30` (full convergence).
- **Regularization:** Dropout in final classification head (`p = 0.2` to `0.4`), train-only rotations/flips.
- **Checkpoint Selection Rule:** Checkpoint with highest **Validation Macro F1** is preserved. The test set is frozen and never accessed during training or hyperparameter tuning.

---

## 3. Evaluation Metrics Framework

### 1. Macro F1 Score (Primary Selection Criterion)
$$\text{Macro F1} = \frac{1}{C} \sum_{c=1}^{C} F1_c$$
Unlike micro-accuracy which is skewed by the majority `nv` class, Macro F1 gives equal weight to all 7 classes, penalizing models that fail to identify rare malignant lesions like Melanoma or Dermatofibroma.

### 2. Recall / Sensitivity for Malignant Classes
$$\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$$
In clinical triage, false negatives for Melanoma (`mel`) can be life-threatening. Maximizing recall for `mel` is prioritized over specificity.

### 3. Expected Calibration Error (ECE)
Checks whether the model's raw softmax output reflects true empirical probability:
$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$$
Raw softmax scores are uncalibrated heuristics. High softmax confidence does not equal diagnostic certainty.

### 4. Inference Latency Benchmark
Single-image execution time on standard CPU measured over 30 runs (median milliseconds per image).

---

## 4. Ensemble Strategy & Disagreement Handling

1. **Equal Probability Averaging:**
   $$P_{\text{ensemble}}(c | x) = \frac{1}{K} \sum_{k=1}^{K} P_k(c | x)$$
2. **Explicit Disagreement Surfacing:**
   If candidate models predict conflicting top classes (e.g. EfficientNet predicts `bkl` while ResNet50 predicts `mel`), the system flags a **Clinical Disagreement Warning**. Rather than arbitrarily hiding the conflict, the disagreement is highlighted in the clinician dossier to prompt human dermoscopic review.

---

## 5. Truthful Status & Viva Defense Rule

> [!IMPORTANT]
> - If no fine-tuned checkpoint file exists on disk in `models/checkpoints/`, the system states:
>   `Vision status: scaffold ready — no fine-tuned checkpoint loaded`.
> - Never invent fake accuracy numbers (e.g., claiming "our model achieved 98.4% accuracy") without real test-set logs and saved `.pt` weights.
> - The starter training pipeline includes synthetic benchmarking to demonstrate the training and evaluation mechanics live during examinations.

