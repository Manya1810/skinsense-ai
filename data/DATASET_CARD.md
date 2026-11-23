# SkinSense AI — Dataset Card: HAM10000

## 1. Dataset Overview & Official Source
- **Dataset Name:** HAM10000 ("Human Against Machine with 10000 training images")
- **Primary Host / Access Points:**
  - Harvard Dataverse: [https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
  - ISIC Archive: [https://www.isic-archive.com](https://www.isic-archive.com)
- **Official Citation:**
  > Tschandl, P., Rosendahl, C. & Kittler, H. *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions*. Sci Data 5, 180161 (2018). doi:10.1038/sdata.2018.161.
- **License / Permitted Use:** Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0). Intended for non-commercial academic research and educational evaluation.

---

## 2. Intended Classification Task & Scope
- **Task Scope:** Multi-class classification of **pigmented skin lesions** captured via dermatoscopy.
- **Explicit Negative Scope:** HAM10000 does **not** cover non-pigmented rash, viral exanthems, acne vulgaris, atopic eczema, psoriasis, urticaria, or deep cutaneous fungal infections. It must never be described as an "all skin disease" model.

---

## 3. Label Categories & Diagnostic Definitions (7 Classes)

| Class Code | Full Diagnostic Name | Clinical Category | Ground Truth Verification Method |
|---|---|---|---|
| `mel` | **Melanoma** | Malignant Neoplasm | Histopathology (Biopsy confirmed) |
| `bcc` | **Basal Cell Carcinoma** | Malignant Neoplasm | Histopathology |
| `akiec` | **Actinic Keratosis / Intraepithelial Carcinoma** | Pre-malignant / In Situ | Histopathology |
| `bkl` | **Benign Keratosis-like Lesions** (Solar lentigines / Seborrheic keratoses) | Benign Neoplasm | Histopathology / Confocal microscopy |
| `nv` | **Melanocytic Nevi** (Common Moles) | Benign Melanocytic | Follow-up / Dermoscopic consensus / Histo |
| `df` | **Dermatofibroma** | Benign Fibrohistiocytic | Dermoscopic consensus / Pinch sign |
| `vasc` | **Vascular Lesions** (Angiomas, pyogenic granulomas, hemorrhage) | Benign Vascular | Dermoscopic consensus |

---

## 4. Dataset Statistics & Extreme Class Imbalance
- **Total Images:** 10,015 dermoscopic JPEG images.
- **Unique Lesions:** 7,470 unique lesions across 10,015 images (meaning multiple images exist for single lesions).
- **Class Distribution:**
  - `nv` (Melanocytic Nevus): **6,705 images (67.0%)** — Dominant majority class.
  - `bkl` (Benign Keratosis): **1,099 images (11.0%)**
  - `mel` (Melanoma): **1,113 images (11.1%)**
  - `bcc` (Basal Cell Carcinoma): **514 images (5.1%)**
  - `akiec` (Actinic Keratosis): **327 images (3.3%)**
  - `vasc` (Vascular Lesion): **142 images (1.4%)**
  - `df` (Dermatofibroma): **115 images (1.1%)** — Extreme minority class.

> [!IMPORTANT]
> Because `nv` represents 67% of the dataset, a naive model predicting only `nv` would achieve 67% raw accuracy while having **0% sensitivity on life-threatening Melanoma**. Therefore, models MUST be evaluated using **Macro F1**, class-wise recall, and balanced loss weighting!

---

## 5. Patient-Aware Leakage Prevention Strategy
### Leakage Hazard
Multiple dermoscopic captures exist for the same lesion (different magnifications, lighting, or angles) and the same patient has multiple distinct lesions. Random splitting on image filenames spreads the same lesion across training and test partitions, resulting in artificial memory inflation rather than true generalization.

### Resolution
The dataset loader groups by `lesion_id`:
1. All images sharing a `lesion_id` or `patient_id` are assigned exclusively to a single split (Train, Val, or Test).
2. Train split: ~70% of unique lesions.
3. Validation split: ~15% of unique lesions (used for hyperparameter tuning & best checkpoint selection).
4. Held-out Test split: ~15% of unique lesions (frozen until single final evaluation).

---

## 6. Preprocessing & Data Augmentation Specifications
1. **Resolution & Decoding:** Decoded as 3-channel RGB, resized to `224 × 224` pixels.
2. **Standardization:** PyTorch ImageNet tensor normalization:
   - Mean: `[0.485, 0.456, 0.406]`
   - Std: `[0.229, 0.224, 0.225]`
3. **Training-Only Augmentations:**
   - Random Horizontal & Vertical Flips (`p = 0.5`)
   - Small Rotation (`degrees = 15°`)
   - Mild Color Jitter (Brightness/Contrast ± 10%)
4. **Test & Validation Splits:** Deterministic resizing and normalization only. No destructive color distortions that alter melanin pigmentation patterns.

---

## 7. Known Biases & Clinical Limitations
1. **Demographic / Skin Tone Bias:** Images predominantly collected from clinical dermatology centers in Vienna (Austria) and Queensland (Australia). The cohort has an overwhelming majority of light Fitzpatrick skin types (FST I–III) and minimal representation of darker phototypes (FST V–VI).
2. **Imaging Modality Gap:** Dermoscopic images differ substantially from ordinary smartphone camera photos (dermoscopy uses polarized epiluminescence microscopy to eliminate stratum corneum surface reflection).
3. **Clinical Validation Status:** For research and educational decision support only. Not approved as a medical device for standalone clinical diagnosis.

