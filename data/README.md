# SkinSense AI — Dataset Guide & Data Sources

## Overview
SkinSense AI benchmarks multi-model vision architectures on dermatological lesions.

## Supported Sources
1. **HAM10000 Dataset** (Human Against Machine with 10,000 training images):
   - **Primary scope:** Dermatoscopic pigmented skin lesions.
   - **Classes (7):**
     - `mel`: Cutaneous Melanoma (Malignant)
     - `nv`: Melanocytic Nevus (Benign Mole)
     - `bcc`: Basal Cell Carcinoma (Malignant)
     - `akiec`: Actinic Keratoses and Intraepithelial Carcinoma
     - `bkl`: Benign Keratosis-like Lesions
     - `df`: Dermatofibroma
     - `vasc`: Vascular Lesions
   - **Source:** Tschandl et al., *Scientific Data* (2018), ISIC Archive: https://challenge.isic-archive.com/data/

2. **Clinical Inflammatory & Fungal Conditions:**
   - `tinea_corporis`: Tinea Corporis / Dermatophytosis (Ringworm)
   - `acne`: Acne Vulgaris (Comedones, Pustules)
   - `psoriasis`: Plaque Psoriasis (Silvery-white scaly plaques)
   - `eczema`: Atopic Dermatitis (Pruritic xerotic patches)

## Data Governance & Leakage Prevention
- Splitting is performed **strictly by patient identifier** (`lesion_id` or `patient_id`) to prevent data leakage between train, validation, and held-out test splits.
- Data augmentations (random horizontal/vertical flips, rotations, slight color jitter) are applied **only** to training splits.
- Evaluation split remains completely untouched until final benchmarking.

