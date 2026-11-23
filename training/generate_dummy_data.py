"""SkinSense AI - Synthetic Mini HAM10000 Dataset Generator
Generates a small valid HAM10000-structured dataset for smoke testing,
unit testing, and live presentation/viva demos without needing a 3GB download.
"""

from pathlib import Path
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

HAM10000_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
LOCALIZATIONS = ["back", "lower extremity", "trunk", "upper extremity", "face", "abdomen", "chest"]
SEXES = ["male", "female", "unknown"]


def generate_synthetic_lesion_image(cls: str, size: int = 224) -> Image.Image:
    """Generates a synthetic dermoscopic-like circular lesion on skin-tone background."""
    # Base skin tone
    bg_r = random.randint(210, 245)
    bg_g = random.randint(180, 215)
    bg_b = random.randint(160, 195)
    img = Image.new("RGB", (size, size), color=(bg_r, bg_g, bg_b))
    draw = ImageDraw.Draw(img)

    # Center lesion
    cx, cy = size // 2 + random.randint(-10, 10), size // 2 + random.randint(-10, 10)
    rx = random.randint(30, 65)
    ry = random.randint(30, 65)

    # Color signature by class
    if cls == "mel":
        # Dark, irregular, dark brown/black with blue-grey hints
        lesion_color = (random.randint(30, 60), random.randint(20, 45), random.randint(20, 40))
    elif cls == "bcc":
        # Pearly pink/reddish translucent
        lesion_color = (random.randint(180, 210), random.randint(100, 130), random.randint(110, 140))
    elif cls == "vasc":
        # Ruby red or dark purple
        lesion_color = (random.randint(160, 210), random.randint(20, 50), random.randint(30, 60))
    elif cls == "bkl":
        # Tan/yellow-brown 'stuck on'
        lesion_color = (random.randint(110, 140), random.randint(85, 115), random.randint(50, 75))
    elif cls == "akiec":
        # Rough scaly erythematous pink/sand
        lesion_color = (random.randint(190, 220), random.randint(130, 160), random.randint(110, 130))
    elif cls == "df":
        # Pale center, brown halo
        lesion_color = (random.randint(140, 165), random.randint(100, 125), random.randint(80, 100))
    else:  # nv (nevus)
        # Uniform light/medium brown
        lesion_color = (random.randint(90, 120), random.randint(60, 90), random.randint(40, 65))

    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=lesion_color)

    # Add a little texture/noise
    img_np = np.array(img).astype(np.int16)
    noise = np.random.normal(0, 8, img_np.shape).astype(np.int16)
    img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(img_np)


def create_demo_dataset(output_dir: str = "data/ham10000_sample", samples_per_class: int = 6) -> Path:
    """Creates a mini HAM10000 dataset with official schema and patient IDs."""
    out_path = Path(output_dir)
    images_dir = out_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    records = []
    image_idx = 1
    patient_idx = 100

    for cls in HAM10000_CLASSES:
        for i in range(samples_per_class):
            image_id = f"ISIC_{image_idx:07d}"
            # Group every 2 images into the same patient/lesion to test patient-aware split
            lesion_id = f"HAM_000{patient_idx + (i // 2)}"
            img_file = images_dir / f"{image_id}.jpg"

            img = generate_synthetic_lesion_image(cls)
            img.save(img_file, format="JPEG", quality=90)

            records.append({
                "lesion_id": lesion_id,
                "image_id": image_id,
                "dx": cls,
                "dx_type": "histo",
                "age": random.randint(25, 75),
                "sex": random.choice(SEXES),
                "localization": random.choice(LOCALIZATIONS),
                "image_path": str(img_file.resolve()),
            })
            image_idx += 1
        patient_idx += 10

    metadata_df = pd.DataFrame(records)
    csv_path = out_path / "HAM10000_metadata.csv"
    metadata_df.to_csv(csv_path, index=False)
    print(f"Created synthetic dataset at '{out_path}' with {len(records)} images across 7 classes.")
    return out_path


if __name__ == "__main__":
    create_demo_dataset()

