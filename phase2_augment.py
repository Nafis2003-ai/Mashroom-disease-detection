"""
Phase 2: Preprocessing & Data Augmentation
- Resize all images to 224x224
- Normalize pixel values to [0, 1]
- Augment TRAINING set only to reach ~2,000-2,500 images
  Techniques: horizontal flip, vertical flip, rotation, zoom,
              shear, brightness, width/height shift, Gaussian noise
- Val/Test sets are only resized + normalized (no augmentation)
- Outputs:
    dataset_augmented/  — augmented structured dataset
    phase2_report.txt   — full report
"""

import os
import shutil
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR     = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR    = os.path.join(BASE_DIR, "dataset_augmented")
REPORT_PATH   = os.path.join(BASE_DIR, "phase2_report.txt")

IMG_SIZE      = (224, 224)
RANDOM_SEED   = 42
TARGET_TRAIN  = 2400          # total augmented training images (across all classes)

CLASSES       = ["Healthy", "Single_Infected", "Mixed_Infected"]
SPLITS        = ["train", "val", "test"]

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(img: Image.Image) -> Image.Image:
    """Resize to 224x224 and ensure RGB."""
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    return img

def normalize_array(arr: np.ndarray) -> np.ndarray:
    """Normalize pixel values to [0, 1]."""
    return arr.astype(np.float32) / 255.0

# ── Augmentation ops (PIL-based, no heavy deps) ───────────────────────────────
def aug_hflip(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)

def aug_vflip(img):
    return img.transpose(Image.FLIP_TOP_BOTTOM)

def aug_rotate(img):
    angle = random.uniform(-30, 30)
    return img.rotate(angle, resample=Image.BILINEAR, expand=False)

def aug_zoom(img):
    factor = random.uniform(0.85, 1.15)
    w, h = img.size
    new_w, new_h = int(w * factor), int(h * factor)
    img_r = img.resize((new_w, new_h), Image.LANCZOS)
    # Center-crop or pad back to original size
    result = Image.new("RGB", (w, h))
    left = (new_w - w) // 2
    top  = (new_h - h) // 2
    if factor >= 1.0:
        result = img_r.crop((left, top, left + w, top + h))
    else:
        result.paste(img_r, (-left, -top))
    return result

def aug_brightness(img):
    factor = random.uniform(0.7, 1.3)
    return ImageEnhance.Brightness(img).enhance(factor)

def aug_contrast(img):
    factor = random.uniform(0.8, 1.2)
    return ImageEnhance.Contrast(img).enhance(factor)

def aug_shift(img):
    w, h = img.size
    dx = int(random.uniform(-0.1, 0.1) * w)
    dy = int(random.uniform(-0.1, 0.1) * h)
    return img.transform(img.size, Image.AFFINE,
                         (1, 0, -dx, 0, 1, -dy),
                         resample=Image.BILINEAR)

def aug_noise(img):
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, 8, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def aug_shear(img):
    shear = random.uniform(-0.15, 0.15)
    w, h = img.size
    return img.transform(img.size, Image.AFFINE,
                         (1, shear, 0, 0, 1, 0),
                         resample=Image.BILINEAR)

AUGMENTATIONS = [
    aug_hflip, aug_vflip, aug_rotate, aug_zoom,
    aug_brightness, aug_contrast, aug_shift, aug_noise, aug_shear
]

def augment_image(img: Image.Image) -> Image.Image:
    """Apply 2-4 random augmentation ops to an image."""
    ops = random.sample(AUGMENTATIONS, k=random.randint(2, 4))
    for op in ops:
        img = op(img)
    return img

# ── Save image ────────────────────────────────────────────────────────────────
def save_img(img: Image.Image, path: str):
    img.save(path, "JPEG", quality=95)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    lines = []
    log = lambda s="": (lines.append(s), print(s))

    log("=" * 72)
    log("  PHASE 2 REPORT — PREPROCESSING & DATA AUGMENTATION")
    log("=" * 72)

    # Clean output dir
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    for split in SPLITS:
        for cls in CLASSES:
            os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

    # ── Val & Test: preprocess only (resize + normalize stored as JPG) ──────
    log()
    log("STEP 2.1 — PREPROCESS VAL & TEST SETS (resize 224x224, RGB)")
    log("-" * 72)

    for split in ("val", "test"):
        for cls in CLASSES:
            src_dir  = os.path.join(INPUT_DIR,  split, cls)
            dst_dir  = os.path.join(OUTPUT_DIR, split, cls)
            files    = sorted(f for f in os.listdir(src_dir)
                              if f.lower().endswith(".jpg"))
            for fname in files:
                img = Image.open(os.path.join(src_dir, fname))
                img = preprocess(img)
                save_img(img, os.path.join(dst_dir, fname))
            log(f"  {split:<5} / {cls:<22} preprocessed {len(files):>3} images")

    # ── Train: preprocess originals + augment to reach TARGET_TRAIN ─────────
    log()
    log("STEP 2.2 — PREPROCESS + AUGMENT TRAINING SET")
    log(f"          Target total training images: {TARGET_TRAIN}")
    log("-" * 72)

    # Count original training images per class
    orig_counts = {}
    for cls in CLASSES:
        src_dir = os.path.join(INPUT_DIR, "train", cls)
        orig_counts[cls] = len([f for f in os.listdir(src_dir)
                                 if f.lower().endswith(".jpg")])

    total_orig = sum(orig_counts.values())

    # Calculate per-class target (proportional to original distribution)
    per_class_target = {}
    for cls in CLASSES:
        per_class_target[cls] = round(TARGET_TRAIN * orig_counts[cls] / total_orig)

    log(f"  Original training counts: " +
        " | ".join(f"{c}: {orig_counts[c]}" for c in CLASSES))
    log(f"  Per-class targets:        " +
        " | ".join(f"{c}: {per_class_target[c]}" for c in CLASSES))
    log()

    aug_stats = {}

    for cls in CLASSES:
        src_dir = os.path.join(INPUT_DIR,  "train", cls)
        dst_dir = os.path.join(OUTPUT_DIR, "train", cls)

        src_files = sorted(f for f in os.listdir(src_dir)
                           if f.lower().endswith(".jpg"))

        # Copy & preprocess originals first
        for fname in src_files:
            img = Image.open(os.path.join(src_dir, fname))
            img = preprocess(img)
            save_img(img, os.path.join(dst_dir, fname))

        originals_saved = len(src_files)
        need = per_class_target[cls] - originals_saved
        aug_saved = 0

        if need > 0:
            # Cycle through source images and augment
            pool = src_files * (need // len(src_files) + 2)
            random.shuffle(pool)
            for i, fname in enumerate(pool[:need]):
                img = Image.open(os.path.join(src_dir, fname))
                img = preprocess(img)
                img = augment_image(img)
                aug_name = f"aug_{i:04d}_{fname}"
                save_img(img, os.path.join(dst_dir, aug_name))
                aug_saved += 1

        total_cls = originals_saved + aug_saved
        aug_stats[cls] = {
            "original": originals_saved,
            "augmented": aug_saved,
            "total": total_cls
        }

        log(f"  {cls:<22}  orig={originals_saved:>3}  "
            f"augmented={aug_saved:>3}  total={total_cls:>4}")

    # ── Summary ──────────────────────────────────────────────────────────────
    log()
    log("STEP 2.3 — FINAL DATASET SUMMARY")
    log("-" * 72)

    grand_total = 0
    for split in SPLITS:
        split_total = 0
        for cls in CLASSES:
            d = os.path.join(OUTPUT_DIR, split, cls)
            n = len(os.listdir(d))
            split_total += n
            log(f"  dataset_augmented/{split:<5}/{cls:<22} {n:>4} images")
        log(f"  {'>> ' + split + ' subtotal':<33} {split_total:>4} images")
        log()
        grand_total += split_total

    log(f"  GRAND TOTAL: {grand_total} images")
    log()

    log("STEP 2.4 — AUGMENTATION TECHNIQUES APPLIED")
    log("-" * 72)
    techniques = [
        "Horizontal flip",
        "Vertical flip",
        "Rotation (up to ±30 degrees)",
        "Zoom (0.85x – 1.15x)",
        "Brightness adjustment (0.7x – 1.3x)",
        "Contrast adjustment (0.8x – 1.2x)",
        "Width/height shift (up to ±10%)",
        "Gaussian noise (std=8)",
        "Shear transformation (up to ±15%)",
    ]
    log("  Each augmented image receives 2–4 randomly selected ops from:")
    for t in techniques:
        log(f"    - {t}")

    log()
    log("STEP 2.5 — CLASS IMBALANCE NOTE")
    log("-" * 72)
    log("  Imbalance ratio (max/min) in training set:")
    train_counts = [aug_stats[c]["total"] for c in CLASSES]
    ratio = max(train_counts) / min(train_counts)
    log(f"    {ratio:.2f}  (threshold for concern: >2.0)")
    if ratio > 2.0:
        log("  >> Still imbalanced after augmentation.")
        log("     Action: Use class_weight={'Healthy':w0,'Single_Infected':w1,"
            "'Mixed_Infected':w2} in model.fit()")
        log("     Weights will be computed automatically in Phase 3.")
    else:
        log("  >> Class balance is acceptable after augmentation.")

    log()
    log("NORMALIZATION NOTE")
    log("-" * 72)
    log("  Images stored as JPG (uint8 0-255) in dataset_augmented/.")
    log("  During model training (Phase 3), normalization to [0,1] will be")
    log("  applied via rescale=1./255 in Keras ImageDataGenerator.")
    log("  This avoids storing float images and saves disk space.")

    log()
    log("=" * 72)
    log("  PHASE 2 COMPLETE")
    log(f"  Augmented dataset at : {OUTPUT_DIR}")
    log(f"  Report saved to      : {REPORT_PATH}")
    log("=" * 72)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
