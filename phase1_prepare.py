"""
Phase 1: Dataset Preparation
- Step 1.1: Audit & inventory (verify all images readable, detect corruption)
- Step 1.2: Class distribution report
- Step 1.3: Stratified 70/15/15 train/val/test split
- Outputs: dataset/ folder structure + phase1_report.txt
"""

import os
import shutil
import random
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CLASS_FOLDERS = {
    "Healthy":          os.path.join(BASE_DIR, "Healthy"),
    "Single_Infected":  os.path.join(BASE_DIR, "Single_Infected"),
    "Mixed_Infected":   os.path.join(BASE_DIR, "Mixed_Infected"),
}

OUTPUT_DIR   = os.path.join(BASE_DIR, "dataset")
REPORT_PATH  = os.path.join(BASE_DIR, "phase1_report.txt")

SPLITS       = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED  = 42

# ── Helpers ───────────────────────────────────────────────────────────────────
def verify_image(path):
    """Return (ok, reason). Tries to open and verify the image."""
    try:
        with Image.open(path) as img:
            img.verify()          # catches truncated / corrupt files
        return True, "ok"
    except Exception as e:
        return False, str(e)

def split_list(items, ratios, seed=42):
    """Stratified split of a list into train/val/test."""
    rng = random.Random(seed)
    items = items[:]
    rng.shuffle(items)
    n = len(items)
    n_train = int(n * ratios["train"])
    n_val   = int(n * ratios["val"])
    train   = items[:n_train]
    val     = items[n_train:n_train + n_val]
    test    = items[n_train + n_val:]
    return train, val, test

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    lines = []   # accumulate report lines
    log = lambda s="": (lines.append(s), print(s))

    log("=" * 72)
    log("  PHASE 1 REPORT — MUSHROOM DISEASE DATASET PREPARATION")
    log("=" * 72)

    # ── 1.1  Audit ─────────────────────────────────────────────────────────
    log()
    log("STEP 1.1 — AUDIT & INVENTORY")
    log("-" * 72)

    all_valid   = {}   # class → [valid paths]
    all_corrupt = {}   # class → [(path, reason)]

    total_found = 0
    total_valid = 0
    total_bad   = 0

    for cls, folder in CLASS_FOLDERS.items():
        valid   = []
        corrupt = []
        files   = sorted(f for f in os.listdir(folder)
                         if f.lower().endswith(".jpg"))
        total_found += len(files)

        for fname in files:
            fpath = os.path.join(folder, fname)
            ok, reason = verify_image(fpath)
            if ok:
                valid.append(fpath)
            else:
                corrupt.append((fpath, reason))

        all_valid[cls]   = valid
        all_corrupt[cls] = corrupt
        total_valid += len(valid)
        total_bad   += len(corrupt)

        log(f"  {cls:<20} Found: {len(files):>3}  |  "
            f"Valid: {len(valid):>3}  |  Corrupt: {len(corrupt):>3}")
        for path, reason in corrupt:
            log(f"    !! CORRUPT: {os.path.basename(path)} — {reason}")

    log()
    log(f"  Total images found : {total_found}")
    log(f"  Total valid        : {total_valid}")
    log(f"  Total corrupt/bad  : {total_bad}")
    if total_bad == 0:
        log("  >> All images passed integrity check.")

    # ── 1.2  Class Distribution ────────────────────────────────────────────
    log()
    log("STEP 1.2 — CLASS DISTRIBUTION")
    log("-" * 72)

    for cls, paths in all_valid.items():
        pct = len(paths) / total_valid * 100
        bar = "#" * int(pct / 2)
        log(f"  {cls:<20} {len(paths):>3} images  ({pct:5.1f}%)  |{bar}")

    log()
    counts = [len(v) for v in all_valid.values()]
    min_c, max_c = min(counts), max(counts)
    imbalance_ratio = max_c / min_c
    log(f"  Imbalance ratio (max/min): {imbalance_ratio:.2f}")
    if imbalance_ratio > 2.0:
        log("  >> WARNING: Significant class imbalance detected.")
        log("     Recommendation: Use class_weight in Keras or oversample.")
    else:
        log("  >> Class balance is acceptable.")

    # ── 1.3  Stratified Split ──────────────────────────────────────────────
    log()
    log("STEP 1.3 — STRATIFIED TRAIN / VAL / TEST SPLIT (70 / 15 / 15)")
    log("-" * 72)

    split_summary = {split: {} for split in SPLITS}

    for cls, paths in all_valid.items():
        train_p, val_p, test_p = split_list(paths, SPLITS, seed=RANDOM_SEED)
        split_summary["train"][cls] = train_p
        split_summary["val"][cls]   = val_p
        split_summary["test"][cls]  = test_p
        log(f"  {cls:<20}  train={len(train_p):>3}  val={len(val_p):>3}  "
            f"test={len(test_p):>3}  (total={len(paths)})")

    # Totals
    for split_name in ("train", "val", "test"):
        n = sum(len(v) for v in split_summary[split_name].values())
        log(f"  {'>> ' + split_name + ' total':<23} {n} images")

    # ── Copy files into dataset/ structure ────────────────────────────────
    log()
    log("STEP 1.3 (cont.) — COPYING FILES INTO dataset/ FOLDER")
    log("-" * 72)

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    copied = 0
    for split_name, cls_dict in split_summary.items():
        for cls, paths in cls_dict.items():
            dest_dir = os.path.join(OUTPUT_DIR, split_name, cls)
            os.makedirs(dest_dir, exist_ok=True)
            for src in paths:
                shutil.copy2(src, dest_dir)
                copied += 1

    log(f"  Copied {copied} images into: {OUTPUT_DIR}")
    log()
    log("  Final dataset/ structure:")
    for split_name in ("train", "val", "test"):
        for cls in CLASS_FOLDERS:
            d = os.path.join(OUTPUT_DIR, split_name, cls)
            n = len(os.listdir(d))
            log(f"    dataset/{split_name}/{cls:<22} {n:>3} images")

    # ── Done ───────────────────────────────────────────────────────────────
    log()
    log("=" * 72)
    log("  PHASE 1 COMPLETE")
    log(f"  Report saved to: {REPORT_PATH}")
    log("=" * 72)

    # Write report file
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
