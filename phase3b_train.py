"""
phase3b_train.py — GPU-Optimised Mushroom Disease Classification
================================================================
Target  : ≥95% validation accuracy
Hardware: Google Colab T4  OR  NVIDIA RTX 4060 (8 GB)  ← recommended
Runtime : ~2 hours on GPU

WHY PHASE 3 FAILED:
  ✗ Training generator had NO online augmentation (rescale only!)
    → model memorised the same 2400 images every epoch → overfitting
  ✗ CPU training → transfer learning models never properly converged
  ✗ Only last 30 layers unfrozen → shallow fine-tuning

WHAT THIS SCRIPT FIXES:
  ✓ Heavy online augmentation  (flip, rotate, zoom, brightness, contrast, shift)
  ✓ MixUp augmentation         (+2-3% on small datasets per papers)
  ✓ EfficientNetV2S + DenseNet121  (paper-proven 95-96% on fungi)
  ✓ 3-stage fine-tuning        (head → top-60% → full unfreeze)
  ✓ Cosine LR decay + warmup   (smoother convergence)
  ✓ Label smoothing 0.1        (reduces overconfidence)
  ✓ L2 regularisation + deeper head
  ✓ Mixed precision fp16       (~2x speed, ~half VRAM on GPU)
  ✓ Test-Time Augmentation     (+1-2% free accuracy at inference)
  ✓ Ensemble of top 2 models   (+1-2% over single best model)

REFERENCES:
  - DenseNet121: 96.11% on fungi  (PMC 2024, DOI:10.3390/s24227189)
  - EfficientNetV2: 97%+ on plant disease  (TechScience CSSE 2024)
  - Swin+DenseNet: 95.57% test accuracy  (Springer 2025)
"""

# ── std lib ──────────────────────────────────────────────────────────────────
import os, json, time, math, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (classification_report, confusion_matrix,
                              ConfusionMatrixDisplay)

# ── TensorFlow ───────────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers
from tensorflow.keras.applications import EfficientNetV2S, DenseNet121

# ════════════════════════════════════════════════════════════════════════════
#  0. GPU SETUP  (run this before any other TF calls)
# ════════════════════════════════════════════════════════════════════════════
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    keras.mixed_precision.set_global_policy("mixed_float16")
    print(f"✓ GPU : {[g.name for g in gpus]}")
    print("✓ Mixed precision (fp16) ON")
else:
    print("⚠  No GPU detected — CPU mode (very slow, not recommended)")

# ════════════════════════════════════════════════════════════════════════════
#  1. CONFIG  ← only section you need to edit
# ════════════════════════════════════════════════════════════════════════════

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "dataset_augmented"   # must have train/ val/ test/
MODELS_DIR  = BASE_DIR / "models"
PLOTS_DIR   = BASE_DIR / "plots"
REPORT_PATH = BASE_DIR / "phase3b_report.txt"
JSON_PATH   = BASE_DIR / "phase3b_results.json"

MODELS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# ── Image / batch ────────────────────────────────────────────────────────────
IMG_SIZE   = 224    # 224×224 — works for EfficientNetV2S + DenseNet121
BATCH_SIZE = 32     # RTX 4060 (8 GB) + Colab T4 safe with fp16
                    # ← reduce to 16 if you get OOM errors

NUM_CLASSES = 3
SEED        = 42

# ── Fine-tuning schedule ─────────────────────────────────────────────────────
#   Stage 1: head only          (fast convergence)
#   Stage 2: top 60% unfrozen   (domain adaptation)
#   Stage 3: full unfreeze      (final squeeze)
S1_EPOCHS, S1_LR, S1_PAT, S1_WARM = 20, 1e-3,  8,  3
S2_EPOCHS, S2_LR, S2_PAT, S2_WARM = 40, 1e-4, 10,  3
S3_EPOCHS, S3_LR, S3_PAT, S3_WARM = 30, 1e-5, 12,  2

# ── Regularisation ───────────────────────────────────────────────────────────
L2_REG          = 1e-4
LABEL_SMOOTHING = 0.1
DROP_1          = 0.40
DROP_2          = 0.30
MIXUP_ALPHA     = 0.30   # 0 = disable MixUp

# ── Evaluation ───────────────────────────────────────────────────────────────
TTA_STEPS = 8     # test-time augmentation passes (+ 1 clean pass)

tf.random.set_seed(SEED)
np.random.seed(SEED)

# ════════════════════════════════════════════════════════════════════════════
#  2. DATA PIPELINE
# ════════════════════════════════════════════════════════════════════════════

# Online augmentation applied during training only (NOT cached → fresh every epoch)
DATA_AUG = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.25),           # ±45°
    layers.RandomZoom((-0.15, 0.15)),
    layers.RandomBrightness(0.25),
    layers.RandomContrast(0.25),
    layers.RandomTranslation(0.1, 0.1),   # ±10% width & height shift
], name="online_aug")


@tf.function
def mixup_batch(images, labels, alpha=MIXUP_ALPHA):
    """Blend pairs within a batch.  alpha=0 disables."""
    lam = tf.random.uniform([], alpha, 1.0 - alpha) if alpha > 0 else tf.constant(1.0)
    idx = tf.random.shuffle(tf.range(tf.shape(images)[0]))
    return (lam * images + (1.0 - lam) * tf.gather(images, idx),
            lam * labels + (1.0 - lam) * tf.gather(labels, idx))


def load_train_ds() -> tf.data.Dataset:
    """Training split: augment online + MixUp — never cached."""
    ds = keras.utils.image_dataset_from_directory(
        str(DATA_DIR / "train"),
        labels="inferred", label_mode="categorical",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, shuffle=True, seed=SEED,
    )
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda x, y: (DATA_AUG(x, training=True), y),
                num_parallel_calls=tf.data.AUTOTUNE)
    if MIXUP_ALPHA > 0:
        ds = ds.map(mixup_batch, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def load_eval_ds(split: str) -> tf.data.Dataset:
    """Val / test split: normalise only, cached for speed."""
    ds = keras.utils.image_dataset_from_directory(
        str(DATA_DIR / split),
        labels="inferred", label_mode="categorical",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, shuffle=False,
    )
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.cache().prefetch(tf.data.AUTOTUNE)


def get_class_weights() -> dict:
    """Balanced class weights from training directory."""
    train_dir = DATA_DIR / "train"
    class_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    exts = {".jpg", ".jpeg", ".png"}
    counts = [sum(1 for f in d.iterdir() if f.suffix.lower() in exts)
              for d in class_dirs]
    total = sum(counts)
    n = len(counts)
    return {i: total / (n * c) for i, c in enumerate(counts)}


def get_class_names() -> list:
    return sorted(d.name for d in (DATA_DIR / "val").iterdir() if d.is_dir())


# ════════════════════════════════════════════════════════════════════════════
#  3. MODEL BUILDERS
# ════════════════════════════════════════════════════════════════════════════

def _head(x) -> layers.Layer:
    """Shared classifier head with L2 regularisation."""
    reg = regularizers.l2(L2_REG)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROP_1)(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(DROP_2)(x)
    # dtype=float32 for numerical stability with mixed precision
    return layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32")(x)


def build_efficientnetv2s():
    base = EfficientNetV2S(weights="imagenet", include_top=False,
                           input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    out = _head(base(inp, training=False))
    return models.Model(inp, out, name="EfficientNetV2S"), base


def build_densenet121():
    base = DenseNet121(weights="imagenet", include_top=False,
                       input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    out = _head(base(inp, training=False))
    return models.Model(inp, out, name="DenseNet121"), base


def build_custom_cnn_v2() -> models.Model:
    """Improved Custom CNN: double conv blocks + L2 regularisation."""
    reg = regularizers.l2(L2_REG)

    def conv_block(x, filters, drop=0.2):
        x = layers.Conv2D(filters, 3, padding="same", kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(2)(x)
        return layers.Dropout(drop)(x)

    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = conv_block(inp,  64, drop=0.2)
    x = conv_block(x,   128, drop=0.2)
    x = conv_block(x,   256, drop=0.3)
    x = conv_block(x,   512, drop=0.3)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROP_1)(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(DROP_2)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32")(x)
    return models.Model(inp, out, name="Custom_CNN_v2")


# ════════════════════════════════════════════════════════════════════════════
#  4. COSINE LR WITH WARMUP
# ════════════════════════════════════════════════════════════════════════════

class CosineWarmup(callbacks.Callback):
    """Sets LR to: linear warmup for `warmup` epochs, then cosine decay."""
    def __init__(self, peak_lr, total_epochs, warmup=5, min_lr=1e-7):
        super().__init__()
        self.peak_lr = peak_lr
        self.total   = total_epochs
        self.warmup  = warmup
        self.min_lr  = min_lr

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup:
            lr = self.peak_lr * (epoch + 1) / max(self.warmup, 1)
        else:
            prog = (epoch - self.warmup) / max(1, self.total - self.warmup)
            lr   = self.min_lr + (self.peak_lr - self.min_lr) * \
                   0.5 * (1.0 + math.cos(math.pi * prog))
        keras.backend.set_value(self.model.optimizer.learning_rate, lr)
        if epoch % 10 == 0:
            print(f"    LR = {lr:.2e}")


# ════════════════════════════════════════════════════════════════════════════
#  5. TRAINING: 3-STAGE FINE-TUNE
# ════════════════════════════════════════════════════════════════════════════

def _loss_fn():
    return keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)


def _run_stage(model, train_ds, val_ds, cw, ckpt,
               lr, epochs, patience, warmup, tag, log):
    model.compile(
        optimizer=optimizers.Adam(lr),
        loss=_loss_fn(),
        metrics=["accuracy"],
    )
    h = model.fit(
        train_ds, validation_data=val_ds,
        epochs=epochs, class_weight=cw,
        callbacks=[
            callbacks.ModelCheckpoint(str(ckpt), monitor="val_accuracy",
                                      save_best_only=True, verbose=0),
            callbacks.EarlyStopping(monitor="val_accuracy", patience=patience,
                                    restore_best_weights=True, verbose=1),
            CosineWarmup(lr, total_epochs=epochs, warmup=warmup),
        ],
        verbose=1,
    )
    best = max(h.history["val_accuracy"])
    log(f"    {tag} best val_acc = {best*100:.2f}%")
    return h, best


def train_transfer(name, build_fn, train_ds, val_ds, cw, log):
    log(f"\n{'='*66}")
    log(f"  TRAINING : {name}")
    log(f"{'='*66}")
    t0   = time.time()
    model, base = build_fn()
    ckpt = MODELS_DIR / f"{name}_best.keras"
    log(f"  Params : {model.count_params():,}")
    histories = []

    # Stage 1 — head only
    log(f"\n  [S1] Head only | LR={S1_LR:.0e} | max {S1_EPOCHS} ep | patience={S1_PAT}")
    h1, b1 = _run_stage(model, train_ds, val_ds, cw, ckpt,
                        S1_LR, S1_EPOCHS, S1_PAT, S1_WARM, "S1", log)
    histories.append(("Stage 1", h1))

    # Stage 2 — unfreeze top 60 %
    log(f"\n  [S2] Top 60% unfrozen | LR={S2_LR:.0e} | max {S2_EPOCHS} ep | patience={S2_PAT}")
    cut = int(len(base.layers) * 0.4)
    for l in base.layers[:cut]: l.trainable = False
    for l in base.layers[cut:]: l.trainable = True
    h2, b2 = _run_stage(model, train_ds, val_ds, cw, ckpt,
                        S2_LR, S2_EPOCHS, S2_PAT, S2_WARM, "S2", log)
    histories.append(("Stage 2", h2))

    # Stage 3 — full unfreeze
    log(f"\n  [S3] Full unfreeze | LR={S3_LR:.0e} | max {S3_EPOCHS} ep | patience={S3_PAT}")
    base.trainable = True
    h3, b3 = _run_stage(model, train_ds, val_ds, cw, ckpt,
                        S3_LR, S3_EPOCHS, S3_PAT, S3_WARM, "S3", log)
    histories.append(("Stage 3", h3))

    elapsed = (time.time() - t0) / 60
    best    = max(b1, b2, b3)

    model = keras.models.load_model(str(ckpt))   # reload best weights
    _save_plot(histories, name)

    log(f"\n  ✓ Best val accuracy : {best*100:.2f}%")
    log(f"  ✓ Training time     : {elapsed:.1f} min")
    return model, {"model": name, "val_accuracy": round(best, 4),
                   "time_min": round(elapsed, 1)}


def train_custom_cnn(train_ds, val_ds, cw, log):
    name = "Custom_CNN_v2"
    log(f"\n{'='*66}")
    log(f"  TRAINING : {name}")
    log(f"{'='*66}")
    t0   = time.time()
    model = build_custom_cnn_v2()
    ckpt  = MODELS_DIR / f"{name}_best.keras"
    log(f"  Params : {model.count_params():,}")

    # Single stage — train from scratch with long schedule
    total_epochs = S1_EPOCHS + S2_EPOCHS   # 60 epochs max
    h, best = _run_stage(model, train_ds, val_ds, cw, ckpt,
                         S1_LR, total_epochs, patience=12,
                         warmup=5, tag="CNN", log=log)

    elapsed = (time.time() - t0) / 60
    model   = keras.models.load_model(str(ckpt))
    _save_plot([("Full", h)], name)

    log(f"\n  ✓ Best val accuracy : {best*100:.2f}%")
    log(f"  ✓ Training time     : {elapsed:.1f} min")
    return model, {"model": name, "val_accuracy": round(best, 4),
                   "time_min": round(elapsed, 1)}


# ════════════════════════════════════════════════════════════════════════════
#  6. EVALUATION: TTA + ENSEMBLE
# ════════════════════════════════════════════════════════════════════════════

def _tta_predict(model, ds, n_tta=TTA_STEPS):
    """
    Predict with Test-Time Augmentation.
    Returns (preds [N, C], labels [N, C]).
    """
    all_preds, all_labels = [], []
    for images, labels in ds:
        # Clean pass
        p = model(images, training=False).numpy().astype(np.float32)
        # TTA passes
        for _ in range(n_tta):
            aug = DATA_AUG(images, training=True)
            p  += model(aug, training=False).numpy().astype(np.float32)
        p /= (n_tta + 1)
        all_preds.append(p)
        all_labels.append(labels.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def evaluate_model(model, ds, name, log, class_names, tta=True):
    tag = f"TTA×{TTA_STEPS}" if tta else "no TTA"
    log(f"\n  Evaluating {name} ({tag}) …")
    preds, labels = _tta_predict(model, ds, n_tta=TTA_STEPS if tta else 0)

    pred_cls = np.argmax(preds,  axis=1)
    true_cls = np.argmax(labels, axis=1)
    acc      = np.mean(pred_cls == true_cls)

    log(f"  Accuracy : {acc*100:.2f}%")
    log(classification_report(true_cls, pred_cls, target_names=class_names))

    # Confusion matrix
    cm   = confusion_matrix(true_cls, pred_cls)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, colorbar=False)
    ax.set_title(f"{name} ({tag})")
    plt.tight_layout()
    fig.savefig(str(PLOTS_DIR / f"{name}_cm.png"), dpi=120)
    plt.close()

    return acc, preds, labels


def evaluate_ensemble(preds_list, labels, log, class_names, split_name):
    avg_preds = np.mean(preds_list, axis=0)
    pred_cls  = np.argmax(avg_preds,  axis=1)
    true_cls  = np.argmax(labels, axis=1)
    acc       = np.mean(pred_cls == true_cls)

    log(f"\n  ╔══════════════════════════════════════════════════╗")
    log(f"  ║  ENSEMBLE {split_name:<8} accuracy (TTA) : {acc*100:>6.2f}%  ║")
    log(f"  ╚══════════════════════════════════════════════════╝")
    log(classification_report(true_cls, pred_cls, target_names=class_names))
    return acc


# ════════════════════════════════════════════════════════════════════════════
#  7. PLOTTING
# ════════════════════════════════════════════════════════════════════════════

def _save_plot(named_histories, model_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f"{model_name} — Training Curves", fontsize=14)
    offset = 0
    for label, h in named_histories:
        ep = range(offset + 1, offset + len(h.history["loss"]) + 1)
        ax1.plot(ep, h.history["loss"],         label=f"{label} train")
        ax1.plot(ep, h.history["val_loss"],     label=f"{label} val", ls="--")
        ax2.plot(ep, h.history["accuracy"],     label=f"{label} train")
        ax2.plot(ep, h.history["val_accuracy"], label=f"{label} val",  ls="--")
        offset += len(h.history["loss"])
    for ax, title in [(ax1, "Loss"), (ax2, "Accuracy")]:
        ax.set(title=title, xlabel="Epoch")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(str(PLOTS_DIR / f"{model_name}_curves.png"), dpi=120)
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
#  8. MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    lines: list = []
    def log(s=""):
        lines.append(str(s)); print(s)

    log("=" * 66)
    log("  PHASE 3B — GPU-OPTIMISED TRAINING  |  Target: ≥95% val acc")
    log("=" * 66)
    log(f"  TensorFlow  : {tf.__version__}")
    log(f"  GPU         : {[g.name for g in tf.config.list_physical_devices('GPU')] or 'None (CPU)'}")
    log(f"  Precision   : {keras.mixed_precision.global_policy().name}")
    log(f"  IMG_SIZE    : {IMG_SIZE}×{IMG_SIZE}")
    log(f"  BATCH_SIZE  : {BATCH_SIZE}")
    log(f"  MIXUP_ALPHA : {MIXUP_ALPHA}")
    log(f"  LABEL_SMOOTH: {LABEL_SMOOTHING}")
    log(f"  TTA_STEPS   : {TTA_STEPS}")

    # ── Load data ────────────────────────────────────────────────────────
    log("\n  Loading datasets …")
    train_ds    = load_train_ds()
    val_ds      = load_eval_ds("val")
    test_ds     = load_eval_ds("test")
    cw          = get_class_weights()
    class_names = get_class_names()

    log(f"  Classes      : {class_names}")
    log(f"  Class weights: { {class_names[i]: round(w, 3) for i, w in cw.items()} }")

    trained: list   = []   # [(name, model)]
    results: list   = []

    # ── Train models ─────────────────────────────────────────────────────
    m1, r1 = train_transfer("EfficientNetV2S", build_efficientnetv2s,
                             train_ds, val_ds, cw, log)
    trained.append(("EfficientNetV2S", m1)); results.append(r1)

    m2, r2 = train_transfer("DenseNet121", build_densenet121,
                             train_ds, val_ds, cw, log)
    trained.append(("DenseNet121", m2)); results.append(r2)

    m3, r3 = train_custom_cnn(train_ds, val_ds, cw, log)
    trained.append(("Custom_CNN_v2", m3)); results.append(r3)

    # ── Evaluate on VAL ──────────────────────────────────────────────────
    log("\n" + "=" * 66)
    log("  EVALUATION — VALIDATION SET (with TTA)")
    log("=" * 66)

    val_preds_list = []
    val_labels_ref = None
    for name, model in trained:
        acc, preds, labels = evaluate_model(
            model, val_ds, name, log, class_names, tta=True)
        val_preds_list.append(preds)
        if val_labels_ref is None:
            val_labels_ref = labels
        next(r for r in results if r["model"] == name)["val_acc_tta"] = round(acc, 4)

    ens_val = evaluate_ensemble(val_preds_list, val_labels_ref, log,
                                class_names, split_name="VAL  ")

    # ── Evaluate on TEST ─────────────────────────────────────────────────
    log("\n" + "=" * 66)
    log("  EVALUATION — TEST SET (held-out)")
    log("=" * 66)

    test_preds_list = []
    test_labels_ref = None
    for name, model in trained:
        acc, preds, labels = evaluate_model(
            model, test_ds, f"{name}_TEST", log, class_names, tta=True)
        test_preds_list.append(preds)
        if test_labels_ref is None:
            test_labels_ref = labels
        next(r for r in results if r["model"] == name)["test_acc_tta"] = round(acc, 4)

    ens_test = evaluate_ensemble(test_preds_list, test_labels_ref, log,
                                 class_names, split_name="TEST ")

    # ── Summary table ────────────────────────────────────────────────────
    log("\n" + "=" * 66)
    log("  PHASE 3B FINAL SUMMARY")
    log("=" * 66)
    log(f"  {'Model':<22} {'Val(raw)':>9} {'Val(TTA)':>9} {'Test(TTA)':>10} {'Min':>6}")
    log(f"  {'-'*22} {'-'*9} {'-'*9} {'-'*10} {'-'*6}")
    for r in sorted(results, key=lambda x: x.get("val_acc_tta", 0), reverse=True):
        log(f"  {r['model']:<22}"
            f"  {r['val_accuracy']*100:>7.2f}%"
            f"  {r.get('val_acc_tta',  0)*100:>7.2f}%"
            f"  {r.get('test_acc_tta', 0)*100:>8.2f}%"
            f"  {r['time_min']:>5.1f}")
    log(f"\n  Ensemble val  (TTA) : {ens_val*100:.2f}%")
    log(f"  Ensemble test (TTA) : {ens_test*100:.2f}%")

    target = ens_val >= 0.95
    log(f"\n  {'✅  TARGET MET  (≥95%)' if target else '⚠️  Target not met — see recommendations below'}")

    if not target:
        log("\n  RECOMMENDATIONS IF STILL BELOW 95%:")
        log("    1. Increase image size to 300×300 (edit IMG_SIZE=300)")
        log("    2. Reduce BATCH_SIZE to 16 and re-run Stage 3 with LR_S3=5e-6")
        log("    3. Add EfficientNetB4 to the registry (larger model)")
        log("    4. Increase S2_EPOCHS to 60 (more time for domain adaptation)")

    # ── Persist ──────────────────────────────────────────────────────────
    results.append({"model": "Ensemble",
                    "val_acc_tta":  round(ens_val,  4),
                    "test_acc_tta": round(ens_test, 4)})
    with open(str(JSON_PATH), "w") as f:
        json.dump(results, f, indent=2)
    with open(str(REPORT_PATH), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    log(f"\n  Report  → {REPORT_PATH}")
    log(f"  JSON    → {JSON_PATH}")
    log(f"  Models  → {MODELS_DIR}/")
    log(f"  Plots   → {PLOTS_DIR}/")
    log("=" * 66)
    log("  PHASE 3B COMPLETE")
    log("=" * 66)


if __name__ == "__main__":
    main()
