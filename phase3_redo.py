"""
phase3_redo.py  —  GPU-Optimised Retraining (Keras 3 compatible)
=================================================================
Run on Google Colab with T4 GPU runtime.

KEY FIX vs previous version:
  - Lambda preprocessing layers REMOVED from model architecture
  - Preprocessing now applied in tf.data pipeline (per model)
  - Models save and load cleanly in Keras 3 with no Lambda issues

Other improvements over phase3_train.py (CPU baseline):
  1. Correct per-model preprocessing  →  fixes EfficientNet 44% bug
  2. Focal Loss + label smoothing 0.1 →  handles 2.16x class imbalance
  3. EfficientNetB3 replaces B0        →  stronger features
  4. Warmup (5 ep) + Cosine LR decay  →  smoother convergence
  5. Mixed precision (fp16)            →  2-3x GPU speedup
  6. Unfreeze last 50 layers           →  deeper fine-tuning
  7. Better head: GAP→BN→512→BN→256  →  stronger classifier
  8. tf.data with prefetch             →  no GPU starvation
  9. Drive backup after EVERY model   →  no data loss on disconnect
 10. Batch size 32, up to 100 epochs  →  larger training budget

COLAB SETUP:
  1. Runtime → Change runtime type → T4 GPU
  2. Mount Drive and unzip dataset:
       from google.colab import drive
       drive.mount('/content/drive')
       !unzip /content/drive/MyDrive/dataset_augmented.zip -d /content/
  3. Paste this script and run
"""

import os
import json
import time
import shutil
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras import mixed_precision
from tensorflow.keras.applications import (
    VGG16, ResNet50, InceptionV3, DenseNet201, EfficientNetB3,
)
from tensorflow.keras.applications import (
    vgg16        as vgg16_mod,
    resnet50     as resnet50_mod,
    inception_v3 as inception_v3_mod,
    densenet     as densenet_mod,
)

# ── Mixed precision ────────────────────────────────────────────────────────────
mixed_precision.set_global_policy("mixed_float16")
print(f"[INFO] Mixed precision: {mixed_precision.global_policy().name}")

# ── Config ────────────────────────────────────────────────────────────────────
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = "/content"

# ↓↓ UPDATE THESE TWO PATHS BEFORE RUNNING ↓↓
DATA_DIR  = "/content/dataset_augmented"
DRIVE_DIR = "/content/drive/MyDrive/mushroom_models"

MODELS_DIR   = os.path.join(BASE_DIR, "models")
PLOTS_DIR    = os.path.join(BASE_DIR, "plots")
RESULTS_PATH = os.path.join(BASE_DIR, "phase3_redo_results.json")
REPORT_PATH  = os.path.join(BASE_DIR, "phase3_redo_report.txt")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)
os.makedirs(DRIVE_DIR,  exist_ok=True)

IMG_SIZE     = (224, 224)
BATCH_SIZE   = 32
NUM_CLASSES  = 3
SEED         = 42
CLASSES      = ["Healthy", "Mixed_Infected", "Single_Infected"]

S1_EPOCHS    = 15
S1_LR        = 1e-3
S2_EPOCHS    = 85
S2_PEAK_LR   = 5e-5
S2_MIN_LR    = 1e-7
S2_WARMUP    = 5
UNFREEZE_N   = 50
PATIENCE_S1  = 8
PATIENCE_S2  = 15

TRAIN_COUNTS = np.array([945, 994, 461])   # Healthy, Mixed, Single (alphabetical)


# ── Model registry ─────────────────────────────────────────────────────────────
# preprocess: function applied to images in tf.data BEFORE feeding to model
#   None  → images passed as-is [0,255]
#   fn    → applied to images (e.g. subtracts ImageNet mean)
#
# EfficientNetB3: preprocess=None, handles [0,255] internally
# Custom_CNN_v2:  preprocess=None, has Rescaling(1/255) inside model
# ResNet50 etc:   preprocess=their preprocess_input function
MODEL_REGISTRY = [
    {
        "name":       "Custom_CNN_v2",
        "type":       "custom",
        "preprocess": None,
        "est_min":    8,
    },
    {
        "name":       "EfficientNetB3",
        "type":       "transfer",
        "base_class": EfficientNetB3,
        "preprocess": None,
        "est_min":    20,
    },
    {
        "name":       "DenseNet201",
        "type":       "transfer",
        "base_class": DenseNet201,
        "preprocess": densenet_mod.preprocess_input,
        "est_min":    30,
    },
    {
        "name":       "InceptionV3",
        "type":       "transfer",
        "base_class": InceptionV3,
        "preprocess": inception_v3_mod.preprocess_input,
        "est_min":    25,
    },
    {
        "name":       "ResNet50",
        "type":       "transfer",
        "base_class": ResNet50,
        "preprocess": resnet50_mod.preprocess_input,
        "est_min":    22,
    },
    {
        "name":       "VGG16",
        "type":       "transfer",
        "base_class": VGG16,
        "preprocess": vgg16_mod.preprocess_input,
        "est_min":    35,
    },
]


# ── Focal Loss ─────────────────────────────────────────────────────────────────
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, label_smoothing=0.1, name="focal_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        n        = tf.cast(tf.shape(y_true)[-1], tf.float32)
        y_smooth = y_true * (1.0 - self.label_smoothing) + self.label_smoothing / n
        y_pred   = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce       = -tf.reduce_sum(y_smooth * tf.math.log(y_pred), axis=-1)
        p_t      = tf.reduce_sum(y_true * y_pred, axis=-1)
        return tf.reduce_mean(tf.pow(1.0 - p_t, self.gamma) * ce)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"gamma": self.gamma, "label_smoothing": self.label_smoothing})
        return cfg


# ── Warmup + Cosine LR ─────────────────────────────────────────────────────────
class WarmupCosineDecay(callbacks.Callback):
    def __init__(self, warmup_epochs, total_epochs, peak_lr, min_lr=1e-7):
        super().__init__()
        self.warmup  = warmup_epochs
        self.total   = total_epochs
        self.peak_lr = peak_lr
        self.min_lr  = min_lr

    def _set_lr(self, lr):
        opt = self.model.optimizer
        try:
            opt.learning_rate.assign(float(lr))
        except AttributeError:
            try:
                opt.lr.assign(float(lr))
            except AttributeError:
                opt.learning_rate = float(lr)

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup:
            lr = self.peak_lr * (epoch + 1) / max(self.warmup, 1)
        else:
            p  = (epoch - self.warmup) / max(self.total - self.warmup, 1)
            lr = self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (1.0 + np.cos(np.pi * p))
        self._set_lr(lr)


# ── tf.data pipeline ───────────────────────────────────────────────────────────
_AUGMENT = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomBrightness(0.15),
    layers.RandomContrast(0.15),
], name="augment")


def make_dataset(split, shuffle=True, augment=False, preprocess_fn=None):
    """Build tf.data dataset for a split.

    Args:
        preprocess_fn: model-specific preprocessing (e.g. resnet50.preprocess_input)
                       applied AFTER augmentation, BEFORE feeding to model.
                       Pass None for EfficientNetB3 and Custom_CNN_v2.
    """
    path = os.path.join(DATA_DIR, split)
    ds   = tf.keras.utils.image_dataset_from_directory(
        path,
        labels="inferred",
        label_mode="categorical",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
    )

    if augment:
        ds = ds.map(
            lambda x, y: (_AUGMENT(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    if preprocess_fn is not None:
        # Apply model-specific preprocessing (converts [0,255] → model's expected range)
        ds = ds.map(
            lambda x, y: (tf.cast(preprocess_fn(x), tf.float32), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    return ds.prefetch(tf.data.AUTOTUNE)


# ── Class weights ──────────────────────────────────────────────────────────────
def get_class_weights():
    total   = TRAIN_COUNTS.sum()
    weights = total / (NUM_CLASSES * TRAIN_COUNTS)
    return {i: float(w) for i, w in enumerate(weights)}


# ── Classifier head ────────────────────────────────────────────────────────────
def build_head(x, prefix):
    x = layers.GlobalAveragePooling2D(name=f"{prefix}_gap")(x)
    x = layers.BatchNormalization(name=f"{prefix}_bn0")(x)
    x = layers.Dense(512, name=f"{prefix}_fc1")(x)
    x = layers.BatchNormalization(name=f"{prefix}_bn1")(x)
    x = layers.Activation("relu", name=f"{prefix}_relu1")(x)
    x = layers.Dropout(0.4, name=f"{prefix}_drop1")(x)
    x = layers.Dense(256, name=f"{prefix}_fc2")(x)
    x = layers.BatchNormalization(name=f"{prefix}_bn2")(x)
    x = layers.Activation("relu", name=f"{prefix}_relu2")(x)
    x = layers.Dropout(0.3, name=f"{prefix}_drop2")(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax",
                       dtype="float32", name=f"{prefix}_predictions")(x)
    return out


# ── Model builders ─────────────────────────────────────────────────────────────
def build_custom_cnn_v2():
    inp = layers.Input(shape=(*IMG_SIZE, 3), name="input")
    x   = layers.Rescaling(1.0 / 255.0, name="rescale")(inp)

    def conv_block(t, f, d):
        reg = tf.keras.regularizers.l2(1e-4)
        t = layers.Conv2D(f, 3, padding="same", kernel_regularizer=reg)(t)
        t = layers.BatchNormalization()(t); t = layers.Activation("relu")(t)
        t = layers.Conv2D(f, 3, padding="same", kernel_regularizer=reg)(t)
        t = layers.BatchNormalization()(t); t = layers.Activation("relu")(t)
        t = layers.MaxPooling2D(2)(t); t = layers.Dropout(d)(t)
        return t

    x = conv_block(x, 32,  0.1)
    x = conv_block(x, 64,  0.2)
    x = conv_block(x, 128, 0.2)
    x = conv_block(x, 256, 0.3)
    x = layers.Conv2D(512, 3, padding="same",
                      kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return models.Model(inp, build_head(x, "cnn"), name="Custom_CNN_v2")


def build_transfer_model(base_class, name):
    """Transfer model WITHOUT any Lambda preprocessing layer.
    Preprocessing is handled in the tf.data pipeline.
    This ensures the model saves/loads cleanly in Keras 3.
    """
    base = base_class(weights="imagenet", include_top=False,
                      input_shape=(*IMG_SIZE, 3))
    base.trainable = False
    inp = layers.Input(shape=(*IMG_SIZE, 3), name="input")
    x   = base(inp, training=False)
    out = build_head(x, name.lower())
    return models.Model(inp, out, name=name), base


# ── Callbacks ──────────────────────────────────────────────────────────────────
def make_callbacks(name, patience):
    ckpt = os.path.join(MODELS_DIR, f"{name}_redo_best.keras")
    return [
        callbacks.ModelCheckpoint(ckpt, monitor="val_accuracy",
                                  save_best_only=True, verbose=0),
        callbacks.EarlyStopping(monitor="val_accuracy", patience=patience,
                                restore_best_weights=True, verbose=1),
    ]


# ── Save training curves ───────────────────────────────────────────────────────
def save_plot(histories, labels, name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{name} — Training Curves", fontsize=13)
    offset = 0
    for hist, label in zip(histories, labels):
        n  = len(hist.history["loss"])
        ep = range(offset + 1, offset + n + 1)
        axes[0].plot(ep, hist.history["loss"],         label=f"{label} Train")
        axes[0].plot(ep, hist.history["val_loss"],     label=f"{label} Val", ls="--")
        axes[1].plot(ep, hist.history["accuracy"],     label=f"{label} Train")
        axes[1].plot(ep, hist.history["val_accuracy"], label=f"{label} Val", ls="--")
        offset += n
    for ax, title in zip(axes, ["Loss", "Accuracy"]):
        ax.set_title(title); ax.set_xlabel("Epoch")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{name}_redo_curves.png")
    plt.savefig(path, dpi=120); plt.close()
    return path


# ── Drive backup ───────────────────────────────────────────────────────────────
def backup_to_drive(src_path, filename, log):
    """Copy a file to DRIVE_DIR immediately. Called after every model."""
    try:
        dst = os.path.join(DRIVE_DIR, filename)
        shutil.copy2(src_path, dst)
        log(f"  >> Drive backup OK  : {dst}")
    except Exception as e:
        log(f"  >> Drive backup FAIL: {e}")


# ── Train one model ────────────────────────────────────────────────────────────
def train_one(cfg, class_weights, log):
    name         = cfg["name"]
    preprocess_fn = cfg.get("preprocess")

    log(f"\n{'='*70}")
    log(f"  MODEL: {name}  (est. ~{cfg['est_min']} min on T4)")
    log(f"{'='*70}")

    # Build model-specific datasets with correct preprocessing
    train_ds = make_dataset("train", shuffle=True,  augment=True,
                            preprocess_fn=preprocess_fn)
    val_ds   = make_dataset("val",   shuffle=False, augment=False,
                            preprocess_fn=preprocess_fn)

    t0    = time.time()
    focal = FocalLoss(gamma=2.0, label_smoothing=0.1)

    if cfg["type"] == "custom":
        model = build_custom_cnn_v2()
        base  = None
    else:
        model, base = build_transfer_model(cfg["base_class"], name)

    log(f"  Parameters: {model.count_params():,}")

    # Stage 1 — head only
    model.compile(optimizer=optimizers.Adam(S1_LR), loss=focal,
                  metrics=["accuracy"])
    log(f"\n  [Stage 1 — frozen base, LR={S1_LR}, max {S1_EPOCHS} ep]")
    h1 = model.fit(train_ds, validation_data=val_ds,
                   epochs=S1_EPOCHS, class_weight=class_weights,
                   callbacks=make_callbacks(name, PATIENCE_S1), verbose=1)

    if cfg["type"] == "custom":
        histories, stage_labels = [h1], ["Train"]
    else:
        # Stage 2 — unfreeze last UNFREEZE_N layers (keep BN frozen)
        for layer in base.layers:
            layer.trainable = False
        for layer in base.layers[-UNFREEZE_N:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True

        n_unf = sum(1 for l in base.layers if l.trainable)
        log(f"\n  [Stage 2 — {n_unf} layers unfrozen, warmup+cosine, max {S2_EPOCHS} ep]")

        model.compile(optimizer=optimizers.Adam(S2_PEAK_LR), loss=focal,
                      metrics=["accuracy"])
        h2 = model.fit(
            train_ds, validation_data=val_ds,
            epochs=S2_EPOCHS, class_weight=class_weights,
            callbacks=make_callbacks(name, PATIENCE_S2) + [
                WarmupCosineDecay(S2_WARMUP, S2_EPOCHS, S2_PEAK_LR, S2_MIN_LR)
            ],
            verbose=1,
        )
        histories, stage_labels = [h1, h2], ["Stage1", "Stage2"]

    best_acc  = max(max(h.history["val_accuracy"]) for h in histories)
    best_loss = min(min(h.history["val_loss"])     for h in histories)
    elapsed   = (time.time() - t0) / 60

    ckpt_path = os.path.join(MODELS_DIR, f"{name}_redo_best.keras")
    plot_path = save_plot(histories, stage_labels, name)

    log(f"\n  >> Best val accuracy : {best_acc*100:.2f}%")
    log(f"  >> Best val loss     : {best_loss:.4f}")
    log(f"  >> Training time     : {elapsed:.1f} min")
    log(f"  >> Weights saved     : {ckpt_path}")

    # ── Save to Drive immediately after this model ─────────────────────────────
    backup_to_drive(ckpt_path, f"{name}_redo_best.keras", log)
    backup_to_drive(plot_path, f"{name}_redo_curves.png", log)

    return {
        "model":        name,
        "val_accuracy": round(float(best_acc), 4),
        "val_loss":     round(float(best_loss), 4),
        "time_min":     round(elapsed, 1),
        "preprocess":   cfg["name"] if preprocess_fn is not None else None,
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    lines = []
    def log(s=""):
        lines.append(str(s)); print(s)

    log("=" * 70)
    log("  PHASE 3 REDO — GPU TRAINING")
    log("=" * 70)
    log(f"  TF      : {tf.__version__}")
    log(f"  GPU     : {tf.config.list_physical_devices('GPU') or 'NONE — switch to GPU!'}")
    log(f"  Dataset : {DATA_DIR}")
    log(f"  Drive   : {DRIVE_DIR}")

    if not tf.config.list_physical_devices("GPU"):
        log("\n  WARNING: No GPU. Training will be very slow.")

    class_weights = get_class_weights()
    log(f"\n  Class weights: {class_weights}")
    log(f"  Total models : {len(MODEL_REGISTRY)}")
    log(f"  Est. time    : ~{sum(m['est_min'] for m in MODEL_REGISTRY)} min on T4")

    results = []
    for cfg in MODEL_REGISTRY:
        result = train_one(cfg, class_weights, log)
        results.append(result)

        # Save running results to Drive after every model
        results_sorted = sorted(results, key=lambda r: r["val_accuracy"], reverse=True)
        with open(RESULTS_PATH, "w") as f:
            json.dump(results_sorted, f, indent=2)
        backup_to_drive(RESULTS_PATH, "phase3_redo_results.json", log)
        log(f"  >> Progress: {len(results)}/{len(MODEL_REGISTRY)} models done")

    # Final summary
    results_sorted = sorted(results, key=lambda r: r["val_accuracy"], reverse=True)
    log(f"\n{'='*70}")
    log("  FINAL RESULTS")
    log(f"{'='*70}")
    log(f"  {'Model':<20} {'Val Acc':>10} {'Val Loss':>10} {'Time':>8}")
    log(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*8}")
    for r in results_sorted:
        tag = " <- BEST" if r is results_sorted[0] else ""
        log(f"  {r['model']:<20} {r['val_accuracy']*100:>9.2f}% "
            f"{r['val_loss']:>10.4f} {r['time_min']:>7.1f}m{tag}")

    log(f"\n  CPU baseline: Custom_CNN 85.84%")
    log(f"  Best GPU    : {results_sorted[0]['model']} "
        f"{results_sorted[0]['val_accuracy']*100:.2f}%")

    # Save final report to Drive
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    backup_to_drive(REPORT_PATH, "phase3_redo_report.txt", log)

    log(f"\n{'='*70}")
    log("  PHASE 3 COMPLETE — all models saved to Drive")
    log(f"  Next: run phase4_evaluate.py")
    log(f"{'='*70}")


if __name__ == "__main__":
    main()
