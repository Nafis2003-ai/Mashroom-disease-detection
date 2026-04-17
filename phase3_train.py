"""
Phase 3: Model Training
Trains 6 models sequentially:
  1. Custom CNN (baseline)
  2. VGG16      (transfer learning)
  3. ResNet50   (transfer learning)
  4. InceptionV3(transfer learning)
  5. DenseNet201(transfer learning)
  6. EfficientNetB0 (transfer learning)

Two-stage fine-tuning per pretrained model:
  Stage 1 — Freeze base, train classifier head (10 epochs)
  Stage 2 — Unfreeze last 30 layers, retrain with low LR (up to 40 epochs)

Outputs:
  models/          — best weights (.keras) for each model
  plots/           — training curves for each model
  phase3_report.txt
"""

import os
import json
import time
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import (
    VGG16, ResNet50, InceptionV3, DenseNet201, EfficientNetB0
)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "dataset_augmented")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
PLOTS_DIR   = os.path.join(BASE_DIR, "plots")
REPORT_PATH = os.path.join(BASE_DIR, "phase3_report.txt")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

IMG_SIZE    = (224, 224)
BATCH_SIZE  = 16          # CPU-friendly
CLASSES     = ["Healthy", "Mixed_Infected", "Single_Infected"]
NUM_CLASSES = 3
SEED        = 42

# Stage 1 (head only)
S1_EPOCHS   = 10
S1_LR       = 1e-3

# Stage 2 (fine-tune)
S2_EPOCHS   = 40
S2_LR       = 1e-4
UNFREEZE_N  = 30          # last N layers to unfreeze

PATIENCE    = 8           # early stopping patience

# ── Data generators ───────────────────────────────────────────────────────────
def make_generators(target_size=IMG_SIZE):
    train_gen = ImageDataGenerator(rescale=1./255)
    val_gen   = ImageDataGenerator(rescale=1./255)

    train = train_gen.flow_from_directory(
        os.path.join(DATA_DIR, "train"),
        target_size=target_size,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
        seed=SEED,
    )
    val = val_gen.flow_from_directory(
        os.path.join(DATA_DIR, "val"),
        target_size=target_size,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )
    return train, val

# ── Class weights ─────────────────────────────────────────────────────────────
def get_class_weights(train_gen):
    labels = train_gen.classes
    cw = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
    return dict(enumerate(cw))

# ── Plot training curves ──────────────────────────────────────────────────────
def save_plot(history_list, model_name, stage_labels):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{model_name} — Training Curves", fontsize=14)

    offset = 0
    for hist, label in zip(history_list, stage_labels):
        epochs = range(offset + 1, offset + len(hist.history["loss"]) + 1)
        axes[0].plot(epochs, hist.history["loss"],     label=f"{label} Train Loss")
        axes[0].plot(epochs, hist.history["val_loss"], label=f"{label} Val Loss",
                     linestyle="--")
        axes[1].plot(epochs, hist.history["accuracy"],     label=f"{label} Train Acc")
        axes[1].plot(epochs, hist.history["val_accuracy"], label=f"{label} Val Acc",
                     linestyle="--")
        offset += len(hist.history["loss"])

    for ax, title in zip(axes, ["Loss", "Accuracy"]):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{model_name}_curves.png")
    plt.savefig(path, dpi=120)
    plt.close()
    return path

# ── Callbacks ─────────────────────────────────────────────────────────────────
def make_callbacks(model_name, stage):
    ckpt_path = os.path.join(MODELS_DIR, f"{model_name}_best.keras")
    return [
        callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_accuracy",
            save_best_only=True, verbose=0
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy", patience=PATIENCE,
            restore_best_weights=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=4, min_lr=1e-7, verbose=1
        ),
    ]

# ── Model builders ────────────────────────────────────────────────────────────
def build_custom_cnn():
    model = models.Sequential([
        layers.Input(shape=(*IMG_SIZE, 3)),

        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),

        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),

        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),

        layers.Conv2D(256, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),

        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name="Custom_CNN")
    return model

def build_transfer_model(base_fn, name, target_size=IMG_SIZE):
    base = base_fn(
        weights="imagenet",
        include_top=False,
        input_shape=(*target_size, 3)
    )
    base.trainable = False

    inp  = layers.Input(shape=(*target_size, 3))
    x    = base(inp, training=False)
    x    = layers.GlobalAveragePooling2D()(x)
    x    = layers.Dense(256, activation="relu")(x)
    x    = layers.Dropout(0.5)(x)
    out  = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inp, out, name=name)
    return model, base

# ── Train one model ───────────────────────────────────────────────────────────
def train_model(name, build_fn, train_gen, val_gen, class_weights, log):
    log(f"\n{'='*72}")
    log(f"  TRAINING: {name}")
    log(f"{'='*72}")

    t0 = time.time()

    if name == "Custom_CNN":
        model = build_fn()
        model.compile(
            optimizer=optimizers.Adam(S1_LR),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        log(f"  Parameters: {model.count_params():,}")
        log(f"\n  [Single-stage training — {S1_EPOCHS + S2_EPOCHS} epochs max]")

        hist = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=S1_EPOCHS + S2_EPOCHS,
            class_weight=class_weights,
            callbacks=make_callbacks(name, "full"),
            verbose=1,
        )
        histories   = [hist]
        stage_labels= ["Train"]

    else:
        model, base = build_fn()
        model.compile(
            optimizer=optimizers.Adam(S1_LR),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        log(f"  Total params : {model.count_params():,}")
        log(f"  Base layers  : {len(base.layers)}")

        # Stage 1 — head only
        log(f"\n  [Stage 1 — frozen base, head only, LR={S1_LR}, max {S1_EPOCHS} epochs]")
        h1 = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=S1_EPOCHS,
            class_weight=class_weights,
            callbacks=make_callbacks(name, "s1"),
            verbose=1,
        )

        # Stage 2 — unfreeze last N layers
        for layer in base.layers[-UNFREEZE_N:]:
            layer.trainable = True
        model.compile(
            optimizer=optimizers.Adam(S2_LR),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        log(f"\n  [Stage 2 — last {UNFREEZE_N} layers unfrozen, LR={S2_LR}, max {S2_EPOCHS} epochs]")
        h2 = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=S2_EPOCHS,
            class_weight=class_weights,
            callbacks=make_callbacks(name, "s2"),
            verbose=1,
        )

        histories    = [h1, h2]
        stage_labels = ["Stage1", "Stage2"]

    elapsed = time.time() - t0
    best_val_acc = max(
        max(h.history["val_accuracy"]) for h in histories
    )
    best_val_loss = min(
        min(h.history["val_loss"]) for h in histories
    )

    plot_path = save_plot(histories, name, stage_labels)

    log(f"\n  >> Best val accuracy  : {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    log(f"  >> Best val loss      : {best_val_loss:.4f}")
    log(f"  >> Training time      : {elapsed/60:.1f} min")
    log(f"  >> Weights saved to   : models/{name}_best.keras")
    log(f"  >> Plot saved to      : plots/{name}_curves.png")

    return {
        "model":        name,
        "val_accuracy": round(best_val_acc, 4),
        "val_loss":     round(best_val_loss, 4),
        "time_min":     round(elapsed / 60, 1),
    }

# ── Model registry ────────────────────────────────────────────────────────────
# InceptionV3 needs 299x299 minimum; we use 224 for all others
INCEPTION_SIZE = (224, 224)   # TF2 allows 224 for InceptionV3

MODEL_REGISTRY = [
    # Custom CNN already trained — skip to avoid retraining
    # ("Custom_CNN",      lambda: build_custom_cnn(),                              IMG_SIZE),
    # VGG16 skipped — too slow on CPU (138M params)
    # ("VGG16",           lambda: build_transfer_model(VGG16,           "VGG16"),          IMG_SIZE),
    # ResNet50 already trained — skip
    # ("ResNet50",        lambda: build_transfer_model(ResNet50,         "ResNet50"),       IMG_SIZE),
    # InceptionV3 already trained — skip
    # ("InceptionV3",     lambda: build_transfer_model(InceptionV3,      "InceptionV3"),   IMG_SIZE),
    # DenseNet201 already trained — skip
    # ("DenseNet201",     lambda: build_transfer_model(DenseNet201,      "DenseNet201"),   IMG_SIZE),
    ("EfficientNetB0",  lambda: build_transfer_model(EfficientNetB0,   "EfficientNetB0"),IMG_SIZE),
]

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    lines = []
    log = lambda s="": (lines.append(str(s)), print(s))

    log("=" * 72)
    log("  PHASE 3 REPORT — MODEL TRAINING")
    log("=" * 72)
    log(f"  TensorFlow : {tf.__version__}")
    log(f"  GPU        : {tf.config.list_physical_devices('GPU') or 'None (CPU only)'}")
    log(f"  Dataset    : {DATA_DIR}")
    log(f"  Image size : {IMG_SIZE}")
    log(f"  Batch size : {BATCH_SIZE}")
    log(f"  Classes    : {CLASSES}")

    # Build generators (default 224x224)
    train_gen, val_gen = make_generators()
    class_weights = get_class_weights(train_gen)

    log(f"\n  Class weights (to handle imbalance):")
    idx_to_cls = {v: k for k, v in train_gen.class_indices.items()}
    for idx, w in class_weights.items():
        log(f"    {idx_to_cls[idx]:<22} weight = {w:.4f}")

    results = []

    for name, build_fn, size in MODEL_REGISTRY:
        # Rebuild generators with correct size if needed
        if size != IMG_SIZE:
            tg, vg = make_generators(target_size=size)
            cw = get_class_weights(tg)
        else:
            tg, vg, cw = train_gen, val_gen, class_weights

        result = train_model(name, build_fn, tg, vg, cw, log)
        results.append(result)

        # Reset generators for next model
        tg.reset()
        vg.reset()

    # ── Summary table ─────────────────────────────────────────────────────
    log(f"\n{'='*72}")
    log("  PHASE 3 SUMMARY — MODEL COMPARISON")
    log(f"{'='*72}")
    log(f"  {'Model':<20} {'Val Accuracy':>14} {'Val Loss':>10} {'Time (min)':>12}")
    log(f"  {'-'*20} {'-'*14} {'-'*10} {'-'*12}")

    results_sorted = sorted(results, key=lambda x: x["val_accuracy"], reverse=True)
    for r in results_sorted:
        marker = " <<< BEST" if r == results_sorted[0] else ""
        log(f"  {r['model']:<20} {r['val_accuracy']*100:>13.2f}% "
            f"{r['val_loss']:>10.4f} {r['time_min']:>11.1f}m{marker}")

    best = results_sorted[0]
    log(f"\n  Best model: {best['model']} with {best['val_accuracy']*100:.2f}% val accuracy")
    log(f"  Best weights: models/{best['model']}_best.keras")

    # Save results JSON
    json_path = os.path.join(BASE_DIR, "phase3_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\n  Results JSON saved to: {json_path}")

    log(f"\n{'='*72}")
    log("  PHASE 3 COMPLETE — Proceed to Phase 4 (Evaluation)")
    log(f"{'='*72}")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
