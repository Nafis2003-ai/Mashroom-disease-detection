"""
phase4_evaluate.py  —  Full Evaluation, Grad-CAM & Ensemble
============================================================
Evaluates all models trained by phase3_redo.py on the held-out test set.

Outputs:
  - Per-model: accuracy, precision, recall, F1, confusion matrix
  - ROC-AUC curves (one-vs-rest, per class)
  - Grad-CAM heatmaps for top 3 models
  - Ensemble (soft-voting, top-3 models)
  - phase4_report.txt + phase4_results.json
  - plots/confusion_matrix_<model>.png
  - plots/roc_auc_<model>.png
  - plots/gradcam_<model>.png
  - plots/comparison_table.png

NOTE: This script evaluates models from phase3_redo.py which have
preprocessing layers inside the model. Pass raw [0,255] images.
"""

import os
import json
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    VGG16, ResNet50, InceptionV3, DenseNet201, EfficientNetB3,
)
from tensorflow.keras.applications import (
    vgg16        as vgg16_mod,
    resnet50     as resnet50_mod,
    inception_v3 as inception_v3_mod,
    densenet     as densenet_mod,
)
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, auc,
    precision_recall_fscore_support,
    accuracy_score,
)

# ── Rebuild helpers (for models that fail to load due to Lambda layer) ────────
# Models that use Lambda preprocessing (ResNet50, VGG16, InceptionV3, DenseNet201)
# cannot be loaded in Keras 3 due to Lambda output_shape issue.
# Fix: rebuild architecture without Lambda, load weights by name.
LAMBDA_MODELS = {
    "ResNet50":    (ResNet50,    resnet50_mod.preprocess_input),
    "VGG16":       (VGG16,       vgg16_mod.preprocess_input),
    "InceptionV3": (InceptionV3, inception_v3_mod.preprocess_input),
    "DenseNet201": (DenseNet201, densenet_mod.preprocess_input),
}

# Preprocessing functions for models from phase3_redo.py (no Lambda layers).
# These models expect preprocessing applied externally in the tf.data pipeline.
# EfficientNetB3 and Custom_CNN_v2 are NOT listed here (they handle [0,255] internally).
PREPROCESS_MAP = {
    "ResNet50":    resnet50_mod.preprocess_input,
    "VGG16":       vgg16_mod.preprocess_input,
    "InceptionV3": inception_v3_mod.preprocess_input,
    "DenseNet201": densenet_mod.preprocess_input,
}

def _build_head(x, prefix):
    """Identical head to phase3_redo.py — must match for weight loading."""
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
    out = layers.Dense(3, activation="softmax", dtype="float32",
                       name=f"{prefix}_predictions")(x)
    return out

def _rebuild_transfer(base_class, name):
    """Rebuild transfer model WITHOUT Lambda preprocessing layer.
    Preprocessing will be applied manually before inference.
    """
    base = base_class(weights=None, include_top=False, input_shape=(224, 224, 3))
    inp  = layers.Input(shape=(224, 224, 3), name="input")
    x    = base(inp, training=False)
    out  = _build_head(x, name.lower())
    return models.Model(inp, out, name=name)


def _load_weights_from_keras_zip(model, keras_path):
    """Extract model.weights.h5 from inside a .keras zip file and load it.
    Keras 3 stores .keras files as zip archives containing model.weights.h5.
    by_name=True works with .h5 files but NOT with .keras files directly.
    """
    import zipfile, tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(keras_path, "r") as zf:
            zf.extractall(tmpdir)
        weights_h5 = os.path.join(tmpdir, "model.weights.h5")
        if not os.path.exists(weights_h5):
            raise FileNotFoundError(f"model.weights.h5 not found inside {keras_path}")
        model.load_weights(weights_h5, by_name=True, skip_mismatch=True)
    return model

# ── Config ────────────────────────────────────────────────────────────────────
# Colab-safe: __file__ is not defined in notebooks
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = "/content"

# ↓↓ UPDATE THESE PATHS FOR COLAB ↓↓
DATA_DIR     = "/content/dataset_augmented"
MODELS_DIR   = "/content/drive/MyDrive/mushroom_models"
PLOTS_DIR    = "/content/drive/MyDrive/mushroom_models/plots"
REPORT_PATH  = "/content/drive/MyDrive/mushroom_models/phase4_report.txt"
RESULTS_PATH = "/content/drive/MyDrive/mushroom_models/phase4_results.json"

os.makedirs(PLOTS_DIR, exist_ok=True)

IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
NUM_CLASSES = 3
CLASSES     = ["Healthy", "Mixed_Infected", "Single_Infected"]

# Models to evaluate (must exist in models/ as *_redo_best.keras)
# Also include old Custom_CNN_best.keras for comparison if it exists
MODEL_FILES = [
    "Custom_CNN_v2_redo_best.keras",
    "EfficientNetB3_redo_best.keras",
    "DenseNet201_redo_best.keras",
    "InceptionV3_redo_best.keras",
    "ResNet50_redo_best.keras",
    "VGG16_redo_best.keras",
    # Old baseline (phase3 CPU training) for comparison:
    "Custom_CNN_best.keras",
]

# Top N models to include in ensemble
ENSEMBLE_TOP_N = 3


# ── Load test dataset ─────────────────────────────────────────────────────────
def load_test_data():
    """Return (images [0,255], one-hot labels, integer labels)."""
    test_path = os.path.join(DATA_DIR, "test")
    ds = tf.keras.utils.image_dataset_from_directory(
        test_path,
        labels="inferred",
        label_mode="categorical",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    ds = ds.prefetch(tf.data.AUTOTUNE)

    images_list, labels_list = [], []
    for imgs, lbls in ds:
        images_list.append(imgs.numpy())
        labels_list.append(lbls.numpy())

    images     = np.concatenate(images_list, axis=0)   # [0, 255] float32
    labels_oh  = np.concatenate(labels_list, axis=0)   # one-hot
    labels_int = np.argmax(labels_oh, axis=1)           # integer
    return images, labels_oh, labels_int


# ── Model loading ─────────────────────────────────────────────────────────────
def load_models(log):
    """Load models from MODELS_DIR.
    Returns dict: {name: (model, preprocess_fn)}
    preprocess_fn is None for models with internal preprocessing (EfficientNet, Custom CNN).
    preprocess_fn is set for Lambda models that were rebuilt from weights.
    """
    tf.keras.config.enable_unsafe_deserialization()
    loaded = {}

    for fname in MODEL_FILES:
        path = os.path.join(MODELS_DIR, fname)
        if not os.path.exists(path):
            log(f"  [SKIP] {fname} not found")
            continue

        name = fname.replace("_redo_best.keras", "").replace("_best.keras", " (baseline)")

        # ── Try normal load first ─────────────────────────────────────────────
        try:
            model = tf.keras.models.load_model(path, compile=False)
            # For phase3_redo models: assign preprocessing by name (no Lambda inside model)
            model_key   = fname.replace("_redo_best.keras", "").replace("_best.keras", "")
            preprocess_fn = PREPROCESS_MAP.get(model_key, None)
            loaded[name] = (model, preprocess_fn)
            log(f"  [OK]   {fname}  →  {name}  (preprocess: {model_key if preprocess_fn else 'internal'})")
            continue
        except Exception:
            pass

        # ── Fallback: rebuild architecture + extract weights from .keras zip ────
        base_name = fname.replace("_redo_best.keras", "")
        if base_name in LAMBDA_MODELS:
            base_class, preprocess_fn = LAMBDA_MODELS[base_name]
            try:
                model = _rebuild_transfer(base_class, base_name)
                _load_weights_from_keras_zip(model, path)
                loaded[name] = (model, preprocess_fn)
                log(f"  [OK-rebuilt]  {fname}  →  {name}  (weights extracted from zip)")
            except Exception as e2:
                log(f"  [ERR]  {fname}: {e2}")
        else:
            log(f"  [ERR]  {fname}: could not load (unknown architecture)")

    return loaded


# ── Per-model evaluation ──────────────────────────────────────────────────────
def evaluate_model(model_tuple, name, images, labels_oh, labels_int, log):
    """Run inference and compute all metrics."""
    model, preprocess_fn = model_tuple
    log(f"\n  Evaluating {name}...")

    if preprocess_fn is not None:
        # Rebuilt model (Lambda removed): apply preprocessing manually
        imgs = preprocess_fn(images.astype(np.float32).copy())
        preds = model.predict(imgs, batch_size=BATCH_SIZE, verbose=0)
    elif _model_has_rescaling(model):
        # EfficientNetB3 / Custom CNN v2: preprocessing inside model
        preds = model.predict(images, batch_size=BATCH_SIZE, verbose=0)
    else:
        # Old phase3 baseline: expects [0,1]
        preds = model.predict(images / 255.0, batch_size=BATCH_SIZE, verbose=0)

    pred_int = np.argmax(preds, axis=1)

    # Accuracy
    acc = accuracy_score(labels_int, pred_int)

    # Per-class precision, recall, F1
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels_int, pred_int, average="weighted", zero_division=0
    )
    prec_cls, rec_cls, f1_cls, _ = precision_recall_fscore_support(
        labels_int, pred_int, average=None, zero_division=0
    )

    # ROC-AUC (one-vs-rest)
    try:
        auc_score = roc_auc_score(labels_oh, preds, multi_class="ovr", average="weighted")
    except Exception:
        auc_score = float("nan")

    result = {
        "model":        name,
        "accuracy":     round(float(acc), 4),
        "precision":    round(float(prec), 4),
        "recall":       round(float(rec), 4),
        "f1":           round(float(f1), 4),
        "roc_auc":      round(float(auc_score), 4) if not np.isnan(auc_score) else "N/A",
        "per_class": {
            cls: {
                "precision": round(float(prec_cls[i]), 4),
                "recall":    round(float(rec_cls[i]), 4),
                "f1":        round(float(f1_cls[i]), 4),
            }
            for i, cls in enumerate(CLASSES)
        },
    }

    log(f"    Accuracy:    {acc*100:.2f}%")
    log(f"    Precision:   {prec*100:.2f}%  (weighted)")
    log(f"    Recall:      {rec*100:.2f}%  (weighted)")
    log(f"    F1-Score:    {f1*100:.2f}%  (weighted)")
    log(f"    ROC-AUC:     {auc_score:.4f}")

    return result, preds


def _model_has_rescaling(model):
    """Check if model has an internal rescaling/preprocessing layer."""
    for layer in model.layers:
        if isinstance(layer, (tf.keras.layers.Rescaling, tf.keras.layers.Lambda)):
            return True
        if hasattr(layer, "layers"):
            for sub in layer.layers:
                if isinstance(sub, tf.keras.layers.Rescaling):
                    return True
    return False


# ── Confusion matrix plot ─────────────────────────────────────────────────────
def plot_confusion_matrix(labels_int, pred_int, name):
    cm = confusion_matrix(labels_int, pred_int)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Confusion Matrix — {name}", fontsize=13)

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Raw counts", "Normalised (row %)"],
        ["d", ".2f"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt,
            xticklabels=CLASSES, yticklabels=CLASSES,
            cmap="Blues", ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"confusion_{name.replace(' ', '_')}.png")
    plt.savefig(path, dpi=120); plt.close()
    return path


# ── ROC-AUC curves ────────────────────────────────────────────────────────────
def plot_roc_auc(labels_oh, preds, name):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors  = ["#e74c3c", "#2ecc71", "#3498db"]

    for i, (cls, color) in enumerate(zip(CLASSES, colors)):
        fpr, tpr, _ = roc_curve(labels_oh[:, i], preds[:, i])
        score       = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{cls}  (AUC={score:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — {name}")
    ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"roc_{name.replace(' ', '_')}.png")
    plt.savefig(path, dpi=120); plt.close()
    return path


# ── Grad-CAM ──────────────────────────────────────────────────────────────────
def get_gradcam(model, img_array, class_idx=None):
    """
    Compute Grad-CAM heatmap using the layer just before GlobalAveragePooling2D.

    Works for both Custom CNN and transfer learning models since all models
    in phase3_redo.py use a GAP layer in the head.

    Args:
        model     : loaded Keras model
        img_array : single image as numpy array, shape (H, W, 3), [0,255]
        class_idx : predicted class index (None = use argmax)

    Returns:
        heatmap   : 2D numpy array [0,1]
        probs     : softmax probabilities
        pred_idx  : predicted class index
    """
    # Find GAP layer
    gap_layer = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
            gap_layer = layer
            break

    if gap_layer is None:
        raise ValueError(f"No GlobalAveragePooling2D layer found in {model.name}")

    # Build grad model: input → (GAP_input_tensor, predictions)
    gap_input_tensor = gap_layer.input
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[gap_input_tensor, model.output],
    )

    img_tensor = tf.cast(img_array[np.newaxis], tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        feature_maps, preds = grad_model(img_tensor, training=False)
        if class_idx is None:
            class_idx = int(tf.argmax(preds[0]))
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, feature_maps)

    # Handle case where feature_maps is 2D (already GAP'd by sub-model edge case)
    if len(feature_maps.shape) == 2:
        # Fallback: return a blank heatmap
        return np.ones((7, 7)), preds[0].numpy(), class_idx

    pooled_grads  = tf.reduce_mean(grads, axis=(0, 1, 2))  # (channels,)
    feature_maps  = feature_maps[0]                         # (H, W, channels)
    heatmap       = feature_maps @ pooled_grads[..., tf.newaxis]
    heatmap       = tf.squeeze(heatmap)
    heatmap       = tf.nn.relu(heatmap)
    heatmap       = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy(), preds[0].numpy(), class_idx


def overlay_gradcam(img, heatmap, alpha=0.45):
    """Overlay a Grad-CAM heatmap on an image."""
    import cv2
    img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
    h, w      = img_uint8.shape[:2]

    heatmap_resized = np.uint8(255 * heatmap)
    heatmap_resized = cv2.resize(heatmap_resized, (w, h))
    heatmap_color   = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_color   = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlaid = np.uint8(alpha * heatmap_color + (1 - alpha) * img_uint8)
    return overlaid


def plot_gradcam_grid(model, name, images, labels_int, preds_int, n_per_class=2):
    """
    Plot a grid of Grad-CAM overlays: n_per_class images per class.
    Shows: original image | Grad-CAM overlay | prediction label
    """
    try:
        import cv2
        has_cv2 = True
    except ImportError:
        has_cv2 = False

    n_cols  = 2   # original + overlay
    n_rows  = NUM_CLASSES * n_per_class
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, n_rows * 3))
    fig.suptitle(f"Grad-CAM Heatmaps — {name}", fontsize=13, y=1.01)

    row = 0
    for cls_idx in range(NUM_CLASSES):
        # Pick n_per_class correctly classified samples from this class
        candidates = np.where(
            (labels_int == cls_idx) & (preds_int == cls_idx)
        )[0]
        if len(candidates) == 0:
            candidates = np.where(labels_int == cls_idx)[0]

        chosen = candidates[:n_per_class]

        for img_idx in chosen:
            img = images[img_idx]  # [0,255] float32

            try:
                heatmap, probs, pred_idx = get_gradcam(model, img, class_idx=cls_idx)
            except Exception as e:
                row += 1
                continue

            # Original
            axes[row, 0].imshow(np.clip(img / 255.0, 0, 1))
            axes[row, 0].set_title(
                f"True: {CLASSES[cls_idx]}\nConf: {probs[cls_idx]*100:.1f}%",
                fontsize=8
            )
            axes[row, 0].axis("off")

            # Grad-CAM overlay
            if has_cv2:
                overlay = overlay_gradcam(img, heatmap)
            else:
                # Without OpenCV: just show the heatmap coloured
                h, w = img.shape[:2]
                import PIL.Image
                hm_resized = np.array(
                    PIL.Image.fromarray(np.uint8(255 * heatmap)).resize((w, h))
                )
                cmap = plt.cm.jet(hm_resized / 255.0)[:, :, :3]
                overlay = np.uint8(0.45 * cmap * 255 + 0.55 * np.clip(img, 0, 255))

            axes[row, 1].imshow(overlay)
            axes[row, 1].set_title(
                f"Pred: {CLASSES[pred_idx]}", fontsize=8
            )
            axes[row, 1].axis("off")
            row += 1

    # Hide unused axes
    for r in range(row, n_rows):
        for c in range(n_cols):
            axes[r, c].axis("off")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"gradcam_{name.replace(' ', '_')}.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    return path


# ── Ensemble ──────────────────────────────────────────────────────────────────
def ensemble_predict(models_dict, top_names, images, log):
    """Soft-voting ensemble of top-N models."""
    log(f"\n  Ensemble ({', '.join(top_names)})...")
    probs_list = []
    for name in top_names:
        model, preprocess_fn = models_dict[name]
        if preprocess_fn is not None:
            imgs = preprocess_fn(images.astype(np.float32).copy())
            p = model.predict(imgs, batch_size=BATCH_SIZE, verbose=0)
        elif _model_has_rescaling(model):
            p = model.predict(images, batch_size=BATCH_SIZE, verbose=0)
        else:
            p = model.predict(images / 255.0, batch_size=BATCH_SIZE, verbose=0)
        probs_list.append(p)
    return np.mean(probs_list, axis=0)


# ── Comparison table plot ─────────────────────────────────────────────────────
def plot_comparison_table(results):
    names = [r["model"] for r in results]
    accs  = [r["accuracy"] * 100 for r in results]

    colors = ["#27ae60" if a == max(accs) else "#3498db" for a in accs]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(names, accs, color=colors, edgecolor="white", height=0.6)
    ax.set_xlabel("Test Accuracy (%)")
    ax.set_title("Model Comparison — Test Set Accuracy")
    ax.set_xlim(0, 100)
    ax.axvline(85.84, color="orange", ls="--", lw=1.5, label="Baseline (Custom CNN 85.84%)")
    ax.legend(fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)

    for bar, val in zip(bars, accs):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}%", va="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "comparison_table.png")
    plt.savefig(path, dpi=120); plt.close()
    return path


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    lines = []
    def log(s=""):
        lines.append(str(s)); print(s)

    log("=" * 70)
    log("  PHASE 4 REPORT — EVALUATION & ANALYSIS")
    log("=" * 70)
    log(f"  Dataset : {DATA_DIR}/test  (117 images)")
    log(f"  Classes : {CLASSES}")

    # Load test data
    log("\n  Loading test set...")
    images, labels_oh, labels_int = load_test_data()
    log(f"  Test images: {len(images)}  shape: {images.shape}")

    # Load models
    log("\n  Loading models...")
    models_dict = load_models(log)

    if not models_dict:
        log("\n  ERROR: No models found. Run phase3_redo.py first.")
        return

    # Evaluate each model
    all_results = []
    all_preds   = {}
    for name, model_tuple in models_dict.items():
        result, preds = evaluate_model(model_tuple, name, images, labels_oh, labels_int, log)
        all_results.append(result)
        all_preds[name] = preds

        pred_int = np.argmax(preds, axis=1)
        plot_confusion_matrix(labels_int, pred_int, name)
        plot_roc_auc(labels_oh, preds, name)

    # Sort by accuracy
    all_results.sort(key=lambda r: r["accuracy"], reverse=True)

    # Ensemble (top N)
    sorted_names = [r["model"] for r in all_results]
    top_names    = sorted_names[:min(ENSEMBLE_TOP_N, len(sorted_names))]

    if len(top_names) >= 2:
        ens_preds   = ensemble_predict(models_dict, top_names, images, log)
        ens_pred_int = np.argmax(ens_preds, axis=1)
        ens_acc     = accuracy_score(labels_int, ens_pred_int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            labels_int, ens_pred_int, average="weighted", zero_division=0
        )
        ens_auc = roc_auc_score(labels_oh, ens_preds, multi_class="ovr", average="weighted")

        ens_result = {
            "model":     f"Ensemble (top {len(top_names)})",
            "accuracy":  round(float(ens_acc), 4),
            "precision": round(float(prec), 4),
            "recall":    round(float(rec), 4),
            "f1":        round(float(f1), 4),
            "roc_auc":   round(float(ens_auc), 4),
        }
        all_results.append(ens_result)
        all_preds["ensemble"] = ens_preds

        plot_confusion_matrix(labels_int, ens_pred_int, f"Ensemble_top{len(top_names)}")
        plot_roc_auc(labels_oh, ens_preds, f"Ensemble_top{len(top_names)}")
        log(f"\n  Ensemble accuracy: {ens_acc*100:.2f}%  |  ROC-AUC: {ens_auc:.4f}")

    # Grad-CAM for top 3 models
    log("\n  Generating Grad-CAM heatmaps...")
    for name in sorted_names[:3]:
        model, _  = models_dict[name]
        preds_int = np.argmax(all_preds[name], axis=1)
        try:
            gradcam_path = plot_gradcam_grid(model, name, images, labels_int, preds_int)
            log(f"  Grad-CAM saved: {gradcam_path}")
        except Exception as e:
            log(f"  Grad-CAM failed for {name}: {e}")

    # Final comparison table
    all_results_sorted = sorted(all_results, key=lambda r: r["accuracy"], reverse=True)
    plot_comparison_table(all_results_sorted)

    # Summary report
    log(f"\n{'='*70}")
    log("  FINAL COMPARISON — TEST SET")
    log(f"{'='*70}")
    log(f"  {'Model':<30} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'AUC':>8}")
    log(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for r in all_results_sorted:
        tag = " ← BEST" if r is all_results_sorted[0] else ""
        log(f"  {r['model']:<30} "
            f"{r['accuracy']*100:>7.2f}% "
            f"{r['precision']*100:>7.2f}% "
            f"{r['recall']*100:>7.2f}% "
            f"{r['f1']*100:>7.2f}% "
            f"{str(r['roc_auc']):>8}"
            f"{tag}")

    log(f"\n  Phase 3 CPU baseline (Custom_CNN):  85.84% val accuracy")
    log(f"  Best GPU model:  {all_results_sorted[0]['model']}  "
        f"{all_results_sorted[0]['accuracy']*100:.2f}% test accuracy")

    # Save
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results_sorted, f, indent=2)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    log(f"\n  Results JSON  → {RESULTS_PATH}")
    log(f"  Report        → {REPORT_PATH}")
    log(f"  Plots         → {PLOTS_DIR}/")
    log(f"\n{'='*70}")
    log("  PHASE 4 COMPLETE")
    log(f"{'='*70}")


if __name__ == "__main__":
    main()
