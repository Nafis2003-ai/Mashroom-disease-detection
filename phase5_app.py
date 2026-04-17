"""
phase5_app.py  —  Streamlit Deployment App
==========================================
Mushroom Disease Classification Web App

Install dependencies:
    pip install streamlit pillow tensorflow opencv-python-headless

Run locally:
    streamlit run phase5_app.py

Run on Colab (via ngrok tunnel):
    !pip install streamlit pyngrok -q
    !ngrok authtoken YOUR_TOKEN
    from pyngrok import ngrok
    public_url = ngrok.connect(8501)
    print(public_url)
    !streamlit run phase5_app.py &

Features:
  - Upload a mushroom image
  - Predict: Healthy / Single_Infected / Mixed_Infected
  - Show confidence scores with colour-coded bar
  - Show Grad-CAM heatmap overlay
  - Show class descriptions and recommended actions
"""

import os
import io
import warnings
import numpy as np

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
from PIL import Image

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mushroom Disease Detector",
    page_icon="🍄",
    layout="centered",
)

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE    = (224, 224)
NUM_CLASSES = 3
CLASSES     = ["Healthy", "Mixed_Infected", "Single_Infected"]

CLASS_INFO = {
    "Healthy": {
        "label": "Healthy",
        "color": "#27ae60",
        "icon":  "✅",
        "desc":  "Mycelium is developing normally with no visible contamination.",
        "action": "Continue standard cultivation protocol.",
    },
    "Single_Infected": {
        "label": "Single Infection",
        "color": "#e67e22",
        "icon":  "⚠️",
        "desc":  "Single mold contamination detected (likely Trichoderma, Aspergillus, or Rhizopus).",
        "action": "Isolate affected bags immediately. Increase ventilation. "
                  "Review substrate sterilisation process.",
    },
    "Mixed_Infected": {
        "label": "Mixed Infection",
        "color": "#e74c3c",
        "icon":  "🚨",
        "desc":  "Multiple contamination types or combined healthy + mold regions detected.",
        "action": "Remove and dispose of affected bags. Disinfect surrounding area. "
                  "Check environmental conditions (humidity, temperature).",
    },
}

# ── Model path — update if needed ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, "models")

# Preference order: best GPU model first, then older baselines
# preprocess_fn=None → model handles internally (EfficientNetB3, Custom CNN)
# preprocess_fn=fn   → apply fn to [0,255] array before inference
def _get_preprocess_map():
    from tensorflow.keras.applications import (
        resnet50 as resnet50_mod,
        vgg16 as vgg16_mod,
        inception_v3 as inception_v3_mod,
        densenet as densenet_mod,
    )
    return {
        "ResNet50":    resnet50_mod.preprocess_input,
        "VGG16":       vgg16_mod.preprocess_input,
        "InceptionV3": inception_v3_mod.preprocess_input,
        "DenseNet201": densenet_mod.preprocess_input,
    }

MODEL_CANDIDATES = [
    ("ResNet50",        os.path.join(MODELS_DIR, "ResNet50_redo_best.keras")),
    ("VGG16",           os.path.join(MODELS_DIR, "VGG16_redo_best.keras")),
    ("EfficientNetB3",  os.path.join(MODELS_DIR, "EfficientNetB3_redo_best.keras")),
    ("DenseNet201",     os.path.join(MODELS_DIR, "DenseNet201_redo_best.keras")),
    ("InceptionV3",     os.path.join(MODELS_DIR, "InceptionV3_redo_best.keras")),
    ("Custom CNN v2",   os.path.join(MODELS_DIR, "Custom_CNN_v2_redo_best.keras")),
    ("Custom CNN",      os.path.join(MODELS_DIR, "Custom_CNN_best.keras")),
]


# ── Load model (cached) ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    import tensorflow as tf
    preprocess_map = _get_preprocess_map()
    for model_name, path in MODEL_CANDIDATES:
        if os.path.exists(path):
            model = tf.keras.models.load_model(path, compile=False)
            preprocess_fn = preprocess_map.get(model_name, None)
            return model, model_name, preprocess_fn
    return None, None, None


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_image(img_pil, preprocess_fn):
    """Resize, convert to array, apply correct preprocessing for the model."""
    img = img_pil.convert("RGB").resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)   # [0, 255]
    if preprocess_fn is not None:
        arr = preprocess_fn(arr)            # e.g. ResNet50 mean subtraction
    return arr


# ── Grad-CAM ──────────────────────────────────────────────────────────────────
def compute_gradcam(model, img_array, class_idx):
    """Compute Grad-CAM heatmap for the given class."""
    import tensorflow as tf
    import numpy as np

    # Find GAP layer
    gap_layer = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
            gap_layer = layer
            break

    if gap_layer is None:
        return None

    gap_input_tensor = gap_layer.input
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[gap_input_tensor, model.output],
    )

    img_tensor = tf.cast(img_array[np.newaxis], tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        feature_maps, preds = grad_model(img_tensor, training=False)
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, feature_maps)

    if len(feature_maps.shape) != 4:
        return None

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    feature_maps = feature_maps[0]
    heatmap      = feature_maps @ pooled_grads[..., tf.newaxis]
    heatmap      = tf.squeeze(heatmap)
    heatmap      = tf.nn.relu(heatmap)
    heatmap      = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(img_pil, heatmap, alpha=0.4):
    """Overlay a Grad-CAM heatmap on a PIL image. Returns PIL Image."""
    import numpy as np
    import matplotlib.pyplot as plt

    img_arr    = np.array(img_pil.resize(IMG_SIZE).convert("RGB"))
    h, w       = img_arr.shape[:2]
    heatmap_pil = Image.fromarray(np.uint8(255 * heatmap)).resize(
        (w, h), Image.LANCZOS
    )
    heatmap_arr = np.array(heatmap_pil)

    # Apply jet colormap
    cmap       = plt.cm.jet(heatmap_arr / 255.0)[:, :, :3]
    overlay    = np.uint8(alpha * cmap * 255 + (1 - alpha) * img_arr)
    return Image.fromarray(overlay)


# ── Confidence bar ────────────────────────────────────────────────────────────
def render_confidence_bars(probs):
    for cls, prob in zip(CLASSES, probs):
        info    = CLASS_INFO[cls]
        pct     = prob * 100
        bar_color = info["color"]
        st.markdown(
            f"**{info['icon']} {info['label']}**",
        )
        st.progress(float(prob))
        st.caption(f"{pct:.1f}%")


# ── Main app ──────────────────────────────────────────────────────────────────
def main():
    # Header
    st.title("🍄 Mushroom Disease Detector")
    st.markdown(
        "Upload an image of oyster mushroom substrate bags to classify as "
        "**Healthy**, **Single Infected**, or **Mixed Infected**."
    )
    st.divider()

    # Load model
    model, model_name, preprocess_fn = load_model()
    if model is None:
        st.error(
            "No model found in `models/`. "
            "Please run `phase3_redo.py` first to train the models."
        )
        return

    st.sidebar.header("Model Info")
    st.sidebar.success(f"Active model: **{model_name}**")
    st.sidebar.caption(f"Parameters: {model.count_params():,}")
    show_gradcam = st.sidebar.checkbox("Show Grad-CAM heatmap", value=True)

    # Upload
    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "heic"],
        help="Upload a mushroom bag/substrate image",
    )

    if uploaded is None:
        st.info("Please upload an image to get started.")
        _show_sample_classes()
        return

    # Display uploaded image
    img_pil = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(img_pil, use_column_width=True)

    # Predict
    with st.spinner("Analysing..."):
        img_arr   = preprocess_image(img_pil, preprocess_fn)
        import tensorflow as tf
        preds     = model.predict(img_arr[np.newaxis], verbose=0)[0]
        pred_idx  = int(np.argmax(preds))
        pred_cls  = CLASSES[pred_idx]
        pred_info = CLASS_INFO[pred_cls]

    # Grad-CAM
    if show_gradcam:
        with st.spinner("Generating Grad-CAM..."):
            heatmap = compute_gradcam(model, img_arr, pred_idx)

        with col2:
            st.subheader("Grad-CAM")
            if heatmap is not None:
                overlay = overlay_heatmap(img_pil, heatmap)
                st.image(overlay, use_column_width=True)
                st.caption("Highlighted regions most influential for the prediction.")
            else:
                st.image(img_pil, use_column_width=True)
                st.caption("Grad-CAM not available for this model architecture.")
    else:
        with col2:
            st.subheader("Resized (224×224)")
            st.image(img_pil.resize(IMG_SIZE), use_column_width=True)

    # Result banner
    st.divider()
    confidence = float(preds[pred_idx]) * 100
    st.markdown(
        f"<div style='background:{pred_info['color']}22; "
        f"border-left:5px solid {pred_info['color']}; "
        f"padding:16px; border-radius:8px;'>"
        f"<h2 style='color:{pred_info['color']}; margin:0'>"
        f"{pred_info['icon']} {pred_info['label']}</h2>"
        f"<p style='margin:8px 0 0'><b>Confidence: {confidence:.1f}%</b></p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Confidence breakdown
    st.subheader("Confidence Scores")
    render_confidence_bars(preds)

    # Description + Action
    st.subheader("Diagnosis")
    st.write(pred_info["desc"])
    st.subheader("Recommended Action")
    st.write(pred_info["action"])

    if confidence < 70:
        st.warning(
            f"Low confidence ({confidence:.1f}%). "
            "Consider capturing a clearer image or consulting an expert."
        )


def _show_sample_classes():
    """Show class descriptions when no image is uploaded."""
    st.subheader("Classes")
    for cls, info in CLASS_INFO.items():
        with st.expander(f"{info['icon']} {info['label']}"):
            st.write(info["desc"])
            st.write(f"**Action:** {info['action']}")


if __name__ == "__main__":
    main()
