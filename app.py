"""
app.py — Mushroom Disease Classification + Expert Chat (Unified App)
====================================================================
Merged Phase 5 (detection) + Phase 6 (RAG chat) into one Streamlit app.

Tabs:
  Disease Detection  — EfficientNetV2S + CORN ordinal grading + Grad-CAM
  Expert Chat        — RAG chatbot (ChromaDB + Groq Llama-3.3-70B)

Grad-CAM strategy (3-tier fallback):
  1. Grad-CAM via _inbound_nodes graph tracing (proper, most informative)
  2. Gradient saliency map (always works, visually similar)
  3. Show error message with reason

Usage:
    pip install streamlit pillow opencv-python-headless pandas \
                chromadb sentence-transformers groq python-dotenv
    streamlit run app.py
"""

import os, io, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import numpy as np
from pathlib import Path
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Page config — MUST be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mushroom Disease AI",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# 0b. Global CSS — full visual redesign
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Root reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* ── App background ── */
.stApp {
    background: linear-gradient(145deg, #F0F4F8 0%, #E8EEF5 100%);
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer { visibility: hidden; }
header { visibility: hidden; }
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1280px !important;
}

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #1B3A5C 0%, #27AE60 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 8px 32px rgba(27,58,92,0.18);
}
.hero-title {
    font-size: 2rem;
    font-weight: 800;
    color: #FFFFFF;
    margin: 0;
    letter-spacing: -0.5px;
}
.hero-sub {
    font-size: 0.92rem;
    color: rgba(255,255,255,0.78);
    margin: 0.3rem 0 0 0;
}
.hero-badge {
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    color: white;
    padding: 0.4rem 1rem;
    border-radius: 24px;
    font-size: 0.8rem;
    font-weight: 600;
    backdrop-filter: blur(6px);
}

/* ── Card ── */
.card {
    background: #FFFFFF;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 2px 16px rgba(27,58,92,0.08);
    margin-bottom: 1.2rem;
    border: 1px solid rgba(27,58,92,0.06);
}
.card-title {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #1B3A5C;
    margin-bottom: 0.8rem;
}

/* ── Upload zone ── */
.upload-zone {
    background: #F8FBFF;
    border: 2.5px dashed #B0C4DE;
    border-radius: 14px;
    padding: 3rem 2rem;
    text-align: center;
    transition: all 0.2s;
}
.upload-icon { font-size: 3.5rem; margin-bottom: 0.8rem; }
.upload-text { color: #5D6D7E; font-size: 1rem; font-weight: 500; }
.upload-hint { color: #ADB5BD; font-size: 0.82rem; margin-top: 0.4rem; }

/* ── Grade cards ── */
.grade-card {
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.07);
}
.grade-icon { font-size: 2.4rem; line-height: 1; }
.grade-label { font-size: 1.25rem; font-weight: 700; margin: 0; }
.grade-sublabel { font-size: 0.82rem; margin: 0.2rem 0 0 0; opacity: 0.8; }
.risk-pill {
    display: inline-block;
    padding: 0.2rem 0.75rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.8px;
    background: rgba(255,255,255,0.3);
    color: white;
    margin-top: 0.5rem;
}

/* ── Confidence bars ── */
.conf-row {
    margin-bottom: 0.9rem;
}
.conf-header {
    display: flex;
    justify-content: space-between;
    font-size: 0.83rem;
    font-weight: 600;
    margin-bottom: 0.35rem;
    color: #2C3E50;
}
.conf-bar-bg {
    background: #EEF2F7;
    border-radius: 8px;
    height: 10px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 10px;
    border-radius: 8px;
    transition: width 0.8s ease;
}

/* ── Advice box ── */
.advice-box {
    background: #F8FBFF;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    border-left: 4px solid #1B3A5C;
    margin-bottom: 0.5rem;
}
.advice-line {
    font-size: 0.9rem;
    color: #2C3E50;
    line-height: 1.65;
    margin: 0.2rem 0;
}

/* ── Section headings ── */
.sec-heading {
    font-size: 1rem;
    font-weight: 700;
    color: #1B3A5C;
    margin: 1.2rem 0 0.7rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.sec-heading::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(to right, #1B3A5C22, transparent);
}

/* ── Severity scale ── */
.scale-card {
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    border: 2px solid transparent;
    transition: all 0.2s;
}
.scale-active { border-color: currentColor !important; box-shadow: 0 4px 16px rgba(0,0,0,0.12); }
.scale-icon { font-size: 1.6rem; }
.scale-grade { font-size: 0.85rem; font-weight: 700; margin-top: 0.3rem; }
.scale-name { font-size: 0.72rem; color: #6C757D; margin-top: 0.15rem; }

/* ── Streamlit file uploader overrides ── */
[data-testid="stFileUploader"] {
    background: transparent !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: #F8FBFF !important;
    border: 2.5px dashed #B0C4DE !important;
    border-radius: 14px !important;
    padding: 1.8rem !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #1B3A5C !important;
    background: #EEF4FF !important;
}
/* ── Upload button inside dropzone ── */
[data-testid="stFileUploaderDropzone"] button {
    background: #FFFFFF !important;
    color: #1B3A5C !important;
    border: 1.5px solid #CBD5E0 !important;
    border-radius: 8px !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    padding: 0.4rem 1.1rem !important;
    box-shadow: 0 1px 4px rgba(27,58,92,0.10) !important;
    transition: all 0.2s !important;
    cursor: pointer !important;
}
[data-testid="stFileUploaderDropzone"] button:hover {
    background: #F0F4FF !important;
    border-color: #1B3A5C !important;
    box-shadow: 0 2px 8px rgba(27,58,92,0.15) !important;
    transform: translateY(-1px) !important;
}
[data-testid="stFileUploaderDropzone"] button span,
[data-testid="stFileUploaderDropzone"] button p,
[data-testid="stFileUploaderDropzone"] button div,
[data-testid="stMarkdownContainer"] [data-testid="stFileUploaderDropzone"] button span,
[data-testid="stMarkdownContainer"] [data-testid="stFileUploaderDropzone"] button p {
    color: #1B3A5C !important;
    opacity: 1 !important;
}

/* ── Tabs redesign ── */
[data-baseweb="tab-list"] {
    background: #FFFFFF !important;
    border-radius: 12px !important;
    padding: 6px !important;
    box-shadow: 0 2px 12px rgba(27,58,92,0.08) !important;
    gap: 4px !important;
    margin-bottom: 1.2rem !important;
}
[data-baseweb="tab"] {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.6rem 1.4rem !important;
    color: #5D6D7E !important;
    transition: all 0.2s !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: linear-gradient(135deg, #1B3A5C, #27AE60) !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 12px rgba(27,58,92,0.25) !important;
}
[data-baseweb="tab-highlight"] { display: none !important; }
[data-baseweb="tab-border"] { display: none !important; }
[aria-selected="true"][data-baseweb="tab"] p,
[aria-selected="true"][data-baseweb="tab"] span,
[aria-selected="true"][data-baseweb="tab"] div {
    color: #FFFFFF !important;
}

/* ── Main content — always dark text ── */
.main .block-container,
[data-testid="stMain"] {
    color: #2C3E50 !important;
}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span {
    color: #2C3E50 !important;
}
[data-testid="stMarkdownContainer"] div:not(.hero-banner):not(.hero-title):not(.hero-sub):not(.hero-badge) {
    color: #2C3E50 !important;
}
/* assistant messages — dark text on light background */
[data-testid="stChatMessage"][data-testid*="assistant"] p,
[data-testid="stChatMessage"][data-testid*="assistant"] li,
[data-testid="stChatMessage"][data-testid*="assistant"] span,
[data-testid="stChatMessage"][data-testid*="assistant"] div {
    color: #2C3E50 !important;
}
/* user messages — white text on dark background */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) p,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) li,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) span,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) div {
    color: #FFFFFF !important;
}
/* fallback — all chat text dark unless overridden above */
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span {
    color: #2C3E50 !important;
}
/* ── Hero banner — white text, placed AFTER dark-text rules ── */
.hero-banner { color: #FFFFFF !important; }
.hero-title  { color: #FFFFFF !important; }
.hero-sub    { color: rgba(255,255,255,0.85) !important; }
.hero-badge  { color: #FFFFFF !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1B3A5C 0%, #162E4A 100%) !important;
}
[data-testid="stSidebar"] > div {
    color: rgba(255,255,255,0.9) !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span:not(.dot-ok):not(.dot-warn),
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown span {
    color: rgba(255,255,255,0.75) !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.12) !important;
}
[data-testid="stSidebar"] [data-testid="stToggleLabel"],
[data-testid="stSidebar"] [data-testid="stSliderLabel"] {
    color: rgba(255,255,255,0.8) !important;
}

/* Sidebar status dots */
.sb-status {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.55rem 0.75rem;
    background: rgba(255,255,255,0.07);
    border-radius: 10px;
    margin: 0.35rem 0;
    font-size: 0.82rem;
}
.dot-ok  { width:9px; height:9px; border-radius:50%; background:#27AE60; flex-shrink:0; }
.dot-warn{ width:9px; height:9px; border-radius:50%; background:#F39C12; flex-shrink:0; }
.sb-section-title {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.4px;
    color: rgba(255,255,255,0.45) !important;
    margin: 1.1rem 0 0.45rem 0.1rem;
}

/* ── Metric override ── */
[data-testid="stMetric"] {
    background: #F0F6FF;
    border-radius: 12px;
    padding: 0.9rem 1.1rem !important;
    border: 1px solid #DDEAFF;
}
[data-testid="stMetricLabel"] { font-size: 0.78rem !important; color: #5D6D7E !important; }
[data-testid="stMetricValue"] { font-size: 1.15rem !important; color: #1B3A5C !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] { font-size: 0.82rem !important; color: #27AE60 !important; }

/* ── Buttons ── */
.stButton button {
    background: linear-gradient(135deg, #1B3A5C, #27AE60) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1.1rem !important;
    transition: all 0.2s !important;
    box-shadow: 0 2px 8px rgba(27,58,92,0.2) !important;
}
.stButton button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(27,58,92,0.3) !important;
}
/* ── Force white text inside ALL buttons (beats stMarkdownContainer p rule) ── */
[data-testid="stButton"] button p,
[data-testid="stButton"] button span,
[data-testid="stButton"] button div,
[data-testid="stBaseButton-secondary"] p,
[data-testid="stBaseButton-secondary"] span,
[data-testid="stBaseButton-secondary"] div {
    color: #FFFFFF !important;
}

/* ── Chat bubbles ── */
[data-testid="stChatMessage"] {
    border-radius: 12px !important;
    margin-bottom: 0.6rem !important;
    border: 1px solid rgba(27,58,92,0.07) !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
}
/* ── Chat input — fixed to bottom like ChatGPT ── */
[data-testid="stBottomBlockContainer"] {
    position: fixed !important;
    bottom: 0 !important;
    left: 0 !important;
    right: 0 !important;
    z-index: 999 !important;
    background: linear-gradient(to top, #F0F4F8 80%, transparent) !important;
    padding: 1rem 1.5rem 1.2rem 1.5rem !important;
}
[data-testid="stChatInput"] {
    border-radius: 24px !important;
    border: 2px solid #B0C4DE !important;
    background: #FFFFFF !important;
    box-shadow: 0 4px 24px rgba(27,58,92,0.12) !important;
    max-width: 860px !important;
    margin: 0 auto !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #1B3A5C !important;
    box-shadow: 0 4px 28px rgba(27,58,92,0.2) !important;
}
/* Push chat content up so it doesn't hide behind fixed input */
[data-testid="stChatMessageContainer"],
.main .block-container {
    padding-bottom: 90px !important;
}

/* ── Example question pills ── */
.example-pill button {
    background: linear-gradient(135deg, #1B3A5C, #27AE60) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 20px !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    box-shadow: 0 2px 8px rgba(27,58,92,0.2) !important;
}
.example-pill button p,
.example-pill button span,
.example-pill button div,
[data-testid="stMarkdownContainer"] .example-pill button p,
[data-testid="stMarkdownContainer"] .example-pill button span,
[data-testid="stMarkdownContainer"] .example-pill button div,
[data-testid="stMarkdownContainer"] .example-pill button {
    color: #FFFFFF !important;
}
.example-pill button:hover {
    background: linear-gradient(135deg, #27AE60, #1B3A5C) !important;
    color: #FFFFFF !important;
    border-color: transparent !important;
    transform: translateY(-1px) !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #1B3A5C !important; }

/* ── Progress bar ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #1B3A5C, #27AE60) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid rgba(27,58,92,0.1) !important;
    border-radius: 10px !important;
    background: #FAFCFF !important;
}
[data-testid="stExpander"] summary {
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    color: #1B3A5C !important;
}

/* ── Toggle / slider ── */
[data-baseweb="toggle"] div { background-color: #27AE60 !important; }
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: #1B3A5C !important;
    border-color: #1B3A5C !important;
}

/* ── Divider ── */
hr { border-color: rgba(27,58,92,0.1) !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 10px !important; overflow: hidden !important; }

/* ── AI insight box ── */
.insight-box {
    background: linear-gradient(135deg, #F0F6FF, #F0FBF4);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    border: 1px solid #DDEAFF;
    font-size: 0.9rem;
    line-height: 1.7;
    color: #2C3E50;
}

/* ── Footer ── */
.app-footer {
    text-align: center;
    padding: 1.5rem;
    color: #ADB5BD;
    font-size: 0.78rem;
    border-top: 1px solid rgba(27,58,92,0.08);
    margin-top: 2rem;
}

/* ── Responsive ── */
@media (max-width: 768px) {
    .hero-title { font-size: 1.4rem !important; }
    .hero-badge { display: none; }
    .block-container { padding: 0.5rem !important; }
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar toggle button (components.v1 for parent window access) ───────────
import streamlit.components.v1 as _components
_components.html("""
<style>
  #stoggle {
    position: fixed;
    top: 50vh;
    left: 0;
    transform: translateY(-50%);
    z-index: 999999;
    background: #1B3A5C;
    color: white;
    border: none;
    border-radius: 0 8px 8px 0;
    padding: 14px 9px;
    font-size: 1.1rem;
    cursor: pointer;
    box-shadow: 2px 0 10px rgba(0,0,0,0.3);
    transition: background 0.2s;
    line-height: 1;
  }
  #stoggle:hover { background: #27AE60; }
</style>
<button id="stoggle" title="Toggle sidebar">&#9776;</button>
<script>
document.getElementById('stoggle').addEventListener('click', function() {
    var p = window.parent;
    // Try the collapsed-control button first
    var btn = p.document.querySelector('[data-testid="stSidebarCollapsedControl"] button')
           || p.document.querySelector('[data-testid="stSidebarCollapsedControl"]')
           || p.document.querySelector('[data-testid="collapsedControl"]');
    if (btn) { btn.click(); return; }
    // Fall back: toggle the sidebar visibility directly
    var sidebar = p.document.querySelector('[data-testid="stSidebar"]');
    if (!sidebar) return;
    var cur = p.getComputedStyle(sidebar).marginLeft;
    sidebar.style.transition = 'margin-left 0.3s ease';
    if (cur === '0px' || cur === '') {
        sidebar.style.marginLeft = '-21rem';
    } else {
        sidebar.style.marginLeft = '0px';
    }
});
</script>
""", height=0)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Constants
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DB_DIR     = str(BASE_DIR / "rag_db" / "rag_db")
IMG_SIZE   = 224
NUM_CLASSES = 3
NUM_TASKS   = NUM_CLASSES - 1

ORDINAL_NAMES = ["Healthy", "Single Infected", "Mixed Infected"]

GRADE_META = {
    0: {
        "label":     "Healthy",
        "full":      "Grade 0 — Healthy",
        "emoji":     "✅",
        "color":     "#27AE60",
        "bg":        "#E8F8F0",
        "risk":      "LOW RISK",
        "rag_query": "healthy oyster mushroom substrate bag signs and maintenance",
        "advice": [
            "No disease detected. Your mushroom culture looks healthy! 🎉",
            "• Keep temperature between 22–28 °C and humidity at 80–90%.",
            "• CO₂ should stay below 1,000 ppm — ensure fresh air exchange.",
            "• Inspect bags every 48–72 hours and log any changes.",
            "• Document your substrate batch and spawn source for traceability.",
        ],
    },
    1: {
        "label":     "Single Infected",
        "full":      "Grade 1 — Single Pathogen",
        "emoji":     "⚠️",
        "color":     "#E67E22",
        "bg":        "#FEF9E7",
        "risk":      "MEDIUM RISK",
        "rag_query": "single pathogen infection oyster mushroom Trichoderma Aspergillus treatment isolation",
        "advice": [
            "One pathogen strain detected at an early or moderate stage.",
            "• **Isolate** affected bags immediately to stop spread.",
            "• Identify the mold: Green = Trichoderma, Black/Yellow = Aspergillus, Soft rot = Rhizopus.",
            "• Apply targeted treatment: fungicide drench or bactericide spray.",
            "• Increase fresh-air exchange; reduce humidity by 5–10%.",
            "• Re-inspect in 24 hours — if spreading, escalate to Grade 2 protocol.",
        ],
    },
    2: {
        "label":     "Mixed Infected",
        "full":      "Grade 2 — Severe Contamination",
        "emoji":     "🚨",
        "color":     "#E74C3C",
        "bg":        "#FDEDEC",
        "risk":      "HIGH RISK",
        "rag_query": "mixed infection severe contamination oyster mushroom disposal sterilization protocol",
        "advice": [
            "Multiple pathogens or advanced contamination detected. Act immediately!",
            "• **Remove and dispose** of all affected bags right away.",
            "• Do NOT compost — use sealed bags for safe disposal.",
            "• Deep-clean the growing chamber with H₂O₂ (3%) or bleach solution.",
            "• Investigate the source: spawn quality, substrate sterilisation, air filtration.",
            "• Halt new inoculations in this room until root cause is found.",
        ],
    },
}

MODEL_PRIORITY = [
    ("EfficientNetV2S_corn.keras",   "EfficientNetV2S + CORN (Phase 3C — 88.03%)", "corn",    "norm_11"),
    ("EfficientNetV2S_CORN_D.keras", "EfficientNetV2S + CORN (Phase 3D diffusion)", "corn",    "raw_255"),
    ("EfficientNetV2S_best.keras",   "EfficientNetV2S (Phase 3B — softmax)",        "softmax", "raw_255"),
]

EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION  = "mushroom_knowledge"
TOP_K       = 4
MAX_HISTORY = 6

RAG_SYSTEM_PROMPT = """You are a knowledgeable mushroom expert assistant specializing in:
- Oyster mushroom cultivation and substrate bag disease management
- Mushroom disease identification (Trichoderma, Aspergillus, Rhizopus, bacterial blotch, etc.)
- Disease prevention, treatment, and cultivation best practices
- Mushroom species — edible, medicinal, poisonous

Use the provided context. Be concise and practical. Use bullet points.
If context is insufficient, use your knowledge but note that.
Never fabricate specific numbers or species classifications."""

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Load .env
# ─────────────────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

# Read from Streamlit secrets (cloud) or .env (local)
GROQ_API_KEY = ""
try:
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
except Exception:
    pass
if not GROQ_API_KEY:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  CORN helpers
# ─────────────────────────────────────────────────────────────────────────────
def corn_predict(logits: np.ndarray) -> np.ndarray:
    import tensorflow as tf
    probs = tf.sigmoid(tf.cast(logits, tf.float32)).numpy()
    B = probs.shape[0]
    out = np.zeros((B, NUM_CLASSES), dtype=np.float32)
    for i in range(B):
        p = probs[i]
        out[i, 0] = 1.0 - p[0]
        out[i, 1] = p[0] * (1.0 - p[1])
        out[i, 2] = p[0] * p[1]
    return out

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Model loading
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading detection model…")
def load_model():
    import tensorflow as tf
    from tensorflow import keras
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    for fname, desc, mode, preproc in MODEL_PRIORITY:
        path = MODELS_DIR / fname
        if path.exists():
            try:
                m = keras.models.load_model(str(path), compile=False)
                return m, desc, mode, preproc
            except Exception as e:
                st.warning(f"Could not load {fname}: {e}")
    return None, "No model found", None, "raw_255"

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Preprocessing + inference
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(pil_img: Image.Image, preproc: str = "raw_255") -> np.ndarray:
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    if preproc == "norm_11":
        arr = arr / 127.5 - 1.0
    return np.expand_dims(arr, 0)

def predict(model, img_array: np.ndarray, mode: str):
    import tensorflow as tf
    logits = model(img_array, training=False).numpy()
    if mode == "corn" or logits.shape[-1] == NUM_TASKS:
        probs = corn_predict(logits)[0]
    else:
        raw   = tf.nn.softmax(logits[0]).numpy()
        probs = np.array([raw[0], raw[2], raw[1]], dtype=np.float32)
    return probs, int(np.argmax(probs))

# ─────────────────────────────────────────────────────────────────────────────
# 6.  Grad-CAM — 3-tier fallback
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Building visual explanation model…")
def build_gradcam_model(_model):
    import tensorflow as tf
    from tensorflow import keras

    backbone = None
    for layer in _model.layers:
        if "efficientnetv2" in layer.name.lower():
            backbone = layer
            break
    if backbone is None:
        return None, "saliency", "Backbone not found"

    last_conv = None
    for layer in backbone.layers:
        if isinstance(layer, (keras.layers.Conv2D, keras.layers.DepthwiseConv2D)):
            last_conv = layer
    if last_conv is None:
        return None, "saliency", "No Conv2D found inside backbone"

    try:
        inbound = getattr(backbone, "_inbound_nodes", [])
        if inbound:
            bb_inp = inbound[0].input_tensors
            if isinstance(bb_inp, (list, tuple)):
                bb_inp = bb_inp[0]
        else:
            bb_inp = _model.input

        conv_sub = keras.Model(
            inputs=backbone.input,
            outputs=backbone.get_layer(last_conv.name).output,
            name="conv_sub",
        )
        conv_tensor = conv_sub(bb_inp)
        grad_m = keras.Model(
            inputs=_model.input,
            outputs=[conv_tensor, _model.output],
            name="grad_model",
        )
        return grad_m, "gradcam", None
    except Exception as e:
        return None, "saliency", f"Grad-CAM graph build failed: {e}"


def compute_visual_explanation(model, grad_model, method,
                                img_array, class_idx, model_mode, model_preproc):
    import tensorflow as tf
    task_idx = min(class_idx, NUM_TASKS - 1) if model_mode == "corn" else class_idx

    if grad_model is not None and method == "gradcam":
        try:
            img_var = tf.Variable(tf.cast(img_array, tf.float32))
            with tf.GradientTape() as tape:
                conv_out, preds = grad_model(img_var, training=False)
                score = preds[:, task_idx]
            grads = tape.gradient(score, conv_out)
            if grads is not None:
                pooled  = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
                heatmap = conv_out[0].numpy() @ pooled
                heatmap = np.maximum(heatmap, 0)
                mx = heatmap.max()
                if mx > 0:
                    return (heatmap / mx).astype(np.float32), "Grad-CAM"
        except Exception:
            pass

    try:
        img_var = tf.Variable(tf.cast(img_array, tf.float32))
        with tf.GradientTape() as tape:
            logits = model(img_var, training=False)
            score  = logits[:, task_idx]
        grads    = tape.gradient(score, img_var)
        saliency = tf.reduce_max(tf.abs(grads[0]), axis=-1).numpy()
        mn, mx   = saliency.min(), saliency.max()
        if mx > mn:
            saliency = (saliency - mn) / (mx - mn)
        return saliency.astype(np.float32), "Saliency"
    except Exception as e:
        return None, f"Error: {e}"


def overlay_heatmap(pil_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.45):
    orig     = np.array(pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE)))
    h        = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    colormap = cv2.applyColorMap(np.uint8(255 * h), cv2.COLORMAP_JET)
    colormap = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
    blended  = (orig * (1 - alpha) + colormap * alpha).astype(np.uint8)
    return Image.fromarray(blended)

# ─────────────────────────────────────────────────────────────────────────────
# 7.  RAG helpers
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading knowledge base…")
def load_rag():
    try:
        import torch, chromadb
        from sentence_transformers import SentenceTransformer
        device = "cpu"
        if torch.cuda.is_available():
            try: torch.cuda.set_per_process_memory_fraction(0.0)
            except Exception: pass
        client     = chromadb.PersistentClient(path=DB_DIR)
        collection = client.get_collection(COLLECTION)
        embedder   = SentenceTransformer(EMBED_MODEL, device=device)
        return collection, embedder
    except Exception:
        return None, None

@st.cache_resource(show_spinner="Connecting to Groq…")
def load_groq_client(api_key: str):
    from groq import Groq
    return Groq(api_key=api_key)

def rag_retrieve(query, collection, embedder, k=TOP_K):
    q_emb   = embedder.encode([query], normalize_embeddings=True, device="cpu").tolist()
    results = collection.query(
        query_embeddings=q_emb, n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    return results["documents"][0], results["metadatas"][0]

def rag_answer(groq_client, query, chunks, history=None):
    history_str = ""
    if history:
        history_str = "\n--- Previous conversation ---\n"
        for m in history[-MAX_HISTORY:]:
            role = "User" if m["role"] == "user" else "Assistant"
            history_str += f"{role}: {m['content']}\n"
        history_str += "---\n"

    context = "\n\n".join(f"[Source {i+1}]\n{c}" for i, c in enumerate(chunks))
    prompt  = f"""{RAG_SYSTEM_PROMPT}

{history_str}
--- Retrieved knowledge ---
{context}
--- End of knowledge ---

User question: {query}
Answer:"""
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=1024,
        )
        return resp.choices[0].message.content
    except Exception as e:
        err = str(e)
        if "401" in err or "403" in err or "api_key" in err.lower():
            return "❌ **API key error.** Check your `GROQ_API_KEY` in `.env`."
        return f"❌ **Generation error:** {err}"

# ─────────────────────────────────────────────────────────────────────────────
# 8.  UI component helpers
# ─────────────────────────────────────────────────────────────────────────────
def grade_card(grade: int):
    m = GRADE_META[grade]
    st.markdown(
        f"""<div class="grade-card" style="background:{m['bg']};border:2px solid {m['color']}33;">
          <div class="grade-icon">{m['emoji']}</div>
          <div>
            <div class="grade-label" style="color:{m['color']}">{m['full']}</div>
            <div class="grade-sublabel" style="color:{m['color']}cc">
              AI detected: <strong>{m['label']}</strong>
            </div>
            <span class="risk-pill" style="background:{m['color']}">{m['risk']}</span>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )

def confidence_bars(probs, grade):
    colors = ["#27AE60", "#E67E22", "#E74C3C"]
    labels = ["Healthy", "Single Infected", "Mixed Infected"]
    st.markdown('<div class="card"><div class="card-title">Confidence Score</div>', unsafe_allow_html=True)
    for i, (name, prob) in enumerate(zip(labels, probs)):
        pct  = prob * 100
        bold = "font-weight:700;" if i == grade else ""
        opacity = "1" if i == grade else "0.55"
        st.markdown(
            f"""<div class="conf-row" style="opacity:{opacity}">
              <div class="conf-header" style="{bold}">
                <span>{name}</span>
                <span style="color:{colors[i]}">{pct:.1f}%</span>
              </div>
              <div class="conf-bar-bg">
                <div class="conf-bar-fill" style="width:{pct:.1f}%;background:{colors[i]};"></div>
              </div>
            </div>""",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

def advice_card(grade: int):
    m = GRADE_META[grade]
    lines_html = "".join(
        f'<div class="advice-line">{line}</div>' for line in m["advice"]
    )
    st.markdown(
        f"""<div class="advice-box" style="border-left-color:{m['color']};">
          {lines_html}
        </div>""",
        unsafe_allow_html=True,
    )

def section_heading(icon: str, title: str):
    st.markdown(
        f'<div class="sec-heading">{icon} {title}</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# 9.  Sidebar
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar(model_desc, rag_ready, rag_chunks):
    with st.sidebar:
        # Logo / brand
        st.markdown(
            """<div style="text-align:center;padding:1rem 0 0.5rem 0">
              <div style="font-size:3rem">🍄</div>
              <div style="font-size:1.1rem;font-weight:800;color:white;letter-spacing:-0.3px">
                Mushroom Disease AI
              </div>
              <div style="font-size:0.75rem;color:rgba(255,255,255,0.5);margin-top:0.2rem">
                Ordinal Grading System
              </div>
            </div>""",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sb-section-title">System Status</div>', unsafe_allow_html=True)

        dot_m = "dot-ok" if model_desc != "No model found" else "dot-warn"
        dot_r = "dot-ok" if rag_ready else "dot-warn"
        dot_g = "dot-ok" if GROQ_API_KEY else "dot-warn"

        st.markdown(
            f"""
            <div class="sb-status"><div class="{dot_m}"></div>
              <span style="font-size:0.82rem">Detection Model</span></div>
            <div class="sb-status"><div class="{dot_r}"></div>
              <span style="font-size:0.82rem">Knowledge Base
                {f'({rag_chunks} chunks)' if rag_ready else '— not set up'}</span></div>
            <div class="sb-status"><div class="{dot_g}"></div>
              <span style="font-size:0.82rem">Groq AI API
                {'connected' if GROQ_API_KEY else '— key missing'}</span></div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sb-section-title">Model</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:0.78rem;color:rgba(255,255,255,0.6);'
            f'background:rgba(255,255,255,0.06);border-radius:8px;padding:0.5rem 0.7rem;">'
            f'{model_desc}</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sb-section-title">Grade Legend</div>', unsafe_allow_html=True)
        for g, m in GRADE_META.items():
            st.markdown(
                f'<div class="sb-status"><div style="width:9px;height:9px;border-radius:50%;'
                f'background:{m["color"]};flex-shrink:0;"></div>'
                f'<span style="font-size:0.82rem">{m["emoji"]} Grade {g}: {m["label"]}</span></div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="sb-section-title">Visual Explanation</div>', unsafe_allow_html=True)
        show_heatmap  = st.toggle("Show Grad-CAM / Saliency", value=True)
        heatmap_alpha = st.slider("Overlay opacity", 0.1, 0.9, 0.45, 0.05,
                                  disabled=not show_heatmap)

        st.markdown(
            '<div style="margin-top:1.5rem;font-size:0.7rem;color:rgba(255,255,255,0.3);'
            'text-align:center;">EfficientNetV2S + CORN Loss<br>CSE 499B · NSU</div>',
            unsafe_allow_html=True,
        )

        return show_heatmap, heatmap_alpha

# ─────────────────────────────────────────────────────────────────────────────
# 10.  Detection tab
# ─────────────────────────────────────────────────────────────────────────────
def render_detection(model, model_mode, model_preproc,
                     grad_model, gradcam_method,
                     rag_collection, rag_embedder, groq_client,
                     show_heatmap, heatmap_alpha):

    # Upload area
    section_heading("📷", "Upload Image")
    uploaded = st.file_uploader(
        "Upload mushroom bag image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed",
    )

    if not uploaded:
        st.markdown(
            """<div class="upload-zone">
              <div class="upload-icon">📸</div>
              <div class="upload-text">Drop your mushroom bag photo here</div>
              <div class="upload-hint">Supports JPG, PNG, WEBP · Works best with clear, well-lit photos</div>
            </div>""",
            unsafe_allow_html=True,
        )
        # Quick guide
        st.markdown("")
        g1, g2, g3 = st.columns(3)
        for col, emoji, tip in zip(
            [g1, g2, g3],
            ["💡", "📐", "🌿"],
            ["Good lighting gives better results",
             "Capture the full bag in frame",
             "Photo from 20–30 cm distance"],
        ):
            with col:
                st.markdown(
                    f'<div class="card" style="text-align:center;padding:1rem;">'
                    f'<div style="font-size:1.8rem">{emoji}</div>'
                    f'<div style="font-size:0.8rem;color:#5D6D7E;margin-top:0.4rem">{tip}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        return

    pil_img = Image.open(uploaded).convert("RGB")

    # ── Analyse ──────────────────────────────────────────────────────────
    with st.spinner("Analysing your image…"):
        img_array    = preprocess(pil_img, model_preproc)
        probs, grade = predict(model, img_array, model_mode)

    # ── Two column layout ─────────────────────────────────────────────────
    col_img, col_res = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown('<div class="card" style="padding:0.8rem;">', unsafe_allow_html=True)
        st.image(pil_img, caption="Uploaded image", use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_res:
        # Grade result card
        grade_card(grade)
        # Confidence bars
        confidence_bars(probs, grade)
        # Metric
        st.metric(
            "AI Confidence",
            f"{probs[grade]*100:.1f}%",
            f"Grade {grade}: {GRADE_META[grade]['label']}",
        )

    # ── Visual explanation ────────────────────────────────────────────────
    if show_heatmap:
        section_heading("🔍", "Visual Explanation — Where the AI looked")
        with st.spinner("Computing heatmap…"):
            heatmap, method_used = compute_visual_explanation(
                model, grad_model, gradcam_method,
                img_array, grade, model_mode, model_preproc,
            )

        if heatmap is not None and isinstance(method_used, str) and not method_used.startswith("Error"):
            overlay = overlay_heatmap(pil_img, heatmap, alpha=heatmap_alpha)
            gc1, gc2, gc3 = st.columns(3)
            captions = ["Original", f"{method_used} Map", "Overlay"]
            images   = [
                pil_img.resize((IMG_SIZE, IMG_SIZE)),
                None,   # heatmap rendered below
                overlay,
            ]
            for col, cap, img_ in zip([gc1, gc2, gc3], captions, images):
                with col:
                    st.markdown(
                        f'<div class="card" style="padding:0.6rem;text-align:center;">'
                        f'<div style="font-size:0.72rem;font-weight:700;color:#1B3A5C;'
                        f'text-transform:uppercase;letter-spacing:1px;margin-bottom:0.4rem">{cap}</div>',
                        unsafe_allow_html=True,
                    )
                    if img_ is None:
                        fig, ax = plt.subplots(figsize=(3, 3))
                        ax.imshow(heatmap, cmap="jet"); ax.axis("off")
                        buf = io.BytesIO()
                        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
                        plt.close(fig); buf.seek(0)
                        st.image(buf.read(), use_column_width=True)
                    else:
                        st.image(img_, use_column_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(
                f'<div style="font-size:0.78rem;color:#5D6D7E !important;margin-top:-0.5rem;'
                f'background:#F8FBFF;padding:0.5rem 0.8rem;border-radius:8px;">'
                f'Method: <strong style="color:#1B3A5C">{method_used}</strong> — '
                f'<span style="color:#5D6D7E">Red/yellow = high AI attention &nbsp;|&nbsp; Blue = low attention</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.warning(f"Visual explanation unavailable: {method_used}")

    # ── Advisory ──────────────────────────────────────────────────────────
    section_heading("💡", "Recommended Actions")
    advice_card(grade)

    # ── Auto RAG Insight ──────────────────────────────────────────────────
    if rag_collection is not None and groq_client is not None:
        section_heading("🤖", "AI Expert Insight")
        st.markdown(
            '<div style="font-size:0.8rem;color:#6C757D;margin:-0.4rem 0 0.6rem 0">'
            'Automatically retrieved from the mushroom knowledge base</div>',
            unsafe_allow_html=True,
        )
        with st.spinner("Fetching expert knowledge…"):
            chunks, metas = rag_retrieve(
                GRADE_META[grade]["rag_query"], rag_collection, rag_embedder, k=3
            )
            insight = rag_answer(groq_client, GRADE_META[grade]["rag_query"], chunks)
        st.markdown(
            f'<div class="insight-box">{insight}</div>',
            unsafe_allow_html=True,
        )
        unique_src = list(dict.fromkeys(
            m["source"].replace(".txt","").replace("_"," ").title() for m in metas
        ))
        with st.expander("📚 View Sources", expanded=False):
            for s in unique_src:
                st.caption(f"• {s}")

    # ── Severity scale ────────────────────────────────────────────────────
    section_heading("📊", "Contamination Severity Scale")
    sc1, sc2, sc3 = st.columns(3)
    for col, (g_id, g_m) in zip([sc1, sc2, sc3], GRADE_META.items()):
        active = g_id == grade
        border = f"2px solid {g_m['color']}" if active else f"1px solid {g_m['color']}44"
        shadow = f"box-shadow:0 4px 16px {g_m['color']}44;" if active else ""
        bg     = g_m["bg"] if active else "#FAFAFA"
        with col:
            current_badge = (
                f'<div style="font-size:0.65rem;color:#fff;background:{g_m["color"]};'
                f'border-radius:10px;padding:2px 8px;margin-top:4px;display:inline-block;">CURRENT</div>'
                if active else ""
            )
            st.markdown(
                f'<div class="scale-card" style="background:{bg};border:{border};{shadow}">'
                f'<div class="scale-icon">{g_m["emoji"]}</div>'
                f'<div class="scale-grade" style="color:{g_m["color"]}">Grade {g_id}</div>'
                f'<div class="scale-name">{g_m["label"]}</div>'
                f'{current_badge}'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Batch mode ────────────────────────────────────────────────────────
    st.markdown("")
    with st.expander("📁 Batch Analysis — Analyse multiple bags at once", expanded=False):
        batch_files = st.file_uploader(
            "Upload multiple images",
            type=["jpg","jpeg","png"],
            accept_multiple_files=True,
            key="batch",
        )
        if batch_files:
            import pandas as pd
            results = []
            prog = st.progress(0, text="Processing images…")
            for i, f in enumerate(batch_files):
                pil  = Image.open(f).convert("RGB")
                arr  = preprocess(pil, model_preproc)
                pr, gr = predict(model, arr, model_mode)
                results.append({
                    "Filename":   f.name,
                    "Grade":      f"Grade {gr}",
                    "Diagnosis":  ORDINAL_NAMES[gr],
                    "Confidence": f"{pr[gr]*100:.1f}%",
                    "Risk":       GRADE_META[gr]["risk"],
                })
                prog.progress((i+1)/len(batch_files), text=f"Processing {i+1}/{len(batch_files)}…")
            prog.empty()
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)
            st.bar_chart(df["Grade"].value_counts())
            st.download_button(
                "⬇ Download Results CSV",
                df.to_csv(index=False).encode(),
                "batch_results.csv", "text/csv",
            )

# ─────────────────────────────────────────────────────────────────────────────
# 11.  Expert Chat tab
# ─────────────────────────────────────────────────────────────────────────────
EXAMPLE_QUESTIONS = [
    "What causes green mold in oyster mushroom bags?",
    "How do I treat Trichoderma contamination?",
    "Difference between single and mixed infected bags?",
    "Best temperature for oyster mushroom fruiting?",
    "How dangerous is Aspergillus in mushroom farms?",
    "What is dry bubble disease?",
    "Signs of a healthy substrate bag?",
    "Which mushrooms are most poisonous?",
]

def render_chat(rag_collection, rag_embedder, groq_client):
    if rag_collection is None:
        st.markdown(
            '<div class="card" style="text-align:center;padding:2.5rem;">'
            '<div style="font-size:3rem">📚</div>'
            '<div style="font-size:1.1rem;font-weight:700;color:#1B3A5C;margin:0.8rem 0 0.4rem">Knowledge Base Not Set Up</div>'
            '<div style="color:#6C757D;font-size:0.88rem">Run these commands to enable the chat:</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.code("python phase6_scrape.py\npython phase6_index.py", language="bash")
        return

    if groq_client is None:
        st.markdown(
            '<div class="card" style="text-align:center;padding:2.5rem;">'
            '<div style="font-size:3rem">🔑</div>'
            '<div style="font-size:1.1rem;font-weight:700;color:#1B3A5C;margin:0.8rem 0 0.4rem">Groq API Key Required</div>'
            '<div style="color:#6C757D;font-size:0.88rem">Add your key to the .env file, then restart.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.code("GROQ_API_KEY=your_key_here  # in .env file", language="bash")
        st.markdown(
            '<div style="text-align:center"><a href="https://console.groq.com" target="_blank" '
            'style="color:#1B3A5C;font-size:0.85rem">Get a free key at console.groq.com →</a></div>',
            unsafe_allow_html=True,
        )
        return

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # ── 1. Render all existing messages first ──────────────────────────────
    if not st.session_state.chat_messages:
        # Welcome card + example pills
        st.markdown(
            """<div class="card" style="background:linear-gradient(135deg,#F0F6FF,#F0FBF4);border:1px solid #DDEAFF;">
              <div style="font-size:1.5rem;margin-bottom:0.5rem">👋 Hello!</div>
              <div style="font-weight:700;font-size:1rem;color:#1B3A5C;">I'm your Mushroom Expert Assistant</div>
              <div style="color:#5D6D7E;font-size:0.88rem;margin-top:0.5rem;line-height:1.6">
                Ask me anything about mushroom diseases, cultivation, treatment, or species.<br>
                I use real scientific knowledge retrieved from my database.
              </div>
            </div>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="font-size:0.82rem;font-weight:700;color:#1B3A5C;'
            'text-transform:uppercase;letter-spacing:1px;margin:1rem 0 0.6rem">'
            'Try asking:</div>',
            unsafe_allow_html=True,
        )
        cols = st.columns(2)
        for i, q in enumerate(EXAMPLE_QUESTIONS):
            with cols[i % 2]:
                st.markdown('<div class="example-pill">', unsafe_allow_html=True)
                if st.button(q, key=f"ex_{i}", use_container_width=True):
                    # Add user message + immediately generate answer, then rerun
                    st.session_state.chat_messages.append({"role": "user", "content": q})
                    with st.spinner("Generating expert answer…"):
                        chunks, metas = rag_retrieve(q, rag_collection, rag_embedder)
                        answer = rag_answer(groq_client, q, chunks)
                    unique_src = list(dict.fromkeys(m["source"] for m in metas))
                    st.session_state.chat_messages.append({
                        "role": "assistant", "content": answer, "sources": unique_src
                    })
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Render chat history
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and "sources" in msg:
                    with st.expander("📚 Sources", expanded=False):
                        for s in msg["sources"]:
                            st.caption(f"• {s.replace('.txt','').replace('_',' ').title()}")

        # Clear button
        col_spacer, col_clear = st.columns([5, 1])
        with col_clear:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.chat_messages = []
                st.rerun()

    # ── 2. Chat input always at the bottom ────────────────────────────────
    query = st.chat_input("Ask about mushroom diseases, cultivation, species…")

    if query:
        # Add user message to state
        st.session_state.chat_messages.append({"role": "user", "content": query})
        # Generate and store assistant reply
        with st.spinner("Searching knowledge base…"):
            chunks, metas = rag_retrieve(query, rag_collection, rag_embedder)
        with st.spinner("Generating expert answer…"):
            answer = rag_answer(
                groq_client, query, chunks,
                history=st.session_state.chat_messages[:-1],
            )
        unique_src = list(dict.fromkeys(m["source"] for m in metas))
        st.session_state.chat_messages.append({
            "role": "assistant", "content": answer, "sources": unique_src,
        })
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# 12.  Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    model, model_desc, model_mode, model_preproc = load_model()
    rag_collection, rag_embedder = load_rag()
    groq_client = load_groq_client(GROQ_API_KEY) if GROQ_API_KEY else None

    rag_ready  = rag_collection is not None
    rag_chunks = rag_collection.count() if rag_ready else 0

    # Sidebar
    show_heatmap, heatmap_alpha = render_sidebar(model_desc, rag_ready, rag_chunks)

    if model is None:
        st.error("❌ No model found in `models/`. Run `python phase3c_ordinal.py` first.")
        st.stop()

    # Build Grad-CAM graph once
    grad_model, gradcam_method, _ = build_gradcam_model(model)

    # ── Hero banner ───────────────────────────────────────────────────────
    st.markdown(
        """<div class="hero-banner">
          <div>
            <div class="hero-title">🍄 Mushroom Disease AI</div>
            <div class="hero-sub">Ordinal Contamination Grading · Expert Knowledge Chat</div>
          </div>
          <div class="hero-badge">EfficientNetV2S + CORN Loss · 88.03% Accuracy</div>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab_detect, tab_chat = st.tabs(["🔬  Disease Detection", "💬  Expert Chat"])

    with tab_detect:
        render_detection(
            model, model_mode, model_preproc,
            grad_model, gradcam_method,
            rag_collection, rag_embedder, groq_client,
            show_heatmap, heatmap_alpha,
        )

    with tab_chat:
        render_chat(rag_collection, rag_embedder, groq_client)

    # ── Footer ────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="app-footer">'
        'EfficientNetV2S + CORN Ordinal Loss &nbsp;·&nbsp; '
        'ChromaDB RAG &nbsp;·&nbsp; Groq Llama-3.3-70B &nbsp;·&nbsp; TensorFlow 2.x'
        '<br>CSE 499B · North South University · 2025'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
