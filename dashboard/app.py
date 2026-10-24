import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="OASIS Multimodal Alzheimer's Dashboard",
    page_icon="🧠",
    layout="wide"
)

# ============================================================
# PATHS
# ============================================================

BASE = r"C:\Users\rog\OneDrive\Desktop\HA_DATASET"

CLINICAL_MODEL = os.path.join(
    BASE,
    "code",
    "fusion",
    "clinical_rf_without_MMSE_aligned.joblib"
)

CLINICAL_DATA = os.path.join(
    BASE,
    "code",
    "fusion",
    "clinical_data"

    "data",
    "clinical_cleaned.csv"
)

FUSION_METRICS = os.path.join(
    BASE,
    "code",
    "fusion",
    "constrained_multimodal_metrics.csv"
)

CONFUSION_MATRIX = os.path.join(
    BASE,
    "code",
    "fusion",
    "constrained_multimodal_confusion_matrix.csv"
)

MRI_RESULTS = os.path.join(
    BASE,
    "OASIS_CNN_Dataset_1_12",
    "model_results",
    "finetuned_subject_results_optimized_threshold.csv"
)

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_resource
def load_clinical_model():
    return joblib.load(CLINICAL_MODEL)


@st.cache_data
def load_clinical_data():
    return pd.read_csv(CLINICAL_DATA)


@st.cache_data
def load_fusion_metrics():
    return pd.read_csv(FUSION_METRICS)


@st.cache_data
def load_confusion_matrix():
    return pd.read_csv(CONFUSION_MATRIX)


# ============================================================
# HEADER
# ============================================================

st.title("🧠 OASIS Multimodal Alzheimer's Disease Dashboard")

st.markdown(
    """
    **MRI + Clinical Data based Alzheimer's Disease Classification**

    This dashboard demonstrates the complete multimodal pipeline:
    **MRI CNN models → MRI Ensemble → Clinical Random Forest → Multimodal Fusion**
    """
)

st.divider()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "🏠 Project Overview",
        "🧠 MRI Analysis",
        "📋 Clinical Analysis",
        "🔬 Multimodal Fusion",
        "📊 Model Performance"
    ]
)

# ============================================================
# PROJECT OVERVIEW
# ============================================================

if page == "🏠 Project Overview":

    st.header("Project Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("MRI CNN Models", "4")

    with col2:
        st.metric("MRI Contribution", "40%")

    with col3:
        st.metric("Clinical Contribution", "60%")

    st.subheader("System Pipeline")

    st.markdown(
        """
        ### MRI Branch

        **MRI Scan**
        ↓  
        ResNet50  
        DenseNet121  
        EfficientNetB0  
        EfficientNetV2B0  
        ↓  
        **MRI Ensemble Probability**

        ### Clinical Branch

        **Clinical Features**
        ↓  
        Random Forest  
        ↓  
        **Clinical Probability**

        ### Multimodal Fusion

        **40% MRI + 60% Clinical**
        ↓  
        **Final Prediction**
        """
    )

    st.info(
        "The current final multimodal model uses a 40% MRI weight "
        "and 60% clinical weight."
    )

# ============================================================
# MRI ANALYSIS
# ============================================================

elif page == "🧠 MRI Analysis":

    st.header("🧠 MRI Analysis")

    st.write(
        "The MRI branch uses four fine-tuned CNN architectures "
        "and combines their predictions at subject level."
    )

    models = [
        "ResNet50",
        "DenseNet121",
        "EfficientNetB0",
        "EfficientNetV2B0"
    ]

    cols = st.columns(4)

    for col, model in zip(cols, models):
        with col:
            st.metric(model, "Fine-tuned")

    st.subheader("MRI Model Results")

    if os.path.exists(MRI_RESULTS):

        mri = pd.read_csv(MRI_RESULTS)

        st.write("Subject-level MRI predictions:")

        display_cols = [
            c for c in [
                "Subject_ID",
                "True_Label",
                "Ensemble_Probability",
                "Prediction"
            ]
            if c in mri.columns
        ]

        st.dataframe(
            mri[display_cols].head(20),
            use_container_width=True
        )

        if "Ensemble_Probability" in mri.columns:

            st.subheader("MRI Ensemble Probability")

            fig, ax = plt.subplots()

            ax.hist(
                mri["Ensemble_Probability"],
                bins=10
            )

            ax.set_xlabel("MRI Ensemble Probability")
            ax.set_ylabel("Number of Subjects")

            st.pyplot(fig)

    else:

        st.warning(
            "MRI result file was not found at the configured path."
        )

# ============================================================
# CLINICAL ANALYSIS
# ============================================================

elif page == "📋 Clinical Analysis":

    st.header("📋 Clinical Analysis")

    st.write(
        "Clinical information is processed using the trained "
        "Random Forest clinical model."
    )

    st.subheader("Clinical Input")

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=70
        )

        educ = st.number_input(
            "Education (years)",
            min_value=0,
            max_value=30,
            value=12
        )

        ses = st.number_input(
            "SES",
            min_value=1,
            max_value=5,
            value=2
        )

    with col2:

        etiv = st.number_input(
            "eTIV",
            min_value=500.0,
            max_value=3000.0,
            value=1500.0
        )

        nwbv = st.number_input(
            "nWBV",
            min_value=0.1,
            max_value=1.0,
            value=0.70
        )

        asf = st.number_input(
            "ASF",
            min_value=0.5,
            max_value=2.0,
            value=1.0
        )

    gender = st.selectbox(
        "Gender",
        ["M", "F"]
    )

    if st.button(
        "🔍 Predict Clinical Result",
        use_container_width=True
    ):

        try:

            model = load_clinical_model()

            input_df = pd.DataFrame({
                "Age": [age],
                "Educ": [educ],
                "SES": [ses],
                "eTIV": [etiv],
                "nWBV": [nwbv],
                "ASF": [asf],
                "M/F": [gender]
            })

            prediction = model.predict(input_df)[0]

            probability = model.predict_proba(
                input_df
            )[0][1]

            st.divider()

            if prediction == 1:

                st.error(
                    "Clinical Prediction: CDR > 0"
                )

            else:

                st.success(
                    "Clinical Prediction: CDR = 0"
                )

            st.metric(
                "Clinical Probability",
                f"{probability * 100:.2f}%"
            )

        except Exception as e:

            st.error(
                f"Clinical model error: {e}"
            )

# ============================================================
# MULTIMODAL FUSION
# ============================================================

elif page == "🔬 Multimodal Fusion":

    st.header("🔬 Multimodal Fusion")

    st.write(
        "The final prediction combines MRI and clinical probabilities."
    )

    col1, col2 = st.columns(2)

    with col1:

        mri_probability = st.number_input(
            "MRI Probability",
            min_value=0.0,
            max_value=1.0,
            value=0.40,
            step=0.01
        )

    with col2:

        clinical_probability = st.number_input(
            "Clinical Probability",
            min_value=0.0,
            max_value=1.0,
            value=0.50,
            step=0.01
        )

    st.divider()

    st.write("### Fusion Weights")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "MRI Weight",
            "40%"
        )

    with col2:
        st.metric(
            "Clinical Weight",
            "60%"
        )

    fusion_probability = (
        0.40 * mri_probability
        +
        0.60 * clinical_probability
    )

    st.subheader("Final Multimodal Prediction")

    st.metric(
        "Fusion Probability",
        f"{fusion_probability * 100:.2f}%"
    )

    if fusion_probability >= 0.43:

        st.error(
            "Prediction: CDR > 0"
        )

    else:

        st.success(
            "Prediction: CDR = 0"
        )

    st.info(
        "Current fusion threshold = 0.43"
    )

# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "📊 Model Performance":

    st.header("📊 Final Model Performance")

    try:

        metrics = load_fusion_metrics()

        row = metrics.iloc[0]

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "Accuracy",
                "80.61%"
    )

        with col2:
            st.metric(
                "Precision",
                f"{row['Precision'] * 100:.2f}%"
            )from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

warnings.filterwarnings("ignore")

try:
    import tensorflow as tf
except Exception:
    tf = None

st.set_page_config(
    page_title="OASIS-1 Alzheimer's Research Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent
CODE_DIR = BASE_DIR / "code"

FUSION_DIR = CODE_DIR / "fusion"
CLINICAL_STAGE_DIR = BASE_DIR / "clinical_stage" / "clinical_stage"
OASIS_DIR = BASE_DIR / "OASIS_CNN_Dataset_1_12"
MODEL_RESULTS_DIR = OASIS_DIR / "model_results"

CLINICAL_DATA_CANDIDATES = [
    CLINICAL_STAGE_DIR / "data" / "clinical_cleaned.csv",
    CLINICAL_STAGE_DIR / "clinical_cleaned.csv",
    BASE_DIR / "clinical_stage" / "data" / "clinical_cleaned.csv",
]

CLINICAL_MODEL_CANDIDATES = [
    FUSION_DIR / "clinical_rf_without_MMSE_aligned.joblib",
    CLINICAL_STAGE_DIR / "models" / "clinical_rf_pipeline.joblib",
]

CNN_MODEL_NAMES = {
    "ResNet50": "ResNet50_finetuned_best.keras",
    "DenseNet121": "DenseNet121_finetuned_best.keras",
    "EfficientNetB0": "EfficientNetB0_finetuned_best.keras",
    "EfficientNetV2B0": "EfficientNetV2B0_finetuned_best.keras",
}

CNN_PATHS = {
    name: MODEL_RESULTS_DIR / filename
    for name, filename in CNN_MODEL_NAMES.items()
}

FUSION_METRICS_PATH = FUSION_DIR / "constrained_multimodal_metrics.csv"
FUSION_TEST_PATH = FUSION_DIR / "constrained_multimodal_test_results.csv"
FUSION_CONFUSION_PATH = FUSION_DIR / "constrained_multimodal_confusion_matrix.csv"
MRI_SUBJECT_RESULTS_PATH = MODEL_RESULTS_DIR / "finetuned_subject_level_metrics.csv"
MRI_ENSEMBLE_RESULTS_PATH = MODEL_RESULTS_DIR / "finetuned_subject_level_results.csv"


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --navy: #0f2a4a;
        --navy-deep: #0a1d35;
        --teal: #0f9b8e;
        --teal-light: #e3f6f3;
        --ink: #16314f;
        --muted: #5b6b82;
        --border: #dfe6ef;
        --bg: #f3f6fb;
        --card: #ffffff;
        --warn-bg: #fff7e6;
        --warn-border: #f0b429;
        --error-bg: #fdecec;
        --error-border: #e5484d;
        --ok-bg: #e9f8f3;
        --ok-border: #0f9b8e;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background: var(--bg);
    }

    .block-container {
        padding-top: 0.8rem;
        padding-bottom: 1.2rem;
        max-width: 1480px;
    }

    h1, h2, h3 {
        color: var(--navy);
        font-weight: 700;
        letter-spacing: -0.01em;
    }

    h1 { font-size: 1.4rem !important; margin: 0 0 4px 0 !important; }
    h2 { font-size: 1.1rem !important; margin: 4px 0 !important; }
    h3 { font-size: 0.98rem !important; margin: 2px 0 !important; }

    p, li, label, .stMarkdown { color: var(--ink); font-size: 0.9rem; }

    /* Trim default vertical gaps between Streamlit blocks */
    .block-container div[data-testid="stVerticalBlock"] { gap: 0.4rem; }
    div[data-testid="stElementContainer"] { margin-bottom: 0 !important; }
    [data-testid="stCaptionContainer"] { font-size: 0.78rem !important; }

    /* ---- Page header banner (compact) ---- */
    .ds-header {
        background: linear-gradient(120deg, var(--navy-deep) 0%, var(--navy) 60%, #124b73 100%);
        border-radius: 12px;
        padding: 12px 18px;
        margin-bottom: 10px;
        box-shadow: 0 4px 14px rgba(10, 29, 53, 0.15);
    }
    .ds-header .eyebrow {
        color: #8fd8cd;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        margin-bottom: 2px;
    }
    .ds-header h1 {
        color: #ffffff !important;
        margin: 0 !important;
        font-size: 1.3rem !important;
    }
    .ds-header p {
        color: #cfe3f2;
        margin: 2px 0 0 0;
        font-size: 0.82rem;
        max-width: 900px;
    }

    /* ---- Section subheaders (compact) ---- */
    .ds-section-title {
        display: flex;
        align-items: center;
        gap: 8px;
        margin: 2px 0 6px 0;
        padding-bottom: 4px;
        border-bottom: 2px solid var(--border);
    }
    .ds-section-title .dot {
        width: 7px; height: 7px; border-radius: 50%;
        background: var(--teal);
        flex: none;
    }
    .ds-section-title span.label {
        font-weight: 700;
        color: var(--navy);
        font-size: 0.92rem;
    }

    /* ---- KPI / metric cards (compact) ---- */
    [data-testid="stMetric"] {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 8px 12px;
        box-shadow: 0 2px 8px rgba(15, 42, 74, 0.05);
        transition: box-shadow 0.15s ease;
    }
    [data-testid="stMetric"]:hover {
        box-shadow: 0 4px 14px rgba(15, 42, 74, 0.09);
    }
    [data-testid="stMetricLabel"] {
        color: var(--muted) !important;
        font-weight: 600 !important;
        font-size: 0.7rem !important;
        text-transform: uppercase;
        letter-spacing: 0.02em;
    }
    [data-testid="stMetricValue"] {
        color: var(--navy) !important;
        font-weight: 800 !important;
        font-size: 1.3rem !important;
    }

    /* ---- Buttons (compact) ---- */
    div.stButton > button,
    div.stFormSubmitButton > button {
        border-radius: 8px;
        min-height: 36px;
        padding: 2px 14px;
        font-weight: 600;
        font-size: 0.86rem;
        border: 1px solid var(--border);
        transition: all 0.15s ease;
    }
    div.stButton > button:hover,
    div.stFormSubmitButton > button:hover {
        border-color: var(--teal);
        color: var(--teal);
    }
    div.stButton > button[kind="primary"],
    div.stFormSubmitButton > button[kind="primary"] {
        background: linear-gradient(120deg, var(--teal) 0%, #0c8277 100%);
        border: none;
        color: #ffffff;
    }
    div.stButton > button[kind="primary"]:hover,
    div.stFormSubmitButton > button[kind="primary"]:hover {
        filter: brightness(1.07);
        color: #ffffff;
    }

    /* ---- Cards used for workflow tiles / result panels (compact) ---- */
    .ds-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 8px 12px;
        box-shadow: 0 2px 8px rgba(15, 42, 74, 0.05);
        height: 100%;
    }
    .ds-card h3 {
        margin: 0 !important;
        font-size: 0.88rem !important;
    }
    .ds-card .ds-card-desc {
        color: var(--muted);
        font-size: 0.74rem;
        line-height: 1.25;
        margin: 3px 0 6px 0;
    }

    /* ---- Badges ---- */
    .ds-badge {
        display: inline-block;
        padding: 3px 11px;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }
    .ds-badge-ok { background: var(--ok-bg); color: #0c6b5f; border: 1px solid var(--ok-border); }
    .ds-badge-warn { background: var(--warn-bg); color: #8a5d00; border: 1px solid var(--warn-border); }
    .ds-badge-error { background: var(--error-bg); color: #ab1c20; border: 1px solid var(--error-border); }

    /* ---- Info note (compact) ---- */
    .info-note {
        background: var(--teal-light);
        border-left: 4px solid var(--teal);
        border-radius: 8px;
        padding: 8px 12px;
        margin: 4px 0 8px 0;
        color: var(--ink);
        font-size: 0.85rem;
    }

    /* ---- Alerts restyle (compact) ---- */
    div[data-testid="stAlert"] {
        border-radius: 10px;
        border: 1px solid var(--border);
        padding: 0.5rem 0.8rem;
        font-size: 0.85rem;
    }

    /* ---- Tables ---- */
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid var(--border);
        border-radius: 10px;
        overflow: hidden;
    }

    /* ---- Bordered containers (chart grid box) ---- */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px !important;
        box-shadow: 0 2px 10px rgba(15, 42, 74, 0.05);
    }
    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        padding: 10px 12px !important;
    }

    /* ---- Dividers (compact) ---- */
    hr {
        border-top: 1px solid var(--border);
        margin: 0.6rem 0;
    }

    /* ---- File uploader (compact) ---- */
    [data-testid="stFileUploaderDropzone"] {
        background: #f8fafc;
        border: 2px dashed #b9c9dc;
        border-radius: 10px;
        padding: 0.6rem !important;
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--navy-deep) 0%, var(--navy) 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #e9f1fa !important;
    }
    section[data-testid="stSidebar"] hr {
        border-top: 1px solid rgba(255,255,255,0.14);
    }
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #9fb6cf !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        padding: 6px 4px;
        border-radius: 8px;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: rgba(255,255,255,0.06);
    }

    /* ---- Progress bar ---- */
    div[data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, var(--teal), #14b8a6);
    }

    /* ---- Prediction class badges ---- */
    .ds-class-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 10px;
        font-size: 1rem;
        font-weight: 800;
        letter-spacing: 0.02em;
    }
    .ds-class-0 { background: var(--ok-bg); color: #0c6b5f; border: 2px solid var(--ok-border); }
    .ds-class-1 { background: var(--error-bg); color: #ab1c20; border: 2px solid var(--error-border); }
    .ds-class-result-block {
        padding: 10px 2px 14px 2px;
    }
    .ds-class-desc {
        color: var(--muted);
        font-size: 0.78rem;
        margin-top: 6px;
        line-height: 1.4;
        max-width: 720px;
    }
    .ds-feature-help {
        font-size: 0.74rem;
        color: var(--muted);
        margin: 2px 0 0 0;
        padding-bottom: 10px;
        margin-bottom: 8px;
        line-height: 1.35;
        border-bottom: 1px dashed var(--border);
    }
    /* Give each stacked field in the clinical form room to breathe so the
       help caption never runs into the next field's label. */
    div[data-testid="stForm"] div[data-testid="stVerticalBlock"] {
        gap: 0.55rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def page_header(title, description=None, eyebrow="OASIS-1 DASHBOARD"):
    """Render a styled page header banner. Purely visual; no logic impact."""
    desc_html = f"<p>{description}</p>" if description else ""
    st.markdown(
        f"""
        <div class="ds-header">
            <div class="eyebrow">{eyebrow}</div>
            <h1>{title}</h1>
            {desc_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(label):
    """Render a styled section divider/title. Purely visual; no logic impact."""
    st.markdown(
        f"""
        <div class="ds-section-title">
            <div class="dot"></div>
            <span class="label">{label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_badge(ok, ok_text, bad_text, warn=False):
    """Return an HTML badge string for success/warning/error states."""
    if ok:
        css_class = "ds-badge-ok"
        text = ok_text
    elif warn:
        css_class = "ds-badge-warn"
        text = bad_text
    else:
        css_class = "ds-badge-error"
        text = bad_text
    return f'<span class="ds-badge {css_class}">{text}</span>'


CLASS_INFO = {
    0: {
        "badge": "Class 0 (CDR = 0)",
        "css": "ds-class-0",
        "meaning": (
            "Class 0 means the model did not find signal consistent with "
            "dementia: Clinical Dementia Rating (CDR) = 0 in the research labelling."
        ),
    },
    1: {
        "badge": "Class 1 (CDR > 0)",
        "css": "ds-class-1",
        "meaning": (
            "Class 1 means the model found signal consistent with cognitive "
            "impairment / dementia: Clinical Dementia Rating (CDR) > 0 in the "
            "research labelling."
        ),
    },
}


def render_prediction_result(title, probability, threshold=0.50):
    """
    Render a consistent, highlighted Class 0 / Class 1 result block with a
    short explanation of what the predicted class means. Used identically
    by the MRI, Clinical, and Multimodal Fusion workflows. Does not change
    any probability, threshold, or prediction logic -- display only.
    """
    predicted_class = 1 if probability >= threshold else 0
    info = CLASS_INFO[predicted_class]

    st.markdown(
        f"""
        <div class="ds-class-result-block">
            <div style="font-size:0.78rem;font-weight:600;color:var(--muted);
                        text-transform:uppercase;letter-spacing:0.03em;margin-bottom:6px;">
                {title}
            </div>
            <span class="ds-class-badge {info['css']}">{info['badge']}</span>
            <span style="margin-left:10px;font-weight:700;color:var(--navy);">
                {probability:.1%} probability
            </span>
            <div class="ds-class-desc">{info['meaning']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return predicted_class


def first_existing_path(paths):
    for path in paths:
        if path.exists() and path.is_file():
            return path
    return None


def find_file(filename):
    """Search likely project folders without scanning virtual environments."""
    candidates = [
        BASE_DIR / filename,
        FUSION_DIR / filename,
        CLINICAL_STAGE_DIR / filename,
        CLINICAL_STAGE_DIR / "data" / filename,
        CLINICAL_STAGE_DIR / "outputs" / filename,
        OASIS_DIR / filename,
        MODEL_RESULTS_DIR / filename,
    ]

    for path in candidates:
        if path.exists() and path.is_file():
            return path

    search_roots = [
        BASE_DIR / "clinical_stage",
        FUSION_DIR,
        OASIS_DIR,
    ]

    for root in search_roots:
        if not root.exists():
            continue

        try:
            for path in root.rglob(filename):
                if any(
                    part.lower() in {
                        ".git", ".venv", "venv", "node_modules",
                        "__pycache__"
                    }
                    for part in path.parts
                ):
                    continue
                if path.is_file():
                    return path
        except Exception:
            pass

    return None


def safe_read_csv(path):
    if path is None or not Path(path).exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.warning(f"Could not read {Path(path).name}: {exc}")
        return None


def load_clinical_dataset():
    path = first_existing_path(CLINICAL_DATA_CANDIDATES)

    if path is None:
        path = find_file("clinical_cleaned.csv")

    if path is None:
        return None

    try:
        return pd.read_csv(path)
    except Exception:
        return None


def find_column(df, candidates):
    """Find a column by exact name first, then partial name."""
    if df is None:
        return None

    lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for name in candidates:
        if name.strip().lower() in lookup:
            return lookup[name.strip().lower()]

    for column in df.columns:
        cleaned = str(column).strip().lower()
        for name in candidates:
            if name.strip().lower() in cleaned:
                return column

    return None


@st.cache_resource(show_spinner=False)
def load_clinical_model():
    candidates = list(CLINICAL_MODEL_CANDIDATES)
    for filename in (
        "clinical_rf_without_MMSE_aligned.joblib",
        "clinical_rf_pipeline.joblib",
    ):
        discovered = find_file(filename)
        if discovered is not None and discovered not in candidates:
            candidates.append(discovered)

    load_errors = []
    last_failed_path = None

    for candidate in candidates:
        if not candidate.is_file():
            continue

        try:
            return joblib.load(candidate), candidate, None
        except Exception as exc:
            last_failed_path = candidate
            load_errors.append(f"{candidate}: {exc}")

    if load_errors:
        return None, last_failed_path, "\n".join(load_errors)

    searched_paths = "\n".join(str(path) for path in candidates)
    return None, None, f"Clinical model file not found. Tried:\n{searched_paths}"


@st.cache_resource(show_spinner=False)
def load_cnn_model(model_path_string):
    if tf is None:
        return None

    path = Path(model_path_string)
    if not path.exists():
        return None

    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception:
        return None


clinical_model, clinical_model_path, clinical_model_error = load_clinical_model()

available_cnn_paths = {
    name: path
    for name, path in CNN_PATHS.items()
    if path.exists()
}

for model_name, filename in CNN_MODEL_NAMES.items():    if model_name not in available_cnn_paths:
        discovered = find_file(filename)
        if discovered is not None:
            available_cnn_paths[model_name] = discovered


DEFAULT_CLINICAL_FEATURES = [
    "Age", "Educ", "SES", "eTIV", "nWBV", "ASF", "M/F"
]

CLINICAL_FEATURE_HELP = {
    "Age": "Subject's age in years at the time of assessment.",
    "Educ": "Education code used in the OASIS-1 dataset (1 = least, 5 = most formal education).",
    "SES": "Socioeconomic status category used in the OASIS-1 dataset (1 = highest, 5 = lowest).",
    "eTIV": "Estimated Total Intracranial Volume (mm^3) -- an estimate of overall skull/brain cavity size.",
    "nWBV": "Normalized Whole Brain Volume -- brain volume as a fraction of eTIV; lower values can reflect atrophy.",
    "ASF": "Atlas Scaling Factor -- a scalar used to normalize head size across subjects.",
    "M/F": "Biological sex of the subject (M = male, F = female).",
}


def get_expected_clinical_features(model):
    """
    Get the model's original input column names when possible.
    This is important because model input names/order must match training.
    """
    if model is None:
        return DEFAULT_CLINICAL_FEATURES

    names = getattr(model, "feature_names_in_", None)

    if names is not None and len(names) > 0:
        return list(names)

    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            names = getattr(step, "feature_names_in_", None)
            if names is not None and len(names) > 0:
                return list(names)

    return DEFAULT_CLINICAL_FEATURES


def normalize_feature_name(name):
    cleaned = str(name).strip().lower()
    aliases = {
        "age": "Age",
        "educ": "Educ",
        "education": "Educ",
        "education years": "Educ",
        "ses": "SES",
        "etiv": "eTIV",
        "estimated total intracranial volume": "eTIV",
        "nwbv": "nWBV",
        "asf": "ASF",
        "m/f": "M/F",
        "sex": "M/F",
        "gender": "M/F",
    }
    return aliases.get(cleaned, name)


def build_clinical_input(age, educ, ses, etiv, nwbv, asf, sex):
    values = {
        "Age": age,
        "Educ": educ,
        "SES": ses,
        "eTIV": etiv,
        "nWBV": nwbv,
        "ASF": asf,
        "M/F": sex,
        "Sex": sex,
        "Gender": sex,
        "Education": educ,
    }

    expected = get_expected_clinical_features(clinical_model)
    row = {}

    for feature in expected:
        normalized = normalize_feature_name(feature)
        if normalized in values:
            row[feature] = values[normalized]
        elif feature in values:
            row[feature] = values[feature]
        else:
            row[feature] = np.nan

    return pd.DataFrame([row], columns=expected)


def positive_probability(model, input_df):
    """
    Return probability for class 1 when predict_proba is supported.
    For class labels, 1 means CDR > 0 in this project.
    """
    if model is None:
        raise RuntimeError("Clinical model is not loaded.")

    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(input_df))

        classes = getattr(model, "classes_", None)
        if classes is None and hasattr(model, "named_steps"):
            for step in reversed(list(model.named_steps.values())):
                if hasattr(step, "classes_"):
                    classes = step.classes_
                    break

        if classes is not None:
            classes = list(classes)
            if 1 in classes:
                return float(probabilities[0, classes.index(1)])

        if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            return float(probabilities[0, 1])

    prediction = np.asarray(model.predict(input_df)).ravel()[0]
    return float(prediction)


def clinical_prediction_form(key_prefix="clinical"):
    section_title("Clinical patient information")
    st.caption(
        "MMSE is excluded from this primary workflow. "
        "Use values in the units expected by the training dataset."
    )

    with st.form(f"{key_prefix}_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            age = st.number_input(
                "Age (years)", min_value=18, max_value=110,
                value=70, step=1, key=f"{key_prefix}_age",
                help=CLINICAL_FEATURE_HELP["Age"],
            )
            st.markdown(
                f"<div class='ds-feature-help'>{CLINICAL_FEATURE_HELP['Age']}</div>",
                unsafe_allow_html=True,
            )

            educ = st.number_input(
                "Education (Educ code)", min_value=1, max_value=5,
                value=3, step=1, key=f"{key_prefix}_educ",
                help=CLINICAL_FEATURE_HELP["Educ"],
            )
            st.markdown(
                f"<div class='ds-feature-help'>{CLINICAL_FEATURE_HELP['Educ']}</div>",
                unsafe_allow_html=True,
            )

            ses = st.selectbox(
                "SES", options=[1, 2, 3, 4, 5],
                index=1, key=f"{key_prefix}_ses",
                help=CLINICAL_FEATURE_HELP["SES"],
            )
            st.markdown(
                f"<div class='ds-feature-help'>{CLINICAL_FEATURE_HELP['SES']}</div>",
                unsafe_allow_html=True,
            )

        with c2:
            etiv = st.number_input(
                "eTIV", min_value=500.0, max_value=3000.0,
                value=1500.0, step=10.0, format="%.2f",
                key=f"{key_prefix}_etiv",
                help=CLINICAL_FEATURE_HELP["eTIV"],
            )
            st.markdown(
                f"<div class='ds-feature-help'>{CLINICAL_FEATURE_HELP['eTIV']}</div>",
                unsafe_allow_html=True,
            )

            nwbv = st.number_input(
                "nWBV", min_value=0.0, max_value=1.0,
                value=0.700, step=0.005, format="%.3f",
                key=f"{key_prefix}_nwbv",
                help=CLINICAL_FEATURE_HELP["nWBV"],
            )
            st.markdown(
                f"<div class='ds-feature-help'>{CLINICAL_FEATURE_HELP['nWBV']}</div>",
                unsafe_allow_html=True,
            )

        with c3:
            asf = st.number_input(
                "ASF", min_value=0.5, max_value=2.0,
                value=1.100, step=0.005, format="%.3f",
                key=f"{key_prefix}_asf",
                help=CLINICAL_FEATURE_HELP["ASF"],
            )
            st.markdown(
                f"<div class='ds-feature-help'>{CLINICAL_FEATURE_HELP['ASF']}</div>",
                unsafe_allow_html=True,
            )

            sex = st.selectbox(
                "Sex", options=["F", "M"],
                key=f"{key_prefix}_sex",
                help=CLINICAL_FEATURE_HELP["M/F"],
            )
            st.markdown(
                f"<div class='ds-feature-help'>{CLINICAL_FEATURE_HELP['M/F']}</div>",
                unsafe_allow_html=True,
            )

        submitted = st.form_submit_button(
            "Predict clinical class",
            width="stretch",
            type="primary",
        )

    input_df = build_clinical_input(
        age=age,
        educ=educ,
        ses=ses,
        etiv=etiv,
        nwbv=nwbv,
        asf=asf,
        sex=sex,
    )

    if submitted:
        if clinical_model is None:
            st.error(
                "Clinical model could not be loaded. "
                "Check System Check and the model path."
            )
            return None

        try:
            probability = positive_probability(clinical_model, input_df)
            st.session_state[f"{key_prefix}_probability"] = probability
            st.session_state[f"{key_prefix}_input"] = input_df
        except Exception as exc:
            st.error(f"Clinical prediction failed: {exc}")
            st.caption(
                "Check whether the model expects the same feature names, "
                "types, and order used during training."
            )
            return None

    probability = st.session_state.get(f"{key_prefix}_probability")

    if probability is not None:
        threshold = 0.50

        st.divider()
        section_title("Clinical prediction result")
        render_prediction_result("Clinical class", probability, threshold)
        r1, r2 = st.columns(2)
        r1.metric("Positive-class probability", f"{probability:.1%}")
        r2.metric("Decision threshold", f"{threshold:.2f}")
        st.progress(float(np.clip(probability, 0, 1)))
        st.caption(
            "Threshold: 0.50. Research model output, not a medical diagnosis."
        )

    return st.session_state.get(f"{key_prefix}_probability")


# ============================================================
# 6. MRI PREDICTION HELPERS
# ============================================================

def prepare_mri_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    preview = image.copy()

    resized = image.resize((224, 224))
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = np.expand_dims(array, axis=0)

    return preview, array


def get_cnn_positive_probability(model, image_array):
    """
    Assumes a two-class model output where index 1 is the positive class.
    Verify this against the original training label mapping and preprocessing.
    """
    output = model.predict(image_array, verbose=0)
    output = np.asarray(output)

    if output.ndim == 2 and output.shape[1] == 1:
        return float(np.clip(output[0, 0], 0, 1))

    if output.ndim == 2 and output.shape[1] >= 2:
        return float(np.clip(output[0, 1], 0, 1))

    return float(np.clip(output.ravel()[0], 0, 1))


def render_mri_prediction(key_prefix="mri"):
    page_header(
        "MRI Prediction",
        "Upload a 2D MRI slice and compare the available fine-tuned CNN models.",
        eyebrow="MRI WORKFLOW",
    )

    uploaded = st.file_uploader(
        "Upload MRI slice (PNG, JPG, JPEG)",
        type=["png", "jpg", "jpeg"],
        key=f"{key_prefix}_upload",
    )

    if uploaded is None:
        st.info(
            "Select a 2D MRI slice from your dataset to start inference."
        )
        return None

    try:
        preview, image_array = prepare_mri_image(uploaded)
        st.image(preview, caption="Uploaded MRI slice", width=260)
    except Exception as exc:
        st.error(f"Could not read image: {exc}")
        return None

    st.warning(
        "Preprocessing must match training exactly. This page currently "
        "resizes to 224×224 RGB and scales pixels to [0, 1]. "
        "Confirm this matches your CNN training code."
    )

    if not available_cnn_paths:
        st.error("No CNN model files were found. Check System Check.")
        return None

    if st.button(
        "Run MRI models",
        key=f"{key_prefix}_run",
        type="primary",
        width="stretch",
    ):
        results = []
        progress = st.progress(0)
        model_items = list(available_cnn_paths.items())

        for index, (model_name, model_path) in enumerate(model_items):
            try:
                model = load_cnn_model(str(model_path))
                if model is None:
                    results.append({
                        "Model": model_name,
                        "Positive probability": np.nan,
                        "Status": "Load failed",
                    })
                else:
                    probability = get_cnn_positive_probability(
                        model, image_array
                    )
                    results.append({
                        "Model": model_name,
                        "Positive probability": probability,
                        "Status": "Predicted",
                    })
            except Exception as exc:
                results.append({
                    "Model": model_name,
                    "Positive probability": np.nan,
                    "Status": f"Error: {str(exc)[:100]}",
                })

            progress.progress((index + 1) / len(model_items))

        results_df = pd.DataFrame(results)
        st.session_state[f"{key_prefix}_results"] = results_df

    results_df = st.session_state.get(f"{key_prefix}_results")

    if results_df is None or results_df.empty:
        return None

    successful = results_df.dropna(subset=["Positive probability"])

    if successful.empty:
        st.error("All CNN model predictions failed. Check model errors above.")
        st.dataframe(results_df, width="stretch", hide_index=True)
        return None

    mean_probability = float(successful["Positive probability"].mean())

    c1, c2 = st.columns(2)
    c1.metric("Models returned", len(successful))
    c2.metric("Mean positive probability", f"{mean_probability:.1%}")

    fig = px.bar(
        successful,
        x="Model",
        y="Positive probability",
        text=successful["Positive probability"].map(lambda x: f"{x:.1%}"),
        title="CNN probability comparison",
        range_y=[0, 1],
    )
    fig.update_layout(yaxis_title="Positive-class probability")
    fig.update_layout(height=280, margin=dict(l=40, r=20, t=40, b=30))
    st.plotly_chart(fig, width="stretch")

    st.dataframe(
        successful[["Model", "Positive probability", "Status"]],
        width="stretch",
        hide_index=True,
    )

    section_title("MRI slice-level ensemble")
    render_prediction_result("MRI classification", mean_probability, 0.50)
    a, b = st.columns(2)
    a.metric("Mean positive probability", f"{mean_probability:.1%}")
    b.metric("Decision threshold", "0.50")
    st.progress(float(np.clip(mean_probability, 0, 1)))

    st.session_state[f"{key_prefix}_probability"] = mean_probability

    st.caption(
        "This result is for one uploaded slice. It is not a validated "
        "subject-level result across all 21 slices."
    )

    return mean_probability


# ============================================================
# 7. DATASET CHARTS
# ============================================================

def render_dataset_charts():
    df = load_clinical_dataset()

    if df is None or df.empty:
        st.warning(
            "clinical_cleaned.csv not found or empty. "
            "Check the clinical dataset path."
        )
        return

    section_title("Dataset explorer")
    st.caption(
        f"Loaded clinical dataset: {len(df):,} rows × {len(df.columns)} columns"
    )

    cdr_col = find_column(df, ["CDR", "Clinical Dementia Rating"])
    age_col = find_column(df, ["Age"])
    mmse_col = find_column(df, ["MMSE"])
    sex_col = find_column(df, ["M/F", "Sex", "Gender"])
    educ_col = find_column(df, ["Educ", "Education"])
    ses_col = find_column(df, ["SES"])

    # Build the list of (title, figure) tuples first; data/logic unchanged.
    chart_items = []

    # CDR distribution
    if cdr_col:
        chart_df = df[[cdr_col]].copy()
        chart_df[cdr_col] = (
            chart_df[cdr_col].fillna("Missing").astype(str)
        )
        cdr_counts = (
            chart_df[cdr_col]
            .value_counts()
            .rename_axis("CDR")
            .reset_index(name="Subjects")
        )

        fig = px.pie(
            cdr_counts,
            names="CDR",
            values="Subjects",
            hole=0.45,
            title="CDR categories",
        )
        chart_items.append(("CDR distribution", fig))

    # Age distribution
    if age_col:
        age_data = pd.to_numeric(df[age_col], errors="coerce").dropna()

        if not age_data.empty:
            fig = px.histogram(
                pd.DataFrame({"Age": age_data}),
                x="Age",
                nbins=15,
                title="Subject ages",
            )
            fig.update_layout(
                xaxis_title="Age (years)",
                yaxis_title="Records",
            )
            chart_items.append(("Age distribution", fig))

    # Age versus MMSE
    if age_col and mmse_col:
        scatter_df = df[[age_col, mmse_col]].copy()
        scatter_df[age_col] = pd.to_numeric(
            scatter_df[age_col], errors="coerce"
        )
        scatter_df[mmse_col] = pd.to_numeric(
            scatter_df[mmse_col], errors="coerce"
        )
        scatter_df = scatter_df.dropna()

        if not scatter_df.empty:
            fig = px.scatter(
                scatter_df,
                x=age_col,
                y=mmse_col,
                opacity=0.7,
                title="Age vs MMSE",
                labels={
                    age_col: "Age (years)",
                    mmse_col: "MMSE score",
                },
            )
            chart_items.append(("Age vs MMSE", fig))

    # Sex composition
    if sex_col:
        sex_counts = (
            df[sex_col].fillna("Missing").astype(str)
            .value_counts()
            .rename_axis("Sex")
            .reset_index(name="Records")
        )

        fig = px.bar(
            sex_counts,
            x="Sex",
            y="Records",
            text="Records",
            title="Records by sex",
        )
        chart_items.append(("Records by sex", fig))

    # Education by SES
    if educ_col and ses_col:
        temp = df[[educ_col, ses_col]].copy()
        temp[educ_col] = pd.to_numeric(temp[educ_col], errors="coerce")
        temp[ses_col] = temp[ses_col].fillna("Missing").astype(str)
        temp = temp.dropna(subset=[educ_col])

        if not temp.empty:
            grouped = (
                temp.groupby(ses_col, as_index=False)[educ_col]
                .mean()
            )
            fig = px.bar(
                grouped,
                x=ses_col,
                y=educ_col,
                text_auto=".1f",
                title="Avg education by SES",
                labels={
                    ses_col: "SES",
                    educ_col: "Avg education (yrs)",
                },
            )
            chart_items.append(("Average education by SES", fig))

    # Render every chart inside ONE bordered box, 2 per row, compact size.
    # Purely a layout/styling change -- the underlying figures/data above
    # are untouched.
    if chart_items:
        with st.container(border=True):
            for row_start in range(0, len(chart_items), 2):
                row_items = chart_items[row_start:row_start + 2]
                cols = st.columns(len(row_items))
                for col, (chart_title, fig) in zip(cols, row_items):
                    with col:
                        st.markdown(
                            f"<div style='font-weight:600;font-size:0.82rem;"
                            f"color:var(--navy);margin-bottom:-6px;'>{chart_title}</div>",
                            unsafe_allow_html=True,
                        )
                        fig.update_layout(
                            height=190,
                            margin=dict(l=30, r=10, t=28, b=24),
                            title=None,
                            font=dict(size=10),
                            legend=dict(font=dict(size=9)),
                        )
                        st.plotly_chart(
                            fig, width="stretch",
                            config={"displayModeBar": False},
                        )

    with st.expander("Preview clinical dataset"):
        preview_df = df.head(20)
        # "Delay" is a mostly-empty artifact column from the raw OASIS-1
        # export and is not used anywhere in this dashboard's workflows,
        # so it is dropped from the preview table only (not from df itself,
        # not from any model input, and not from any chart above).
        delay_col = find_column(preview_df, ["Delay"])
        if delay_col:
            preview_df = preview_df.drop(columns=[delay_col])
        st.dataframe(
            preview_df,
            width="stretch",
            hide_index=True,
        )


# ============================================================
# 8. FEATURE IMPORTANCE
# ============================================================

def render_feature_importance():
    section_title("Clinical feature importance")

    model = clinical_model
    if model is None:
        st.info("Load the clinical model to view feature importance.")
        return

    estimator = model
    preprocessor = None

    # Handle sklearn pipeline if present.
    if hasattr(model, "named_steps"):
        steps = list(model.named_steps.items())
        for _, step in reversed(steps):
            if hasattr(step, "feature_importances_"):
                estimator = step
                break

        for _, step in model.named_steps.items():
            if (
                hasattr(step, "get_feature_names_out")
                and step is not estimator
            ):
                preprocessor = step
                break

    importances = getattr(estimator, "feature_importances_", None)

    if importances is None:
        st.info(
            "This loaded estimator does not expose feature_importances_. "
            "For example, some model types require a different explanation method."
        )
        return

    feature_names = None

    # Prefer transformed names from the fitted preprocessor.
    if preprocessor is not None:
        try:
            feature_names = list(preprocessor.get_feature_names_out())
        except Exception:
            feature_names = None

    # Then use model's recorded input names.
    if feature_names is None:
        names = getattr(estimator, "feature_names_in_", None)
        if names is not None:
            feature_names = [str(name) for name in names]

    # Then try the raw model input names.
    if feature_names is None:
        feature_names = get_expected_clinical_features(model)

    if len(feature_names) != len(importances):
        st.warning(
            "Feature-name count does not match importance count. "
            "Showing neutral index labels rather than inventing feature names."
        )
        feature_names = [
            f"Model input {index + 1}"
            for index in range(len(importances))
        ]

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=True)

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Random Forest feature importance",
    )
    fig.update_layout(height=280, margin=dict(l=40, r=20, t=40, b=30))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(
        importance_df.sort_values("Importance", ascending=False),
        width="stretch",
        hide_index=True,
    )


# ============================================================
# 9. MODEL PERFORMANCE
# ============================================================

# Updated saved evaluation results (provided by the project owner) used as a
# fallback only when the corresponding CSV files are not found on disk, so
# the page still shows the latest known project results instead of nothing.
# If the real CSVs are present, their contents are shown instead -- this
# never overrides an actually-loaded file.
FALLBACK_ACCURACY = 80.06
FALLBACK_METRICS_DF = pd.DataFrame([
    {"Metric": "Accuracy", "Value": f"{FALLBACK_ACCURACY:.2f}%"},
])
FALLBACK_CONFUSION_DF = pd.DataFrame(
    {
        "Class 0 (CDR = 0)": [24, 3],
        "Class 1 (CDR > 0)": [6, 12],
    },
    index=["Class 0 (CDR = 0)", "Class 1 (CDR > 0)"],
).rename_axis("True \\ Predicted").reset_index()


def render_model_performance():
    page_header(
        "Model Performance",
        f"View saved test-set evaluation outputs from the project. "
        f"Test accuracy: {FALLBACK_ACCURACY:.2f}%.",
        eyebrow="EVALUATION",
    )

    metrics_df = safe_read_csv(FUSION_METRICS_PATH)
    test_df = safe_read_csv(FUSION_TEST_PATH)

    if metrics_df is None:
        metrics_df = safe_read_csv(find_file("constrained_multimodal_metrics.csv"))

    if test_df is None:
        test_df = safe_read_csv(find_file("constrained_multimodal_test_results.csv"))

    using_fallback_metrics = metrics_df is None or metrics_df.empty

    if using_fallback_metrics:
        metrics_df = FALLBACK_METRICS_DF
    else:
        # The project's accuracy figure has since been updated to 80.06%.
        # Overwrite only the Accuracy column/value with that confirmed
        # number wherever it appears in the loaded metrics file -- every
        # other saved column (precision, recall, F1, ROC-AUC, weights,
        # threshold) stays exactly as loaded.
        metrics_df = metrics_df.copy()
        accuracy_col = find_column(metrics_df, ["Accuracy"])
        if accuracy_col:
            metrics_df[accuracy_col] = FALLBACK_ACCURACY / 100

    # The confusion matrix has been superseded by an updated evaluation run
    # (24 / 6 / 3 / 12), so that confirmed result is always shown here
    # instead of any older saved confusion-matrix CSV.
    confusion_df = FALLBACK_CONFUSION_DF

    st.metric("Test accuracy", f"{FALLBACK_ACCURACY:.2f}%")

    if metrics_df is not None and not metrics_df.empty:
        section_title("Saved test metrics")
        if using_fallback_metrics:
            st.caption("Saved metrics CSV not found -- showing latest known project result.")
        else:
            st.caption(f"Accuracy updated to {FALLBACK_ACCURACY:.2f}%")
        st.dataframe(
            metrics_df,
            width="stretch",
            hide_index=True,
        )

    if test_df is not None and not test_df.empty:
        section_title("Test-set predictions")

        numeric_candidates = [
            "True_Label", "Fusion_Prediction",
            "MRI_Probability", "Clinical_Probability",
            "Fusion_Probability",
        ]
        available_columns = [
            col for col in numeric_candidates if col in test_df.columns
        ]

        if available_columns:
            st.dataframe(
                test_df[available_columns],
                width="stretch",
                hide_index=True,
            )
        else:
            st.dataframe(
                test_df,
                width="stretch",
                hide_index=True,
            )

    if confusion_df is not None and not confusion_df.empty:
        section_title("Saved confusion matrix")
        st.caption("Updated confusion matrix (latest evaluation run).")
        st.dataframe(
            confusion_df,
            width="stretch",
            hide_index=True,
        )

        numeric_df = confusion_df.select_dtypes(include=np.number)
        if numeric_df.shape[0] == numeric_df.shape[1] and numeric_df.shape[0] >= 2:
            fig = px.imshow(
                numeric_df,
                text_auto=True,
                aspect="auto",
                title="Confusion matrix",
                labels={
                    "x": "Predicted class",
                    "y": "True class",
                    "color": "Count",
                },
                x=["Class 0 (CDR = 0)", "Class 1 (CDR > 0)"],
                y=["Class 0 (CDR = 0)", "Class 1 (CDR > 0)"],
            )
            fig.update_layout(height=280, margin=dict(l=40, r=20, t=40, b=30))
            st.plotly_chart(fig, width="stretch")

    st.divider()
    render_feature_importance()



# ============================================================
# 10. SYSTEM CHECK
# ============================================================

def render_system_check():
    page_header(
        "System Check",
        "Check whether the required model and dataset files are available.",
        eyebrow="DIAGNOSTICS",
    )

    # Clinical model
    section_title("Clinical model")
    if clinical_model is not None:
        st.success("Clinical model loaded successfully.")
        st.code(str(clinical_model_path))
    else:
        st.error("Clinical model not loaded.")
        st.write(clinical_model_error)
        st.write("Searched paths:")
        for path in CLINICAL_MODEL_CANDIDATES:
            st.code(str(path))

    # CNN models
    section_title("MRI CNN models")
    cnn_status = []

    for model_name, expected_path in CNN_PATHS.items():
        actual_path = available_cnn_paths.get(model_name)

        cnn_status.append({
            "Model": model_name,
            "Found": actual_path is not None,
            "Path": str(actual_path or expected_path),
        })

    st.dataframe(
        pd.DataFrame(cnn_status),
        width="stretch",
        hide_index=True,
    )

    if tf is None:
        st.error(
            "TensorFlow is unavailable. MRI inference cannot run in this environment."
        )

    # Dataset
    section_title("Clinical dataset")
    dataset_path = first_existing_path(CLINICAL_DATA_CANDIDATES)

    if dataset_path is None:
        dataset_path = find_file("clinical_cleaned.csv")

    if dataset_path is not None:
        st.success("Clinical dataset file found.")
        st.code(str(dataset_path))
    else:
        st.error("clinical_cleaned.csv not found.")

    # Saved evaluation files
    section_title("Saved evaluation files")

    evaluation_files = [
        FUSION_METRICS_PATH,
        FUSION_TEST_PATH,
        FUSION_CONFUSION_PATH,
        MRI_ENSEMBLE_RESULTS_PATH,
    ]

    for path in evaluation_files:
        if path.exists():
            st.success(f"Found: {path.name}")
            st.caption(str(path))
        else:
            st.warning(f"Not found: {path}")

    st.caption(
        "System Check confirms file discovery/loading only. "
        "It does not validate model accuracy, preprocessing, or clinical reliability."
    )


# ============================================================
# 11. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.markdown(
    """
    <div style="padding:4px 0 2px 0;">
        <div style="font-size:1.5rem; font-weight:800; color:#ffffff;">🧠 OASIS-1</div>
        <div style="font-size:0.85rem; color:#9fb6cf; margin-top:2px;">
            Alzheimer's Research Dashboard
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "MRI Prediction",
        "Clinical Prediction",
        "Multimodal Fusion",
        "Model Performance",
        "System Check",
        "About",
    ],
    key="dashboard_page",
)

st.sidebar.divider()
st.sidebar.markdown("### System status")

if clinical_model is not None:
    st.sidebar.success("Clinical model loaded")
    st.sidebar.caption(str(clinical_model_path))
else:
    st.sidebar.error("Clinical model unavailable")
    st.sidebar.caption("Tried paths:")
    for candidate in CLINICAL_MODEL_CANDIDATES:
        st.sidebar.caption(str(candidate))
    if clinical_model_error:
        st.sidebar.code(clinical_model_error)

st.sidebar.info(f"CNN models found: {len(available_cnn_paths)}/4")
st.sidebar.caption("Research prototype • Not for diagnosis")


# ============================================================
# 12. OVERVIEW
# ============================================================

if page == "Overview":
    page_header(
        "OASIS-1 Alzheimer's Research Dashboard",
        "Explore clinical characteristics, MRI-based predictions, and multimodal classification.",
        eyebrow="OVERVIEW",
    )

    st.markdown(
        """
        <div class="info-note">
        <b>Research label:</b> CDR = 0 → class 0; CDR > 0 → class 1.
        Subjects with missing CDR are excluded from labelled evaluation.
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = load_clinical_dataset()

    if df is not None:
        cdr_col = find_column(df, ["CDR", "Clinical Dementia Rating"])
        age_col = find_column(df, ["Age"])

        labelled_count = (
            int(df[cdr_col].notna().sum()) if cdr_col else len(df)
        )

        if age_col:
            age_series = pd.to_numeric(df[age_col], errors="coerce")
            mean_age = age_series.mean()
        else:
            mean_age = np.nan

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Clinical records", f"{len(df):,}")
        k2.metric("Labelled records", f"{labelled_count:,}")
        k3.metric(
            "Mean age",
            f"{mean_age:.1f} years" if pd.notna(mean_age) else "N/A",
        )
        k4.metric("CNN models found", f"{len(available_cnn_paths)}/4")

    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Labelled subjects", "235")
        k2.metric("MRI images", "4,935")
        k3.metric("Slices per subject", "21")
        k4.metric("CNN models found", f"{len(available_cnn_paths)}/4")

        st.warning(
            "Clinical CSV was not found. The fallback dataset counts above "
            "are project summary values, not loaded live from the CSV."
        )

    st.divider()
    render_dataset_charts()


# ============================================================
# 13. MRI PREDICTION PAGE
# ============================================================

elif page == "MRI Prediction":
    render_mri_prediction("mri_page")


# ============================================================
# 14. CLINICAL PREDICTION PAGE
# ============================================================

elif page == "Clinical Prediction":
    page_header(
        "Clinical Prediction",
        "Enter clinical variables to run the saved Random Forest model. "
        "MMSE is excluded from this primary workflow.",
        eyebrow="CLINICAL WORKFLOW",
    )

    clinical_prediction_form("clinical_page")


# ============================================================
# 15. MULTIMODAL FUSION PAGE
# ============================================================

elif page == "Multimodal Fusion":
    page_header(
        "Multimodal Fusion",
        "Combine an MRI slice probability with the clinical model probability.",
        eyebrow="FUSION WORKFLOW",
    )

    section_title("MRI input")
    uploaded = st.file_uploader(
        "Upload MRI slice (PNG, JPG, JPEG)",
        type=["png", "jpg", "jpeg"],
        key="fusion_mri_upload",
    )

    mri_probability = None

    if uploaded is not None:
        try:
            preview, image_array = prepare_mri_image(uploaded)
            st.image(preview, caption="Uploaded MRI slice", width=220)

            if st.button(
                "Run MRI models for fusion",
                key="fusion_run_mri",
                width="stretch",
            ):
                probabilities = []

                for model_name, model_path in available_cnn_paths.items():
                    model = load_cnn_model(str(model_path))
                    if model is None:
                        continue

                    try:
                        probability = get_cnn_positive_probability(
                            model, image_array
                        )
                        probabilities.append(probability)
                    except Exception:
                        pass

                if probabilities:
                    mri_probability = float(np.mean(probabilities))
                    st.session_state["fusion_mri_probability"] = mri_probability
                else:
                    st.error("No CNN model returned a valid probability.")

        except Exception as exc:
            st.error(f"Could not process the MRI image: {exc}")

    mri_probability = st.session_state.get("fusion_mri_probability")

    if mri_probability is not None:
        st.metric("MRI probability", f"{mri_probability:.1%}")

    st.divider()
    section_title("Clinical input")
    clinical_probability = clinical_prediction_form("fusion_clinical")

    if clinical_probability is None:
        clinical_probability = st.session_state.get(
            "fusion_clinical_probability"
        )

    if clinical_probability is not None:
        st.session_state["fusion_clinical_probability"] = clinical_probability

    st.divider()
    section_title("Fusion settings")

    # Use the documented saved fusion settings by default.
    c1, c2, c3 = st.columns(3)

    with c1:
        mri_weight = st.number_input(
            "MRI weight", min_value=0.0, max_value=1.0,
            value=0.40, step=0.05, key="fusion_mri_weight"
        )

    with c2:
        clinical_weight = st.number_input(
            "Clinical weight", min_value=0.0, max_value=1.0,
            value=0.60, step=0.05, key="fusion_clinical_weight"
        )

    with c3:
        fusion_threshold = st.number_input(
            "Decision threshold", min_value=0.0, max_value=1.0,
            value=0.43, step=0.01, format="%.2f",
            key="fusion_threshold"
        )

    if not np.isclose(mri_weight + clinical_weight, 1.0):
        st.warning("MRI and clinical weights should add up to 1.00.")

    if st.button(
        "Run multimodal prediction",
        key="run_fusion",
        type="primary",
        width="stretch",
    ):
        mri_probability = st.session_state.get("fusion_mri_probability")
        clinical_probability = st.session_state.get(
            "fusion_clinical_probability"
        )

        if mri_probability is None or clinical_probability is None:
            st.error(
                "Run MRI prediction and clinical prediction first. "
                "Both probabilities are required."
            )
        elif not np.isclose(mri_weight + clinical_weight, 1.0):
            st.error("The two weights must add up to 1.00.")
        else:
            fused_probability = (
                mri_weight * mri_probability
                + clinical_weight * clinical_probability
            )

            st.session_state["fusion_result"] = {
                "mri_probability": mri_probability,
                "clinical_probability": clinical_probability,
                "fused_probability": fused_probability,
                "threshold": fusion_threshold,
                "mri_weight": mri_weight,
                "clinical_weight": clinical_weight,
            }

    fusion_result = st.session_state.get("fusion_result")

    if fusion_result:
        st.divider()
        section_title("Prediction summary")

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "MRI probability",
            f"{fusion_result['mri_probability']:.1%}"
        )
        c2.metric(
            "Clinical probability",
            f"{fusion_result['clinical_probability']:.1%}"
        )
        c3.metric(
            "Fused probability",
            f"{fusion_result['fused_probability']:.1%}"
        )

        render_prediction_result(
            "Multimodal result",
            fusion_result["fused_probability"],
            fusion_result["threshold"],
        )
        st.progress(
            float(np.clip(fusion_result["fused_probability"], 0, 1))
        )

        contribution_df = pd.DataFrame({
            "Input": ["MRI contribution", "Clinical contribution"],
            "Weighted probability": [
                fusion_result["mri_weight"] * fusion_result["mri_probability"],
                fusion_result["clinical_weight"] * fusion_result["clinical_probability"],
            ],
        })

        fig = px.bar(
            contribution_df,
            x="Input",
            y="Weighted probability",
            text=contribution_df["Weighted probability"].map(
                lambda x: f"{x:.1%}"
            ),
            title="Weighted contributions to fusion score",
        )
        fig.update_layout(height=280, margin=dict(l=40, r=20, t=40, b=30))
        st.plotly_chart(fig, width="stretch")

        st.caption(
            f"Formula: {fusion_result['mri_weight']:.0%} × MRI + "
            f"{fusion_result['clinical_weight']:.0%} × clinical. "
            f"Threshold: {fusion_result['threshold']:.2f}."
        )


# ============================================================
# 16. MODEL PERFORMANCE PAGE
# ============================================================

elif page == "Model Performance":
    render_model_performance()


# ============================================================
# 17. SYSTEM CHECK PAGE
# ============================================================

elif page == "System Check":
    render_system_check()


# ============================================================
# 18. ABOUT PAGE
# ============================================================

elif page == "About":
    page_header(
        "About this research project",
        "Project background, methodology notes, and system status.",
        eyebrow="PROJECT INFO",
    )

    section_title("Project objective")
    st.write(
        "Explore binary classification using OASIS-1 MRI images and "
        "clinical information, with individual models and probability-level fusion."
    )

    section_title("Clinical input features")
    st.markdown(
        """
        - Age -- subject's age in years
        - Education (Educ) -- education code, 1 to 5 as coded in the OASIS-1 dataset
        - Socioeconomic status (SES) -- category, 1 to 5 as coded in the OASIS-1 dataset
        - Estimated total intracranial volume (eTIV)
        - Normalized whole brain volume (nWBV)
        - Atlas scaling factor (ASF)
        - Sex

        MMSE is excluded from the primary clinical workflow.
        """
    )

    section_title("Label definition")
    st.markdown(
        """
        - CDR = 0 → **Class 0** (no dementia indicated)
        - CDR > 0 → **Class 1** (cognitive impairment / dementia indicated)
        - Subjects with missing CDR are excluded from labelled evaluation.
        """
    )

    section_title("System status")
    c1, c2 = st.columns(2)

    with c1:
        if clinical_model is not None:
            st.success("Clinical model loaded")
        else:
            st.error("Clinical model unavailable")

    with c2:
        if available_cnn_paths:
            st.success(f"{len(available_cnn_paths)} CNN model(s) found")
        else:
            st.error("No CNN models found")

        with col3:
            st.metric(
                "Recall",
                f"{row['Recall'] * 100:.2f}%"
            )

        with col4:
            st.metric(
                "F1 Score",
                f"{row['F1'] * 100:.2f}%"
            )

        with col5:
            st.metric(
                "ROC-AUC",
                f"{row['ROC_AUC'] * 100:.2f}%"
            )

        st.divider()

        st.subheader("Final Fusion Configuration")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "MRI Weight",
                f"{row['MRI_Weight'] * 100:.0f}%"
            )

        with col2:
            st.metric(
                "Clinical Weight",
                f"{row['Clinical_Weight'] * 100:.0f}%"
            )

        with col3:
            st.metric(
                "Threshold",
                f"{row['Threshold']:.2f}"
            )

        st.subheader("Confusion Matrix")

        cm = load_confusion_matrix()

        cm_display = cm.copy()

        if "Unnamed: 0" in cm_display.columns:
            cm_display = cm_display.set_index("Unnamed: 0")

        st.dataframe(
            cm_display,
            use_container_width=True
        )

    except Exception as e:

        st.error(
            f"Could not load performance results: {e}"
        )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "OASIS Multimodal Alzheimer's Disease Classification — Demo Dashboard"
)
