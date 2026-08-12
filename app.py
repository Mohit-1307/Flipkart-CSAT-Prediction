"""
Flipkart Customer Satisfaction (CSAT) Intelligence Suite
==========================================================
A production-grade Streamlit application for predicting and analyzing
customer satisfaction from support-interaction data, powered by a
GridSearchCV-tuned XGBoost classifier trained on 85,907 Flipkart
customer support tickets (structured features + TF-IDF text signal).

Model performance (held-out test set, 20%, stratified):
    ROC-AUC ......... 0.8055
    Accuracy ........ 0.7358
    Precision (macro) 0.6455
    Recall (macro) .. 0.7248
    F1 (macro) ...... 0.6335
    Best params ..... n_estimators=300, max_depth=6, learning_rate=0.1

Run locally:
    streamlit run app.py

Author: ML Deployment Layer for Flipkart-CSAT-Prediction
"""

import os
import re
import string
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.sparse import csr_matrix, hstack

warnings.filterwarnings("ignore")

# ============================================================================
# PAGE CONFIG  (must be the first Streamlit call)
# ============================================================================
st.set_page_config(
    page_title="Flipkart CSAT Intelligence Suite",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/Mohit-1307/Flipkart-CSAT-Prediction",
        "About": "Flipkart Customer Satisfaction Prediction — XGBoost-powered "
                 "decision-support tool for support operations teams.",
    },
)

# ============================================================================
# PATHS  (flat layout — falls back gracefully if a models/ subfolder exists)
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _first_existing(*candidates):
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


MODEL_PATH = _first_existing(
    os.path.join(BASE_DIR, "best_xgboost_classifier.pkl"),
    os.path.join(BASE_DIR, "models", "best_xgboost_classifier.pkl"),
)
TFIDF_PATH = _first_existing(
    os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"),
    os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl"),
)
SCALER_PATH = _first_existing(
    os.path.join(BASE_DIR, "standard_scaler.pkl"),
    os.path.join(BASE_DIR, "models", "standard_scaler.pkl"),
)
PT_PATH = _first_existing(
    os.path.join(BASE_DIR, "power_transformer.pkl"),
    os.path.join(BASE_DIR, "models", "power_transformer.pkl"),
)
DATA_PATH = _first_existing(
    os.path.join(BASE_DIR, "Customer_support_data.csv"),
    os.path.join(BASE_DIR, "data", "Customer_support_data.csv"),
)
IMAGES_DIR = _first_existing(
    os.path.join(BASE_DIR, "images"),
    os.path.join(BASE_DIR, "assets", "images"),
)

# Exact feature order the model was trained on — do not reorder.
ENC_FEATURE_COLS = ["channel_name_enc", "category_enc", "Sub-category_enc",
                     "Tenure Bucket_enc", "Agent Shift_enc"]
NUM_FEATURE_COLS = ["response_time_minutes", "issue_hour", "issue_dayofweek",
                     "agent_csat_encoded", "supervisor_csat_encoded"]
TENURE_ORDER = ["On Job Training", "0-30", "31-60", "61-90", ">90"]
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Verified against the training notebook (grid_xgb.best_params_ / classification_report)
MODEL_METRICS = {
    "accuracy": 0.7358,
    "precision_macro": 0.6455,
    "recall_macro": 0.7248,
    "f1_macro": 0.6335,
    "roc_auc": 0.8055,
}
BEST_PARAMS = {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1,
               "tree_method": "hist", "eval_metric": "logloss"}
DEFAULT_THRESHOLD = 0.5  # notebook's F1-optimal threshold varies by run; 0.5 is the safe production default

# ============================================================================
# THEME — CSS INJECTION
# ============================================================================
PRIMARY = "#1F4FD8"
PRIMARY_LIGHT = "#3667E8"
ACCENT = "#FFC633"
POSITIVE = "#17B26A"
NEGATIVE = "#F04438"
WARNING_C = "#F79009"
SURFACE = "#0B1220"
SURFACE_2 = "#121C30"
TEXT_MUTED = "#93A2C2"

CUSTOM_CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Manrope', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    .stApp {{
        background: radial-gradient(circle at 15% 0%, #101C36 0%, #070B14 55%);
    }}

    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0C1526 0%, #070B14 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }}
    section[data-testid="stSidebar"] * {{ color: #E7ECFA; }}

    .csat-hero {{
        background: linear-gradient(120deg, {SURFACE_2} 0%, #0E1830 60%, #14204A 100%);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 20px;
        padding: 34px 40px;
        margin-bottom: 22px;
        position: relative;
        overflow: hidden;
    }}
    .csat-hero::after {{
        content: ""; position: absolute; top: -60px; right: -60px;
        width: 260px; height: 260px;
        background: radial-gradient(circle, rgba(255,198,51,0.18) 0%, rgba(255,198,51,0) 70%);
    }}
    .csat-hero::before {{
        content: ""; position: absolute; bottom: -80px; left: 20%;
        width: 300px; height: 300px;
        background: radial-gradient(circle, rgba(31,79,216,0.28) 0%, rgba(31,79,216,0) 70%);
    }}
    .csat-eyebrow {{
        color: {ACCENT}; font-weight: 700; letter-spacing: 0.14em;
        font-size: 0.72rem; text-transform: uppercase; margin-bottom: 10px;
        font-family: 'JetBrains Mono', monospace;
    }}
    .csat-title {{
        color: #F5F7FF; font-size: 2.15rem; font-weight: 800;
        margin: 0 0 8px 0; letter-spacing: -0.02em;
    }}
    .csat-subtitle {{
        color: {TEXT_MUTED}; font-size: 1.02rem; max-width: 760px; line-height: 1.55; margin: 0;
    }}

    .kpi-card {{
        background: linear-gradient(160deg, {SURFACE_2} 0%, #0D1526 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px; padding: 18px 20px; height: 100%;
        transition: border-color 0.2s ease;
    }}
    .kpi-card:hover {{ border-color: rgba(255,198,51,0.4); }}
    .kpi-label {{
        color: {TEXT_MUTED}; font-size: 0.74rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px;
    }}
    .kpi-value {{ color: #F5F7FF; font-size: 1.65rem; font-weight: 800; letter-spacing: -0.01em; }}
    .kpi-delta-pos {{ color: {POSITIVE}; font-size: 0.82rem; font-weight: 600; }}
    .kpi-delta-neg {{ color: {NEGATIVE}; font-size: 0.82rem; font-weight: 600; }}

    .verdict-box {{
        border-radius: 16px; padding: 26px 30px; margin: 14px 0 20px 0;
        border: 1px solid rgba(255,255,255,0.08);
    }}
    .verdict-satisfied {{
        background: linear-gradient(120deg, rgba(23,178,106,0.16) 0%, rgba(23,178,106,0.03) 100%);
        border-left: 4px solid {POSITIVE};
    }}
    .verdict-risk {{
        background: linear-gradient(120deg, rgba(240,68,56,0.18) 0%, rgba(240,68,56,0.03) 100%);
        border-left: 4px solid {NEGATIVE};
    }}
    .verdict-label {{ font-size: 1.35rem; font-weight: 800; color: #F5F7FF; margin-bottom: 4px; }}
    .verdict-sub {{ color: {TEXT_MUTED}; font-size: 0.92rem; }}

    .section-tag {{
        display: inline-block; font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem; letter-spacing: 0.12em; text-transform: uppercase;
        color: {ACCENT}; border: 1px solid rgba(255,198,51,0.35);
        border-radius: 999px; padding: 4px 12px; margin-bottom: 10px;
    }}

    .factor-chip {{
        display: inline-flex; align-items: center; gap: 6px;
        background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px; padding: 10px 14px; margin: 4px 6px 4px 0;
        font-size: 0.85rem; color: #E7ECFA;
    }}

    .alert-box {{
        border-radius: 12px; padding: 14px 18px; margin: 10px 0;
        border: 1px solid rgba(255,255,255,0.08); font-size: 0.88rem;
    }}
    .alert-warn {{
        background: rgba(247,144,9,0.1); border-left: 3px solid {WARNING_C}; color: #FEEBC8;
    }}
    .alert-info {{
        background: rgba(31,79,216,0.1); border-left: 3px solid {PRIMARY}; color: #DCE5FF;
    }}

    div[data-testid="stMetric"] {{
        background: linear-gradient(160deg, {SURFACE_2} 0%, #0D1526 100%);
        border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 14px 16px;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px; background: rgba(255,255,255,0.03); padding: 6px; border-radius: 14px;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 10px; color: {TEXT_MUTED}; font-weight: 600; padding: 8px 18px;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(120deg, {PRIMARY}, {PRIMARY_LIGHT}) !important; color: white !important;
    }}

    .stButton > button {{
        background: linear-gradient(120deg, {PRIMARY} 0%, {PRIMARY_LIGHT} 100%);
        color: white; border: none; border-radius: 10px; font-weight: 700;
        padding: 0.6rem 1.4rem; transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}
    .stButton > button:hover {{
        transform: translateY(-1px); box-shadow: 0 8px 20px rgba(31,79,216,0.35);
    }}
    .stDownloadButton > button {{
        background: linear-gradient(120deg, {POSITIVE} 0%, #12965A 100%);
        color: white; border: none; border-radius: 10px; font-weight: 700;
    }}

    code {{ color: {ACCENT} !important; }}

    .footer-note {{
        text-align: center; color: {TEXT_MUTED}; font-size: 0.78rem;
        padding: 24px 0 10px 0; border-top: 1px solid rgba(255,255,255,0.06); margin-top: 30px;
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Manrope, sans-serif", color="#E7ECFA", size=13),
        colorway=[PRIMARY, ACCENT, POSITIVE, NEGATIVE, "#8E7CFF", "#33C3D6"],
        xaxis=dict(gridcolor="rgba(255,255,255,0.07)", zerolinecolor="rgba(255,255,255,0.12)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.07)", zerolinecolor="rgba(255,255,255,0.12)"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=50, b=10),
    )
)

# ============================================================================
# CACHED LOADERS
# ============================================================================
@st.cache_resource(show_spinner="Loading model artifacts...")
def load_artifacts():
    """Load the trained XGBoost model and fitted preprocessors."""
    missing = [p for p in [MODEL_PATH, TFIDF_PATH, SCALER_PATH, PT_PATH] if not os.path.exists(p)]
    if missing:
        return None, None, None, None, missing
    try:
        model = joblib.load(MODEL_PATH)
        tfidf = joblib.load(TFIDF_PATH)
        scaler = joblib.load(SCALER_PATH)
        pt = joblib.load(PT_PATH)
    except Exception as e:  # pragma: no cover - defensive
        return None, None, None, None, [f"Failed to load artifacts: {e}"]
    return model, tfidf, scaler, pt, []


@st.cache_data(show_spinner="Loading historical support data...")
def load_data():
    """Load raw dataset and compute the encoding lookup tables the model relies on.

    Mirrors the training notebook's feature-engineering pipeline exactly:
    response-time computation, IQR winsorization, smoothed target encoding
    (K=10) for Agent_name / Supervisor, and label-encoding for nominal
    categoricals (fit in the same alphabetical order LabelEncoder uses).
    """
    if not os.path.exists(DATA_PATH):
        return None, None

    df = pd.read_csv(DATA_PATH)
    df1 = df.copy()

    df1["Issue_reported at"] = pd.to_datetime(df1["Issue_reported at"], dayfirst=True, errors="coerce")
    df1["issue_responded"] = pd.to_datetime(df1["issue_responded"], dayfirst=True, errors="coerce")
    df1["response_time_minutes"] = ((df1["issue_responded"] - df1["Issue_reported at"]).dt.total_seconds() / 60).clip(lower=0)
    df1["issue_hour"] = df1["Issue_reported at"].dt.hour
    df1["issue_dayofweek"] = df1["Issue_reported at"].dt.dayofweek
    df1.drop_duplicates(inplace=True)
    if "connected_handling_time" in df1.columns:
        df1.drop(columns=["connected_handling_time"], inplace=True)
    df1["CSAT_label"] = (df1["CSAT Score"] >= 4).astype(int)

    for col in ["Sub-category", "category", "channel_name", "Tenure Bucket", "Agent Shift"]:
        if df1[col].isnull().sum() > 0:
            df1[col] = df1[col].fillna(df1[col].mode()[0])
    df1["Customer Remarks"] = df1["Customer Remarks"].fillna("no remarks").astype(str)
    for col in ["Item_price", "response_time_minutes", "issue_hour", "issue_dayofweek"]:
        if df1[col].isnull().sum() > 0:
            df1[col] = df1[col].fillna(df1[col].median())
    for col in ["Customer_City", "Product_category"]:
        if col in df1.columns:
            df1[col] = df1[col].fillna("Unknown").astype(str)

    # Winsorize response time (IQR capping — identical to training)
    q1, q3 = df1["response_time_minutes"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df1["response_time_minutes"] = df1["response_time_minutes"].clip(lower=lo, upper=hi)

    # Agent / Supervisor smoothed target-encoding lookups (K=10, matches notebook)
    K = 10
    global_mean = df1["CSAT Score"].mean()

    agent_mean = df1.groupby("Agent_name")["CSAT Score"].mean()
    agent_count = df1.groupby("Agent_name")["CSAT Score"].count()
    agent_lookup = ((agent_count * agent_mean + K * global_mean) / (agent_count + K)).to_dict()

    sup_mean = df1.groupby("Supervisor")["CSAT Score"].mean()
    sup_count = df1.groupby("Supervisor")["CSAT Score"].count()
    sup_lookup = ((sup_count * sup_mean + K * global_mean) / (sup_count + K)).to_dict()

    df1["agent_csat_encoded"] = df1["Agent_name"].map(agent_lookup).fillna(global_mean)
    df1["supervisor_csat_encoded"] = df1["Supervisor"].map(sup_lookup).fillna(global_mean)

    lookups = {
        "global_mean": global_mean,
        "agent_lookup": agent_lookup,
        "sup_lookup": sup_lookup,
        "agents": sorted(df1["Agent_name"].dropna().unique().tolist()),
        "supervisors": sorted(df1["Supervisor"].dropna().unique().tolist()),
        "channels": sorted(df1["channel_name"].dropna().unique().tolist()),
        "categories": sorted(df1["category"].dropna().unique().tolist()),
        "subcategory_by_category": {
            cat: sorted(g["Sub-category"].dropna().unique().tolist())
            for cat, g in df1.groupby("category")
        },
        "cities": sorted([c for c in df1["Customer_City"].dropna().unique().tolist() if c != "Unknown"]),
        "response_time_median": df1["response_time_minutes"].median(),
        "response_time_p90": df1["response_time_minutes"].quantile(0.90),
        "item_price_median": df1["Item_price"].median(skipna=True),
    }
    return df1, lookups


def build_label_maps(df1):
    """Recreate the exact LabelEncoder mappings (alphabetical fit order) used in training."""
    from sklearn.preprocessing import LabelEncoder
    maps = {}
    for col in ["channel_name", "category", "Sub-category", "Agent Shift"]:
        le = LabelEncoder()
        le.fit(df1[col].astype(str))
        maps[col] = {cls: idx for idx, cls in enumerate(le.classes_)}
    return maps


# ============================================================================
# TEXT CLEANING PIPELINE (mirrors notebook exactly)
# ============================================================================
_NLTK_READY = False


def ensure_nltk():
    global _NLTK_READY
    if _NLTK_READY:
        return
    import nltk
    for pkg, path in [
        ("stopwords", "corpora/stopwords"),
        ("wordnet", "corpora/wordnet"),
        ("omw-1.4", "corpora/omw-1.4"),
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass
    _NLTK_READY = True


CONTRACTIONS_MAP = {"can't": "cannot", "won't": "will not", "n't": " not", "re": "are",
                     "'s": "is", "'d": " would", "ll": " will", "'ve": "have", "'m": "am"}


def expand_contractions(text):
    for key, value in CONTRACTIONS_MAP.items():
        text = text.replace(key, value)
    return text


def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))


def clean_text_urls_digits(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\w*\d\w*", "", text)
    return text


def rephrase_text(text):
    text = text.replace("delivery late", "late delivery")
    text = text.replace("not received", "undelivered")
    text = text.replace("didnt receive", "undelivered")
    return text


def full_text_pipeline(raw_text: str) -> str:
    """Reproduces the notebook's exact NLP cleaning sequence for a single remark:
    expand contractions -> lowercase -> strip punctuation -> strip URLs/digit-tokens
    -> remove stopwords -> rephrase -> lemmatize.
    """
    ensure_nltk()
    try:
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        stop_words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()
    except Exception:
        stop_words = set()
        lemmatizer = None

    text = str(raw_text) if raw_text and str(raw_text).strip() else "no remarks"
    text = expand_contractions(text)
    text = text.lower()
    text = remove_punctuation(text)
    text = clean_text_urls_digits(text)

    words = text.split()
    text = " ".join([w for w in words if w not in stop_words])
    text = text.strip() if text.strip() else "no remarks"
    text = rephrase_text(text)

    tokens = text.split()
    if lemmatizer is not None:
        tokens = [lemmatizer.lemmatize(w) for w in tokens]
    clean = " ".join(tokens).strip()
    return clean if clean else "no remarks"


# ============================================================================
# INFERENCE
# ============================================================================
def build_structured_vector(record: dict, label_maps, lookups):
    """Build the raw 10-column structured feature row (before PT/scaling)."""
    tenure_idx = TENURE_ORDER.index(record["tenure_bucket"]) if record["tenure_bucket"] in TENURE_ORDER else -1

    enc_vals = [
        label_maps["channel_name"].get(record["channel_name"], 0),
        label_maps["category"].get(record["category"], 0),
        label_maps["Sub-category"].get(record["sub_category"], 0),
        tenure_idx,
        label_maps["Agent Shift"].get(record["agent_shift"], 0),
    ]

    agent_csat = lookups["agent_lookup"].get(record.get("agent_name"), lookups["global_mean"])
    sup_csat = lookups["sup_lookup"].get(record.get("supervisor"), lookups["global_mean"])

    num_vals = [
        record["response_time_minutes"],
        record["issue_hour"],
        record["issue_dayofweek"],
        agent_csat,
        sup_csat,
    ]
    return enc_vals, num_vals, agent_csat, sup_csat


def predict_single(record: dict, model, tfidf, scaler, pt, label_maps, lookups, threshold: float = 0.5):
    enc_vals, num_vals, agent_csat, sup_csat = build_structured_vector(record, label_maps, lookups)

    X_struct = np.array([enc_vals + num_vals], dtype=float)
    X_cat = X_struct[:, :len(ENC_FEATURE_COLS)]
    X_num = X_struct[:, len(ENC_FEATURE_COLS):]
    X_num_tf = pt.transform(X_num)
    X_struct_transformed = np.hstack([X_cat, X_num_tf])
    X_struct_scaled = scaler.transform(X_struct_transformed)

    clean_remark = full_text_pipeline(record.get("customer_remarks", ""))
    X_text = tfidf.transform([clean_remark])

    X_combined = hstack([csr_matrix(X_struct_scaled), X_text])

    proba = model.predict_proba(X_combined)[0]
    pred = int(proba[1] >= threshold)
    return {
        "prediction": pred,
        "prob_satisfied": float(proba[1]),
        "prob_dissatisfied": float(proba[0]),
        "clean_remark": clean_remark,
        "agent_csat_encoded": agent_csat,
        "supervisor_csat_encoded": sup_csat,
    }


def predict_batch(df_input: pd.DataFrame, model, tfidf, scaler, pt, label_maps, lookups,
                   threshold: float = 0.5) -> pd.DataFrame:
    """Vectorized batch prediction for an uploaded CSV of tickets."""
    df = df_input.copy()

    for target, source in [("channel_name", "channel_name"), ("category", "category"),
                            ("sub_category", "Sub-category"), ("tenure_bucket", "Tenure Bucket"),
                            ("agent_shift", "Agent Shift")]:
        if source in df.columns and target not in df.columns:
            df[target] = df[source]

    required = ["channel_name", "category", "sub_category", "tenure_bucket", "agent_shift"]
    for col in required:
        if col not in df.columns:
            df[col] = "Unknown"
        df[col] = df[col].fillna("Unknown").astype(str)

    if "Issue_reported at" in df.columns and "issue_responded" in df.columns:
        ir = pd.to_datetime(df["Issue_reported at"], dayfirst=True, errors="coerce")
        rr = pd.to_datetime(df["issue_responded"], dayfirst=True, errors="coerce")
        df["response_time_minutes"] = ((rr - ir).dt.total_seconds() / 60).clip(lower=0)
        df["issue_hour"] = ir.dt.hour
        df["issue_dayofweek"] = ir.dt.dayofweek

    for col, default in [("response_time_minutes", lookups["response_time_median"]),
                          ("issue_hour", 12), ("issue_dayofweek", 2)]:
        if col not in df.columns:
            df[col] = default
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

    if "Agent_name" not in df.columns:
        df["Agent_name"] = None
    if "Supervisor" not in df.columns:
        df["Supervisor"] = None
    if "Customer Remarks" not in df.columns:
        df["Customer Remarks"] = "no remarks"
    df["Customer Remarks"] = df["Customer Remarks"].fillna("no remarks").astype(str)

    df["agent_csat_encoded"] = df["Agent_name"].map(lookups["agent_lookup"]).fillna(lookups["global_mean"])
    df["supervisor_csat_encoded"] = df["Supervisor"].map(lookups["sup_lookup"]).fillna(lookups["global_mean"])

    def enc(col_key, series):
        mp = label_maps[col_key]
        return series.map(lambda v: mp.get(v, 0))

    tenure_map = {t: i for i, t in enumerate(TENURE_ORDER)}
    enc_vals = np.column_stack([
        enc("channel_name", df["channel_name"]).values,
        enc("category", df["category"]).values,
        enc("Sub-category", df["sub_category"]).values,
        df["tenure_bucket"].map(lambda v: tenure_map.get(v, -1)).values,
        enc("Agent Shift", df["agent_shift"]).values,
    ]).astype(float)

    num_vals = df[["response_time_minutes", "issue_hour", "issue_dayofweek",
                    "agent_csat_encoded", "supervisor_csat_encoded"]].values.astype(float)

    X_num_tf = pt.transform(num_vals)
    X_struct_transformed = np.hstack([enc_vals, X_num_tf])
    X_struct_scaled = scaler.transform(X_struct_transformed)

    ensure_nltk()
    clean_remarks = df["Customer Remarks"].apply(full_text_pipeline)
    X_text = tfidf.transform(clean_remarks)

    X_combined = hstack([csr_matrix(X_struct_scaled), X_text])
    proba = model.predict_proba(X_combined)
    df["predicted_label"] = np.where(proba[:, 1] >= threshold, "Satisfied", "Dissatisfied")
    df["prob_satisfied"] = proba[:, 1]
    df["prob_dissatisfied"] = proba[:, 0]
    df["risk_tier"] = pd.cut(
        df["prob_dissatisfied"], bins=[-0.01, 0.3, 0.6, 1.01],
        labels=["Low Risk", "Medium Risk", "High Risk"]
    )
    return df


# ============================================================================
# SMALL UI HELPERS
# ============================================================================
def kpi_card(label, value, help_text=None):
    st.markdown(
        f"""<div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                {f'<div style="color:{TEXT_MUTED};font-size:0.78rem;margin-top:4px;">{help_text}</div>' if help_text else ''}
            </div>""",
        unsafe_allow_html=True,
    )


def section_tag(text):
    st.markdown(f'<span class="section-tag">{text}</span>', unsafe_allow_html=True)


def alert_box(text, kind="info"):
    cls = "alert-warn" if kind == "warn" else "alert-info"
    icon = "⚠️" if kind == "warn" else "ℹ️"
    st.markdown(f'<div class="alert-box {cls}">{icon}&nbsp; {text}</div>', unsafe_allow_html=True)


def render_hero(eyebrow, title, subtitle):
    st.markdown(
        f"""<div class="csat-hero">
                <div class="csat-eyebrow">{eyebrow}</div>
                <div class="csat-title">{title}</div>
                <p class="csat-subtitle">{subtitle}</p>
            </div>""",
        unsafe_allow_html=True,
    )


def gauge_chart(prob_satisfied: float):
    color = POSITIVE if prob_satisfied >= 0.5 else NEGATIVE
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_satisfied * 100,
        number={"suffix": "%", "font": {"size": 40, "color": "#F5F7FF"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": TEXT_MUTED},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(255,255,255,0.04)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "rgba(240,68,56,0.18)"},
                {"range": [40, 65], "color": "rgba(247,144,9,0.16)"},
                {"range": [65, 100], "color": "rgba(23,178,106,0.16)"},
            ],
            "threshold": {"line": {"color": "#F5F7FF", "width": 3}, "thickness": 0.8, "value": 50},
        },
    ))
    fig.update_layout(template=PLOTLY_TEMPLATE, height=260, margin=dict(l=20, r=20, t=30, b=10))
    return fig


def factor_chips(record: dict, result: dict, lookups: dict):
    """Simple, transparent rule-based contribution flags (no SHAP dependency at inference)."""
    chips = []
    rt = record["response_time_minutes"]
    if rt > lookups["response_time_p90"]:
        chips.append(("🐢", f"Response time {rt:.0f} min — above P90 ({lookups['response_time_p90']:.0f} min)", "neg"))
    elif rt <= lookups["response_time_median"]:
        chips.append(("⚡", f"Response time {rt:.0f} min — at/below median ({lookups['response_time_median']:.0f} min)", "pos"))

    if result["agent_csat_encoded"] < lookups["global_mean"] - 0.15:
        chips.append(("🎯", f"Agent historical CSAT below team average ({result['agent_csat_encoded']:.2f})", "neg"))
    elif result["agent_csat_encoded"] > lookups["global_mean"] + 0.15:
        chips.append(("🌟", f"Agent historical CSAT above team average ({result['agent_csat_encoded']:.2f})", "pos"))

    if result["supervisor_csat_encoded"] < lookups["global_mean"] - 0.15:
        chips.append(("📉", f"Supervisor team CSAT trending low ({result['supervisor_csat_encoded']:.2f})", "neg"))

    if record["tenure_bucket"] == "On Job Training":
        chips.append(("🎓", "Agent still in On-Job-Training bucket", "neg"))
    elif record["tenure_bucket"] == ">90":
        chips.append(("🏅", "Agent tenure > 90 days — high experience bucket", "pos"))

    remark = record.get("customer_remarks", "").lower()
    negative_terms = ["refund", "not received", "undelivered", "late", "damaged", "worst", "cancel", "complaint"]
    if any(t in remark for t in negative_terms):
        chips.append(("💬", "Customer remark contains dissatisfaction-linked terms", "neg"))

    return chips


# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
with st.sidebar:
    st.markdown(
        f"""<div style="display:flex;align-items:center;gap:10px;padding:6px 0 18px 0;">
                <div style="font-size:1.8rem;">🛍️</div>
                <div>
                    <div style="font-weight:800;font-size:1.05rem;color:#F5F7FF;line-height:1.2;">CSAT Intelligence</div>
                    <div style="font-size:0.72rem;color:{TEXT_MUTED};letter-spacing:0.04em;">FLIPKART SUPPORT SUITE</div>
                </div>
            </div>""",
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigate",
        ["🔮 Single Prediction", "📦 Batch Prediction", "📊 Data Explorer",
         "🧠 Model Performance", "ℹ️ About"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    threshold = st.slider(
        "Decision threshold", min_value=0.10, max_value=0.90, value=DEFAULT_THRESHOLD, step=0.01,
        help="Probability of 'Satisfied' above which a ticket is classified Satisfied. "
             "Lower it to catch more at-risk tickets (higher recall); raise it to reduce false alarms (higher precision)."
    )
    st.caption(f"Current threshold: **{threshold:.2f}** · default 0.50")

    st.markdown("---")
    model, tfidf, scaler, pt, missing_artifacts = load_artifacts()
    df_data, lookups = load_data()

    if missing_artifacts:
        st.error("⚠️ Model artifacts missing")
        with st.expander("Details"):
            for m in missing_artifacts:
                st.code(m)
    else:
        st.success("✅ Model artifacts loaded")

    if df_data is None:
        st.warning("⚠️ Dataset not found")
    else:
        st.success(f"✅ {len(df_data):,} tickets loaded")

    label_maps = build_label_maps(df_data) if df_data is not None else None

# ============================================================================
# PAGE: SINGLE PREDICTION
# ============================================================================
if page == "🔮 Single Prediction":
    render_hero(
        "REAL-TIME SCORING",
        "Will this customer be satisfied?",
        "Enter the details of a live support ticket to get an instant CSAT-risk score, "
        "powered by a tuned XGBoost model combining structured ticket metadata with "
        "NLP signal extracted from the customer's own words.",
    )

    if model is None or df_data is None:
        alert_box("Model or dataset artifacts are missing — predictions are unavailable until "
                   "best_xgboost_classifier.pkl, tfidf_vectorizer.pkl, standard_scaler.pkl, "
                   "power_transformer.pkl and Customer_support_data.csv are all present next to app.py.", "warn")
    else:
        with st.form("predict_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                channel_name = st.selectbox("Support Channel", lookups["channels"])
                category = st.selectbox("Issue Category", lookups["categories"])
                sub_options = lookups["subcategory_by_category"].get(category, [])
                sub_category = st.selectbox("Sub-Category", sub_options if sub_options else ["Unknown"])
            with c2:
                agent_shift = st.selectbox("Agent Shift", sorted(df_data["Agent Shift"].dropna().unique().tolist()))
                tenure_bucket = st.selectbox("Agent Tenure Bucket", TENURE_ORDER)
                response_time_minutes = st.number_input(
                    "Response Time (minutes)", min_value=0.0,
                    value=float(round(lookups["response_time_median"], 1)), step=1.0
                )
            with c3:
                issue_dt = st.time_input("Issue Reported Time", value=datetime.now().time())
                issue_day = st.selectbox("Day of Week", DAY_NAMES, index=2)
                agent_name = st.selectbox("Agent (optional)", ["— Unknown —"] + lookups["agents"])
                supervisor = st.selectbox("Supervisor (optional)", ["— Unknown —"] + lookups["supervisors"])

            customer_remarks = st.text_area(
                "Customer Remarks (optional)",
                placeholder="e.g. My order was delayed by 5 days and the refund still hasn't been processed...",
                height=90,
            )

            submitted = st.form_submit_button("Predict CSAT Outcome →", use_container_width=True)

        if submitted:
            record = {
                "channel_name": channel_name,
                "category": category,
                "sub_category": sub_category,
                "agent_shift": agent_shift,
                "tenure_bucket": tenure_bucket,
                "response_time_minutes": response_time_minutes,
                "issue_hour": issue_dt.hour,
                "issue_dayofweek": DAY_NAMES.index(issue_day),
                "agent_name": None if agent_name == "— Unknown —" else agent_name,
                "supervisor": None if supervisor == "— Unknown —" else supervisor,
                "customer_remarks": customer_remarks,
            }
            with st.spinner("Scoring ticket..."):
                result = predict_single(record, model, tfidf, scaler, pt, label_maps, lookups, threshold=threshold)

            st.markdown("---")
            vc1, vc2 = st.columns([1.4, 1])
            with vc1:
                if result["prediction"] == 1:
                    st.markdown(
                        f"""<div class="verdict-box verdict-satisfied">
                                <div class="verdict-label">✅ Predicted: Satisfied</div>
                                <div class="verdict-sub">Model confidence: {result['prob_satisfied']*100:.1f}%
                                (threshold {threshold:.2f}) — this ticket is unlikely to need proactive escalation.</div>
                            </div>""",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""<div class="verdict-box verdict-risk">
                                <div class="verdict-label">🚨 Predicted: Dissatisfied — Escalation Risk</div>
                                <div class="verdict-sub">Model confidence: {result['prob_dissatisfied']*100:.1f}%
                                (threshold {threshold:.2f}) — consider proactive service recovery before the
                                customer submits a low rating.</div>
                            </div>""",
                        unsafe_allow_html=True,
                    )

                section_tag("Contributing Factors")
                chips = factor_chips(record, result, lookups)
                if chips:
                    chip_html = "".join(
                        f'<span class="factor-chip">{icon} {text}</span>' for icon, text, _ in chips
                    )
                    st.markdown(chip_html, unsafe_allow_html=True)
                else:
                    st.caption("No strong signal flags detected — this ticket looks typical.")

                with st.expander("🔍 View cleaned text sent to the NLP model"):
                    st.code(result["clean_remark"] or "(empty)", language="text")

            with vc2:
                st.plotly_chart(gauge_chart(result["prob_satisfied"]), use_container_width=True)
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("P(Satisfied)", f"{result['prob_satisfied']*100:.1f}%")
                with m2:
                    st.metric("P(Dissatisfied)", f"{result['prob_dissatisfied']*100:.1f}%")

# ============================================================================
# PAGE: BATCH PREDICTION
# ============================================================================
elif page == "📦 Batch Prediction":
    render_hero(
        "BULK SCORING",
        "Score a whole queue of tickets at once",
        "Upload a CSV of support tickets to get CSAT-risk predictions and risk tiers for "
        "every row — ideal for prioritizing a support queue or auditing a day's tickets.",
    )

    if model is None or df_data is None:
        alert_box("Model or dataset artifacts are missing — batch scoring is unavailable.", "warn")
    else:
        alert_box(
            "Expected columns (best effort matching, missing ones default sensibly): "
            "<code>channel_name</code>, <code>category</code>, <code>Sub-category</code>, "
            "<code>Tenure Bucket</code>, <code>Agent Shift</code>, <code>Issue_reported at</code>, "
            "<code>issue_responded</code>, <code>Agent_name</code>, <code>Supervisor</code>, "
            "<code>Customer Remarks</code>.", "info"
        )

        uploaded = st.file_uploader("Upload tickets CSV", type=["csv"])
        sample = st.checkbox("Use a random 200-row sample from the training data instead")

        input_df = None
        if uploaded is not None:
            try:
                input_df = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
        elif sample:
            input_df = df_data.sample(n=min(200, len(df_data)), random_state=42).drop(
                columns=["CSAT_label", "response_time_minutes", "issue_hour", "issue_dayofweek",
                         "agent_csat_encoded", "supervisor_csat_encoded"], errors="ignore"
            )

        if input_df is not None:
            st.caption(f"Loaded {len(input_df):,} rows · {len(input_df.columns)} columns")
            st.dataframe(input_df.head(10), use_container_width=True, height=250)

            if st.button("Run Batch Prediction →", use_container_width=True):
                with st.spinner(f"Scoring {len(input_df):,} tickets..."):
                    scored = predict_batch(input_df, model, tfidf, scaler, pt, label_maps, lookups,
                                            threshold=threshold)
                st.session_state["batch_result"] = scored

        if "batch_result" in st.session_state:
            scored = st.session_state["batch_result"]
            st.markdown("---")
            section_tag("Results Summary")

            k1, k2, k3, k4 = st.columns(4)
            with k1:
                kpi_card("Total Scored", f"{len(scored):,}")
            with k2:
                sat_pct = (scored["predicted_label"] == "Satisfied").mean() * 100
                kpi_card("Predicted Satisfied", f"{sat_pct:.1f}%")
            with k3:
                high_risk = (scored["risk_tier"] == "High Risk").sum()
                kpi_card("High Risk Tickets", f"{high_risk:,}")
            with k4:
                kpi_card("Avg P(Satisfied)", f"{scored['prob_satisfied'].mean()*100:.1f}%")

            cc1, cc2 = st.columns(2)
            with cc1:
                section_tag("Risk Tier Distribution")
                tier_counts = scored["risk_tier"].value_counts().reindex(["Low Risk", "Medium Risk", "High Risk"]).fillna(0)
                fig = px.bar(x=tier_counts.index, y=tier_counts.values,
                             color=tier_counts.index,
                             color_discrete_map={"Low Risk": POSITIVE, "Medium Risk": WARNING_C, "High Risk": NEGATIVE})
                fig.update_layout(template=PLOTLY_TEMPLATE, height=340, showlegend=False,
                                   xaxis_title="", yaxis_title="Tickets")
                st.plotly_chart(fig, use_container_width=True)
            with cc2:
                section_tag("Predicted Satisfaction Split")
                fig = px.pie(scored, names="predicted_label", hole=0.55,
                             color="predicted_label",
                             color_discrete_map={"Satisfied": POSITIVE, "Dissatisfied": NEGATIVE})
                fig.update_layout(template=PLOTLY_TEMPLATE, height=340)
                st.plotly_chart(fig, use_container_width=True)

            section_tag("Scored Tickets (High Risk First)")
            display_cols = [c for c in ["Unique id", "channel_name", "category", "sub_category",
                                         "Customer Remarks", "predicted_label", "prob_satisfied",
                                         "prob_dissatisfied", "risk_tier"] if c in scored.columns]
            sorted_view = scored[display_cols].sort_values("prob_dissatisfied", ascending=False)
            st.dataframe(sorted_view, use_container_width=True, height=400)

            csv_bytes = scored.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Full Scored CSV", data=csv_bytes,
                file_name=f"csat_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv", use_container_width=True,
            )

# ============================================================================
# PAGE: DATA EXPLORER
# ============================================================================
elif page == "📊 Data Explorer":
    render_hero(
        "OPERATIONAL ANALYTICS",
        "Explore the underlying support data",
        "Slice 85,907 historical Flipkart support tickets by channel, category, shift, and "
        "tenure to surface systemic drivers of dissatisfaction.",
    )

    if df_data is None:
        alert_box("Dataset not found — place Customer_support_data.csv next to app.py.", "warn")
    else:
        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1:
            f_channel = st.multiselect("Channel", lookups["channels"], default=[])
        with fc2:
            f_category = st.multiselect("Category", lookups["categories"], default=[])
        with fc3:
            f_shift = st.multiselect("Agent Shift", sorted(df_data["Agent Shift"].dropna().unique().tolist()), default=[])
        with fc4:
            f_tenure = st.multiselect("Tenure Bucket", TENURE_ORDER, default=[])

        filtered = df_data.copy()
        if f_channel:
            filtered = filtered[filtered["channel_name"].isin(f_channel)]
        if f_category:
            filtered = filtered[filtered["category"].isin(f_category)]
        if f_shift:
            filtered = filtered[filtered["Agent Shift"].isin(f_shift)]
        if f_tenure:
            filtered = filtered[filtered["Tenure Bucket"].isin(f_tenure)]

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            kpi_card("Filtered Tickets", f"{len(filtered):,}")
        with k2:
            kpi_card("Avg CSAT Score", f"{filtered['CSAT Score'].mean():.2f} / 5")
        with k3:
            sat_rate = (filtered["CSAT_label"] == 1).mean() * 100
            kpi_card("Satisfaction Rate", f"{sat_rate:.1f}%")
        with k4:
            kpi_card("Median Response Time", f"{filtered['response_time_minutes'].median():.0f} min")

        st.write("")
        tab1, tab2, tab3, tab4 = st.tabs(["Satisfaction Breakdown", "Response Time", "Agent & Supervisor", "Raw Data"])

        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                section_tag("CSAT Score Distribution")
                fig = px.histogram(filtered, x="CSAT Score", nbins=5, color="CSAT Score",
                                    color_discrete_sequence=px.colors.sequential.Blues_r)
                fig.update_layout(template=PLOTLY_TEMPLATE, height=380, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                section_tag("Satisfaction Rate by Category")
                cat_sat = filtered.groupby("category")["CSAT_label"].mean().sort_values(ascending=True) * 100
                fig = px.bar(cat_sat, orientation="h", color=cat_sat.values,
                             color_continuous_scale=[NEGATIVE, ACCENT, POSITIVE],
                             labels=dict(color="Satisfaction"))
                fig.update_layout(template=PLOTLY_TEMPLATE, height=460, coloraxis_showscale=False,
                                   xaxis_title="Satisfaction Rate (%)", yaxis_title="")
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            c1, c2 = st.columns(2)
            with c1:
                section_tag("Response Time Distribution (clipped at 500 min)")
                rt = filtered["response_time_minutes"]
                rt = rt[rt <= 500]
                fig = px.histogram(rt, nbins=50, color_discrete_sequence=[PRIMARY])
                fig.add_vline(x=filtered["response_time_minutes"].median(), line_dash="dash", line_color=ACCENT)
                fig.update_layout(template=PLOTLY_TEMPLATE, height=380, xaxis_title="Minutes", yaxis_title="Tickets",
                                   showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                section_tag("Response Time by CSAT Score")
                box_df = filtered[filtered["response_time_minutes"] <= 500]
                fig = px.box(box_df, x="CSAT Score", y="response_time_minutes",
                             color="CSAT Score", color_discrete_sequence=px.colors.sequential.Blues_r)
                fig.update_layout(template=PLOTLY_TEMPLATE, height=380, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            section_tag("Issue Volume by Hour of Day")
            hourly = filtered["issue_hour"].value_counts().sort_index().reset_index()
            hourly.columns = ["hour", "count"]
            fig = px.bar(hourly, x="hour", y="count", color_discrete_sequence=[PRIMARY])
            fig.update_layout(template=PLOTLY_TEMPLATE, height=320, xaxis_title="Hour", yaxis_title="Tickets")
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            c1, c2 = st.columns(2)
            with c1:
                section_tag("Top 10 Agents by Avg CSAT (min 20 tickets)")
                agent_stats = filtered.groupby("Agent_name")["CSAT Score"].agg(["mean", "count"])
                agent_stats = agent_stats[agent_stats["count"] >= 20].sort_values("mean", ascending=False).head(10)
                fig = px.bar(agent_stats.reset_index(), x="mean", y="Agent_name", orientation="h",
                             color="mean", color_continuous_scale=[ACCENT, POSITIVE])
                fig.update_layout(template=PLOTLY_TEMPLATE, height=420, coloraxis_showscale=False,
                                   yaxis_title="", xaxis_title="Avg CSAT Score")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                section_tag("Top 10 Supervisors by Avg Team CSAT")
                sup_stats = filtered.groupby("Supervisor")["CSAT Score"].agg(["mean", "count"])
                sup_stats = sup_stats[sup_stats["count"] >= 20].sort_values("mean", ascending=False).head(10)
                fig = px.bar(sup_stats.reset_index(), x="mean", y="Supervisor", orientation="h",
                             color="mean", color_continuous_scale=[ACCENT, PRIMARY])
                fig.update_layout(template=PLOTLY_TEMPLATE, height=420, coloraxis_showscale=False,
                                   yaxis_title="", xaxis_title="Avg CSAT Score")
                st.plotly_chart(fig, use_container_width=True)

        with tab4:
            st.dataframe(
                filtered[["channel_name", "category", "Sub-category", "Customer Remarks",
                          "Agent_name", "Supervisor", "Agent Shift", "Tenure Bucket",
                          "response_time_minutes", "CSAT Score"]].head(500),
                use_container_width=True, height=460,
            )
            st.caption("Showing first 500 filtered rows.")

# ============================================================================
# PAGE: MODEL PERFORMANCE
# ============================================================================
elif page == "🧠 Model Performance":
    render_hero(
        "MODEL TRANSPARENCY",
        "How the model was built, tuned, and evaluated",
        "Full evaluation summary from the training notebook — model comparison, "
        "hyperparameters, and (when available) saved evaluation artifacts such as "
        "confusion matrices and SHAP plots.",
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    metrics = [
        ("Accuracy", f"{MODEL_METRICS['accuracy']:.3f}"),
        ("Precision (macro)", f"{MODEL_METRICS['precision_macro']:.3f}"),
        ("Recall (macro)", f"{MODEL_METRICS['recall_macro']:.3f}"),
        ("F1 (macro)", f"{MODEL_METRICS['f1_macro']:.3f}"),
        ("ROC-AUC", f"{MODEL_METRICS['roc_auc']:.3f}"),
    ]
    for col, (label, val) in zip([c1, c2, c3, c4, c5], metrics):
        with col:
            kpi_card(label, val)

    st.write("")
    section_tag("Model Ranking (Test Set ROC-AUC)")
    rank_df = pd.DataFrame({
        "Model": ["XGBoost (Tuned) 🥇", "Logistic Regression 🥈", "Random Forest 🥉"],
        "ROC-AUC": [0.8055, 0.7947, 0.7862],
        "Type": ["Boosting", "Linear", "Bagging"],
    })
    fig = px.bar(rank_df, x="ROC-AUC", y="Model", orientation="h", color="Model",
                 color_discrete_sequence=[POSITIVE, ACCENT, PRIMARY], range_x=[0.75, 0.82])
    fig.update_layout(template=PLOTLY_TEMPLATE, height=280, showlegend=False, yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Evaluation Artifacts")

    image_groups = {
        "Model Comparison & ROC/PR": ["model_comparison.png", "roc_curve.png", "precision_recall_and_threshold.png"],
        "Confusion Matrices": ["logistic_regression_confusion_matrix_and_evaluation_metrics.png",
                                "random_forest_classification_confusion_matrix_and_evaluation_metrics.png",
                                "xgboost_classification_confusion_matrix_and_evaluation_metrics.png"],
        "Feature Importance & Explainability (SHAP)": ["top_20_feature_importance_xgboost_classifier.png",
                                                          "shap_bar_plot.png", "shap_beeswarm_plot.png",
                                                          "shap_force_plot.png"],
        "Correlation & Class Balance": ["correlation_heatmap_numerical_values.png", "class_distribution.png"],
    }

    any_images_found = False
    for group_name, files in image_groups.items():
        existing = [f for f in files if os.path.exists(os.path.join(IMAGES_DIR, f))]
        if not existing:
            continue
        any_images_found = True
        section_tag(group_name)
        cols = st.columns(min(3, len(existing)))
        for i, fname in enumerate(existing):
            with cols[i % len(cols)]:
                st.image(os.path.join(IMAGES_DIR, fname), use_container_width=True,
                          caption=fname.replace("_", " ").replace(".png", "").title())
        st.write("")

    if not any_images_found:
        alert_box(f"No saved evaluation images found in <code>{IMAGES_DIR}</code>. Export the notebook's "
                  "plots to an <code>images/</code> folder next to app.py to display them here.", "info")

    with st.expander("⚙️ Best Hyperparameters (GridSearchCV, 3-fold CV, scoring=roc_auc)"):
        hp_df = pd.DataFrame({
            "Parameter": ["n_estimators", "max_depth", "learning_rate", "scale_pos_weight",
                          "tree_method", "eval_metric"],
            "Value": [str(BEST_PARAMS["n_estimators"]), str(BEST_PARAMS["max_depth"]),
                      str(BEST_PARAMS["learning_rate"]), "class-imbalance ratio (auto-computed)",
                      BEST_PARAMS["tree_method"], BEST_PARAMS["eval_metric"]],
        })
        st.table(hp_df)

    with st.expander("📐 Feature Set Used at Inference"):
        st.markdown(f"""
- **Label-encoded categoricals:** `{'`, `'.join(ENC_FEATURE_COLS)}`
- **Numerical / target-encoded:** `{'`, `'.join(NUM_FEATURE_COLS)}`
- **Text:** TF-IDF on cleaned `Customer Remarks` — 500 features, 1–3 n-grams, `min_df=3`, `sublinear_tf=True`
- **Transform order:** PowerTransformer (Yeo-Johnson, numeric cols only) → StandardScaler
  (`with_mean=False`, all 10 structured cols) → sparse-hstack with TF-IDF matrix
        """)

# ============================================================================
# PAGE: ABOUT
# ============================================================================
elif page == "ℹ️ About":
    render_hero(
        "PROJECT INFO",
        "About this application",
        "A deployable Streamlit front-end for the Flipkart Customer Satisfaction "
        "Prediction project — combining structured ticket metadata with NLP on "
        "customer remarks to flag dissatisfaction risk before it escalates.",
    )

    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.markdown("""
#### Pipeline Summary
1. **Data Cleaning** — missing values imputed, duplicates removed, response-time outliers winsorized (IQR method).
2. **Feature Engineering** — response time, issue hour/day-of-week, ordinal tenure encoding, label-encoded categoricals, and smoothed target encoding (K=10) for agent/supervisor historical CSAT.
3. **NLP** — customer remarks cleaned (contraction expansion, lowercasing, punctuation/URL/digit removal, stopword removal, rephrasing, lemmatization) and vectorized with TF-IDF (500 features, 1–3 n-grams).
4. **Modeling** — Logistic Regression, Random Forest, and XGBoost compared; XGBoost selected after GridSearchCV tuning (ROC-AUC optimized, `scale_pos_weight` for class imbalance).
5. **Threshold Tuning** — F1-macro optimized decision threshold explored via precision-recall curve (adjustable in this app's sidebar).
6. **Explainability** — SHAP TreeExplainer used in the notebook to validate feature importances beyond native gain-based rankings.

#### Business Use Cases
- **Proactive escalation**: flag high dissatisfaction-risk tickets before a customer rates them.
- **Queue prioritization**: route the riskiest tickets to senior agents first.
- **Root-cause analysis**: use the Data Explorer to identify systemic drivers (slow categories, weak shifts, underperforming supervisors).
        """)
    with c2:
        st.markdown(f"""
#### Tech Stack
- **Model**: XGBoost Classifier (tuned)
- **NLP**: NLTK + TF-IDF
- **Scaling**: PowerTransformer (Yeo-Johnson) + StandardScaler
- **App**: Streamlit + Plotly

#### Dataset
- 85,907 Flipkart customer support interactions
- Binary target: CSAT ≥ 4 → Satisfied

#### Held-out Test Performance
- ROC-AUC: **{MODEL_METRICS['roc_auc']:.4f}**
- Accuracy: **{MODEL_METRICS['accuracy']:.4f}**
- F1 (macro): **{MODEL_METRICS['f1_macro']:.4f}**

#### Disclaimer
Flipkart is a trademark of its respective owner. This project is for
educational and portfolio purposes only and is not affiliated with
or endorsed by Flipkart.
        """)

    st.markdown("---")
    st.markdown("#### Author")
    st.markdown(
        "**Mohit Singh Rajput** — AI/ML Engineer &nbsp;·&nbsp; "
        "[LinkedIn](https://linkedin.com/in/mohitsingh1307) &nbsp;·&nbsp; "
        "[GitHub](https://github.com/Mohit-1307) &nbsp;·&nbsp; "
        "[Kaggle](https://www.kaggle.com/mohitsinghrajput1307)"
    )

st.markdown(
    """<div class="footer-note">© 2026 Flipkart CSAT Intelligence Suite ·
    Portfolio / educational project · Not affiliated with Flipkart</div>""",
    unsafe_allow_html=True,
)