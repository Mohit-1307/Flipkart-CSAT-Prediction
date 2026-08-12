"""
Flipkart Customer Satisfaction (CSAT) Intelligence Suite
==========================================================
A premium, production-grade Streamlit application for predicting and
analyzing customer satisfaction from support-interaction data, built on
top of a tuned XGBoost classifier trained on 85,907 Flipkart customer
support tickets.

Run locally:
    streamlit run app.py

Author: ML Deployment Layer for Flipkart-CSAT-Prediction
"""

import os
import re
import string
import warnings
from datetime import datetime, time as dtime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.sparse import csr_matrix, hstack

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# PAGE CONFIG  (must be the first Streamlit call)
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Flipkart CSAT Intelligence Suite",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/Mohit-1307/Flipkart-CSAT-Prediction",
        "About": "Flipkart Customer Satisfaction Prediction — XGBoost powered "
                  "decision-support tool for support operations teams.",
    },
)

# ----------------------------------------------------------------------------
# PATHS
# ----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "Customer_support_data.csv")
IMAGES_DIR = os.path.join(BASE_DIR, "images")

MODEL_PATH = os.path.join(MODELS_DIR, "best_xgboost_classifier.pkl")
TFIDF_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "standard_scaler.pkl")
PT_PATH = os.path.join(MODELS_DIR, "power_transformer.pkl")

ENC_FEATURE_COLS = ["channel_name_enc", "category_enc", "Sub-category_enc",
                     "Tenure Bucket_enc", "Agent Shift_enc"]
NUM_FEATURE_COLS = ["response_time_minutes", "issue_hour", "issue_dayofweek",
                     "agent_csat_encoded", "supervisor_csat_encoded"]
TENURE_ORDER = ["On Job Training", "0-30", "31-60", "61-90", ">90"]
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# ----------------------------------------------------------------------------
# PREMIUM THEME — CSS INJECTION
# ----------------------------------------------------------------------------
PRIMARY = "#1F4FD8"      # Flipkart-inspired royal blue, deepened for premium feel
ACCENT = "#FFC633"       # Flipkart gold/yellow accent
POSITIVE = "#17B26A"
NEGATIVE = "#F04438"
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

    section[data-testid="stSidebar"] * {{
        color: #E7ECFA;
    }}

    /* Hero header */
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
        content: "";
        position: absolute;
        top: -60px; right: -60px;
        width: 260px; height: 260px;
        background: radial-gradient(circle, rgba(255,198,51,0.18) 0%, rgba(255,198,51,0) 70%);
    }}
    .csat-hero::before {{
        content: "";
        position: absolute;
        bottom: -80px; left: 20%;
        width: 300px; height: 300px;
        background: radial-gradient(circle, rgba(31,79,216,0.28) 0%, rgba(31,79,216,0) 70%);
    }}
    .csat-eyebrow {{
        color: {ACCENT};
        font-weight: 700;
        letter-spacing: 0.14em;
        font-size: 0.72rem;
        text-transform: uppercase;
        margin-bottom: 10px;
        font-family: 'JetBrains Mono', monospace;
    }}
    .csat-title {{
        color: #F5F7FF;
        font-size: 2.15rem;
        font-weight: 800;
        margin: 0 0 8px 0;
        letter-spacing: -0.02em;
    }}
    .csat-subtitle {{
        color: {TEXT_MUTED};
        font-size: 1.02rem;
        max-width: 720px;
        line-height: 1.55;
        margin: 0;
    }}

    /* KPI cards */
    .kpi-card {{
        background: linear-gradient(160deg, {SURFACE_2} 0%, #0D1526 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 18px 20px;
        height: 100%;
        transition: border-color 0.2s ease;
    }}
    .kpi-card:hover {{ border-color: rgba(255,198,51,0.4); }}
    .kpi-label {{
        color: {TEXT_MUTED};
        font-size: 0.74rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
    }}
    .kpi-value {{
        color: #F5F7FF;
        font-size: 1.65rem;
        font-weight: 800;
        letter-spacing: -0.01em;
    }}
    .kpi-delta-pos {{ color: {POSITIVE}; font-size: 0.82rem; font-weight: 600; }}
    .kpi-delta-neg {{ color: {NEGATIVE}; font-size: 0.82rem; font-weight: 600; }}

    /* Verdict banners */
    .verdict-box {{
        border-radius: 16px;
        padding: 26px 30px;
        margin: 14px 0 20px 0;
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
    .verdict-label {{
        font-size: 1.35rem;
        font-weight: 800;
        color: #F5F7FF;
        margin-bottom: 4px;
    }}
    .verdict-sub {{
        color: {TEXT_MUTED};
        font-size: 0.92rem;
    }}

    .section-tag {{
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: {ACCENT};
        border: 1px solid rgba(255,198,51,0.35);
        border-radius: 999px;
        padding: 4px 12px;
        margin-bottom: 10px;
    }}

    .factor-chip {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 10px 14px;
        margin: 4px 6px 4px 0;
        font-size: 0.85rem;
        color: #E7ECFA;
    }}

    div[data-testid="stMetric"] {{
        background: linear-gradient(160deg, {SURFACE_2} 0%, #0D1526 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 14px 16px;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        background: rgba(255,255,255,0.03);
        padding: 6px;
        border-radius: 14px;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 10px;
        color: {TEXT_MUTED};
        font-weight: 600;
        padding: 8px 18px;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(120deg, {PRIMARY}, #3667E8) !important;
        color: white !important;
    }}

    .stButton > button {{
        background: linear-gradient(120deg, {PRIMARY} 0%, #3667E8 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        padding: 0.6rem 1.4rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}
    .stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 8px 20px rgba(31,79,216,0.35);
    }}

    .footer-note {{
        text-align: center;
        color: {TEXT_MUTED};
        font-size: 0.78rem;
        padding: 24px 0 10px 0;
        border-top: 1px solid rgba(255,255,255,0.06);
        margin-top: 30px;
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

# ----------------------------------------------------------------------------
# CACHED LOADERS
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model artifacts...")
def load_artifacts():
    """Load the trained XGBoost model and fitted preprocessors."""
    missing = [p for p in [MODEL_PATH, TFIDF_PATH, SCALER_PATH, PT_PATH] if not os.path.exists(p)]
    if missing:
        return None, None, None, None, missing
    model = joblib.load(MODEL_PATH)
    tfidf = joblib.load(TFIDF_PATH)
    scaler = joblib.load(SCALER_PATH)
    pt = joblib.load(PT_PATH)
    return model, tfidf, scaler, pt, []


@st.cache_data(show_spinner="Loading historical support data...")
def load_data():
    """Load raw dataset and compute the encoding lookup tables the model relies on."""
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

    # Winsorize response time (same IQR capping as training)
    q1, q3 = df1["response_time_minutes"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df1["response_time_minutes"] = df1["response_time_minutes"].clip(lower=lo, upper=hi)

    # Agent / Supervisor smoothed target-encoding lookups (K=10 smoothing, identical to training)
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


# ----------------------------------------------------------------------------
# TEXT CLEANING PIPELINE (mirrors notebook exactly)
# ----------------------------------------------------------------------------
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
            nltk.download(pkg, quiet=True)
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
    """Reproduces the notebook's exact NLP cleaning sequence for a single remark."""
    ensure_nltk()
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    text = str(raw_text) if raw_text and str(raw_text).strip() else "no remarks"
    text = expand_contractions(text)
    text = remove_punctuation(text)
    text = clean_text_urls_digits(text)

    stop_words = set(stopwords.words("english"))
    words = text.split()
    text = " ".join([w for w in words if w.lower() not in stop_words])
    text = text.strip() if text.strip() else "no remarks"
    text = rephrase_text(text)

    lemmatizer = WordNetLemmatizer()
    tokens = text.split()
    lemmatized = [lemmatizer.lemmatize(w) for w in tokens]
    clean = " ".join(lemmatized).strip()
    return clean if clean else "no remarks"


# ----------------------------------------------------------------------------
# INFERENCE
# ----------------------------------------------------------------------------
def predict_single(record: dict, model, tfidf, scaler, pt, label_maps, lookups):
    """
    record keys required:
      channel_name, category, sub_category, tenure_bucket, agent_shift,
      response_time_minutes, issue_hour, issue_dayofweek,
      agent_name (optional), supervisor (optional), customer_remarks
    """
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
    pred = int(proba[1] >= 0.5)
    return {
        "prediction": pred,
        "prob_satisfied": float(proba[1]),
        "prob_dissatisfied": float(proba[0]),
        "clean_remark": clean_remark,
    }


def predict_batch(df_input: pd.DataFrame, model, tfidf, scaler, pt, label_maps, lookups) -> pd.DataFrame:
    """Vectorized batch prediction for an uploaded CSV of tickets."""
    df = df_input.copy()

    rename_guess = {
        "channel_name": "channel_name", "category": "category", "Sub-category": "sub_category",
        "Tenure Bucket": "tenure_bucket", "Agent Shift": "agent_shift",
    }
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
    df["predicted_label"] = np.where(proba[:, 1] >= 0.5, "Satisfied", "Dissatisfied")
    df["prob_satisfied"] = proba[:, 1]
    df["prob_dissatisfied"] = proba[:, 0]
    df["risk_tier"] = pd.cut(
        df["prob_dissatisfied"], bins=[-0.01, 0.3, 0.6, 1.01],
        labels=["Low Risk", "Medium Risk", "High Risk"]
    )
    return df


# ----------------------------------------------------------------------------
# SMALL UI HELPERS
# ----------------------------------------------------------------------------
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


def gauge_chart(prob_satisfied: float):
    color = POSITIVE if prob_satisfied >= 0.5 else NEGATIVE
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_satisfied * 100,
        number={"suffix": "%", "font": {"size": 44, "color": "#F5F7FF"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": TEXT_MUTED, "tickfont": {"color": TEXT_MUTED}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(255,255,255,0.04)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "rgba(240,68,56,0.18)"},
                {"range": [40, 65], "color": "rgba(255,198,51,0.14)"},
                {"range": [65, 100], "color": "rgba(23,178,106,0.16)"},
            ],
            "threshold": {"line": {"color": "white", "width": 3}, "thickness": 0.8, "value": 50},
        },
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(template=PLOTLY_TEMPLATE, height=280, margin=dict(l=20, r=20, t=10, b=10))
    return fig


# ----------------------------------------------------------------------------
# LOAD EVERYTHING
# ----------------------------------------------------------------------------
model, tfidf, scaler, pt, missing_artifacts = load_artifacts()
df1, lookups = load_data()
label_maps = build_label_maps(df1) if df1 is not None else None

ARTIFACTS_OK = model is not None and df1 is not None

# ----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------------------------
with st.sidebar:
    logo_path = os.path.join(IMAGES_DIR, "flipkart_logo.png")
    if os.path.exists(logo_path):
        st.image(logo_path, width=140)
    st.markdown("### CSAT Intelligence Suite")
    st.caption("XGBoost · TF-IDF · SHAP-explained")

    page = st.radio(
        "Navigate",
        ["🏠 Overview", "🎯 Predict Single Ticket", "📂 Batch Scoring",
         "📊 Data Explorer", "🧠 Model Performance", "ℹ️ About"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    if ARTIFACTS_OK:
        st.success("Model & data loaded", icon="✅")
        st.caption(f"Training records: **{len(df1):,}**")
        sat_rate = df1["CSAT_label"].mean()
        st.caption(f"Historical satisfaction rate: **{sat_rate:.1%}**")
    else:
        st.error("Artifacts missing", icon="⚠️")
        if missing_artifacts:
            for m in missing_artifacts:
                st.caption(f"Missing: `{os.path.basename(m)}`")

    st.markdown("---")
    st.caption("Built for support operations · ROC-AUC ≈ 0.78")

# ----------------------------------------------------------------------------
# HERO
# ----------------------------------------------------------------------------
def render_hero(eyebrow, title, subtitle):
    st.markdown(
        f"""<div class="csat-hero">
                <div class="csat-eyebrow">{eyebrow}</div>
                <div class="csat-title">{title}</div>
                <p class="csat-subtitle">{subtitle}</p>
            </div>""",
        unsafe_allow_html=True,
    )


if not ARTIFACTS_OK:
    render_hero(
        "SETUP REQUIRED",
        "Model artifacts not found",
        "Place the `models/` folder (best_xgboost_classifier.pkl, tfidf_vectorizer.pkl, "
        "standard_scaler.pkl, power_transformer.pkl) and `Customer_support_data.csv` "
        "alongside this app.py file, then redeploy.",
    )
    st.stop()

# ============================================================================
# PAGE: OVERVIEW
# ============================================================================
if page == "🏠 Overview":
    render_hero(
        "FLIPKART · CUSTOMER SUPPORT ANALYTICS",
        "Predict, prioritize, and understand customer satisfaction",
        "A decision-support layer over Flipkart's support pipeline. Score individual "
        "tickets in real time, batch-score entire queues, and explore the exact drivers "
        "of satisfaction and churn risk across 85,907 historical interactions.",
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Total Tickets Analyzed", f"{len(df1):,}", "Full training corpus")
    with c2:
        kpi_card("Historical CSAT Rate", f"{df1['CSAT_label'].mean():.1%}", "Score ≥ 4 = Satisfied")
    with c3:
        kpi_card("Model ROC-AUC", "0.780", "Tuned XGBoost, test set")
    with c4:
        kpi_card("Median Response Time", f"{df1['response_time_minutes'].median():.0f} min", "Issue → first response")

    st.write("")
    left, right = st.columns([1.35, 1])

    with left:
        section_tag("Satisfaction by Issue Category")
        cat_sat = (df1.groupby("category")["CSAT_label"].mean().sort_values(ascending=True) * 100).reset_index()
        cat_sat.columns = ["category", "satisfaction_pct"]
        fig = px.bar(
            cat_sat, x="satisfaction_pct", y="category", orientation="h",
            color="satisfaction_pct", color_continuous_scale=[NEGATIVE, ACCENT, POSITIVE],
            labels={"satisfaction_pct": "Satisfaction %", "category": ""},
        )
        fig.update_layout(template=PLOTLY_TEMPLATE, height=420, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        section_tag("Overall Class Split")
        split = df1["CSAT_label"].value_counts().rename({0: "Dissatisfied", 1: "Satisfied"})
        fig = px.pie(
            values=split.values, names=split.index, hole=0.62,
            color=split.index, color_discrete_map={"Satisfied": POSITIVE, "Dissatisfied": NEGATIVE},
        )
        fig.update_traces(textinfo="percent+label", textfont_size=13)
        fig.update_layout(template=PLOTLY_TEMPLATE, height=420, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        section_tag("Support Channel Mix")
        ch = df1["channel_name"].value_counts().reset_index()
        ch.columns = ["channel", "count"]
        fig = px.bar(ch, x="channel", y="count", color="channel",
                      color_discrete_sequence=[PRIMARY, ACCENT, "#8E7CFF"])
        fig.update_layout(template=PLOTLY_TEMPLATE, height=340, showlegend=False,
                           xaxis_title="", yaxis_title="Tickets")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        section_tag("Issue Volume by Hour of Day")
        hourly = df1["issue_hour"].value_counts().sort_index().reset_index()
        hourly.columns = ["hour", "count"]
        fig = px.area(hourly, x="hour", y="count", markers=True)
        fig.update_traces(line_color=PRIMARY, fillcolor="rgba(31,79,216,0.22)")
        fig.update_layout(template=PLOTLY_TEMPLATE, height=340, xaxis_title="Hour", yaxis_title="Tickets")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"""<div class="footer-note">Flipkart CSAT Intelligence Suite · Educational/portfolio project,
        not affiliated with Flipkart · Model: tuned XGBoost (ROC-AUC 0.78)</div>""",
        unsafe_allow_html=True,
    )

# ============================================================================
# PAGE: PREDICT SINGLE TICKET
# ============================================================================
elif page == "🎯 Predict Single Ticket":
    render_hero(
        "REAL-TIME SCORING",
        "Predict satisfaction for a single ticket",
        "Fill in the ticket details a support agent would have on hand. The model returns "
        "a satisfaction probability instantly, using the same feature pipeline as training.",
    )

    with st.form("single_predict_form"):
        st.markdown("#### Ticket Details")
        col1, col2, col3 = st.columns(3)
        with col1:
            channel_name = st.selectbox("Support Channel", lookups["channels"])
            category = st.selectbox("Issue Category", lookups["categories"])
        with col2:
            sub_options = lookups["subcategory_by_category"].get(category, [])
            sub_category = st.selectbox("Sub-Category", sub_options if sub_options else ["General Enquiry"])
            tenure_bucket = st.selectbox("Agent Tenure Bucket", TENURE_ORDER, index=2)
        with col3:
            agent_shift = st.selectbox("Agent Shift", ["Morning", "Afternoon", "Evening", "Night", "Split"])
            issue_dow_name = st.selectbox("Day Issue Reported", DAY_NAMES, index=2)

        st.markdown("#### Timing")
        col4, col5, col6 = st.columns(3)
        with col4:
            issue_time = st.time_input("Issue Reported Time", value=dtime(10, 0))
        with col5:
            response_minutes = st.slider(
                "Response Time (minutes)", min_value=0, max_value=1440,
                value=int(min(lookups["response_time_median"], 120)), step=5,
                help="Time between issue being reported and agent's first response.",
            )
        with col6:
            st.metric("Historical Median Response", f"{lookups['response_time_median']:.0f} min")

        st.markdown("#### Agent / Supervisor (optional — improves accuracy)")
        col7, col8 = st.columns(2)
        with col7:
            agent_name = st.selectbox(
                "Agent Name", ["(Unknown / New Agent)"] + lookups["agents"],
                help="If known, uses the agent's historical average CSAT as a signal.",
            )
        with col8:
            supervisor = st.selectbox(
                "Supervisor", ["(Unknown)"] + lookups["supervisors"],
            )

        st.markdown("#### Customer Remarks")
        customer_remarks = st.text_area(
            "What did the customer say?",
            placeholder="e.g. 'Product delivered late and packaging was damaged, very disappointed'",
            height=100,
        )

        submitted = st.form_submit_button("🔮 Predict Satisfaction", use_container_width=True)

    if submitted:
        record = {
            "channel_name": channel_name,
            "category": category,
            "sub_category": sub_category,
            "tenure_bucket": tenure_bucket,
            "agent_shift": agent_shift,
            "response_time_minutes": float(response_minutes),
            "issue_hour": issue_time.hour,
            "issue_dayofweek": DAY_NAMES.index(issue_dow_name),
            "agent_name": None if agent_name == "(Unknown / New Agent)" else agent_name,
            "supervisor": None if supervisor == "(Unknown)" else supervisor,
            "customer_remarks": customer_remarks,
        }

        with st.spinner("Scoring ticket..."):
            result = predict_single(record, model, tfidf, scaler, pt, label_maps, lookups)

        st.markdown("---")
        is_satisfied = result["prediction"] == 1

        if is_satisfied:
            st.markdown(
                f"""<div class="verdict-box verdict-satisfied">
                        <div class="verdict-label">✅ Likely Satisfied</div>
                        <div class="verdict-sub">Model confidence: {result['prob_satisfied']:.1%} — this ticket looks
                        low-risk based on historical patterns.</div>
                    </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""<div class="verdict-box verdict-risk">
                        <div class="verdict-label">⚠️ Dissatisfaction Risk</div>
                        <div class="verdict-sub">Model confidence: {result['prob_dissatisfied']:.1%} — recommend
                        prioritizing this ticket for proactive follow-up.</div>
                    </div>""",
                unsafe_allow_html=True,
            )

        colA, colB = st.columns([1, 1.4])
        with colA:
            st.plotly_chart(gauge_chart(result["prob_satisfied"]), use_container_width=True)
        with colB:
            st.markdown("#### Contributing Signals")
            chips = []
            if response_minutes > lookups["response_time_p90"]:
                chips.append(("🔴", f"Response time ({response_minutes} min) is in the slowest 10% historically"))
            elif response_minutes <= lookups["response_time_median"]:
                chips.append(("🟢", f"Response time ({response_minutes} min) is at/below the historical median"))
            else:
                chips.append(("🟡", f"Response time ({response_minutes} min) is above median"))

            cat_sat_rate = df1[df1["category"] == category]["CSAT_label"].mean()
            if cat_sat_rate < df1["CSAT_label"].mean() - 0.05:
                chips.append(("🔴", f"'{category}' historically under-performs on satisfaction ({cat_sat_rate:.0%})"))
            elif cat_sat_rate > df1["CSAT_label"].mean() + 0.05:
                chips.append(("🟢", f"'{category}' historically over-performs on satisfaction ({cat_sat_rate:.0%})"))

            if record["agent_name"]:
                a_score = lookups["agent_lookup"].get(record["agent_name"], lookups["global_mean"])
                if a_score >= lookups["global_mean"]:
                    chips.append(("🟢", f"Agent's historical avg CSAT ({a_score:.2f}) is above global average"))
                else:
                    chips.append(("🔴", f"Agent's historical avg CSAT ({a_score:.2f}) is below global average"))

            if customer_remarks.strip():
                neg_words = {"late", "damaged", "worst", "bad", "disappointed", "delay", "refund",
                             "not received", "poor", "rude", "cancel", "fraud", "undelivered"}
                if any(w in customer_remarks.lower() for w in neg_words):
                    chips.append(("🔴", "Customer remarks contain negative sentiment keywords"))
                else:
                    chips.append(("🟢", "No strong negative keywords detected in remarks"))

            for icon, txt in chips:
                st.markdown(f'<div class="factor-chip">{icon} {txt}</div>', unsafe_allow_html=True)

            with st.expander("View processed text features"):
                st.code(result["clean_remark"] or "(empty — treated as 'no remarks')", language=None)

# ============================================================================
# PAGE: BATCH SCORING
# ============================================================================
elif page == "📂 Batch Scoring":
    render_hero(
        "BULK OPERATIONS",
        "Score an entire queue of tickets at once",
        "Upload a CSV in the same schema as the Flipkart support export. The app engineers "
        "identical features to training and returns a ranked, exportable risk report.",
    )

    with st.expander("📋 Expected CSV columns (missing ones are safely defaulted)", expanded=False):
        st.code(
            "channel_name, category, Sub-category, Tenure Bucket, Agent Shift,\n"
            "Issue_reported at, issue_responded, Agent_name, Supervisor, Customer Remarks",
            language=None,
        )
        st.caption("You can also use your original `Customer_support_data.csv` schema directly.")

    uploaded = st.file_uploader("Upload ticket CSV", type=["csv"])

    demo_col1, demo_col2 = st.columns([1, 3])
    with demo_col1:
        use_sample = st.button("Use sample of historical data instead", use_container_width=True)

    source_df = None
    if uploaded is not None:
        try:
            source_df = pd.read_csv(uploaded)
            st.success(f"Loaded {len(source_df):,} rows from upload.")
        except Exception as e:
            st.error(f"Could not read file: {e}")
    elif use_sample:
        source_df = df1.sample(n=min(200, len(df1)), random_state=42).drop(
            columns=["CSAT_label", "agent_csat_encoded", "supervisor_csat_encoded"], errors="ignore"
        )
        st.info("Using a random sample of 200 historical tickets for demonstration.")

    if source_df is not None:
        st.markdown("#### Preview")
        st.dataframe(source_df.head(10), use_container_width=True, height=220)

        if st.button("🚀 Run Batch Prediction", type="primary", use_container_width=True):
            with st.spinner(f"Scoring {len(source_df):,} tickets..."):
                scored = predict_batch(source_df, model, tfidf, scaler, pt, label_maps, lookups)
            st.session_state["batch_scored"] = scored

    if "batch_scored" in st.session_state:
        scored = st.session_state["batch_scored"]
        st.markdown("---")
        st.markdown("### Results")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card("Tickets Scored", f"{len(scored):,}")
        with c2:
            kpi_card("Predicted Satisfied", f"{(scored['predicted_label']=='Satisfied').mean():.1%}")
        with c3:
            kpi_card("High Risk Tickets", f"{(scored['risk_tier']=='High Risk').sum():,}")
        with c4:
            kpi_card("Avg. Dissatisfaction Prob.", f"{scored['prob_dissatisfied'].mean():.1%}")

        colL, colR = st.columns([1, 1])
        with colL:
            section_tag("Risk Tier Distribution")
            tier_counts = scored["risk_tier"].value_counts().reindex(["Low Risk", "Medium Risk", "High Risk"]).fillna(0)
            fig = px.bar(
                x=tier_counts.index, y=tier_counts.values,
                color=tier_counts.index,
                color_discrete_map={"Low Risk": POSITIVE, "Medium Risk": ACCENT, "High Risk": NEGATIVE},
            )
            fig.update_layout(template=PLOTLY_TEMPLATE, height=320, showlegend=False,
                               xaxis_title="", yaxis_title="Tickets")
            st.plotly_chart(fig, use_container_width=True)
        with colR:
            section_tag("Predicted Probability Distribution")
            fig = px.histogram(scored, x="prob_satisfied", nbins=30, color_discrete_sequence=[PRIMARY])
            fig.add_vline(x=0.5, line_dash="dash", line_color=ACCENT)
            fig.update_layout(template=PLOTLY_TEMPLATE, height=320,
                               xaxis_title="P(Satisfied)", yaxis_title="Tickets")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 🔥 Highest-Risk Tickets (prioritize these first)")
        priority_cols = [c for c in ["Unique id", "category", "sub_category", "channel_name",
                                      "Agent_name", "Customer Remarks", "prob_dissatisfied",
                                      "predicted_label", "risk_tier"] if c in scored.columns]
        top_risk = scored.sort_values("prob_dissatisfied", ascending=False)[priority_cols].head(25)
        st.dataframe(
            top_risk.style.background_gradient(subset=["prob_dissatisfied"], cmap="Reds"),
            use_container_width=True, height=400,
        )

        st.markdown("#### Full Results")
        st.dataframe(scored, use_container_width=True, height=350)

        csv_bytes = scored.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Full Results (CSV)", data=csv_bytes,
            file_name=f"csat_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv", use_container_width=True,
        )

# ============================================================================
# PAGE: DATA EXPLORER
# ============================================================================
elif page == "📊 Data Explorer":
    render_hero(
        "EXPLORATORY ANALYTICS",
        "Explore the historical support dataset",
        "Slice 85,907 Flipkart support interactions by channel, category, agent shift, "
        "and more to understand what drives satisfaction outcomes.",
    )

    with st.expander("🔍 Filters", expanded=True):
        f1, f2, f3 = st.columns(3)
        with f1:
            channel_filter = st.multiselect("Channel", lookups["channels"], default=lookups["channels"])
        with f2:
            category_filter = st.multiselect("Category", lookups["categories"], default=lookups["categories"])
        with f3:
            shift_filter = st.multiselect(
                "Agent Shift", sorted(df1["Agent Shift"].dropna().unique().tolist()),
                default=sorted(df1["Agent Shift"].dropna().unique().tolist()),
            )

    filtered = df1[
        df1["channel_name"].isin(channel_filter)
        & df1["category"].isin(category_filter)
        & df1["Agent Shift"].isin(shift_filter)
    ]

    st.caption(f"Showing **{len(filtered):,}** of {len(df1):,} tickets")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Filtered Tickets", f"{len(filtered):,}")
    with c2:
        kpi_card("Satisfaction Rate", f"{filtered['CSAT_label'].mean():.1%}" if len(filtered) else "—")
    with c3:
        kpi_card("Median Response Time", f"{filtered['response_time_minutes'].median():.0f} min" if len(filtered) else "—")
    with c4:
        kpi_card("Avg CSAT Score", f"{filtered['CSAT Score'].mean():.2f} / 5" if len(filtered) else "—")

    tab1, tab2, tab3, tab4 = st.tabs(["Satisfaction Drivers", "Response Time", "Agents & Supervisors", "Raw Data"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            section_tag("Satisfaction by Sub-Category (Top 15 by volume)")
            top_sub = filtered["Sub-category"].value_counts().head(15).index
            sub_sat = (filtered[filtered["Sub-category"].isin(top_sub)]
                       .groupby("Sub-category")["CSAT_label"].mean().sort_values() * 100).reset_index()
            fig = px.bar(sub_sat, x="CSAT_label", y="Sub-category", orientation="h",
                         color="CSAT_label", color_continuous_scale=[NEGATIVE, ACCENT, POSITIVE],
                         labels={"CSAT_label": "Satisfaction %"})
            fig.update_layout(template=PLOTLY_TEMPLATE, height=460, coloraxis_showscale=False, yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            section_tag("Satisfaction by Channel x Shift")
            pivot = filtered.pivot_table(index="Agent Shift", columns="channel_name",
                                          values="CSAT_label", aggfunc="mean")
            fig = px.imshow(pivot, color_continuous_scale=[NEGATIVE, ACCENT, POSITIVE], aspect="auto",
                             labels=dict(color="Satisfaction"))
            fig.update_layout(template=PLOTLY_TEMPLATE, height=460)
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
        "Full evaluation artifacts from the training notebook — model comparison, "
        "confusion matrices, ROC/PR curves, and SHAP-based explainability.",
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    metrics = [("Accuracy", "0.726"), ("Precision (macro)", "0.632"), ("Recall (macro)", "0.701"),
               ("F1 (macro)", "0.638"), ("ROC-AUC", "0.780")]
    for col, (label, val) in zip([c1, c2, c3, c4, c5], metrics):
        with col:
            kpi_card(label, val)

    st.write("")
    section_tag("Model Ranking")
    rank_df = pd.DataFrame({
        "Model": ["XGBoost 🥇", "Random Forest 🥈", "Logistic Regression 🥉"],
        "ROC-AUC": [0.780, 0.761, 0.757],
        "Type": ["Boosting", "Bagging", "Linear"],
    })
    fig = px.bar(rank_df, x="ROC-AUC", y="Model", orientation="h", color="Model",
                 color_discrete_sequence=[POSITIVE, ACCENT, PRIMARY], range_x=[0.7, 0.8])
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

    for group_name, files in image_groups.items():
        existing = [f for f in files if os.path.exists(os.path.join(IMAGES_DIR, f))]
        if not existing:
            continue
        section_tag(group_name)
        cols = st.columns(min(3, len(existing)))
        for i, fname in enumerate(existing):
            with cols[i % len(cols)]:
                st.image(os.path.join(IMAGES_DIR, fname), use_container_width=True,
                          caption=fname.replace("_", " ").replace(".png", "").title())
        st.write("")

    with st.expander("⚙️ Hyperparameters"):
        hp_df = pd.DataFrame({
            "Parameter": ["n_estimators", "max_depth", "learning_rate", "scale_pos_weight", "tree_method"],
            "Value": ["300", "4–6 (tuned)", "0.05–0.1 (tuned)", "class-imbalance ratio", "hist"],
        })
        st.table(hp_df)

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
2. **Feature Engineering** — response time, issue hour/day-of-week, ordinal tenure encoding, label-encoded categoricals, and smoothed target encoding for agent/supervisor historical CSAT.
3. **NLP** — customer remarks cleaned (contraction expansion, punctuation/URL/digit removal, stopword removal, lemmatization) and vectorized with TF-IDF (500 features, 1–3 n-grams).
4. **Modeling** — Logistic Regression, Random Forest, and XGBoost compared; XGBoost selected after GridSearchCV tuning (ROC-AUC optimized, `scale_pos_weight` for class imbalance).
5. **Explainability** — SHAP TreeExplainer used to validate feature importances beyond native gain-based rankings.

#### Business Use Cases
- **Proactive escalation**: flag high dissatisfaction-risk tickets before a customer rates them.
- **Queue prioritization**: route the riskiest tickets to senior agents first.
- **Root-cause analysis**: use the Data Explorer to identify systemic drivers (slow categories, weak shifts, underperforming supervisors).
        """)
    with c2:
        st.markdown("""
#### Tech Stack
- **Model**: XGBoost Classifier
- **NLP**: NLTK + TF-IDF
- **Scaling**: PowerTransformer (Yeo-Johnson) + StandardScaler
- **App**: Streamlit + Plotly

#### Dataset
- 85,907 Flipkart customer support interactions
- Binary target: CSAT ≥ 4 → Satisfied

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