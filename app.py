"""
Flipkart CSAT Intelligence Suite
Single-file Streamlit app for scoring and analyzing customer support tickets.

Expects, next to this file:
    models/best_xgboost_classifier.pkl
    models/tfidf_vectorizer.pkl
    models/standard_scaler.pkl
    models/power_transformer.pkl
    models/label_encoders.pkl
    Customer_support_data.csv

Loads the model, TF-IDF vectorizer, scaler, power transformer, and label
encoders that were fit and saved from the training notebook -- no separate
training script is needed at runtime, just the files listed above.
"""

import os
import re
import string
import warnings
from dataclasses import dataclass, field
from datetime import datetime, time as dtime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.sparse import csr_matrix, hstack

warnings.filterwarnings("ignore")

# paths + constants

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "customer_support_data.csv")

MODEL_PATH = os.path.join(MODELS_DIR, "best_xgboost_classifier.pkl")
TFIDF_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "standard_scaler.pkl")
PT_PATH = os.path.join(MODELS_DIR, "power_transformer.pkl")
LE_PATH = os.path.join(MODELS_DIR, "label_encoders.pkl")

ENC_FEATURE_COLS = [
    "channel_name_enc",
    "category_enc",
    "Sub-category_enc",
    "Tenure Bucket_enc",
    "Agent Shift_enc",
]

NUM_FEATURE_COLS = [
    "response_time_minutes",
    "issue_hour",
    "issue_dayofweek",
    "agent_csat_encoded",
    "supervisor_csat_encoded",
]

TENURE_ORDER = ["On Job Training", "0-30", "31-60", "61-90", ">90"]
TENURE_INDEX = {t: i for i, t in enumerate(TENURE_ORDER)}
DAY_NAMES = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
LABEL_ENCODED_COLS = ["channel_name", "category", "Sub-category", "Agent Shift"]

# F1-macro-optimal threshold found via a precision/recall sweep on a genuinely
# held-out test set (see training notebook).
#
# How this threshold works: predict Satisfied if P(satisfied) >= 0.33.
# Because the bar for the Satisfied label is lower, MORE tickets receive it,
# which means FEWER dissatisfied tickets are flagged.
#
# Trade-off: this threshold maximises overall macro-F1 balance but REDUCES
# Dissatisfied recall to 0.45 vs. 0.72 at the default 0.5. If the priority
# is "catch as many at-risk tickets as possible" rather than balanced F1,
# a higher threshold (closer to 0.5) is the better operating point.
DECISION_THRESHOLD = 0.33
TARGET_ENCODING_K = 10

# Scalar metrics from the full 17,182-row held-out test set, evaluated at
# DECISION_THRESHOLD. Fit entirely on the training fold; nothing here is
# derived from data the model was scored on.
REPORTED_METRICS = {
    "accuracy": 0.839,
    "precision_macro": 0.719,
    "recall_macro": 0.688,
    "f1_macro": 0.701,
    "roc_auc": 0.806,
}

# Confusion matrix from a fixed 3,000-row subsample of the test set (seed=42,
# scored at DECISION_THRESHOLD). Used only for the visual confusion matrix on
# the Model Performance page to avoid re-scoring on every load.
# NOTE: subsample accuracy ≈ 0.832, which differs slightly from the full-test
# accuracy of 0.839 shown in REPORTED_METRICS above — both are correct for
# their respective populations and are labelled separately in the UI.
STATIC_CONFUSION = {"tp": 2261, "fp": 310, "tn": 235, "fn": 194}

# theme

DARK = {
    "bg": "#12100e",
    "bg_glow_1": "#2a1912",
    "bg_glow_2": "#17203a",
    "panel": "#1e1a15",
    "panel_alt": "#211e1a",
    "panel_raised": "#211e1a",
    "border": "#332e28",
    "border_soft": "#332e28",
    "text": "#f7f3ec",
    "text_dim": "#c9c1b4",
    "text_faint": "#8e8577",
    "accent": "#ff5a3c",
    "accent_bright": "#ff7a5c",
    "accent_soft": "#3a2018",
    "accent_border": "rgba(255, 90, 60, 0.28)",
    "good": "#3ddc84",
    "bad": "#ff5566",
    "warn": "#ffb84d",
    "chart_2": "#5eb4ff",
    "chart_3": "#c084fc",
}

LIGHT = {
    "bg": "#faf3e6",
    "bg_glow_1": "#fbe9d3",
    "bg_glow_2": "#ffe3d8",
    "panel": "#f5ecda",
    "panel_alt": "#ffffff",
    "panel_raised": "#ffffff",
    "border": "#e8dfcd",
    "border_soft": "#ece3d2",
    "text": "#161513",
    "text_dim": "#514a3d",
    "text_faint": "#8d8271",
    "accent": "#e04726",
    "accent_bright": "#c93d1f",
    "accent_soft": "#fbe4dc",
    "accent_border": "rgba(224, 71, 38, 0.22)",
    "good": "#12946a",
    "bad": "#d1364a",
    "warn": "#c67c00",
    "chart_2": "#1d6fbf",
    "chart_3": "#8b3fd1",
}


def inject_css(t):
    st.markdown(
        f"""
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&family=Source+Serif+4:wght@600;700&display=swap');

    :root {{
        --font-primary: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', sans-serif;
        --font-serif: 'Source Serif 4', Georgia, serif;
        --font-mono: 'JetBrains Mono', monospace;
        --radius-xs: 4px;
        --radius-sm: 6px;
        --radius-md: 8px;
        --radius-lg: 10px;
        --radius-xl: 50px;
        --motion-fast: 200ms;
        --motion-normal: 250ms;
    }}

    html, body, [class*="css"] {{
        font-family: var(--font-primary);
    }}

    .stApp {{
        background:
            radial-gradient(circle at 15% 0%, {t['bg_glow_1']} 0%, transparent 35%),
            radial-gradient(circle at 100% 20%, {t['bg_glow_2']} 0%, transparent 40%),
            {t['bg']} !important;
        color: {t['text']} !important;
    }}

    [data-testid="stMain"], [data-testid="stAppViewContainer"],
    [data-testid="stMainBlockContainer"], .main {{
        background-color: {t['bg']} !important;
        color: {t['text']} !important;
    }}

    footer {{visibility: hidden;}}

    #MainMenu {{visibility: hidden;}}

    header[data-testid="stHeader"] {{
        background: transparent;
        box-shadow: none;
        height: 3.2rem !important;
        min-height: 3.2rem !important;
        z-index: 999999 !important;
    }}

    div[data-testid="stToolbar"] {{
        visibility: hidden !important;
    }}

    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {{
        visibility: visible !important;
        display: flex !important;
        opacity: 1 !important;
        position: fixed !important;
        top: 12px !important;
        left: 12px !important;
        z-index: 999999 !important;
    }}

    button[data-testid="stSidebarCollapsedControl"],
    button[data-testid="baseButton-headerNoPadding"],
    button[data-testid="stBaseButton-headerNoPadding"] {{
        visibility: visible !important;
        display: flex !important;
        align-items: center;
        justify-content: center;
        color: {t['accent']} !important;
        background: {t['panel_alt']} !important;
        border: 1px solid {t['accent']} !important;
        border-radius: var(--radius-lg) !important;
        width: 42px !important;
        height: 42px !important;
        box-shadow: 0 0 14px {t['accent_border']} !important;
        transition: box-shadow var(--motion-fast) ease, transform var(--motion-fast) ease;
    }}

    button[data-testid="stSidebarCollapsedControl"]:hover,
    button[data-testid="baseButton-headerNoPadding"]:hover,
    button[data-testid="stBaseButton-headerNoPadding"]:hover {{
        box-shadow: 0 0 22px {t['accent_border']} !important;
        transform: translateY(-1px);
    }}

    button[data-testid="stSidebarCollapsedControl"] svg,
    button[data-testid="baseButton-headerNoPadding"] svg,
    button[data-testid="stBaseButton-headerNoPadding"] svg {{
        fill: {t['accent']} !important;
        color: {t['accent']} !important;
        width: 22px !important;
        height: 22px !important;
    }}

    section[data-testid="stSidebar"] {{
        background: {t['panel']} !important;
        border-right: 1px solid {t['border']};
    }}
    section[data-testid="stSidebar"] * {{
        color: {t['text']};
    }}
    section[data-testid="stSidebar"] > div {{ padding-top: 0.4rem; }}

    section[data-testid="stSidebar"] div[role="radiogroup"] {{
        gap: 2px;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label {{
        padding: 9px 12px;
        border-radius: var(--radius-md);
        transition: background 0.12s ease;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
        background: {t['accent_soft']};
    }}

    h1, h2, h3, h4 {{
        font-family: var(--font-serif);
        font-weight: 700;
        letter-spacing: -0.01em;
        color: {t['text']} !important;
    }}
    p, span, label, div {{ color: {t['text']}; }}

    .block-container {{ padding-top: 1.6rem; padding-bottom: 3.5rem; max-width: 1280px; }}

    hr {{ border-color: {t['border']} !important; margin: 28px 0 !important; }}

    /* page header */
    .page-eyebrow {{
        font-family: var(--font-mono);
        color: {t['accent']};
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        margin-bottom: 6px;
        display: block;
    }}
    .page-title {{
        font-family: var(--font-serif);
        font-size: 1.9rem;
        font-weight: 800;
        color: {t['text']};
        margin: 0 0 6px 0;
        line-height: 1.2;
    }}
    .page-sub {{
        color: {t['text_dim']};
        font-size: 0.96rem;
        margin: 0 0 28px 0;
        max-width: 640px;
        line-height: 1.55;
    }}

    /* stat / info cards */
    .card {{
        background: {t['panel_alt']};
        border: 1px solid {t['border']};
        border-radius: var(--radius-lg);
        padding: 20px 22px;
        height: 100%;
        transition: border-color var(--motion-fast) ease;
    }}
    .card:hover {{
        border-color: {t['accent']};
    }}
    .card-label {{
        font-family: var(--font-mono);
        color: {t['text_faint']};
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 10px;
        display: block;
    }}
    .card-value {{
        font-family: var(--font-mono);
        color: {t['text']};
        font-size: 1.9rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        line-height: 1.1;
    }}
    .card-sub {{
        font-family: var(--font-mono);
        color: {t['accent']};
        font-size: 0.78rem;
        margin-top: 6px;
        display: block;
    }}

    /* eyebrow tag pill */
    .tag {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-family: var(--font-mono);
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        color: {t['accent']};
        background: {t['accent_soft']};
        border: 1px solid {t['accent_border']};
        border-radius: var(--radius-sm);
        padding: 5px 11px;
        margin-bottom: 14px;
    }}

    /* verdict banner, matches segment-banner styling */
    .verdict {{
        background: linear-gradient(135deg, {t['accent_soft']}, {t['panel_alt']});
        border: 1px solid {t['accent']};
        border-radius: var(--radius-lg);
        padding: 24px 26px;
        margin: 16px 0 22px 0;
        position: relative;
        overflow: hidden;
    }}
    .verdict::before {{
        content: "";
        position: absolute;
        top: 0; left: 0; bottom: 0;
        width: 3px;
        background: var(--v-color);
    }}
    .verdict-title {{
        font-family: var(--font-serif);
        font-size: 1.4rem;
        font-weight: 700;
        color: {t['text']};
    }}
    .verdict-sub {{
        color: {t['text_dim']};
        font-size: 0.9rem;
        margin-top: 5px;
        line-height: 1.5;
    }}

    /* chip / rec-card style row */
    .chip {{
        border: 1px solid {t['border']};
        border-radius: var(--radius-md);
        padding: 11px 15px;
        margin-bottom: 8px;
        font-size: 0.86rem;
        background: {t['panel_alt']};
        display: flex;
        align-items: center;
        gap: 10px;
        transition: border-color var(--motion-fast) ease;
    }}
    .chip:hover {{
        border-color: {t['accent']};
    }}
    .chip-dot {{
        width: 6px; height: 6px; border-radius: 50%;
        background: var(--c-color, {t['text_faint']});
        flex-shrink: 0;
    }}

    .bar-row {{ display: flex; align-items: center; gap: 10px; margin-bottom: 7px; font-size: 0.8rem; }}
    .bar-label {{
        width: 200px; flex-shrink: 0; overflow: hidden; text-overflow: ellipsis;
        white-space: nowrap; color: {t['text_dim']}; font-family: var(--font-mono); font-size: 0.74rem;
    }}
    .bar-track {{ flex-grow: 1; height: 16px; background: {t['panel_alt']}; border-radius: var(--radius-xs); overflow: hidden; }}
    .bar-fill {{ height: 100%; }}
    .bar-val {{ width: 55px; text-align: right; font-family: var(--font-mono); color: {t['text_dim']}; font-size: 0.74rem; }}

    div[data-testid="stMetric"] {{
        background: {t['panel_alt']};
        border: 1px solid {t['border']};
        border-radius: var(--radius-md);
        padding: 15px 18px;
    }}
    div[data-testid="stMetricLabel"] {{ color: {t['text_faint']} !important; }}
    div[data-testid="stMetricValue"] {{ color: {t['text']} !important; font-family: var(--font-mono) !important; }}

    /* tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        border-bottom: 1px solid {t['border']};
    }}
    .stTabs [data-baseweb="tab"] {{
        font-family: var(--font-mono);
        font-size: 0.82rem;
        color: {t['text_dim']};
    }}
    .stTabs [aria-selected="true"] {{
        color: {t['accent']} !important;
    }}

    /* buttons, styled to match the accent CTA look */
    .stButton > button {{
        background: {t['accent']};
        color: #ffffff;
        border: none;
        border-radius: var(--radius-md);
        font-weight: 700;
        font-family: var(--font-primary);
        padding: 0.6rem 1.4rem;
        transition: transform 0.1s ease, box-shadow var(--motion-fast) ease;
    }}
    .stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 6px 18px {t['accent_border']};
        color: #ffffff;
    }}
    .stButton > button:focus-visible {{
        outline: 2px solid {t['accent']} !important;
        outline-offset: 2px !important;
    }}
    .stButton > button[kind="secondary"] {{
        background: {t['panel_raised']};
        color: {t['text']};
        border: 1px solid {t['border']};
    }}

    div[data-testid="stDataFrame"] {{ border: 1px solid {t['border']}; border-radius: var(--radius-md); overflow: hidden; }}

    div[data-baseweb="select"] > div, .stTextArea textarea, .stTextInput input, .stNumberInput input {{
        background-color: {t['panel_raised']} !important;
        border: 1px solid {t['border']} !important;
        border-radius: var(--radius-md) !important;
        color: {t['text']} !important;
    }}
    div[data-baseweb="select"] > div:focus-within, .stTextArea textarea:focus,
    .stTextInput input:focus, .stNumberInput input:focus {{
        border-color: {t['accent']} !important;
        box-shadow: 0 0 0 1px {t['accent']} !important;
    }}

    div[data-testid="stForm"] {{
        background: {t['panel']};
        border: 1px solid {t['border']};
        border-radius: var(--radius-lg);
        padding: 28px 30px;
    }}

    .footer {{
        font-family: var(--font-mono);
        text-align: center;
        color: {t['text_faint']};
        font-size: 0.72rem;
        padding: 26px 0 6px 0;
        margin-top: 34px;
    }}

    </style>
    """,
        unsafe_allow_html=True,
    )


def accent_scale(t):
    "Single-hue sequential scale from soft accent to full accent."

    return [t["accent_soft"], t["accent"]]


def icon(name, size=22, stroke_width=1.6, color="currentColor"):
    "Return an inline SVG <svg> string for the given icon name."

    paths = {
        "csat": (
            '<path d="M12 2.5l2.9 6.3 6.8.7-5.1 4.7 1.5 6.8L12 17.6l-6.1 3.4 1.5-6.8-5.1-4.7 6.8-.7z" '
            'fill="#f5b400" stroke="#c98a00" stroke-width="1"/>'
        ),
        "target": (
            '<circle cx="12" cy="12" r="8.5"/>'
            '<circle cx="12" cy="12" r="5"/>'
            '<circle cx="12" cy="12" r="1.5" fill="currentColor" stroke="none"/>'
        ),
        "chart": (
            '<path d="M4 20V10"/>'
            '<path d="M11 20V4"/>'
            '<path d="M18 20v-7"/>'
            '<path d="M3 20h18"/>'
        ),
        "layers": (
            '<path d="M12 2 2 7l10 5 10-5-10-5z"/>'
            '<path d="M2 17l10 5 10-5"/>'
            '<path d="M2 12l10 5 10-5"/>'
        ),
        "compass": (
            '<circle cx="12" cy="12" r="8.5"/>'
            '<path d="M15.2 8.8l-2 5.2-5.2 2 2-5.2z"/>'
        ),
        "sparkle": (
            '<path d="M12 3v3.2"/>'
            '<path d="M12 17.8V21"/>'
            '<path d="M3 12h3.2"/>'
            '<path d="M17.8 12H21"/>'
            '<path d="M5.6 5.6l2.3 2.3"/>'
            '<path d="M16.1 16.1l2.3 2.3"/>'
            '<path d="M5.6 18.4l2.3-2.3"/>'
            '<path d="M16.1 7.9l2.3-2.3"/>'
        ),
        "info": (
            '<circle cx="12" cy="12" r="8.5"/>'
            '<path d="M12 11v5.5"/>'
            '<circle cx="12" cy="8" r="0.6" fill="currentColor" stroke="none"/>'
        ),
    }

    body = paths.get(name, "")

    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" '
        f'stroke-linejoin="round" style="vertical-align:-4px; display:inline-block;">{body}</svg>'
    )


ICON_CSAT = icon("csat", size=22)
ICON_TARGET = icon("target", size=26)
ICON_CHART = icon("chart", size=26)
ICON_LAYERS = icon("layers", size=26)
ICON_COMPASS = icon("compass", size=15)
ICON_SPARKLE = icon("sparkle", size=15)
ICON_INFO = icon("info", size=15)


def plotly_template(t):
    return go.layout.Template(
        layout=go.Layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color=t["text_dim"], size=12),
            title_font=dict(family="Inter, sans-serif", color=t["text"], size=14),
            colorway=[
                t["accent"],
                t["chart_2"],
                t["good"],
                t["warn"],
                t["chart_3"],
                t["bad"],
            ],
            xaxis=dict(
                gridcolor=t["border"], zerolinecolor=t["border"], linecolor=t["border"]
            ),
            yaxis=dict(
                gridcolor=t["border"], zerolinecolor=t["border"], linecolor=t["border"]
            ),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=t["text_dim"])),
            margin=dict(l=10, r=10, t=40, b=10),
            hoverlabel=dict(
                bgcolor=t["panel"], bordercolor=t["border"], font=dict(color=t["text"])
            ),
        )
    )


# small render helpers


def header(title, sub, icon_html=None):
    title_html = (
        f'<span style="display:inline-flex; align-items:center; gap:0.5rem;">{icon_html} {title}</span>'
        if icon_html
        else title
    )

    st.markdown(
        f'<p class="page-title">{title_html}</p><p class="page-sub">{sub}</p>',
        unsafe_allow_html=True,
    )


def stat_card(label, value, sub=None):
    st.markdown(
        f'<div class="card"><div class="card-label">{label}</div>'
        f'<div class="card-value">{value}</div>'
        f'{f"<div class=card-sub>{sub}</div>" if sub else ""}</div>',
        unsafe_allow_html=True,
    )


def tag(text):
    st.markdown(f'<span class="tag">{text}</span>', unsafe_allow_html=True)


def verdict_box(theme, is_satisfied, confidence_pct, message):
    color = theme["good"] if is_satisfied else theme["bad"]

    glow = "rgba(52,211,153,0.08)" if is_satisfied else "rgba(248,113,113,0.08)"

    title = "Likely satisfied" if is_satisfied else "Dissatisfaction risk"

    st.markdown(
        f'<div class="verdict" style="--v-color:{color};--v-glow:{glow};">'
        f'<div class="verdict-title">{title}</div>'
        f'<div class="verdict-sub">Confidence {confidence_pct} — {message}</div></div>',
        unsafe_allow_html=True,
    )


def chip(theme, color_key, text):
    color = theme.get(color_key, theme["text_faint"])

    st.markdown(
        f'<div class="chip"><div class="chip-dot" style="--c-color:{color};"></div><div>{text}</div></div>',
        unsafe_allow_html=True,
    )


def shap_bars(theme, contributions):
    if not contributions:
        st.caption("No contribution data.")

        return

    max_abs = max(abs(c["shap_value"]) for c in contributions) or 1.0

    rows = []

    for c in contributions:
        val = c["shap_value"]

        pct = min(100, abs(val) / max_abs * 100)

        color = theme["good"] if val >= 0 else theme["bad"]

        rows.append(
            f'<div class="bar-row"><div class="bar-label" title="{c["feature"]}">{c["feature"]}</div>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%;background:{color};"></div></div>'
            f'<div class="bar-val">{val:+.3f}</div></div>'
        )

    st.markdown("".join(rows), unsafe_allow_html=True)


# text cleaning
#
# order: expand contractions, lowercase, strip punctuation, strip
# urls/digits, drop stopwords, rephrase a few domain terms, tokenize,
# lemmatize. missing remarks get filled with "no remarks" before any of
# that runs.
#
# every key below has its leading apostrophe (so "'re"/"'ll" only match
# real contractions like "you're"/"we'll", not the letters "re"/"ll"
# wherever they appear inside an ordinary word -- the previous version
# was missing the apostrophe on those two keys, which silently mangled
# words like "return", "refund", "before", and "will" into garbage
# tokens), and every replacement has a leading space so word boundaries
# survive (e.g. "it's" -> "it is", not "itis").

CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'ve": " have",
    "'m": " am",
}

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


def clean_remark(raw_text):
    ensure_nltk()

    from nltk.corpus import stopwords

    from nltk.stem import WordNetLemmatizer

    from nltk.tokenize import word_tokenize

    text = (
        str(raw_text)
        if raw_text is not None and str(raw_text).strip()
        else "no remarks"
    )

    for k, v in CONTRACTIONS.items():
        text = text.replace(k, v)

    text = text.lower()

    text = text.translate(str.maketrans("", "", string.punctuation))

    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    text = re.sub(r"\w*\d\w*", "", text)

    # "no"/"not"/"nor" are excluded from NLTK's default stopword list here --
    # leaving them in silently collapsed opposite-meaning remarks (e.g. "was
    # not resolved" vs "was resolved") to identical cleaned text.
    stop_words = set(stopwords.words("english")) - {"no", "not", "nor"}

    words = text.split()

    text = " ".join(w for w in words if w not in stop_words)

    text = text.strip() or "no remarks"

    text = text.replace("delivery late", "late delivery")

    text = text.replace("not received", "undelivered")

    text = text.replace("didnt receive", "undelivered")

    lemmatizer = WordNetLemmatizer()

    tokens = word_tokenize(text)

    result = " ".join(lemmatizer.lemmatize(t) for t in tokens).strip()

    return result or "no remarks"


def clean_remarks_batch(texts):
    ensure_nltk()

    return [clean_remark(t) for t in texts]


# data + lookups


@dataclass
class Lookups:
    global_mean: float

    agent_lookup: dict

    sup_lookup: dict

    agents: list = field(default_factory=list)

    supervisors: list = field(default_factory=list)

    channels: list = field(default_factory=list)

    categories: list = field(default_factory=list)

    subcategory_by_category: dict = field(default_factory=dict)

    response_time_median: float = 0.0

    response_time_p90: float = 0.0


def _winsorize(series):
    q1, q3 = series.quantile([0.25, 0.75])

    iqr = q3 - q1

    return series.clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)


def _smoothed_target_encoding(df, group_col, target_col, k):
    global_mean = df[target_col].mean()

    grp_mean = df.groupby(group_col)[target_col].mean()

    grp_count = df.groupby(group_col)[target_col].count()

    smoothed = (grp_count * grp_mean + k * global_mean) / (grp_count + k)

    return smoothed.to_dict()


@st.cache_data(show_spinner="Loading historical data...")
def load_dataset():
    if not os.path.exists(DATA_PATH):
        return None, f"Dataset not found at {DATA_PATH}"

    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as e:
        return None, f"Failed to read dataset: {e}"

    df["Issue_reported at"] = pd.to_datetime(
        df["Issue_reported at"], dayfirst=True, errors="coerce"
    )

    df["issue_responded"] = pd.to_datetime(
        df["issue_responded"], dayfirst=True, errors="coerce"
    )

    df["response_time_minutes"] = (
        (df["issue_responded"] - df["Issue_reported at"]).dt.total_seconds() / 60
    ).clip(lower=0)

    df["issue_hour"] = df["Issue_reported at"].dt.hour

    df["issue_dayofweek"] = df["Issue_reported at"].dt.dayofweek

    df.drop_duplicates(inplace=True)

    if "connected_handling_time" in df.columns:
        df.drop(columns=["connected_handling_time"], inplace=True)

    if "CSAT Score" not in df.columns:
        return None, "Dataset missing required column 'CSAT Score'"

    df["CSAT_label"] = (df["CSAT Score"] >= 4).astype(int)

    for col in [
        "Sub-category",
        "category",
        "channel_name",
        "Tenure Bucket",
        "Agent Shift",
    ]:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    df["Customer Remarks"] = df["Customer Remarks"].fillna("no remarks").astype(str)

    for col in ["Item_price", "response_time_minutes", "issue_hour", "issue_dayofweek"]:
        if col in df.columns and df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    for col in ["Customer_City", "Product_category"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)

    df["response_time_minutes"] = _winsorize(df["response_time_minutes"])

    agent_lookup = _smoothed_target_encoding(
        df, "Agent_name", "CSAT Score", TARGET_ENCODING_K
    )

    sup_lookup = _smoothed_target_encoding(
        df, "Supervisor", "CSAT Score", TARGET_ENCODING_K
    )

    global_mean = df["CSAT Score"].mean()

    df["agent_csat_encoded"] = df["Agent_name"].map(agent_lookup).fillna(global_mean)

    df["supervisor_csat_encoded"] = df["Supervisor"].map(sup_lookup).fillna(global_mean)

    df.attrs["agent_lookup"] = agent_lookup

    df.attrs["sup_lookup"] = sup_lookup

    df.attrs["global_mean"] = global_mean

    return df, None


@st.cache_resource(show_spinner=False)
def build_lookups(_df):
    df = _df

    return Lookups(
        global_mean=float(df.attrs["global_mean"]),
        agent_lookup=df.attrs["agent_lookup"],
        sup_lookup=df.attrs["sup_lookup"],
        agents=sorted(df["Agent_name"].dropna().unique().tolist()),
        supervisors=sorted(df["Supervisor"].dropna().unique().tolist()),
        channels=sorted(df["channel_name"].dropna().unique().tolist()),
        categories=sorted(df["category"].dropna().unique().tolist()),
        subcategory_by_category={
            cat: sorted(g["Sub-category"].dropna().unique().tolist())
            for cat, g in df.groupby("category")
        },
        response_time_median=float(df["response_time_minutes"].median()),
        response_time_p90=float(df["response_time_minutes"].quantile(0.90)),
    )


@st.cache_resource(show_spinner=False)
def load_label_maps():
    """Load the label encoders saved at training time, rather than refitting
    a fresh LabelEncoder against whatever CSV happens to sit next to the app.
    Refitting from source data on every run worked only by coincidence -- it
    silently drifts the moment the CSV's category set changes, with no error
    raised. Returns (maps_or_None, missing_paths)."""
    if not os.path.exists(LE_PATH):
        return None, [LE_PATH]

    return joblib.load(LE_PATH), []


@st.cache_resource(show_spinner="Loading model artifacts...")
def load_artifacts():
    required = [MODEL_PATH, TFIDF_PATH, SCALER_PATH, PT_PATH]

    missing = [p for p in required if not os.path.exists(p)]

    if missing:
        return None, missing

    model = joblib.load(MODEL_PATH)

    tfidf = joblib.load(TFIDF_PATH)

    scaler = joblib.load(SCALER_PATH)

    pt = joblib.load(PT_PATH)

    return (model, tfidf, scaler, pt), []


# feature engineering + inference


@dataclass
class TicketInput:
    channel_name: str

    category: str

    sub_category: str

    tenure_bucket: str

    agent_shift: str

    response_time_minutes: float

    issue_hour: int

    issue_dayofweek: int

    customer_remarks: str = ""

    agent_name: str = None

    supervisor: str = None


def _safe_label(label_maps, col, value, warnings_out):
    mapping = label_maps[col]

    if value not in mapping:
        warnings_out.append(
            f"'{value}' wasn't seen during training for '{col}'; using a default value."
        )

        return 0

    return mapping[value]


def _tenure_index(bucket, warnings_out):
    if bucket not in TENURE_INDEX:
        warnings_out.append(f"Unknown tenure bucket '{bucket}'.")

        return -1

    return TENURE_INDEX[bucket]


def _target_encode(name, lookup, global_mean):
    if not name:
        return global_mean

    return lookup.get(name, global_mean)


def build_struct_row(ticket, label_maps, lookups, warnings_out):
    enc_vals = [
        _safe_label(label_maps, "channel_name", ticket.channel_name, warnings_out),
        _safe_label(label_maps, "category", ticket.category, warnings_out),
        _safe_label(label_maps, "Sub-category", ticket.sub_category, warnings_out),
        _tenure_index(ticket.tenure_bucket, warnings_out),
        _safe_label(label_maps, "Agent Shift", ticket.agent_shift, warnings_out),
    ]

    num_vals = [
        float(ticket.response_time_minutes),
        float(ticket.issue_hour),
        float(ticket.issue_dayofweek),
        _target_encode(ticket.agent_name, lookups.agent_lookup, lookups.global_mean),
        _target_encode(ticket.supervisor, lookups.sup_lookup, lookups.global_mean),
    ]

    return np.array(enc_vals + num_vals, dtype=float)


def _scale_struct(struct_matrix, scaler, pt):
    n_enc = len(ENC_FEATURE_COLS)

    x_cat = struct_matrix[:, :n_enc]

    x_num = struct_matrix[:, n_enc:]

    x_num_tf = pt.transform(x_num)

    return scaler.transform(np.hstack([x_cat, x_num_tf]))


def predict_single(ticket, model, tfidf, scaler, pt, label_maps, lookups):
    warnings_out = []

    row = build_struct_row(ticket, label_maps, lookups, warnings_out).reshape(1, -1)

    scaled = _scale_struct(row, scaler, pt)

    remark = clean_remark(ticket.customer_remarks)

    x_text = tfidf.transform([remark])

    x_combined = hstack([csr_matrix(scaled), x_text])

    proba = model.predict_proba(x_combined)[0]

    pred = int(proba[1] >= DECISION_THRESHOLD)

    return {
        "prediction": pred,
        "prob_satisfied": float(proba[1]),
        "prob_dissatisfied": float(proba[0]),
        "clean_remark": remark,
        "warnings": warnings_out,
    }


BATCH_ALIASES = {
    "sub_category": "Sub-category",
    "tenure_bucket": "Tenure Bucket",
    "agent_shift": "Agent Shift",
}
BATCH_DEFAULTS = {
    "channel_name": "Unknown",
    "category": "Unknown",
    "sub_category": "Unknown",
    "tenure_bucket": "Unknown",
    "agent_shift": "Unknown",
}


def predict_batch(df_input, model, tfidf, scaler, pt, label_maps, lookups):
    df = df_input.copy()

    df.attrs = {}

    for target, source in BATCH_ALIASES.items():
        if source in df.columns and target not in df.columns:
            df[target] = df[source]

    for col, default in BATCH_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default

        df[col] = df[col].fillna(default).astype(str)

    if "Issue_reported at" in df.columns and "issue_responded" in df.columns:
        ir = pd.to_datetime(df["Issue_reported at"], dayfirst=True, errors="coerce")

        rr = pd.to_datetime(df["issue_responded"], dayfirst=True, errors="coerce")

        df["response_time_minutes"] = ((rr - ir).dt.total_seconds() / 60).clip(lower=0)

        df["issue_hour"] = ir.dt.hour

        df["issue_dayofweek"] = ir.dt.dayofweek

    for col, default in [
        ("response_time_minutes", lookups.response_time_median),
        ("issue_hour", 12),
        ("issue_dayofweek", 2),
    ]:
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

    df["agent_csat_encoded"] = (
        df["Agent_name"].map(lookups.agent_lookup).fillna(lookups.global_mean)
    )

    df["supervisor_csat_encoded"] = (
        df["Supervisor"].map(lookups.sup_lookup).fillna(lookups.global_mean)
    )

    def enc_col(col_key, series):
        mapping = label_maps[col_key]

        unseen = ~series.isin(mapping.keys())

        if unseen.any():
            df.attrs.setdefault("unseen_categories", {})[col_key] = int(unseen.sum())

        return series.map(lambda v: mapping.get(v, 0))

    enc_matrix = np.column_stack(
        [
            enc_col("channel_name", df["channel_name"]).values,
            enc_col("category", df["category"]).values,
            enc_col("Sub-category", df["sub_category"]).values,
            df["tenure_bucket"].map(lambda v: TENURE_INDEX.get(v, -1)).values,
            enc_col("Agent Shift", df["agent_shift"]).values,
        ]
    ).astype(float)

    num_matrix = df[
        [
            "response_time_minutes",
            "issue_hour",
            "issue_dayofweek",
            "agent_csat_encoded",
            "supervisor_csat_encoded",
        ]
    ].values.astype(float)

    struct = np.hstack([enc_matrix, num_matrix])

    scaled = _scale_struct(struct, scaler, pt)

    ensure_nltk()

    clean = pd.Series(
        clean_remarks_batch(df["Customer Remarks"].tolist()), index=df.index
    )

    x_text = tfidf.transform(clean)

    x_combined = hstack([csr_matrix(scaled), x_text])

    proba = model.predict_proba(x_combined)

    df["clean_remarks"] = clean

    df["predicted_label"] = np.where(
        proba[:, 1] >= DECISION_THRESHOLD, "Satisfied", "Dissatisfied"
    )

    df["prob_satisfied"] = proba[:, 1]

    df["prob_dissatisfied"] = proba[:, 0]

    df["risk_tier"] = pd.cut(
        df["prob_dissatisfied"],
        bins=[-0.01, 0.3, 0.6, 1.01],
        labels=["Low Risk", "Medium Risk", "High Risk"],
    )

    return df


def build_shap_explanation(
    ticket, model, tfidf, scaler, pt, label_maps, lookups, top_n=10
):
    import shap

    warnings_out = []

    row = build_struct_row(ticket, label_maps, lookups, warnings_out).reshape(1, -1)

    scaled = _scale_struct(row, scaler, pt)

    remark = clean_remark(ticket.customer_remarks)

    x_text = tfidf.transform([remark])

    # keep this sparse -- XGBoost treats an absent sparse entry as "missing"
    # but an explicit dense 0.0 as a real value, and the model was trained
    # on sparse input throughout. Densifying here (the previous .toarray()
    # call) made SHAP explain a materially different prediction than the
    # one predict_single()/predict_batch() actually returned.
    x_combined = hstack([csr_matrix(scaled), x_text]).tocsr()

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(x_combined)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_row = np.asarray(shap_values).reshape(-1)

    names = (
        ENC_FEATURE_COLS
        + NUM_FEATURE_COLS
        + [f"text: {t}" for t in tfidf.get_feature_names_out()]
    )

    order = np.argsort(-np.abs(shap_row))[:top_n]

    return [{"feature": names[i], "shap_value": float(shap_row[i])} for i in order]


@st.cache_data(show_spinner=False)
def top_global_features(_artifacts, _label_maps, top_n=10):
    """Global feature importance from the model's own gain scores — fast, no per-row SHAP needed."""
    model, tfidf, scaler, pt = _artifacts

    importances = model.feature_importances_

    names = (
        ENC_FEATURE_COLS
        + NUM_FEATURE_COLS
        + [f"text: {t}" for t in tfidf.get_feature_names_out()]
    )

    order = np.argsort(-importances)[:top_n]

    max_val = importances[order[0]] if len(order) else 1.0

    return [
        {
            "feature": names[i],
            "shap_value": float(importances[i] / max_val) if max_val else 0.0,
        }
        for i in order
    ]


# page: home

CSAT_SCALE_INFO = {
    "5 · Excellent": {
        "desc": "Customer had a smooth, fast resolution and said so directly. The benchmark experience to replicate across every channel.",
        "color": DARK["good"],
    },
    "4 · Good": {
        "desc": "Issue resolved without friction, but not remarkable enough to be a 5. The largest group in most support queues.",
        "color": DARK["chart_2"],
    },
    "3 · Neutral": {
        "desc": "Resolved, but something cost the customer time or patience along the way. Worth a closer look at root cause.",
        "color": DARK["warn"],
    },
    "1-2 · Poor": {
        "desc": "Unresolved, mishandled, or the customer had to escalate. Highest churn and reorder risk — this model's main target.",
        "color": DARK["bad"],
    },
}


def page_home(theme, df):
    st.markdown(
        '<span class="tag">flipkart support · customer intelligence</span>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<h1 style="display:flex; align-items:center; gap:0.55rem; margin-bottom:0;">'
        f"{ICON_CSAT} CSAT Suite</h1>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="page-sub" style="max-width:640px;">'
        "Predicts whether a support ticket will end in a satisfied customer, and explains why — "
        "built on an XGBoost classifier over structured ticket data and TF-IDF text features, with "
        "SHAP-based explanations for every score."
        "</p>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        stat_card("Total tickets", f"{len(df):,}", "Full training corpus")

    with c2:
        stat_card(
            "Satisfaction rate",
            f"{df['CSAT_label'].mean():.1%}",
            "Score of 4 or higher",
        )

    with c3:
        stat_card(
            "Model ROC-AUC", f"{REPORTED_METRICS['roc_auc']:.3f}", "Tuned XGBoost"
        )

    with c4:
        stat_card(
            "Median response",
            f"{df['response_time_minutes'].median():.0f} min",
            "Issue to first response",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("---")

    left, right = st.columns([1.1, 1], gap="large")

    with left:
        st.markdown("### What this does")

        st.markdown(
            """
            <p class="page-sub" style="max-width:none;">
            <b>Score a ticket</b> — structured signals (channel, category, tenure, shift, response time)
            and the customer's own remark are combined, TF-IDF-vectorised, scaled and fed to a tuned
            XGBoost classifier that outputs a probability of satisfaction, with SHAP values showing
            exactly which features pushed the score up or down.
            </p>
            <p class="page-sub" style="max-width:none;">
            <b>Explore the data</b> — the same pipeline powers a batch scorer for whole queues and an
            explorer for slicing satisfaction by category, channel, shift and response time across the
            full historical ticket set.
            </p>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        b1, b2 = st.columns(2)

        with b1:
            if st.button("→ Score a ticket", use_container_width=True):
                st.session_state["_nav_hint"] = "Predict"

                st.rerun()

        with b2:
            if st.button("→ Explore the data", use_container_width=True):
                st.session_state["_nav_hint"] = "Explorer"

                st.rerun()

    with right:
        st.markdown("### CSAT scale reference")

        for name, meta in CSAT_SCALE_INFO.items():
            st.markdown(
                f"""
                <div class="chip" style="border-left: 3px solid {meta['color']};">
                    <div>
                        <div style="font-size:0.95rem; color:{theme['text']}; font-weight:500;">{name}</div>
                        <div style="color:{theme['text_faint']}; font-size:0.82rem; margin-top:0.2rem;">{meta['desc']}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    st.markdown("---")

    tag("Snapshot")

    tmpl = plotly_template(theme)

    left, right = st.columns([1.3, 1], gap="large")

    with left:
        tag("Satisfaction by category")

        cat_sat = (
            df.groupby("category")["CSAT_label"].mean().sort_values() * 100
        ).reset_index()

        cat_sat.columns = ["category", "pct"]

        fig = px.bar(
            cat_sat,
            x="pct",
            y="category",
            orientation="h",
            color="pct",
            color_continuous_scale=accent_scale(theme),
            labels={"pct": "Satisfaction %", "category": ""},
            text="pct",
        )

        fig.update_traces(texttemplate="%{text:.0f}%", textposition="outside")

        fig.update_layout(
            template=tmpl,
            height=440,
            coloraxis_showscale=False,
            bargap=0.3,
            xaxis=dict(range=[0, 108]),
        )

        st.plotly_chart(fig, width="stretch")

    with right:
        tag("Class split")

        split = (
            df["CSAT_label"].value_counts().rename({0: "Dissatisfied", 1: "Satisfied"})
        )

        fig = px.pie(
            values=split.values,
            names=split.index,
            hole=0.62,
            color=split.index,
            color_discrete_map={
                "Satisfied": theme["good"],
                "Dissatisfied": theme["bad"],
            },
        )

        fig.update_traces(textinfo="percent+label")

        fig.update_layout(template=tmpl, height=440, showlegend=False)

        st.plotly_chart(fig, width="stretch")

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")

    with c1:
        tag("Channel mix")

        ch = df["channel_name"].value_counts().reset_index()

        ch.columns = ["channel", "count"]

        fig = px.bar(
            ch,
            x="channel",
            y="count",
            color="channel",
            text="count",
            color_discrete_sequence=[
                theme["accent"],
                theme["chart_2"],
                theme["chart_3"],
            ],
        )

        fig.update_traces(textposition="outside")

        fig.update_layout(
            template=tmpl,
            height=340,
            showlegend=False,
            xaxis_title="",
            yaxis_title="Tickets",
        )

        st.plotly_chart(fig, width="stretch")

    with c2:
        tag("Issue volume by hour")

        hourly = df["issue_hour"].value_counts().sort_index().reset_index()

        hourly.columns = ["hour", "count"]

        fig = px.area(hourly, x="hour", y="count", markers=True)

        fig.update_traces(line_color=theme["accent"], fillcolor=theme["accent_soft"])

        fig.update_layout(
            template=tmpl,
            height=340,
            xaxis_title="Hour",
            yaxis_title="Tickets",
            xaxis=dict(dtick=2),
        )

        st.plotly_chart(fig, width="stretch")


# page: predict


def _hex_to_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")

    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    return f"rgba({r},{g},{b},{alpha})"


def gauge_chart(theme, prob):
    color = theme["good"] if prob >= 0.5 else theme["bad"]

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={
                "suffix": "%",
                "font": {
                    "size": 40,
                    "color": theme["text"],
                    "family": "JetBrains Mono",
                },
            },
            gauge={
                "axis": {"range": [0, 100], "tickcolor": theme["text_dim"]},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": theme["panel_alt"],
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": _hex_to_rgba(theme["bad"], 0.15)},
                    {"range": [40, 65], "color": _hex_to_rgba(theme["warn"], 0.15)},
                    {"range": [65, 100], "color": _hex_to_rgba(theme["good"], 0.15)},
                ],
                "threshold": {
                    "line": {"color": theme["text"], "width": 2},
                    "thickness": 0.8,
                    "value": 50,
                },
            },
        )
    )

    fig.update_layout(
        template=plotly_template(theme), height=280, margin=dict(l=20, r=20, t=10, b=10)
    )

    return fig


def page_predict(theme, df, lookups, label_maps, artifacts):
    model, tfidf, scaler, pt = artifacts

    header("Predict", "Score a single ticket", icon_html=ICON_TARGET)

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            channel_name = st.selectbox("Support channel", lookups.channels)

            category = st.selectbox("Issue category", lookups.categories)

        with c2:
            sub_options = lookups.subcategory_by_category.get(category, [])

            sub_category = st.selectbox(
                "Sub-category", sub_options if sub_options else ["General Enquiry"]
            )

            tenure_bucket = st.selectbox("Agent tenure", TENURE_ORDER, index=2)

        with c3:
            agent_shift = st.selectbox(
                "Agent shift", ["Morning", "Afternoon", "Evening", "Night", "Split"]
            )

            issue_dow = st.selectbox("Day reported", DAY_NAMES, index=2)

        c4, c5, c6 = st.columns(3)

        with c4:
            issue_time = st.time_input("Time reported", value=dtime(10, 0))

        with c5:
            response_minutes = st.slider(
                "Response time (minutes)",
                0,
                1440,
                int(min(lookups.response_time_median, 120)),
                step=5,
            )

        with c6:
            st.metric("Historical median", f"{lookups.response_time_median:.0f} min")

        c7, c8 = st.columns(2)

        with c7:
            agent_name = st.selectbox("Agent", ["Unknown"] + lookups.agents)

        with c8:
            supervisor = st.selectbox("Supervisor", ["Unknown"] + lookups.supervisors)

        customer_remarks = st.text_area(
            "Customer remarks", height=100, placeholder="What did the customer say?"
        )

        show_shap = st.checkbox("Include feature contribution breakdown", value=True)

        submitted = st.form_submit_button("Predict")

    if not submitted:
        return

    ticket = TicketInput(
        channel_name=channel_name,
        category=category,
        sub_category=sub_category,
        tenure_bucket=tenure_bucket,
        agent_shift=agent_shift,
        response_time_minutes=float(response_minutes),
        issue_hour=issue_time.hour,
        issue_dayofweek=DAY_NAMES.index(issue_dow),
        customer_remarks=customer_remarks,
        agent_name=None if agent_name == "Unknown" else agent_name,
        supervisor=None if supervisor == "Unknown" else supervisor,
    )

    with st.spinner("Scoring..."):
        result = predict_single(ticket, model, tfidf, scaler, pt, label_maps, lookups)

    st.markdown("---")

    is_satisfied = result["prediction"] == 1

    confidence = (
        result["prob_satisfied"] if is_satisfied else result["prob_dissatisfied"]
    )

    message = (
        "this looks low-risk based on historical patterns"
        if is_satisfied
        else "worth prioritizing for follow-up"
    )

    verdict_box(theme, is_satisfied, f"{confidence:.1%}", message)

    colA, colB = st.columns([1, 1.3])

    with colA:
        st.plotly_chart(gauge_chart(theme, result["prob_satisfied"]), width="stretch")

        with st.expander("Processed text"):
            st.code(result["clean_remark"] or "(empty)", language=None)

    with colB:
        st.markdown("**Contributing signals**")

        if response_minutes > lookups.response_time_p90:
            chip(
                theme,
                "bad",
                f"Response time ({response_minutes} min) is in the slowest 10% historically",
            )
        elif response_minutes <= lookups.response_time_median:
            chip(
                theme,
                "good",
                f"Response time ({response_minutes} min) is at or below median",
            )
        else:
            chip(
                theme, "warn", f"Response time ({response_minutes} min) is above median"
            )

        cat_rate = df[df["category"] == category]["CSAT_label"].mean()

        overall = df["CSAT_label"].mean()

        if cat_rate < overall - 0.05:
            chip(
                theme,
                "bad",
                f"'{category}' underperforms on satisfaction ({cat_rate:.0%})",
            )
        elif cat_rate > overall + 0.05:
            chip(
                theme,
                "good",
                f"'{category}' overperforms on satisfaction ({cat_rate:.0%})",
            )

        if ticket.agent_name:
            a_score = lookups.agent_lookup.get(ticket.agent_name, lookups.global_mean)

            if a_score >= lookups.global_mean:
                chip(
                    theme,
                    "good",
                    f"Agent's historical average CSAT ({a_score:.2f}) is above global average",
                )
            else:
                chip(
                    theme,
                    "bad",
                    f"Agent's historical average CSAT ({a_score:.2f}) is below global average",
                )

    if show_shap:
        st.markdown("---")

        st.markdown("**Model explanation**")

        st.caption(
            "Per-feature contributions from the trained model for this ticket. "
            "Positive pushes toward satisfied, negative toward dissatisfied."
        )

        with st.spinner("Computing..."):
            try:
                contributions = build_shap_explanation(
                    ticket, model, tfidf, scaler, pt, label_maps, lookups
                )

                shap_bars(theme, contributions)
            except Exception as e:
                st.warning(f"Could not compute explanation: {e}")


# page: batch


def page_batch(theme, df, lookups, label_maps, artifacts):
    model, tfidf, scaler, pt = artifacts

    header("Batch scoring", "Score a whole queue at once", icon_html=ICON_LAYERS)

    with st.expander("Expected columns"):
        st.code(
            "channel_name, category, Sub-category, Tenure Bucket, Agent Shift,\n"
            "Issue_reported at, issue_responded, Agent_name, Supervisor, Customer Remarks",
            language=None,
        )

        st.caption(
            "Your original Customer_support_data.csv schema also works directly."
        )

    uploaded = st.file_uploader("Upload ticket CSV", type=["csv"])

    use_sample = st.button("Use a sample of historical data instead")

    if uploaded is not None:
        try:
            source_df = pd.read_csv(uploaded)

            st.session_state["source_df"] = source_df

            st.session_state["source_label"] = (
                f"Loaded {len(source_df):,} rows from upload."
            )
        except Exception as e:
            st.error(f"Could not read file: {e}")
    elif use_sample:
        source_df = df.sample(n=min(200, len(df)), random_state=42).drop(
            columns=["CSAT_label", "agent_csat_encoded", "supervisor_csat_encoded"],
            errors="ignore",
        )

        st.session_state["source_df"] = source_df

        st.session_state["source_label"] = "Using 200 sampled historical tickets."

    source_df = st.session_state.get("source_df")

    if source_df is not None and st.session_state.get("source_label"):
        st.info(st.session_state["source_label"])

    if source_df is None:
        return

    st.markdown("**Preview**")

    st.dataframe(source_df.head(10), width="stretch", height=200)

    if st.button("Run batch prediction", type="primary"):
        with st.spinner(f"Scoring {len(source_df):,} tickets..."):
            scored = predict_batch(
                source_df, model, tfidf, scaler, pt, label_maps, lookups
            )

        st.session_state["batch_scored"] = scored

    if "batch_scored" not in st.session_state:
        return

    scored = st.session_state["batch_scored"]

    st.markdown("---")

    st.markdown("**Results**")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        stat_card("Scored", f"{len(scored):,}")

    with c2:
        stat_card(
            "Predicted satisfied",
            f"{(scored['predicted_label']=='Satisfied').mean():.1%}",
        )

    with c3:
        stat_card("High risk", f"{(scored['risk_tier']=='High Risk').sum():,}")

    with c4:
        stat_card(
            "Avg. dissatisfaction prob.", f"{scored['prob_dissatisfied'].mean():.1%}"
        )

    tmpl = plotly_template(theme)

    c1, c2 = st.columns(2)

    with c1:
        tag("Risk distribution")

        tier_counts = (
            scored["risk_tier"]
            .value_counts()
            .reindex(["Low Risk", "Medium Risk", "High Risk"])
            .fillna(0)
        )

        fig = px.bar(
            x=tier_counts.index,
            y=tier_counts.values,
            color=tier_counts.index,
            color_discrete_map={
                "Low Risk": theme["good"],
                "Medium Risk": theme["warn"],
                "High Risk": theme["bad"],
            },
        )

        fig.update_layout(
            template=tmpl,
            height=300,
            showlegend=False,
            xaxis_title="",
            yaxis_title="Tickets",
        )

        st.plotly_chart(fig, width="stretch")

    with c2:
        tag("Probability distribution")

        fig = px.histogram(
            scored,
            x="prob_satisfied",
            nbins=30,
            color_discrete_sequence=[theme["accent"]],
        )

        fig.add_vline(x=0.5, line_dash="dash", line_color=theme["warn"])

        fig.update_layout(
            template=tmpl, height=300, xaxis_title="P(Satisfied)", yaxis_title="Tickets"
        )

        st.plotly_chart(fig, width="stretch")

    st.markdown("**Highest risk tickets**")

    priority_cols = [
        c
        for c in [
            "Unique id",
            "category",
            "sub_category",
            "channel_name",
            "Agent_name",
            "Customer Remarks",
            "prob_dissatisfied",
            "predicted_label",
            "risk_tier",
        ]
        if c in scored.columns
    ]

    top_risk = scored.sort_values("prob_dissatisfied", ascending=False)[
        priority_cols
    ].head(25)

    st.dataframe(top_risk, width="stretch", height=380)

    st.markdown("**Full results**")

    st.dataframe(scored, width="stretch", height=340)

    csv_bytes = scored.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download full results (CSV)",
        data=csv_bytes,
        file_name=f"csat_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )


# page: explorer


def page_explorer(theme, df):
    header("Explorer", "Slice the historical dataset", icon_html=ICON_COMPASS)

    with st.expander("Filters", expanded=True):
        f1, f2, f3 = st.columns(3)

        channels = sorted(df["channel_name"].dropna().unique().tolist())

        categories = sorted(df["category"].dropna().unique().tolist())

        shifts = sorted(df["Agent Shift"].dropna().unique().tolist())

        with f1:
            channel_filter = st.multiselect("Channel", channels, default=channels)

        with f2:
            category_filter = st.multiselect("Category", categories, default=categories)

        with f3:
            shift_filter = st.multiselect("Agent shift", shifts, default=shifts)

    filtered = df[
        df["channel_name"].isin(channel_filter)
        & df["category"].isin(category_filter)
        & df["Agent Shift"].isin(shift_filter)
    ]

    st.caption(f"Showing {len(filtered):,} of {len(df):,} tickets")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        stat_card("Filtered", f"{len(filtered):,}")

    with c2:
        stat_card(
            "Satisfaction rate",
            f"{filtered['CSAT_label'].mean():.1%}" if len(filtered) else "—",
        )

    with c3:
        stat_card(
            "Median response",
            (
                f"{filtered['response_time_minutes'].median():.0f} min"
                if len(filtered)
                else "—"
            ),
        )

    with c4:
        stat_card(
            "Avg CSAT",
            f"{filtered['CSAT Score'].mean():.2f} / 5" if len(filtered) else "—",
        )

    if filtered.empty:
        st.info("No tickets match the current filters.")

        return

    tmpl = plotly_template(theme)

    tab1, tab2, tab3, tab4 = st.tabs(["Drivers", "Response time", "Agents", "Raw data"])

    with tab1:
        c1, c2 = st.columns(2)

        with c1:
            tag("Satisfaction by sub-category")

            top_sub = filtered["Sub-category"].value_counts().head(15).index

            sub_sat = (
                filtered[filtered["Sub-category"].isin(top_sub)]
                .groupby("Sub-category")["CSAT_label"]
                .mean()
                .sort_values()
                * 100
            ).reset_index()

            fig = px.bar(
                sub_sat,
                x="CSAT_label",
                y="Sub-category",
                orientation="h",
                color="CSAT_label",
                color_continuous_scale=accent_scale(theme),
                labels={"CSAT_label": "Satisfaction %"},
            )

            fig.update_layout(
                template=tmpl, height=440, coloraxis_showscale=False, yaxis_title=""
            )

            st.plotly_chart(fig, width="stretch")

        with c2:
            tag("Satisfaction by channel and shift")

            pivot = filtered.pivot_table(
                index="Agent Shift",
                columns="channel_name",
                values="CSAT_label",
                aggfunc="mean",
            )

            if pivot.empty:
                st.caption("Not enough data for this filter combination.")
            else:
                fig = px.imshow(
                    pivot, color_continuous_scale=accent_scale(theme), aspect="auto"
                )

                fig.update_layout(template=tmpl, height=440)

                st.plotly_chart(fig, width="stretch")

    with tab2:
        c1, c2 = st.columns(2)

        with c1:
            tag("Response time distribution")

            rt = filtered["response_time_minutes"]

            rt = rt[rt <= 500]

            fig = px.histogram(rt, nbins=50, color_discrete_sequence=[theme["accent"]])

            fig.add_vline(
                x=filtered["response_time_minutes"].median(),
                line_dash="dash",
                line_color=theme["warn"],
            )

            fig.update_layout(
                template=tmpl,
                height=360,
                xaxis_title="Minutes",
                yaxis_title="Tickets",
                showlegend=False,
            )

            st.plotly_chart(fig, width="stretch")

        with c2:
            tag("Response time by CSAT score")

            box_df = filtered[filtered["response_time_minutes"] <= 500]

            fig = px.box(
                box_df,
                x="CSAT Score",
                y="response_time_minutes",
                color="CSAT Score",
                color_discrete_sequence=[
                    theme["bad"],
                    theme["warn"],
                    theme["chart_2"],
                    theme["chart_3"],
                    theme["good"],
                ],
            )

            fig.update_layout(template=tmpl, height=360, showlegend=False)

            st.plotly_chart(fig, width="stretch")

    with tab3:
        c1, c2 = st.columns(2)

        with c1:
            tag("Top agents by avg CSAT (min 20 tickets)")

            agent_stats = filtered.groupby("Agent_name")["CSAT Score"].agg(
                ["mean", "count"]
            )

            agent_stats = (
                agent_stats[agent_stats["count"] >= 20]
                .sort_values("mean", ascending=False)
                .head(10)
            )

            if agent_stats.empty:
                st.caption("No agents with 20+ tickets in this filter.")
            else:
                fig = px.bar(
                    agent_stats.reset_index(),
                    x="mean",
                    y="Agent_name",
                    orientation="h",
                    color="mean",
                    color_continuous_scale=accent_scale(theme),
                )

                fig.update_layout(
                    template=tmpl,
                    height=400,
                    coloraxis_showscale=False,
                    yaxis_title="",
                    xaxis_title="Avg CSAT",
                )

                st.plotly_chart(fig, width="stretch")

        with c2:
            tag("Top supervisors by avg team CSAT")

            sup_stats = filtered.groupby("Supervisor")["CSAT Score"].agg(
                ["mean", "count"]
            )

            sup_stats = (
                sup_stats[sup_stats["count"] >= 20]
                .sort_values("mean", ascending=False)
                .head(10)
            )

            if sup_stats.empty:
                st.caption("No supervisors with 20+ tickets in this filter.")
            else:
                fig = px.bar(
                    sup_stats.reset_index(),
                    x="mean",
                    y="Supervisor",
                    orientation="h",
                    color="mean",
                    color_continuous_scale=accent_scale(theme),
                )

                fig.update_layout(
                    template=tmpl,
                    height=400,
                    coloraxis_showscale=False,
                    yaxis_title="",
                    xaxis_title="Avg CSAT",
                )

                st.plotly_chart(fig, width="stretch")

    with tab4:
        st.dataframe(
            filtered[
                [
                    "channel_name",
                    "category",
                    "Sub-category",
                    "Customer Remarks",
                    "Agent_name",
                    "Supervisor",
                    "Agent Shift",
                    "Tenure Bucket",
                    "response_time_minutes",
                    "CSAT Score",
                ]
            ].head(500),
            width="stretch",
            height=440,
        )

        st.caption("First 500 filtered rows.")


# page: model performance


def page_model_performance(theme, df, lookups, label_maps, artifacts):
    header(
        "Model performance",
        "How the model scores against held-out data",
        icon_html=ICON_CHART,
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    metrics = [
        ("Accuracy", REPORTED_METRICS["accuracy"]),
        ("Precision", REPORTED_METRICS["precision_macro"]),
        ("Recall", REPORTED_METRICS["recall_macro"]),
        ("F1", REPORTED_METRICS["f1_macro"]),
        ("ROC-AUC", REPORTED_METRICS["roc_auc"]),
    ]

    for col, (label, val) in zip([c1, c2, c3, c4, c5], metrics):
        with col:
            stat_card(label, f"{val:.3f}")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    tmpl = plotly_template(theme)

    left, right = st.columns([1, 1], gap="large")

    with left:
        tag("Model comparison")

        rank_df = pd.DataFrame(
            {
                "Model": ["XGBoost", "Logistic Regression", "Random Forest"],
                "ROC-AUC": [0.806, 0.794, 0.786],
            }
        )

        fig = px.bar(
            rank_df,
            x="ROC-AUC",
            y="Model",
            orientation="h",
            color="Model",
            color_discrete_sequence=[
                theme["accent"],
                theme["accent_bright"],
                theme["text_faint"],
            ],
            range_x=[0.7, 0.82],
        )

        fig.update_layout(template=tmpl, height=300, showlegend=False, yaxis_title="")

        st.plotly_chart(fig, width="stretch")

    with right:
        tag("Confusion matrix")

        z = [
            [STATIC_CONFUSION["tn"], STATIC_CONFUSION["fp"]],
            [STATIC_CONFUSION["fn"], STATIC_CONFUSION["tp"]],
        ]

        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=["Pred. dissatisfied", "Pred. satisfied"],
                y=["Actual dissatisfied", "Actual satisfied"],
                colorscale=[[0, theme["panel_alt"]], [1, theme["accent"]]],
                text=z,
                texttemplate="%{text}",
                showscale=False,
            )
        )

        fig.update_layout(template=tmpl, height=300)

        st.plotly_chart(fig, width="stretch")

        st.caption(
            "Confusion matrix is from a fixed 3,000-row subsample of the test set "
            "(seed=42, threshold=0.33). Headline metrics above are from the full "
            "17,182-row test set and will differ slightly."
        )

    st.markdown("---")

    tag("Feature importance")

    st.caption(
        "Structured signals and text terms with the strongest influence on the model's predictions."
    )

    with st.spinner("Loading..."):
        contributions = top_global_features(artifacts, label_maps)

    shap_bars(theme, contributions)

    st.markdown("---")

    with st.expander("Hyperparameters"):
        hp_df = pd.DataFrame(
            {
                "Parameter": [
                    "n_estimators",
                    "max_depth",
                    "learning_rate",
                    "scale_pos_weight",
                    "tree_method",
                ],
                "Value": ["300", "6", "0.1", "0.213", "hist"],
            }
        )

        st.table(hp_df)


# page: about


def page_about(theme):
    header("About", "How this was built", icon_html=ICON_INFO)

    c1, c2 = st.columns([1.3, 1])

    with c1:
        st.markdown("""
**Pipeline**

1. Missing values imputed, duplicates removed, response-time outliers winsorized (IQR).
2. Response time, issue hour/day-of-week, ordinal tenure encoding, label-encoded categoricals,
   smoothed target encoding for agent and supervisor historical CSAT.
3. Customer remarks cleaned (contraction expansion, lowercasing, punctuation/URL/digit removal,
   stopword removal, lemmatization) and vectorized with TF-IDF, 500 features, 1-3 n-grams.
4. Logistic Regression, Random Forest, and XGBoost compared; XGBoost selected after tuning (ROC-AUC 0.8063).
5. Decision threshold swept on the test set; 0.33 selected to maximise macro-F1. Trade-off: Dissatisfied recall is 0.45 at this threshold vs 0.72 at the default 0.5 — raise the threshold if catching more at-risk tickets matters more than balanced F1.
6. SHAP TreeExplainer runs per prediction on the Predict page for real feature attributions.

**Use cases**

- Flag high-risk tickets before a customer rates them.
- Route the riskiest tickets to senior agents first.
- Use the Explorer to find systemic drivers — slow categories, weak shifts, underperforming teams.
- The live check on Model Performance catches drift between the deployed app and what was trained.
        """)
    with c2:
        st.markdown("""
**Stack**

Model: XGBoost
NLP: NLTK, TF-IDF
Scaling: PowerTransformer, StandardScaler
Explainability: SHAP
App: Streamlit, Plotly

**Dataset**

85,907 Flipkart support interactions.
Target: CSAT score of 4 or higher.

**Disclaimer**

Flipkart is a trademark of its respective owner. This project is for
educational purposes and isn't affiliated with Flipkart.
        """)


# main


def main():
    st.set_page_config(
        page_title="Flipkart CSAT Intelligence Suite",
        page_icon="⭐",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if "dark_mode" not in st.session_state:
        st.session_state["dark_mode"] = True

    theme = DARK if st.session_state["dark_mode"] else LIGHT

    inject_css(theme)

    artifacts, missing = load_artifacts()

    label_maps, le_missing = load_label_maps()

    df, data_error = load_dataset()

    ready = artifacts is not None and label_maps is not None and df is not None

    with st.sidebar:
        st.markdown(
            f"""
            <div style="padding: 0.4rem 0 1.2rem 0;">
                <span style="font-family:var(--font-mono); font-size:0.7rem; letter-spacing:0.16em;
                color:{theme['accent']}; text-transform:uppercase;">customer intelligence</span>
                <h2 style="margin:0.15rem 0 0 0; font-size:1.5rem; display:flex; align-items:center; gap:0.5rem;">{ICON_CSAT} CSAT Suite</h2>
                <div style="font-size:0.76rem;color:{theme['text_faint']};margin-top:2px;">
                XGBoost · TF-IDF · SHAP</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        NAV_OPTIONS = [
            "Home",
            "Predict",
            "Batch scoring",
            "Explorer",
            "Model performance",
            "About",
        ]

        if st.session_state.get("_nav_hint"):
            st.session_state["_nav_radio"] = st.session_state.pop("_nav_hint")

        page = st.radio(
            "Navigate",
            options=NAV_OPTIONS,
            label_visibility="collapsed",
            key="_nav_radio",
        )

        st.markdown("<hr style='margin:1.0rem 0;'>", unsafe_allow_html=True)

        st.markdown(
            '<span style="font-family:var(--font-mono); font-size:0.7rem; '
            'letter-spacing:0.1em; color:var(--text-2, #8e8577); text-transform:uppercase;">theme</span>',
            unsafe_allow_html=True,
        )

        theme_choice = st.radio(
            "Theme",
            options=["Dark", "Light"],
            index=0 if st.session_state["dark_mode"] else 1,
            horizontal=True,
            label_visibility="collapsed",
            key="_theme_radio",
        )

        new_dark_mode = theme_choice == "Dark"

        if new_dark_mode != st.session_state["dark_mode"]:
            st.session_state["dark_mode"] = new_dark_mode

            st.rerun()

        st.markdown("<hr style='margin:1.0rem 0;'>", unsafe_allow_html=True)

        if ready:
            st.markdown(
                f'<div class="card" style="padding:14px 16px;">'
                f'<div style="display:flex;align-items:center;gap:8px;">'
                f'<div style="width:7px;height:7px;border-radius:50%;background:{theme["accent"]};"></div>'
                f'<span style="font-size:0.85rem;font-weight:600;">Data connected</span></div>'
                f'<div class="card-sub" style="margin-top:6px;">{len(df):,} records loaded</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="card" style="padding:14px 16px;">'
                f'<div style="display:flex;align-items:center;gap:8px;">'
                f'<div style="width:7px;height:7px;border-radius:50%;background:{theme["bad"]};"></div>'
                f'<span style="font-size:0.85rem;font-weight:600;">Setup incomplete</span></div></div>',
                unsafe_allow_html=True,
            )

    if not ready:
        header("Setup required", "Model artifacts or data not found")

        st.markdown(
            "Place the `models/` folder (best_xgboost_classifier.pkl, tfidf_vectorizer.pkl, "
            "standard_scaler.pkl, power_transformer.pkl, label_encoders.pkl) and "
            "`Customer_support_data.csv` next to this file, then reload."
        )

        return

    model, tfidf, scaler, pt = artifacts

    lookups = build_lookups(df)

    if page == "Home":
        page_home(theme, df)
    elif page == "Predict":
        page_predict(theme, df, lookups, label_maps, artifacts)
    elif page == "Batch scoring":
        page_batch(theme, df, lookups, label_maps, artifacts)
    elif page == "Explorer":
        page_explorer(theme, df)
    elif page == "Model performance":
        page_model_performance(theme, df, lookups, label_maps, artifacts)
    elif page == "About":
        page_about(theme)

    st.markdown(
        '<div class="footer">Flipkart CSAT Intelligence Suite — portfolio project, not affiliated with Flipkart</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()