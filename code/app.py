import os
import re
import string
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import torch
import shap

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

BASE_PATH     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_FILE  = os.path.join(BASE_PATH, "data", "processed", "merged_dataset.csv")
FEATURES_FILE = os.path.join(BASE_PATH, "data", "processed", "sadilar_morph_features.csv")
SADILAR_PATH  = os.path.join(BASE_PATH, "data", "sadilar")

MORPH_FILES = {
    "zu": os.path.join(SADILAR_PATH, "zulu_morph.txt"),
    "xh": os.path.join(SADILAR_PATH, "xhosa_morph.txt"),
    "ss": os.path.join(SADILAR_PATH, "siswati_morph.txt"),
}

#update this to wherever your best_model folder is saved locally
AFROXLMR_MODEL_PATH = os.environ.get(
    "AFROXLMR_MODEL_PATH",
    os.path.join(BASE_PATH, "models", "best_model")
)
MAX_LENGTH = 512

SADILAR_FEATURE_COLUMNS = [
    "word_count", "matched_words", "unmatched_words", "sadilar_coverage",
    "avg_word_length", "unique_word_ratio", "unique_morph_analysis_count",
    "morph_diversity_ratio", "word_repetition_rate", "bigram_repetition_rate",
]
ALL_FEATURE_COLUMNS = ["prob_human", "prob_machine"] + SADILAR_FEATURE_COLUMNS

FEATURE_LABELS = {
    "prob_human":                  "AfroXLMR: P(Human)",
    "prob_machine":                "AfroXLMR: P(Machine)",
    "word_count":                  "Word Count",
    "matched_words":               "SADiLaR Matched Words",
    "unmatched_words":             "SADiLaR Unmatched Words",
    "sadilar_coverage":            "SADiLaR Coverage",
    "avg_word_length":             "Avg Word Length",
    "unique_word_ratio":           "Lexical Diversity (Unique Word Ratio)",
    "unique_morph_analysis_count": "Unique Morphological Analyses",
    "morph_diversity_ratio":       "Morphological Diversity Ratio",
    "word_repetition_rate":        "Word Repetition Rate",
    "bigram_repetition_rate":      "Bigram Repetition Rate",
}

st.set_page_config(
    page_title="MGT Detection-Bantu Languages",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .stApp { background-color: #ffffff; }

    h1 { font-family: Georgia, serif; color: #1a1a1a; font-size: 26px; }
    h2, h3 { font-family: Georgia, serif; color: #1a1a1a; }

    .section-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #888888;
        margin-bottom: 4px;
    }

    .phase-header {
        font-size: 14px;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #1a1a1a;
        border-bottom: 1px solid #e0e0e0;
        padding-bottom: 6px;
        margin-bottom: 14px;
    }

    .pred-human {
        display: inline-block;
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
        border-radius: 3px;
        padding: 6px 18px;
        font-weight: bold;
        font-size: 16px;
    }

    .pred-machine {
        display: inline-block;
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
        border-radius: 3px;
        padding: 6px 18px;
        font-weight: bold;
        font-size: 16px;
    }

    .metric-block { margin-top: 8px; }
    .metric-lbl { font-size: 11px; color: #888888; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-val { font-size: 20px; font-weight: bold; color: #1a1a1a; }

    .note-text { font-size: 12px; color: #777777; margin-top: 6px; font-style: italic; }

    hr { border: none; border-top: 1px solid #e0e0e0; margin: 24px 0; }
</style>
""", unsafe_allow_html=True)


def tokenize_text(text):
    return re.findall(r"[a-zA-ZÀ-ÿ]+", str(text).lower())

def clean_for_tfidf(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())

def prediction_html(pred, conf):
    if pred == 0:
        return f'<span class="pred-human">Human-Written &nbsp;({conf:.1%})</span>'
    return f'<span class="pred-machine">Machine-Generated &nbsp;({conf:.1%})</span>'

def extract_sadilar_features(text, lookup):
    tokens     = tokenize_text(text)
    word_count = len(tokens)

    if word_count == 0:
        return {col: 0.0 for col in SADILAR_FEATURE_COLUMNS}

    matched, analyses = 0, []
    for token in tokens:
        if token in lookup:
            matched += 1
            analyses.append(lookup[token])

    unique_words    = len(set(tokens))
    unique_analyses = len(set(analyses))

    counts = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    repeated             = sum(1 for c in counts.values() if c > 1)
    word_repetition_rate = repeated / unique_words if unique_words > 0 else 0.0

    bigrams = list(zip(tokens[:-1], tokens[1:]))
    bigram_rep = (1 - len(set(bigrams)) / len(bigrams)) if bigrams else 0.0

    return {
        "word_count":                  word_count,
        "matched_words":               matched,
        "unmatched_words":             word_count - matched,
        "sadilar_coverage":            matched / word_count,
        "avg_word_length":             sum(len(t) for t in tokens) / word_count,
        "unique_word_ratio":           unique_words / word_count,
        "unique_morph_analysis_count": unique_analyses,
        "morph_diversity_ratio":       unique_analyses / word_count,
        "word_repetition_rate":        word_repetition_rate,
        "bigram_repetition_rate":      bigram_rep,
    }

#to make faster for demo

@st.cache_resource
def load_morph_lookups():
    lookups = {}
    for lang, path in MORPH_FILES.items():
        if not os.path.exists(path):
            lookups[lang] = {}
            continue
        lookup = {}
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line or "\t" not in line:
                    continue
                parts = line.split("\t")
                lookup[parts[0].lower()] = "\t".join(parts[1:])
        lookups[lang] = lookup
    return lookups

@st.cache_resource
def load_baseline():
    df       = pd.read_csv(DATASET_FILE)
    train_df = df[df["Language_Code"].isin(["zu", "xh"])].copy()
    X        = train_df["Text_Generated"].apply(clean_for_tfidf)
    y        = train_df["Label"]
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    tfidf = TfidfVectorizer(
        max_features=10000, ngram_range=(3, 6),
        sublinear_tf=True, analyzer="char_wb", min_df=5,
    )
    clf = LogisticRegression(
        class_weight="balanced", random_state=42, max_iter=1000, solver="lbfgs",
    )
    tfidf.fit(X_tr)
    clf.fit(tfidf.transform(X_tr), y_tr)
    return tfidf, clf

@st.cache_resource
def load_afroxlmr():
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(AFROXLMR_MODEL_PATH)
    model     = AutoModelForSequenceClassification.from_pretrained(
        AFROXLMR_MODEL_PATH
    ).to(device)
    model.eval()
    return tokenizer, model, device

@st.cache_resource
def load_phase3_rf():
    """Train the augmented RF on the precomputed features file."""
    if not os.path.exists(FEATURES_FILE):
        raise FileNotFoundError(
            f"SADiLaR features file not found at:\n  {FEATURES_FILE}\n"
            "Please run code/sadilar_morph_features.py first to generate it."
        )
    df = pd.read_csv(FEATURES_FILE)
    df["Language_Code"] = df["Language_Code"].str.strip().str.lower()
    train_df = df[df["Language_Code"].isin(["zu", "xh"])].copy()

    tokenizer, afroxlmr, device = load_afroxlmr()
    texts, all_probs = train_df["Text_Generated"].tolist(), []
    afroxlmr.eval()

    for i in range(0, len(texts), 16):
        batch  = texts[i : i + 16]
        inputs = tokenizer(
            batch, padding=True, truncation=True,
            max_length=MAX_LENGTH, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = afroxlmr(**inputs).logits
        all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())

    probs = np.vstack(all_probs)
    train_df = train_df.copy()
    train_df["prob_human"]   = probs[:, 0]
    train_df["prob_machine"] = probs[:, 1]

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(train_df[ALL_FEATURE_COLUMNS], train_df["Label"])
    return rf

def get_xlmr_probs(text, tokenizer, model, device):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    return torch.softmax(logits, dim=-1).squeeze().cpu().numpy()


st.title("Detecting Machine-Generated Text in African Languages")
st.markdown(
    "**COS760-Group 45** &nbsp;|&nbsp; "
    "Transfer Learning and Linguistic Feature Analysis in Bantu Languages"
)
st.markdown("<hr>", unsafe_allow_html=True)

col_lang, col_input = st.columns([1, 4])

with col_lang:
    language = st.selectbox(
        "Language",
        options=["zu", "xh", "ss"],
        format_func=lambda x: {"zu": "isiZulu", "xh": "isiXhosa", "ss": "Siswati"}[x],
    )

with col_input:
    input_text = st.text_area(
        "Input Text",
        height=120,
        placeholder="Paste isiZulu, isiXhosa, or Siswati text here...",
        label_visibility="visible",
    )

run = st.button("Run Analysis", type="primary")


if run and input_text.strip():

    with st.spinner("Loading models-this may take a moment on first run..."):
        tfidf, baseline_clf          = load_baseline()
        tokenizer, afroxlmr, device  = load_afroxlmr()
        rf_model                     = load_phase3_rf()
        morph_lookups                = load_morph_lookups()

    #Phase 1
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="phase-header">Phase 1-Baseline: TF-IDF + Logistic Regression</div>', unsafe_allow_html=True)

    X_vec    = tfidf.transform([clean_for_tfidf(input_text)])
    p1_pred  = baseline_clf.predict(X_vec)[0]
    p1_proba = baseline_clf.predict_proba(X_vec)[0]
    p1_conf  = p1_proba[p1_pred]

    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        st.markdown('<div class="section-label">Prediction</div>', unsafe_allow_html=True)
        st.markdown(prediction_html(p1_pred, p1_conf), unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-block"><div class="metric-lbl">P(Human)</div>'
                    f'<div class="metric-val">{p1_proba[0]:.3f}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-block"><div class="metric-lbl">P(Machine)</div>'
                    f'<div class="metric-val">{p1_proba[1]:.3f}</div></div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="note-text">Character n-gram TF-IDF matching-lexical surface patterns only. '
        'No language understanding. No transfer learning.</div>',
        unsafe_allow_html=True,
    )

    #Phase 2
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="phase-header">Phase 2-Transfer Learning: Fine-tuned AfroXLMR</div>', unsafe_allow_html=True)

    with st.spinner("Running AfroXLMR inference..."):
        p2_proba = get_xlmr_probs(input_text, tokenizer, afroxlmr, device)

    p2_pred = int(np.argmax(p2_proba))
    p2_conf = float(p2_proba[p2_pred])

    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        st.markdown('<div class="section-label">Prediction</div>', unsafe_allow_html=True)
        st.markdown(prediction_html(p2_pred, p2_conf), unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-block"><div class="metric-lbl">P(Human)</div>'
                    f'<div class="metric-val">{p2_proba[0]:.3f}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-block"><div class="metric-lbl">P(Machine)</div>'
                    f'<div class="metric-val">{p2_proba[1]:.3f}</div></div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="note-text">Multilingual contextual embeddings from AfroXLMR, '
        'pre-trained on African language corpora and fine-tuned on isiZulu and isiXhosa.</div>',
        unsafe_allow_html=True,
    )

    #Phase 3
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<div class="phase-header">Phase 3-Feature Augmentation: AfroXLMR + SADiLaR Linguistic Features</div>',
        unsafe_allow_html=True,
    )

    sadilar_feats = extract_sadilar_features(input_text, morph_lookups.get(language, {}))
    feature_vec   = pd.DataFrame([{"prob_human": p2_proba[0], "prob_machine": p2_proba[1], **sadilar_feats}])[ALL_FEATURE_COLUMNS]

    p3_pred  = rf_model.predict(feature_vec)[0]
    p3_proba = rf_model.predict_proba(feature_vec)[0]
    p3_conf  = float(p3_proba[p3_pred])

    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        st.markdown('<div class="section-label">Prediction</div>', unsafe_allow_html=True)
        st.markdown(prediction_html(p3_pred, p3_conf), unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-block"><div class="metric-lbl">P(Human)</div>'
                    f'<div class="metric-val">{p3_proba[0]:.3f}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-block"><div class="metric-lbl">P(Machine)</div>'
                    f'<div class="metric-val">{p3_proba[1]:.3f}</div></div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="note-text">AfroXLMR probabilities augmented with SADiLaR morphological '
        'and stylistic features. Random Forest classifier trained on isiZulu + isiXhosa, '
        'evaluated zero-shot on Siswati.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.markdown("**Extracted Linguistic Features**")
        feat_df = pd.DataFrame({
            "Feature": [FEATURE_LABELS.get(k, k) for k in ALL_FEATURE_COLUMNS],
            "Value":   [round(float(feature_vec[k].iloc[0]), 4) for k in ALL_FEATURE_COLUMNS],
            "Source":  ["AfroXLMR" if k.startswith("prob_") else "SADiLaR" for k in ALL_FEATURE_COLUMNS],
        })
        st.dataframe(feat_df, use_container_width=True, hide_index=True, height=380)

    with right_col:
        st.markdown("**SHAP Feature Contributions-Why this prediction?**")

        explainer = shap.TreeExplainer(rf_model)
        shap_vals = explainer.shap_values(feature_vec)

        sv_arr = np.array(shap_vals)
        if isinstance(shap_vals, list):
            sv = shap_vals[1][0]
        elif sv_arr.ndim == 3:
            sv = sv_arr[0, :, 1]
        else:
            sv = sv_arr[0]

        labels    = [FEATURE_LABELS.get(c, c) for c in ALL_FEATURE_COLUMNS]
        sort_idx  = np.argsort(np.abs(sv))
        sv_sorted = sv[sort_idx]
        lb_sorted = [labels[i] for i in sort_idx]
        bar_colors = ["#c0392b" if v > 0 else "#2d6a4f" for v in sv_sorted]

        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.barh(lb_sorted, sv_sorted, color=bar_colors, height=0.6)
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.set_xlabel("SHAP Value", fontsize=10)
        ax.set_title("Feature Contributions (Machine-Generated class)", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown(
            '<div class="note-text">'
            '<span style="color:#c0392b; font-weight:bold;">Red</span> = pushes towards Machine-Generated. '
            '<span style="color:#2d6a4f; font-weight:bold;">Green</span> = pushes towards Human-Written.'
            '</div>',
            unsafe_allow_html=True,
        )

    #comparison table
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="phase-header">Phase Comparison</div>', unsafe_allow_html=True)

    pred_label = {0: "Human-Written", 1: "Machine-Generated"}

    comparison = pd.DataFrame({
        "Phase": [
            "Phase 1: TF-IDF + LR",
            "Phase 2: AfroXLMR",
            "Phase 3: Augmented",
        ],
        "Approach": [
            "Lexical surface pattern matching",
            "Multilingual transfer learning",
            "Transfer learning + linguistic features",
        ],
        "Prediction": [pred_label[p1_pred], pred_label[p2_pred], pred_label[p3_pred]],
        "P(Human)":   [f"{p1_proba[0]:.3f}", f"{p2_proba[0]:.3f}", f"{p3_proba[0]:.3f}"],
        "P(Machine)": [f"{p1_proba[1]:.3f}", f"{p2_proba[1]:.3f}", f"{p3_proba[1]:.3f}"],
        "Confidence": [f"{p1_conf:.1%}", f"{p2_conf:.1%}", f"{p3_conf:.1%}"],
    })

    def color_pred(val):
        if val == "Human-Written":
            return "background-color: #d4edda; color: #155724;"
        if val == "Machine-Generated":
            return "background-color: #f8d7da; color: #721c24;"
        return ""

    styled = comparison.style.map(color_pred, subset=["Prediction"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

elif run and not input_text.strip():
    st.warning("Please enter some text before running the analysis.")