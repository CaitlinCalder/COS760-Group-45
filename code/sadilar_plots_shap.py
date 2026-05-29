#NEW
"""
Visualisation and SHAP Analysis for phase 3

Creates plots and SHAP explanations for the Phase 3 augmented classifier.
SHAP is applied to the combined model (AfroXLMR probabilities + SADiLaR
features) to reveal whether detection relies on AfroXLMR's learned
representations or deeper linguistic/morphological structure.

Also produces a 3-phase comparison chart (Phase 1 vs 2 vs 3).
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import seaborn as sns

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, roc_auc_score, average_precision_score, precision_score, recall_score, matthews_corrcoef

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FEATURES_FILE = os.path.join(
    BASE_PATH, "data", "processed", "sadilar_morph_features.csv"
)

AFROXLMR_MODEL_PATH = os.environ.get(
    "AFROXLMR_MODEL_PATH",
    os.path.join(BASE_PATH, "models", "best_model")
)

PHASE1_METRICS_JSON = os.path.join(
    "/content/drive/MyDrive/afroxlmr_detector", "baseline_metrics.json"
)

PHASE2_METRICS_JSON = os.path.join(
    "/content/drive/MyDrive/afroxlmr_detector", "phase2_metrics.json"
)

RESULTS_DIR = os.path.join(BASE_PATH, "results", "sadilar_analysis")
os.makedirs(RESULTS_DIR, exist_ok=True)

MAX_LENGTH = 512

SADILAR_FEATURE_COLUMNS = [
    "word_count",
    "matched_words",
    "unmatched_words",
    "sadilar_coverage",
    "avg_word_length",
    "unique_word_ratio",
    "unique_morph_analysis_count",
    "morph_diversity_ratio",
    "word_repetition_rate",
    "bigram_repetition_rate",
]

ALL_FEATURE_COLUMNS = ["prob_human", "prob_machine"] + SADILAR_FEATURE_COLUMNS

#labels for SHAP plots
FEATURE_LABELS = {
    "prob_human": "AfroXLMR: P(Human)",
    "prob_machine": "AfroXLMR: P(Machine)",
    "word_count": "Word Count",
    "matched_words": "SADiLaR Matched Words",
    "unmatched_words": "SADiLaR Unmatched Words",
    "sadilar_coverage": "SADiLaR Coverage",
    "avg_word_length": "Avg Word Length",
    "unique_word_ratio": "Lexical Diversity (Unique Word Ratio)",
    "unique_morph_analysis_count": "Unique Morphological Analyses",
    "morph_diversity_ratio": "Morphological Diversity Ratio",
    "word_repetition_rate": "Word Repetition Rate",
    "bigram_repetition_rate": "Bigram Repetition Rate",
}


def extract_afroxlmr_probabilities(texts, model, tokenizer, device, batch_size=16):
    all_probs = []
    model.eval()
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i: i + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
    return np.vstack(all_probs)


print("Loading SADiLaR feature dataset...")
df = pd.read_csv(FEATURES_FILE)
df["Language_Code"] = df["Language_Code"].str.strip().str.lower()

train_df = df[df["Language_Code"].isin(["zu", "xh"])].copy()
test_df = df[df["Language_Code"] == "ss"].copy()

print("Loading fine-tuned AfroXLMR from Phase 2...")
device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer= AutoTokenizer.from_pretrained(AFROXLMR_MODEL_PATH)
afroxlmr = AutoModelForSequenceClassification.from_pretrained(
    AFROXLMR_MODEL_PATH
).to(device)

print("Extracting AfroXLMR probabilities...")
train_probs = extract_afroxlmr_probabilities(
    train_df["Text_Generated"].tolist(), afroxlmr, tokenizer, device
)
test_probs = extract_afroxlmr_probabilities(
    test_df["Text_Generated"].tolist(), afroxlmr, tokenizer, device
)

train_df = train_df.copy()
test_df  = test_df.copy()
train_df["prob_human"] = train_probs[:, 0]
train_df["prob_machine"] = train_probs[:, 1]
test_df["prob_human"] = test_probs[:, 0]
test_df["prob_machine"] = test_probs[:, 1]

X_train = train_df[ALL_FEATURE_COLUMNS]
y_train = train_df["Label"]
X_test = test_df[ALL_FEATURE_COLUMNS]
y_test = test_df["Label"]

print("Training augmented Random Forest...")
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

phase1, phase2 = {}, {}

if os.path.exists(PHASE1_METRICS_JSON):
    with open(PHASE1_METRICS_JSON) as f:
        phase1 = json.load(f).get("siswati_zeroshot", {})

if os.path.exists(PHASE2_METRICS_JSON):
    with open(PHASE2_METRICS_JSON) as f:
        phase2 = json.load(f).get("siswati_crosslingual", {})

from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred, output_dict=True)

phase3 = {
    "macro_f1":round(float(f1_score(y_test, y_pred, average="macro")), 4),
    "precision":round(float(report["macro avg"]["precision"]), 4),
    "recall":round(float(report["macro avg"]["recall"]), 4),
    "auc_roc":round(float(roc_auc_score(y_test, y_proba)), 4),
    "auc_pr":round(float(average_precision_score(y_test, y_proba)), 4),
}

#confusion matrix
print("Plotting confusion matrix...")
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=["Human", "Machine"],
    ax=ax,
)
ax.set_title("Phase 3 Augmented Classifier - Confusion Matrix\n(Siswati Zero-Shot)")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "p3_confusion_matrix.png"), dpi=300)
plt.close()

#feature importance
print("Plotting feature importance...")
importance_df = pd.DataFrame({
    "Feature":    [FEATURE_LABELS.get(f, f) for f in ALL_FEATURE_COLUMNS],
    "Importance": model.feature_importances_,
    "Source":     ["AfroXLMR" if f.startswith("prob_") else "SADiLaR"
                   for f in ALL_FEATURE_COLUMNS],
}).sort_values("Importance", ascending=True)

colours = importance_df["Source"].map({"AfroXLMR": "#55A868", "SADiLaR": "#4C72B0"})

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(importance_df["Feature"], importance_df["Importance"], color=colours)
ax.set_xlabel("Feature Importance")
ax.set_title("Phase 3 Feature Importance: AfroXLMR vs SADiLaR Linguistic Features\n"
             "(Green = AfroXLMR transfer learning, Blue = SADiLaR linguistic)")

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#55A868", label="AfroXLMR (Transfer Learning)"),
    Patch(facecolor="#4C72B0", label="SADiLaR (Linguistic Features)"),
]
ax.legend(handles=legend_elements)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "p3_feature_importance.png"), dpi=300)
plt.close()

#three-phase comparison bar chart 
print("Plotting 3-phase comparison...")
metrics_labels = ["Precision", "Recall", "Macro F1", "AUC-ROC", "AUC-PR"]
metric_keys    = ["precision", "recall", "macro_f1", "auc_roc", "auc_pr"]

p1_vals= [phase1.get(k, float("nan")) for k in metric_keys]
p2_vals= [phase2.get(k, float("nan")) for k in metric_keys]
p3_vals= [phase3.get(k, float("nan")) for k in metric_keys]

x= np.arange(len(metrics_labels))
width= 0.25

fig, ax = plt.subplots(figsize=(10, 6))
b1= ax.bar(x - width,     p1_vals, width, label="Phase 1: TF-IDF + LR (Baseline)",
            color="#4C72B0", edgecolor="white")
b2= ax.bar(x,             p2_vals, width, label="Phase 2: AfroXLMR (Transfer Learning)",
            color="#55A868", edgecolor="white")
b3= ax.bar(x + width,     p3_vals, width, label="Phase 3: Augmented (AfroXLMR + SADiLaR)",
            color="#C44E52", edgecolor="white")

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(metrics_labels)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Score")
ax.set_title("Phase 1 vs Phase 2 vs Phase 3 - Siswati Zero-Shot (Cross-Lingual)",
             fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "p3_phase_comparison.png"), dpi=300)
plt.close()

#box plots of linguistic features by label
print("Plotting feature distribution boxplots...")
test_df_plot = test_df.copy()
test_df_plot["Label_Name"] = test_df_plot["Label"].map({0: "Human", 1: "Machine"})

PLOT_FEATURES = [
    ("sadilar_coverage","SADiLaR Coverage"),
    ("unique_word_ratio","Lexical Diversity (Unique Word Ratio)"),
    ("word_repetition_rate","Word Repetition Rate"),
    ("bigram_repetition_rate","Bigram Repetition Rate"),
    ("morph_diversity_ratio","Morphological Diversity Ratio"),
    ("avg_word_length","Average Word Length"),
]

for col, title in PLOT_FEATURES:
    plt.figure(figsize=(7, 4))
    test_df_plot.boxplot(column=col, by="Label_Name", rot=0)
    plt.title(f"{title} by Label - Siswati Test Set")
    plt.suptitle("")
    plt.xlabel("Label")
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"p3_{col}_boxplot.png"), dpi=300)
    plt.close()

#cross-LLM generalisation bar chart
print("Plotting cross-LLM generalisation...")
if "Model_Identifier" in test_df.columns:
    llm_f1s= {}
    llm_aucs= {}
    for llm in sorted(test_df["Model_Identifier"].unique()):
        if llm == "human":
            continue
        human_mask = test_df["Label"] == 0
        machine_mask = (test_df["Label"] == 1) & (test_df["Model_Identifier"] == llm)
        subset_idx = test_df[human_mask | machine_mask].index
        X_llm = test_df.loc[subset_idx, ALL_FEATURE_COLUMNS]
        y_llm = test_df.loc[subset_idx, "Label"]
        if len(y_llm.unique()) < 2:
            continue
        y_llm_pred = model.predict(X_llm)
        y_llm_proba = model.predict_proba(X_llm)[:, 1]
        llm_f1s[llm] = round(f1_score(y_llm, y_llm_pred, average="macro"), 4)
        llm_aucs[llm] = round(roc_auc_score(y_llm, y_llm_proba), 4)
 
    if llm_f1s:
        llm_names = list(llm_f1s.keys())
        f1_vals = [llm_f1s[l] for l in llm_names]
        auc_vals = [llm_aucs[l] for l in llm_names]
 
        x = np.arange(len(llm_names))
        width = 0.35
 
        fig, ax = plt.subplots(figsize=(9, 5))
        b1 = ax.bar(x - width / 2, f1_vals,  width, label="Macro F1",  color="#55A868", edgecolor="white")
        b2 = ax.bar(x + width / 2, auc_vals, width, label="AUC-ROC",   color="#C44E52", edgecolor="white")
 
        for bars in [b1, b2]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=9)
 
        ax.set_xticks(x)
        ax.set_xticklabels(llm_names, fontsize=11)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score")
        ax.set_title("Phase 3 Cross-LLM Generalisation - Siswati Zero-Shot\n"
                     "(Augmented AfroXLMR + SADiLaR)", fontweight="bold")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "p3_cross_llm_bar.png"), dpi=300)
        plt.close()
        print("  saved: p3_cross_llm_bar.png")
 
#SHAP analysis on the augmented model reveals whether detection decisions rely on AfroXLMR's representations (prob_human/prob_machine) or linguistic features
print("Running SHAP analysis on augmented model...")

X_test_labelled = X_test.copy()
X_test_labelled.columns = [FEATURE_LABELS.get(c, c) for c in X_test_labelled.columns]

explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_labelled)

if isinstance(shap_values, list):
    shap_machine = shap_values[1]
elif len(shap_values.shape) == 3:
    shap_machine = shap_values[:, :, 1]
else:
    shap_machine = shap_values

plt.figure()
shap.summary_plot(shap_machine, X_test_labelled, show=False)
plt.title("SHAP Summary - Phase 3 Augmented Model (Machine-Generated Class)")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "p3_shap_summary.png"), dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_machine, X_test_labelled, plot_type="bar", show=False)
plt.title("SHAP Feature Importance - Phase 3 Augmented Model")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "p3_shap_bar.png"), dpi=300)
plt.close()

print("\nAll Phase 3 plots saved to:")
print(RESULTS_DIR)
print("\nDone.")