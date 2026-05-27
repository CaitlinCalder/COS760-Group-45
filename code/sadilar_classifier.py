#NEW
"""
SADiLaR Phase 3 — Feature-Augmented Classifier

Builds on Phase 2 (fine-tuned AfroXLMR) by combining its predicted
probabilities with SADiLaR morphological features. This is the actual
feature augmentation described in the proposal, Phase 2's transfer
learning output is used as input here, not replaced.

The Random Forest is trained on:
  - AfroXLMR predicted probabilities (prob_human, prob_machine)
  - SADiLaR morphological and stylistic features

Training: isiZulu + isiXhosa
Test (zero-shot): Siswati (cross-language generalisation)

Performance is evaluated using Macro F1, MCC, and compared against
both Phase 1 (baseline) and Phase 2 (AfroXLMR alone).
"""

import os
import json
import torch
import numpy as np
import pandas as pd

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score
)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE_PATH, "data", "processed", "sadilar_morph_features.csv"
)

# Phase 2 fine-tuned AfroXLMR model — loaded to extract probabilities
AFROXLMR_MODEL_PATH = r"C:\Users\caitl\Downloads\best_model\best_model"

PHASE1_METRICS_JSON = os.path.join(
    "/content/drive/MyDrive/afroxlmr_detector", "baseline_metrics.json"
)

PHASE2_METRICS_JSON = os.path.join(
    "/content/drive/MyDrive/afroxlmr_detector", "phase2_metrics.json"
)

RESULTS_DIR = os.path.join(BASE_PATH, "results", "sadilar_analysis")
os.makedirs(RESULTS_DIR, exist_ok=True)

RESULTS_FILE = os.path.join(RESULTS_DIR, "sadilar_results.json")

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

#run the fine-tuned AfroXLMR model over all texts and return softmax probabilities [prob_human, prob_machine] for each sample to make Phase 3 an augmentation of Phase 2
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
df = pd.read_csv(INPUT_FILE)
df["Language_Code"] = df["Language_Code"].str.strip().str.lower()

train_df = df[df["Language_Code"].isin(["zu", "xh"])].copy()
test_df = df[df["Language_Code"] == "ss"].copy()

print(f"Train (zu+xh): {len(train_df)} samples")
print(f"Test  (ss)   : {len(test_df)} samples")

print("\nLoading fine-tuned AfroXLMR from Phase 2...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(AFROXLMR_MODEL_PATH)
afroxlmr = AutoModelForSequenceClassification.from_pretrained(
    AFROXLMR_MODEL_PATH
).to(device)

print("Extracting AfroXLMR probabilities for training set...")
train_texts = train_df["Text_Generated"].tolist()
train_probs = extract_afroxlmr_probabilities(train_texts, afroxlmr, tokenizer, device)

print("Extracting AfroXLMR probabilities for Siswati test set...")
test_texts = test_df["Text_Generated"].tolist()
test_probs = extract_afroxlmr_probabilities(test_texts, afroxlmr, tokenizer, device)

train_df = train_df.copy()
test_df = test_df.copy()

train_df["prob_human"] = train_probs[:, 0]
train_df["prob_machine"] = train_probs[:, 1]
test_df["prob_human"] = test_probs[:, 0]
test_df["prob_machine"] = test_probs[:, 1]

X_train = train_df[ALL_FEATURE_COLUMNS]
y_train = train_df["Label"]

X_test = test_df[ALL_FEATURE_COLUMNS]
y_test = test_df["Label"]

print("\nTraining augmented Random Forest (Phase 2 probs + SADiLaR features)...")
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

print("Evaluating on Siswati (zero-shot)...")
y_pred = model.predict(X_test)
y_proba= model.predict_proba(X_test)[:, 1]

report = classification_report(y_test, y_pred, output_dict=True)
report_text = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")
mcc = matthews_corrcoef(y_test, y_pred)
auc_roc     = round(roc_auc_score(y_test, y_proba), 4)
auc_pr      = round(average_precision_score(y_test, y_proba), 4)

feature_importance = {
    feature: float(importance)
    for feature, importance in zip(ALL_FEATURE_COLUMNS, model.feature_importances_)
}
sorted_feature_importance = dict(
    sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)
)

phase1, phase2 = {}, {}

if os.path.exists(PHASE1_METRICS_JSON):
    with open(PHASE1_METRICS_JSON) as f:
        phase1_all = json.load(f)
    phase1 = phase1_all.get("siswati_zeroshot", {})
    print(f"\nLoaded Phase 1 metrics from {PHASE1_METRICS_JSON}")

if os.path.exists(PHASE2_METRICS_JSON):
    with open(PHASE2_METRICS_JSON) as f:
        phase2_all = json.load(f)
    phase2 = phase2_all.get("siswati_crosslingual", {})
    print(f"Loaded Phase 2 metrics from {PHASE2_METRICS_JSON}")

phase3 = {
    "macro_f1": round(float(macro_f1), 4),
    "mcc": round(float(mcc), 4),
    "precision": round(float(report["macro avg"]["precision"]), 4),
    "recall": round(float(report["macro avg"]["recall"]), 4),
    "auc_roc":   auc_roc,
    "auc_pr":    auc_pr,
}

print("\n" + "=" * 65)
print("PHASE COMPARISON — Siswati Zero-Shot (Cross-Lingual)")
print("=" * 65)
header = f"  {'Metric':<14} {'TF-IDF+LR (P1)':>16} {'AfroXLMR (P2)':>16} {'Augmented (P3)':>16}"
print(header)
print("  " + "-" * (len(header) - 2))

for m in ["precision", "recall", "macro_f1", "mcc"]:
    p1_val = phase1.get(m, float("nan"))
    p2_val = phase2.get(m, float("nan"))
    p3_val = phase3.get(m, float("nan"))
    print(f"  {m:<14} {p1_val:>16.4f} {p2_val:>16.4f} {p3_val:>16.4f}")

print("\nClassification Report (Phase 3 — Siswati):")
print(report_text)
print(f"Macro F1 : {macro_f1:.4f}")
print(f"MCC      : {mcc:.4f}")
print(f"AUC-ROC  : {auc_roc:.4f}")
print(f"AUC-PR   : {auc_pr:.4f}")

#cross-LLM breakdown on Siswati to check if the augmented model generalises equally across ChatGPT, Claude, Gemini
print("\n" + "=" * 65)
print("CROSS-LLM GENERALISATION — Siswati Zero-Shot (Phase 3)")
print("=" * 65)

if "Model_Identifier" in test_df.columns:
    llm_results = {}
    for llm in sorted(test_df["Model_Identifier"].unique()):
        if llm == "human":
            continue

        human_mask   = test_df["Label"] == 0
        machine_mask = (test_df["Label"] == 1) & (test_df["Model_Identifier"] == llm)
        subset_idx   = test_df[human_mask | machine_mask].index

        X_llm = test_df.loc[subset_idx, ALL_FEATURE_COLUMNS]
        y_llm = test_df.loc[subset_idx, "Label"]

        if len(y_llm.unique()) < 2:
            continue

        y_llm_pred  = model.predict(X_llm)
        y_llm_proba = model.predict_proba(X_llm)[:, 1]

        llm_metrics = {
            "n_machine":  int(machine_mask.sum()),
            "precision":  round(precision_score(y_llm, y_llm_pred, average="macro", zero_division=0), 4),
            "recall":     round(recall_score(y_llm, y_llm_pred, average="macro", zero_division=0), 4),
            "macro_f1":   round(f1_score(y_llm, y_llm_pred, average="macro", zero_division=0), 4),
            "mcc":        round(matthews_corrcoef(y_llm, y_llm_pred), 4),
            "auc_roc":    round(roc_auc_score(y_llm, y_llm_proba), 4),
            "auc_pr":     round(average_precision_score(y_llm, y_llm_proba), 4),
        }
        llm_results[llm] = llm_metrics

        print(f"\n  {llm} (n_machine={llm_metrics['n_machine']})")
        for k, v in llm_metrics.items():
            if k != "n_machine":
                print(f"    {k:<12}: {v}")
else:
    print("  Model_Identifier column not found in test set — skipping cross-LLM breakdown")
    llm_results = {}

print("\nFeature Importance (what the augmented model relies on):")
for feature, importance in sorted_feature_importance.items():
    label = "(AfroXLMR)" if feature.startswith("prob_") else "(SADiLaR)"
    print(f"  {feature:<35} {label} {importance:.6f}")

results = {
    "experiment": "Phase 3 — SADiLaR feature-augmented AfroXLMR classifier",
    "model": "RandomForestClassifier + AfroXLMR probabilities + SADiLaR features",
    "train_languages": ["zu", "xh"],
    "test_language": "ss",
    "feature_columns": ALL_FEATURE_COLUMNS,
    "phase3_metrics": phase3,
    "phase_comparison": {
        "phase1_tfidf_lr": phase1,
        "phase2_afroxlmr": phase2,
        "phase3_augmented": phase3,
    },
    "classification_report": report,
    "confusion_matrix": cm.tolist(),
    "feature_importance": sorted_feature_importance,
}

with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

print(f"\nSaved all results to: {RESULTS_FILE}")
print("\nDone.")