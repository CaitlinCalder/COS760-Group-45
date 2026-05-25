"""
SADiLaR Morphology-Based Text Detection Classifier

This script trains and evaluates a Random Forest classifier
using morphology-informed linguistic features extracted from
SADiLaR resources for isiZulu, isiXhosa, and Siswati.

The model predicts whether text is:
0 = Human-written
1 = Machine-generated

Performance is evaluated using Macro F1-score,
MCC, classification reports, and feature importance.
"""

import os
import json
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    f1_score
)

from sklearn.ensemble import RandomForestClassifier


BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE_PATH,
    "data",
    "processed",
    "sadilar_morph_features.csv"
)

RESULTS_DIR = os.path.join(
    BASE_PATH,
    "results",
    "sadilar_analysis"
)

os.makedirs(RESULTS_DIR, exist_ok=True)

RESULTS_FILE = os.path.join(
    RESULTS_DIR,
    "sadilar_results.json"
)

FEATURE_COLUMNS = [
    "word_count",
    "matched_words",
    "unmatched_words",
    "sadilar_coverage",
    "avg_word_length",
    "unique_word_ratio",
    "unique_morph_analysis_count",
    "morph_diversity_ratio"
]


print("Loading SADiLaR feature dataset...")
df = pd.read_csv(INPUT_FILE)

X = df[FEATURE_COLUMNS]
y = df["Label"]

print("\nFeature columns:")
print(FEATURE_COLUMNS)

print("\nSplitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Random Forest classifier...")
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

print("Making predictions...")
y_pred = model.predict(X_test)

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

report_text = classification_report(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

macro_f1 = f1_score(y_test, y_pred, average="macro")
mcc = matthews_corrcoef(y_test, y_pred)

feature_importance = {
    feature: float(importance)
    for feature, importance in zip(
        FEATURE_COLUMNS,
        model.feature_importances_
    )
}

sorted_feature_importance = dict(
    sorted(
        feature_importance.items(),
        key=lambda item: item[1],
        reverse=True
    )
)

results = {
    "experiment": "SADiLaR morphology-based classifier",
    "model": "RandomForestClassifier",
    "label_mapping": {
        "0": "Human-written",
        "1": "Machine-generated"
    },
    "feature_columns": FEATURE_COLUMNS,
    "test_size": 0.2,
    "random_state": 42,
    "macro_f1": float(macro_f1),
    "mcc": float(mcc),
    "classification_report": report,
    "confusion_matrix": cm.tolist(),
    "feature_importance": sorted_feature_importance
}

with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

print("\nClassification Report:")
print(report_text)

print(f"\nMacro F1: {macro_f1:.4f}")
print(f"MCC: {mcc:.4f}")

print("\nFeature Importance:")
for feature, importance in sorted_feature_importance.items():
    print(f"{feature}: {importance:.6f}")

print(f"\nSaved all results to: {RESULTS_FILE}")
print("\nDone.")