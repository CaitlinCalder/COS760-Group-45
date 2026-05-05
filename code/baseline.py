#phase 1 baseline model, a TF-IDF and logistic regression classifier to detect machine generated text in isiZulu and isiXhosa, then tests it zero-shot on Siswati and the results from this phase serve as the benchmark that phases 2 and 3 must beat
#pip install scikit-learn pandas matplotlib seaborn
#need to run dataset_prep.py first

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score,
    average_precision_score, confusion_matrix,
    classification_report
)

print("all imports successful")

#build paths relative to the repo root so the script works from any machine
BASE_PATH    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(BASE_PATH, "data", "processed")
RESULTS_PATH = os.path.join(BASE_PATH, "results")

os.makedirs(os.path.join(RESULTS_PATH, "metrics"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_PATH, "plots"),   exist_ok=True)

#load the merged dataset produced by dataset_prep.py
DATASET_FILE = os.path.join(DATA_PATH, "merged_dataset.csv")

if not os.path.exists(DATASET_FILE):
    raise FileNotFoundError(
        f"processed dataset not found at:\n  {DATASET_FILE}\n"
        "please run code/dataset_prep.py first"
    )

print(f"loading dataset from: {DATASET_FILE}")
df = pd.read_csv(DATASET_FILE)

print(f"\ntotal records: {len(df)}")
print(f"\nby language:")
print(df["Language_Code"].value_counts())
print(f"\nby label (0=human, 1=machine):")
print(df["Label"].value_counts())

#separate the data by language
#isiZulu and isiXhosa are used for training and in-language testing
#Siswati is completely held out and only used for zero-shot cross-lingual evaluation
#this means the model never sees any Siswati text during training
train_df   = df[df["Language_Code"].isin(["zu", "xh"])].copy()
siswati_df = df[df["Language_Code"] == "ss"].copy()

print(f"\ntraining pool (isiZulu + isiXhosa) : {len(train_df)} records")
print(train_df["Label"].value_counts())
print(f"\nsiswati held-out (zero-shot)        : {len(siswati_df)} records")
print(siswati_df["Label"].value_counts())

#split the training pool into 80% train and 20% test
#stratify=y ensures both classes are proportionally represented in each split
X = train_df["Text_Generated"]
y = train_df["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\ntrain size: {len(X_train)}")
print(f"test size : {len(X_test)}")

#convert raw text into numerical TF-IDF features
#max_features=10000 keeps the vocabulary at a manageable size
#ngram_range=(1,2) includes both single words and two-word phrases as features
#sublinear_tf=True applies log normalisation to reduce the weight of very frequent terms
#strip_accents=None preserves special characters used in Bantu languages
#min_df=2 removes terms that only appear in one document since they are likely noise
tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    strip_accents=None,
    analyzer="word",
    min_df=2
)

#fit the vectoriser on training data only, then transform all three sets
#fitting on test or Siswati data would cause data leakage
X_train_tfidf   = tfidf.fit_transform(X_train)
X_test_tfidf    = tfidf.transform(X_test)
X_siswati_tfidf = tfidf.transform(siswati_df["Text_Generated"])
y_siswati       = siswati_df["Label"]

print(f"\nTF-IDF vocabulary size : {len(tfidf.vocabulary_)}")
print(f"train matrix shape     : {X_train_tfidf.shape}")
print(f"test matrix shape      : {X_test_tfidf.shape}")
print(f"siswati matrix shape   : {X_siswati_tfidf.shape}")

#train a logistic regression classifier on the TF-IDF features
#class_weight='balanced' adjusts for any imbalance between human and machine label counts
#lbfgs is a standard solver that works well for this size of problem
clf = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42,
    solver="lbfgs"
)

clf.fit(X_train_tfidf, y_train)
print("\nmodel trained successfully")

#metrics include everything mentioned in proposal
def evaluate_model(clf, X_vec, y_true, label="evaluation"):
    y_pred  = clf.predict(X_vec)
    y_proba = clf.predict_proba(X_vec)[:, 1]

    metrics = {
        "label"     : label,
        "precision" : round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall"    : round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "macro_f1"  : round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "mcc"       : round(matthews_corrcoef(y_true, y_pred), 4),
        "auc_roc"   : round(roc_auc_score(y_true, y_proba), 4),
        "auc_pr"    : round(average_precision_score(y_true, y_proba), 4),
    }

    print(f"\n{label}")
    for k, v in metrics.items():
        if k != "label":
            print(f"  {k:<12}: {v}")
    print(f"\n  classification report:")
    print(classification_report(y_true, y_pred, target_names=["Human", "Machine"]))

    return metrics, y_pred, y_proba


#evaluate on the in-language test set (isiZulu + isiXhosa)
#this tells us how well the model performs on languages it was trained on
metrics_inlang, y_pred_inlang, y_proba_inlang = evaluate_model(
    clf, X_test_tfidf, y_test,
    label="in-language test (isiZulu + isiXhosa)"
)

#evaluate on Siswati which the model has never seen during training
#this tests whether the lexical patterns learned from isiZulu and isiXhosa
#transfer to a related but unseen Bantu language
metrics_siswati, y_pred_siswati, y_proba_siswati = evaluate_model(
    clf, X_siswati_tfidf, y_siswati,
    label="zero-shot cross-lingual test (Siswati)"
)

#run stratified 5-fold cross validation on the full training pool
#we wrap TF-IDF and logistic regression in a pipeline so the vectoriser
#is re-fitted on each fold's training data, which prevents data leakage
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents=None,
        analyzer="word",
        min_df=2
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
        solver="lbfgs"
    ))
])

skf      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X_all    = train_df["Text_Generated"]
y_all    = train_df["Label"]

cv_preds  = cross_val_predict(pipeline, X_all, y_all, cv=skf)
cv_probas = cross_val_predict(pipeline, X_all, y_all, cv=skf, method="predict_proba")[:, 1]

cv_metrics = {
    "label"    : "5-fold cross validation",
    "macro_f1" : round(f1_score(y_all, cv_preds, average="macro"), 4),
    "mcc"      : round(matthews_corrcoef(y_all, cv_preds), 4),
    "auc_roc"  : round(roc_auc_score(y_all, cv_probas), 4),
}

print("\n5-fold cross validation results")
for k, v in cv_metrics.items():
    if k != "label":
        print(f"  {k:<12}: {v}")

#plot confusion matrices side by side for both test sets
#this shows where the model makes errors, e.g. labelling human text as machine or vice versa
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, y_true, y_pred, title in [
    (axes[0], y_test,    y_pred_inlang,  "In-Language\n(isiZulu + isiXhosa)"),
    (axes[1], y_siswati, y_pred_siswati, "Zero-Shot\n(Siswati)"),
]:
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", ax=ax,
        xticklabels=["Human", "Machine"],
        yticklabels=["Human", "Machine"],
        cmap="Blues"
    )
    ax.set_title(f"confusion matrix\n{title}", fontsize=12)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")

plt.tight_layout()
cm_path = os.path.join(RESULTS_PATH, "plots", "baseline_confusion_matrices.png")
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"saved: {cm_path}")

#plot a grouped bar chart comparing all metrics between the two test sets
#this gives a quick visual summary of how much performance drops on the unseen language
metric_keys  = ["precision", "recall", "macro_f1", "mcc", "auc_roc", "auc_pr"]
inlang_vals  = [metrics_inlang[k]  for k in metric_keys]
siswati_vals = [metrics_siswati[k] for k in metric_keys]

x     = np.arange(len(metric_keys))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 5))
bars1 = ax.bar(x - width / 2, inlang_vals,  width, label="in-language (isiZulu + isiXhosa)", color="steelblue")
bars2 = ax.bar(x + width / 2, siswati_vals, width, label="zero-shot (Siswati)",               color="coral")

ax.set_xticks(x)
ax.set_xticklabels([m.replace("_", " ").upper() for m in metric_keys])
ax.set_ylim(0, 1.1)
ax.set_ylabel("score")
ax.set_title("baseline TF-IDF + logistic regression: metric comparison")
ax.legend()
ax.bar_label(bars1, fmt="%.2f", padding=2, fontsize=8)
ax.bar_label(bars2, fmt="%.2f", padding=2, fontsize=8)

plt.tight_layout()
bar_path = os.path.join(RESULTS_PATH, "plots", "baseline_metrics_comparison.png")
plt.savefig(bar_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"saved: {bar_path}")

#save all metrics to a json file so they can be referenced when comparing phases
all_metrics = {
    "in_language"      : metrics_inlang,
    "siswati_zeroshot" : metrics_siswati,
    "cross_validation" : cv_metrics,
}

metrics_path = os.path.join(RESULTS_PATH, "metrics", "baseline_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(all_metrics, f, indent=2)

print(f"\nmetrics saved to: {metrics_path}")

#print a final summary of the key numbers
#these are the scores that the AfroXLMR model in phase 2 needs to improve on
print("\nbaseline summary (TF-IDF + logistic regression)")
print(f"  in-language macro-F1  : {metrics_inlang['macro_f1']}")
print(f"  in-language MCC       : {metrics_inlang['mcc']}")
print(f"  in-language AUC-ROC   : {metrics_inlang['auc_roc']}")
print(f"  siswati macro-F1      : {metrics_siswati['macro_f1']}  (zero-shot)")
print(f"  siswati MCC           : {metrics_siswati['mcc']}  (zero-shot)")
print(f"  CV macro-F1 (5-fold)  : {cv_metrics['macro_f1']}")
print("phase 2 AfroXLMR must beat these numbers")
