#Phase 1 baseline: TF-IDF + Logistic Regression (SGD) for MGT detection, trains on isiZulu and isiXhosa, tests zero-shot on Siswati
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.metrics import (precision_score, recall_score, f1_score,matthews_corrcoef, roc_auc_score,average_precision_score, confusion_matrix,classification_report)

print("all imports successful")

BASE_PATH= os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH= os.path.join(BASE_PATH, "data", "processed")
RESULTS_PATH= os.path.join(BASE_PATH, "results")

os.makedirs(os.path.join(RESULTS_PATH, "metrics"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_PATH, "plots"),   exist_ok=True)

DATASET_FILE = os.path.join(DATA_PATH, "merged_dataset.csv")

if not os.path.exists(DATASET_FILE):
    raise FileNotFoundError(f"processed dataset not found at:\n  {DATASET_FILE}, please run code/dataset_prep.py first")

print(f"loading dataset from: {DATASET_FILE}")
df = pd.read_csv(DATASET_FILE)

print(f"\ntotal records: {len(df)}")
print(df["Language_Code"].value_counts())
print(df["Label"].value_counts())

print(f"\ntotal records: {len(df)}")
print(f"by language: {df['Language_Code'].value_counts().to_dict()}")
print(f"by label: {df['Label'].value_counts().to_dict()}")

#Texts are already truncated to ~800 chars at sentence boundaries in dataset_prep.py
train_df= df[df["Language_Code"].isin(["zu", "xh"])].copy()
siswati_df= df[df["Language_Code"] == "ss"].copy()

print(f"\ntraining pool (isiZulu + isiXhosa): {len(train_df)}")
print(f"siswati held-out (zero-shot): {len(siswati_df)}")

X = train_df["Text_Generated"]
y = train_df["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\ntrain: {len(X_train)}  test: {len(X_test)}")

#Preprocessing: lowercase and remove punctuation to eliminate capitalization
#and punctuation leakage. This forces the model to focus on morphological and
#lexical patterns rather than superficial formatting differences.
#Human texts have ~18 capitals vs MGT ~6 capitals (major leakage)
#Human texts have ~4 commas vs MGT ~2 commas (structural leakage)
import string

def clean_for_tfidf(text):
    """Remove punctuation and lowercase to eliminate formatting leakage."""
    text= text.lower()
    text= text.translate(str.maketrans('', '', string.punctuation))
    text= ' '.join(text.split())  # Normalize whitespace
    return text

print("\npreprocessing: lowercasing and removing punctuation...")
X_train_clean= X_train.apply(clean_for_tfidf)
X_test_clean= X_test.apply(clean_for_tfidf)
X_siswati_clean= siswati_df["Text_Generated"].apply(clean_for_tfidf)

tfidf= TfidfVectorizer(
    max_features=10000,
    ngram_range=(3, 6),
    sublinear_tf=True,
    strip_accents=None,
    analyzer="char_wb",
    min_df=5
)

X_train_tfidf= tfidf.fit_transform(X_train_clean)
X_test_tfidf= tfidf.transform(X_test_clean)
X_siswati_tfidf= tfidf.transform(X_siswati_clean)
y_siswati= siswati_df["Label"]

print(f"\nTF-IDF vocab size: {len(tfidf.vocabulary_)}")
print(f"train matrix: {X_train_tfidf.shape}")
print(f"test matrix: {X_test_tfidf.shape}")
print(f"siswati matrix: {X_siswati_tfidf.shape}")


clf = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000, solver="lbfgs")
clf.fit(X_train_tfidf, y_train)
print("\nmodel trained")


def evaluate_model(clf, X_vec, y_true, label="evaluation"):
    y_pred= clf.predict(X_vec)
    y_proba= clf.predict_proba(X_vec)[:, 1]

    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        auc_roc= float("nan")
        auc_pr = float("nan")
    else:
        auc_roc= round(roc_auc_score(y_true, y_proba), 4)
        auc_pr = round(average_precision_score(y_true, y_proba), 4)

    metrics = {
        "label": label,
        "precision": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "mcc": round(matthews_corrcoef(y_true, y_pred), 4),
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
    }

    print(f"\n{label}")
    for k, v in metrics.items():
        if k != "label":
            print(f"  {k:<12}: {v}")

    present_labels = sorted(unique_classes)
    target_names   = [["Human", "Machine"][i] for i in present_labels]
    print(classification_report(y_true, y_pred, labels=present_labels, target_names=target_names))

    return metrics, y_pred, y_proba


metrics_inlang, y_pred_inlang, _ = evaluate_model(
    clf, X_test_tfidf, y_test,
    label="in-language test (isiZulu + isiXhosa)"
)

metrics_siswati, y_pred_siswati, _ = evaluate_model(
    clf, X_siswati_tfidf, y_siswati,
    label="zero-shot cross-lingual test (Siswati)"
)

#5-fold stratified cross-validation on the full training pool, apply same preprocessing (lowercase + remove punctuation)
X_all_clean= train_df["Text_Generated"].apply(clean_for_tfidf)
y_all= train_df["Label"]

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=10000,
        ngram_range=(3, 6),
        sublinear_tf=True,
        strip_accents=None,
        analyzer="char_wb",
        min_df=5
    )),
    ("clf", SGDClassifier(
        loss="log_loss",
        class_weight="balanced",
        random_state=42,
        max_iter=1000,
        tol=1e-3
    ))
])

skf= StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_preds= cross_val_predict(pipeline, X_all_clean, y_all, cv=skf)
cv_probas= cross_val_predict(pipeline, X_all_clean, y_all, cv=skf, method="predict_proba")[:, 1]

cv_metrics = {
    "label": "5-fold cross validation",
    "macro_f1": round(f1_score(y_all, cv_preds, average="macro"), 4),
    "mcc": round(matthews_corrcoef(y_all, cv_preds), 4),
    "auc_roc": round(roc_auc_score(y_all, cv_probas), 4),
}

print("\n5-fold cross-validation")
for k, v in cv_metrics.items():
    if k != "label":
        print(f"  {k:<12}: {v}")

#cross-LLM generalisation
print("\ncross-LLM generalisation (in-language test set)")

llm_metrics = {}
for llm in sorted(train_df["Model_Identifier"].unique()):
    if llm == "human":
        continue
    mask= train_df.loc[X_test.index, "Model_Identifier"] == llm
    X_test_llm= X_test[mask]
    y_test_llm= y_test[mask]
    if len(X_test_llm) == 0:
        continue
    #apply same preprocessing
    X_test_llm_clean = X_test_llm.apply(clean_for_tfidf)
    m, _, _ = evaluate_model(
        clf, tfidf.transform(X_test_llm_clean), y_test_llm,
        label=f"cross-LLM in-language ({llm})"
    )
    llm_metrics[llm] = m
print("\ncross-LLM generalisation (Siswati zero-shot)")

llm_siswati_metrics = {}
for llm in sorted(siswati_df["Model_Identifier"].unique()):
    if llm == "human":
        continue
    mask= siswati_df["Model_Identifier"] == llm
    X_ss_llm = siswati_df.loc[mask, "Text_Generated"]
    y_ss_llm = siswati_df.loc[mask, "Label"]
    if len(X_ss_llm) == 0:
        continue
    # Apply same preprocessing
    X_ss_llm_clean = X_ss_llm.apply(clean_for_tfidf)
    m, _, _ = evaluate_model(
        clf, tfidf.transform(X_ss_llm_clean), y_ss_llm,
        label=f"cross-LLM Siswati zero-shot ({llm})"
    )
    llm_siswati_metrics[llm] = m

#shows the generalisation gap across Precision, Recall, Macro-F1 and MCC, include the four metrics the proposal uses to evaluate the baseline.
metric_keys= ["precision", "recall", "macro_f1", "mcc"]
metric_labels= ["Precision", "Recall", "Macro-F1", "MCC"]
inlang_vals= [metrics_inlang[k]  for k in metric_keys]
siswati_vals = [metrics_siswati[k] for k in metric_keys]

x= np.arange(len(metric_keys))
width = 0.35

fig, ax= plt.subplots(figsize=(9, 5))
bars1= ax.bar(x - width / 2, inlang_vals,  width, color="#4C72B0", label="In-language (isiZulu + isiXhosa)")
bars2= ax.bar(x + width / 2, siswati_vals, width, color="#DD8452", label="Zero-shot (Siswati)")

ax.set_xticks(x)
ax.set_xticklabels(metric_labels, fontsize=11)
ax.set_ylim(0, 1.15)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
ax.set_ylabel("Score", fontsize=11)
ax.set_title("Baseline (TF-IDF + LR): In-Language vs Zero-Shot Generalisation", fontsize=12)
ax.legend(fontsize=10)
ax.bar_label(bars1, fmt="%.2f", padding=3, fontsize=9)
ax.bar_label(bars2, fmt="%.2f", padding=3, fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plot1_path = os.path.join(RESULTS_PATH, "plots", "baseline_generalisation_gap.png")
plt.savefig(plot1_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"saved: {plot1_path}")

#confusion matrix for Siswati zero-shot only, shows the actual error pattern: the model is biased toward predicting Machine, causing it to miss most Human texts.
cm = confusion_matrix(y_siswati, y_pred_siswati)
labels = ["Human", "Machine"]

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
plt.colorbar(im, ax=ax)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(labels, fontsize=11)
ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel("Predicted label", fontsize=11)
ax.set_ylabel("True label", fontsize=11)
ax.set_title("Confusion Matrix-Zero-Shot Siswati\n(baseline TF-IDF + LR)", fontsize=11)

thresh = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=13)

plt.tight_layout()
plot2_path = os.path.join(RESULTS_PATH, "plots", "baseline_siswati_confusion.png")
plt.savefig(plot2_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"saved: {plot2_path}")

all_metrics = {
    "in_language": metrics_inlang,
    "siswati_zeroshot": metrics_siswati,
    "cross_validation": cv_metrics,
}
for llm, m in llm_metrics.items():
    all_metrics[f"cross_llm_{llm}"] = m
for llm, m in llm_siswati_metrics.items():
    all_metrics[f"cross_llm_siswati_{llm}"] = m

metrics_path = os.path.join(RESULTS_PATH, "metrics", "baseline_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(all_metrics, f, indent=2)
print(f"\nmetrics saved: {metrics_path}")

print("\nbaseline summary")
print(f"in-language macro-F1: {metrics_inlang['macro_f1']}")
print(f"in-language MCC: {metrics_inlang['mcc']}")
print(f"siswati macro-F1: {metrics_siswati['macro_f1']}  (zero-shot)")
print(f"siswati MCC: {metrics_siswati['mcc']}  (zero-shot)")
print(f"CV macro-F1 (5-fold): {cv_metrics['macro_f1']}")
print("Phase 2 AfroXLMR must beat these numbers")
