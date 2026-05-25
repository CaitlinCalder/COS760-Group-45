"""
SADiLaR Visualisation and SHAP Analysis

Creates plots and SHAP explanations for the SADiLaR
morphology-based classifier.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay


BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FEATURES_FILE = os.path.join(
    BASE_PATH,
    "data",
    "processed",
    "sadilar_morph_features.csv"
)

# Create dedicated SADiLaR results folder
RESULTS_DIR = os.path.join(
    BASE_PATH,
    "results",
    "sadilar_analysis"
)

os.makedirs(RESULTS_DIR, exist_ok=True)

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
df = pd.read_csv(FEATURES_FILE)

X = df[FEATURE_COLUMNS]
y = df["Label"]

print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Random Forest model...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

print("Making predictions...")
y_pred = model.predict(X_test)

# confusion matrix

print("Creating confusion matrix plot...")

fig, ax = plt.subplots(figsize=(6, 5))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["Human", "Machine"],
    ax=ax
)

plt.title("SADiLaR Classifier Confusion Matrix")
plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "sadilar_confusion_matrix.png"
    ),
    dpi=300
)

plt.close()

#feature importance

print("Creating feature importance plot...")

importance_df = pd.DataFrame({
    "Feature": FEATURE_COLUMNS,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(8, 5))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xlabel("Importance")
plt.title("SADiLaR Feature Importance")

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "sadilar_feature_importance.png"
    ),
    dpi=300
)

plt.close()

# box plots

print("Creating boxplots...")

df["Label_Name"] = df["Label"].map({
    0: "Human",
    1: "Machine"
})

PLOT_FEATURES = [
    "sadilar_coverage",
    "avg_word_length",
    "unique_word_ratio",
    "morph_diversity_ratio"
]

for feature in PLOT_FEATURES:

    plt.figure(figsize=(8, 5))

    df.boxplot(
        column=feature,
        by=["Language_Code", "Label_Name"],
        rot=45
    )

    plt.title(f"{feature} by Language and Label")
    plt.suptitle("")

    plt.xlabel("Language and Label")
    plt.ylabel(feature)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            f"{feature}_boxplot.png"
        ),
        dpi=300
    )

    plt.close()

# SHAP analysis

print("Running SHAP analysis...")

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_test)

# Binary classification
if isinstance(shap_values, list):
    shap_values_machine = shap_values[1]

# Some SHAP versions return 3D arrays
elif len(shap_values.shape) == 3:
    shap_values_machine = shap_values[:, :, 1]

else:
    shap_values_machine = shap_values

# SHAP summary plot
plt.figure()

shap.summary_plot(
    shap_values_machine,
    X_test,
    show=False
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "sadilar_shap_summary.png"
    ),
    dpi=300
)

plt.close()

# SHAP bar plot
plt.figure()

shap.summary_plot(
    shap_values_machine,
    X_test,
    plot_type="bar",
    show=False
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "sadilar_shap_bar.png"
    ),
    dpi=300
)

plt.close()

print("\nAll SADiLaR plots saved to:")
print(RESULTS_DIR)

print("\nDone.")