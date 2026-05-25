#sadilar visualisations on preprocessed merged dataset
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE_PATH,
    "data",
    "processed",
    "sadilar_morph_features.csv"
)

RESULTS_DIR = os.path.join(BASE_PATH, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

label_map = {
    0: "Human",
    1: "Machine"
}

df["Label_Name"] = df["Label"].map(label_map)

sns.set(style="whitegrid")

plots = [
    "sadilar_coverage",
    "morph_diversity_ratio",
    "unique_word_ratio",
    "avg_word_length"
]

for feature in plots:

    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=df,
        x="Language_Code",
        y=feature,
        hue="Label_Name"
    )

    plt.title(f"{feature} by Language and Label")
    plt.tight_layout()

    output_path = os.path.join(
        RESULTS_DIR,
        f"{feature}_boxplot.png"
    )

    plt.savefig(output_path)
    plt.close()

    print(f"Saved: {output_path}")

print("All plots generated.")