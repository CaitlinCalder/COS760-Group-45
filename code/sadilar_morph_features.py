#sadilar morphological feature extraction
import os
import re
import pandas as pd

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_FILE = os.path.join(BASE_PATH, "data", "processed", "merged_dataset.csv")
SADILAR_PATH = os.path.join(BASE_PATH, "data", "sadilar")
OUTPUT_FILE = os.path.join(BASE_PATH, "data", "processed", "sadilar_morph_features.csv")

MORPH_FILES = {
    "zu": os.path.join(SADILAR_PATH, "zulu_morph.txt"),
    "xh": os.path.join(SADILAR_PATH, "xhosa_morph.txt"),
    "ss": os.path.join(SADILAR_PATH, "siswati_morph.txt"),
}


def tokenize(text):
    text = str(text).lower()
    return re.findall(r"[a-zA-ZÀ-ÿ]+", text)


def load_morph_lookup(filepath):
    lookup = {}

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing SADiLaR file: {filepath}")

    with open(filepath, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            line = line.strip()

            if not line or "\t" not in line:
                continue

            parts = line.split("\t")
            token = parts[0].lower()
            analysis = "\t".join(parts[1:])

            lookup[token] = analysis

    return lookup

#added repitition feature 
def extract_features(text, lookup):
    tokens = tokenize(text)
    word_count = len(tokens)

    if word_count == 0:
        return {
            "word_count": 0,
            "matched_words": 0,
            "unmatched_words": 0,
            "sadilar_coverage": 0,
            "avg_word_length": 0,
            "unique_word_ratio": 0,
            "unique_morph_analysis_count": 0,
            "morph_diversity_ratio": 0,
            "word_repetition_rate": 0,
            "bigram_repetition_rate": 0,
        }

    matched = 0
    analyses = []

    for token in tokens:
        if token in lookup:
            matched += 1
            analyses.append(lookup[token])

    unique_words = len(set(tokens))
    unique_analyses = len(set(analyses))

    word_counts = {}
    for token in tokens:
        word_counts[token] = word_counts.get(token, 0) + 1
    repeated_words = sum(1 for count in word_counts.values() if count > 1)
    word_repetition_rate = repeated_words / unique_words if unique_words > 0 else 0

    bigrams = list(zip(tokens[:-1], tokens[1:]))
    if bigrams:
        unique_bigrams = len(set(bigrams))
        bigram_repetition_rate = 1 - (unique_bigrams / len(bigrams))
    else:
        bigram_repetition_rate = 0

    return {
        "word_count": word_count,
        "matched_words": matched,
        "unmatched_words": word_count - matched,
        "sadilar_coverage": matched / word_count,
        "avg_word_length": sum(len(token) for token in tokens) / word_count,
        "unique_word_ratio": unique_words / word_count,
        "unique_morph_analysis_count": unique_analyses,
        "morph_diversity_ratio": unique_analyses / word_count,
        "word_repetition_rate": word_repetition_rate,
        "bigram_repetition_rate": bigram_repetition_rate,
    }


def main():
    print("Loading merged dataset...")
    df = pd.read_csv(DATASET_FILE)

    print("Loading SADiLaR morphology files...")
    lookups = {}

    for lang, path in MORPH_FILES.items():
        lookups[lang] = load_morph_lookup(path)
        print(f"{lang}: loaded {len(lookups[lang])} words")

    rows = []

    print("Extracting morphology features...")

    for index, row in df.iterrows():
        lang = row["Language_Code"]

        if lang not in lookups:
            continue

        text = row["Text_Generated"]
        features = extract_features(text, lookups[lang])

        rows.append({
            "row_id": index,
            "Text_Generated": text,
            "Language_Code": lang,
            "Language_Name": row.get("Language_Name", ""),
            "Domain": row.get("Domain", ""),
            "Model_Identifier": row.get("Model_Identifier", ""),
            "Label": row.get("Label", ""),
            **features
        })

    features_df = pd.DataFrame(rows)
    features_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print("Done.")
    print(f"Saved file to: {OUTPUT_FILE}")

    print("\nAverage features by language and label:")
    print(
        features_df.groupby(["Language_Code", "Label"])[
            [
                "sadilar_coverage",
                "avg_word_length",
                "unique_word_ratio",
                "morph_diversity_ratio",
            ]
        ].mean()
    )


if __name__ == "__main__":
    main()