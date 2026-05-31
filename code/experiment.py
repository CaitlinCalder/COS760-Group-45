import os
import re
import sys
import json
import string
import argparse
import textwrap
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

BASE_PATH= os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH= os.path.join(BASE_PATH, "data", "processed")
SADILAR_PATH= os.path.join(BASE_PATH, "data", "sadilar")
MODEL_PATH= os.environ.get(
    "AFROXLMR_MODEL_PATH",
    os.path.join(BASE_PATH, "models", "best_model")
)
MERGED_CSV= os.path.join(DATA_PATH, "merged_dataset.csv")
RESULTS_PATH = os.path.join(BASE_PATH, "results", "metrics")

MORPH_FILES = {
    "zu": os.path.join(SADILAR_PATH, "zulu_morph.txt"),
    "xh": os.path.join(SADILAR_PATH, "xhosa_morph.txt"),
    "ss": os.path.join(SADILAR_PATH, "siswati_morph.txt"),
}

MAX_LENGTH = 512


def _banner(title: str):
    width = 64
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def _result_row(phase: str, label: str, confidence: float | None, note: str = ""):
    conf_str = f"{confidence:.1%}" if confidence is not None else "  N/A  "
    verdict  = f"{'🟥 MACHINE' if label == 'Machine' else '🟦 Human  '}" if label not in ("N/A", "ERROR") else f"  {label:<9}"
    print(f"  {phase:<14}  {verdict}   confidence {conf_str}   {note}")


def _clean_for_tfidf(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def run_phase1(text: str):
    """Re-train the baseline on the full isiZulu + isiXhosa pool, then predict."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        if not os.path.exists(MERGED_CSV):
            return "N/A", None, "merged_dataset.csv not found"

        df = pd.read_csv(MERGED_CSV)
        train_df = df[df["Language_Code"].isin(["zu", "xh"])].copy()

        X_train = train_df["Text_Generated"].apply(_clean_for_tfidf)
        y_train = train_df["Label"]

        tfidf = TfidfVectorizer(
            max_features=10000, ngram_range=(3, 6),
            sublinear_tf=True, strip_accents=None,
            analyzer="char_wb", min_df=5
        )
        clf = LogisticRegression(
            class_weight="balanced", random_state=42,
            max_iter=1000, solver="lbfgs"
        )

        X_vec = tfidf.fit_transform(X_train)
        clf.fit(X_vec, y_train)

        x_input = tfidf.transform([_clean_for_tfidf(text)])
        prob     = clf.predict_proba(x_input)[0]
        label    = "Machine" if prob[1] >= 0.5 else "Human"
        return label, float(max(prob)), "TF-IDF + Logistic Regression"

    except Exception as exc:
        return "ERROR", None, str(exc)[:60]



def run_phase2(text: str):
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        if not os.path.exists(MODEL_PATH):
            return "N/A", None, f"model not found at {MODEL_PATH}"

        device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer= AutoTokenizer.from_pretrained(MODEL_PATH)
        model= AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
        model.eval()

        inputs= tokenizer(text, padding=True, truncation=True,max_length=MAX_LENGTH, return_tensors="pt").to(device)

        with torch.no_grad():
            logits = model(**inputs).logits

        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        label = "Machine" if probs[1] >= 0.5 else "Human"
        return label, float(max(probs)), "AfroXLMR (fine-tuned)"

    except Exception as exc:
        return "ERROR", None, str(exc)[:60]


def _tokenize(text: str):
    return re.findall(r"[a-zA-ZÀ-ÿ]+", text.lower())


def _load_morph_lookup(filepath: str) -> dict:
    lookup = {}
    if not os.path.exists(filepath):
        return lookup
    with open(filepath, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            parts= line.split("\t")
            token= parts[0].lower()
            lookup[token] = "\t".join(parts[1:])
    return lookup


def _extract_morph_features(text: str, lookup: dict) -> dict:
    tokens= _tokenize(text)
    word_count= len(tokens)

    if word_count == 0:
        return {k: 0 for k in [
            "word_count","matched_words","unmatched_words","sadilar_coverage",
            "avg_word_length","unique_word_ratio","unique_morph_analysis_count",
            "morph_diversity_ratio","word_repetition_rate","bigram_repetition_rate"
        ]}

    matched= 0
    analyses= []
    for tok in tokens:
        if tok in lookup:
            matched += 1
            analyses.append(lookup[tok])

    unique_words = len(set(tokens))
    unique_ana= len(set(analyses))

    counts = {}
    for tok in tokens:
        counts[tok] = counts.get(tok, 0) + 1
    repeated = sum(1 for c in counts.values() if c > 1)
    word_rep_rate = repeated / unique_words if unique_words > 0 else 0

    bigrams = list(zip(tokens[:-1], tokens[1:]))
    bigram_rep_rate = (1 - len(set(bigrams)) / len(bigrams)) if bigrams else 0

    return {
        "word_count":word_count,
        "matched_words":matched,
        "unmatched_words":word_count - matched,
        "sadilar_coverage":matched / word_count,
        "avg_word_length":sum(len(t) for t in tokens) / word_count,
        "unique_word_ratio":unique_words / word_count,
        "unique_morph_analysis_count": unique_ana,
        "morph_diversity_ratio":unique_ana / word_count,
        "word_repetition_rate":word_rep_rate,
        "bigram_repetition_rate":bigram_rep_rate,
    }


SADILAR_FEATURE_COLUMNS = [
    "word_count","matched_words","unmatched_words","sadilar_coverage",
    "avg_word_length","unique_word_ratio","unique_morph_analysis_count",
    "morph_diversity_ratio","word_repetition_rate","bigram_repetition_rate",
]
ALL_FEATURE_COLUMNS = ["prob_human", "prob_machine"] + SADILAR_FEATURE_COLUMNS


def run_phase3(text: str, lang_hint: str = "ss"):
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from sklearn.ensemble import RandomForestClassifier

        if not os.path.exists(MODEL_PATH):
            return "N/A", None, f"model not found at {MODEL_PATH}"

        if not os.path.exists(MERGED_CSV):
            return "N/A", None, "merged_dataset.csv not found"

        morph_path = MORPH_FILES.get(lang_hint, MORPH_FILES["ss"])
        lookup= _load_morph_lookup(morph_path)

        if not lookup:
            return "N/A", None, f"SADiLaR morph file missing for lang={lang_hint}"

        #phase 2 probabilities
        device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        afroxlmr  = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
        afroxlmr.eval()

        def _get_probs(texts):
            all_probs = []
            for i in range(0, len(texts), 16):
                batch = texts[i:i+16]
                enc   = tokenizer(batch, padding=True, truncation=True,
                                  max_length=MAX_LENGTH, return_tensors="pt").to(device)
                with torch.no_grad():
                    logits = afroxlmr(**enc).logits
                all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
            return np.vstack(all_probs)

        #training features
        df= pd.read_csv(MERGED_CSV)
        train_df = df[df["Language_Code"].isin(["zu", "xh"])].copy()

        print("[Phase 3] Extracting AfroXLMR probs for training set …", flush=True)
        train_texts = train_df["Text_Generated"].tolist()
        train_probs = _get_probs(train_texts)
        train_df= train_df.copy()
        train_df["prob_human"]   = train_probs[:, 0]
        train_df["prob_machine"] = train_probs[:, 1]

        #SADiLaR features for training set 
        train_morph_rows = []
        for lang in ["zu", "xh"]:
            lk= _load_morph_lookup(MORPH_FILES.get(lang, morph_path))
            mask = train_df["Language_Code"] == lang
            for txt in train_df.loc[mask, "Text_Generated"]:
                train_morph_rows.append(_extract_morph_features(txt, lk))

        #Re-order to match train_df row order
        morph_features_df = pd.DataFrame(train_morph_rows, index=train_df.index)
        for col in SADILAR_FEATURE_COLUMNS:
            train_df[col] = morph_features_df[col]

        X_train = train_df[ALL_FEATURE_COLUMNS]
        y_train = train_df["Label"]

        rf = RandomForestClassifier(n_estimators=200, random_state=42)
        rf.fit(X_train, y_train)

        input_probs = _get_probs([text])[0]
        morph_feats = _extract_morph_features(text, lookup)

        row = {
            "prob_human":   input_probs[0],
            "prob_machine": input_probs[1],
            **morph_feats,
        }
        X_input = pd.DataFrame([row])[ALL_FEATURE_COLUMNS]

        prob    = rf.predict_proba(X_input)[0]
        label   = "Machine" if prob[1] >= 0.5 else "Human"
        return label, float(max(prob)), "AfroXLMR + SADiLaR (Random Forest)"

    except Exception as exc:
        return "ERROR", None, str(exc)[:80]


def parse_args():
    parser = argparse.ArgumentParser(
        description="MGT Detection Demo -> runs text through all three phases."
    )
    parser.add_argument("--text", type=str, default=None,
                        help="Text to classify (wrap in quotes)")
    parser.add_argument("--file", type=str, default=None,
                        help="Path to a .txt file containing the text to classify")
    parser.add_argument("--lang", type=str, default="ss",
                        choices=["zu", "xh", "ss"],
                        help="Language hint for SADiLaR morphology lookup (default: ss)")
    return parser.parse_args()


def get_text(args) -> str:
    if args.file:
        with open(args.file, "r", encoding="utf-8") as fh:
            return fh.read().strip()
    if args.text:
        return args.text.strip()

    print("  MGT Detection Experiment -> paste your text below.")
    print("  Type or paste the text, then press Enter twice to submit.")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "" and lines and lines[-1] == "":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def main():
    args = parse_args()
    text = get_text(args)

    if not text:
        print("No text provided. Exiting.")
        sys.exit(1)

    print(f"\nInput ({len(text)} chars):")
    preview = textwrap.fill(text[:300], width=64)
    print("  " + preview.replace("\n", "\n  "))
    if len(text) > 300:
        print("  … [truncated for display]")

    lang = args.lang

    _banner("Phase 1 -> TF-IDF + Logistic Regression  (baseline)")
    label1, conf1, note1 = run_phase1(text)
    _result_row("Phase 1", label1, conf1, note1)

    _banner("Phase 2 -> AfroXLMR  (fine-tuned transfer learning)")
    print("  Loading fine-tuned model …", flush=True)
    label2, conf2, note2 = run_phase2(text)
    _result_row("Phase 2", label2, conf2, note2)

    _banner("Phase 3 -> AfroXLMR + SADiLaR  (augmented classifier)")
    print(f"  Language hint: {lang.upper()}  |  Building training features …", flush=True)
    label3, conf3, note3 = run_phase3(text, lang_hint=lang)
    _result_row("Phase 3", label3, conf3, note3)

    _banner("Summary")
    print(f"  {'Phase':<14}  {'Verdict':<14}  {'Confidence':<12}  Notes")
    print("  " + "-" * 58)
    for phase, lbl, conf, note in [
        ("Phase 1 (P1)", label1, conf1, note1),
        ("Phase 2 (P2)", label2, conf2, note2),
        ("Phase 3 (P3)", label3, conf3, note3),
    ]:
        conf_str = f"{conf:.1%}" if conf is not None else "N/A"
        print(f"  {phase:<14}  {lbl:<14}  {conf_str:<12}  {note}")
    print()


    votes = [l for l in [label1, label2, label3] if l in ("Human", "Machine")]
    if votes:
        machine_votes = votes.count("Machine")
        human_votes   = votes.count("Human")
        majority      = "Machine" if machine_votes > human_votes else "Human"
        print(f"  Majority vote ({machine_votes}M / {human_votes}H across {len(votes)} phases):  "
              f"{'MACHINE-GENERATED' if majority == 'Machine' else 'HUMAN-WRITTEN'}")
    print()


if __name__ == "__main__":
    main()