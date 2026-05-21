"""
Vuk'uzenzele Dataset Analysis Script
Group 45 - Machine-Generated Text Detection in Bantu Languages

PURPOSE:
    Analyses the Vuk'uzenzele corpus to profile what real human-written text
    looks like so your synthetic (LLM-generated) dataset can be made to match
    it as closely as possible. If your baseline is giving perfect scores it means
    the two classes are trivially separable — most likely because the LLM outputs
    differ in length, vocabulary, or domain from the real articles.

OUTPUT:
    - Printed summary stats
    - Four saved plots: distributions, vocabulary, readability, topic clusters
    - A RECOMMENDATIONS section at the end that tells you exactly what to
      constrain when prompting your LLMs.

USAGE:
    pip install datasets pandas numpy matplotlib seaborn scikit-learn langdetect
    python analyse_vukuzenzele.py
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import re
import string
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import load_dataset
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")

# ── Language config mapping (HuggingFace config name → short code) ────────────
# The dataset uses ISO 639-2/3 codes as config names
HF_CONFIGS  = {"zul": "zu", "xho": "xh", "ssw": "ss"}
PALETTE     = {"zu": "#2E86AB", "xh": "#A23B72", "ss": "#F18F01"}
LANG_NAMES  = {"zu": "isiZulu",  "xh": "isiXhosa", "ss": "Siswati"}

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATASET  (one config per language)
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("  Vuk'uzenzele Corpus Analyser — Group 45")
print("=" * 70)
print("\n[1/6] Loading dataset from HuggingFace …")
print("      Loading configs: zul, xho, ssw\n")

frames = []
for hf_code, short_code in HF_CONFIGS.items():
    print(f"      → dsfsi/vukuzenzele-monolingual  [{hf_code}]", end=" ", flush=True)
    try:
        ds = load_dataset("dsfsi/vukuzenzele-monolingual", hf_code)
        # Grab whichever split exists (train / test / validation)
        split_name = list(ds.keys())[0]
        part = ds[split_name].to_pandas()
        part["lang"] = short_code          # add a unified lang column
        frames.append(part)
        print(f"✓  ({len(part):,} rows, split='{split_name}')")
    except Exception as e:
        print(f"✗  SKIPPED: {e}")

if not frames:
    raise RuntimeError("No language configs loaded — check your internet connection.")

df = pd.concat(frames, ignore_index=True)
print(f"\n      Total rows loaded: {len(df):,}")
print(f"      Columns: {list(df.columns)}")

# ── Identify the text column (should be 'text', 'article', 'content', etc.)
text_col_candidates = [c for c in df.columns if c.lower() in ("text", "article", "content", "body")]
if not text_col_candidates:
    # Fallback: pick the column with the longest average string
    str_cols = [c for c in df.columns if df[c].dtype == object and c != "lang"]
    TEXT_COL = max(str_cols, key=lambda c: df[c].str.len().mean())
    print(f"  ⚠  No obvious text column — using '{TEXT_COL}' (longest strings)")
else:
    TEXT_COL = text_col_candidates[0]

print(f"      Text column used: '{TEXT_COL}'")

df.rename(columns={TEXT_COL: "text"}, inplace=True)
df.dropna(subset=["text"], inplace=True)
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"].str.len() > 0]

print(f"\n  Records per language after cleaning:")
print(df["lang"].value_counts().rename(LANG_NAMES).to_string())

# ══════════════════════════════════════════════════════════════════════════════
# 2. TEXT-LEVEL FEATURES
# ══════════════════════════════════════════════════════════════════════════════

print("\n[2/6] Computing text-level features …")


def tokenise(text: str) -> list[str]:
    """Whitespace tokeniser — avoids NLTK dependency for low-resource langs."""
    return text.lower().split()


def sentences(text: str) -> list[str]:
    """Simple sentence splitter on . ! ?"""
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


def type_token_ratio(tokens: list[str]) -> float:
    return len(set(tokens)) / len(tokens) if tokens else 0.0


def avg_word_len(tokens: list[str]) -> float:
    words = [w.strip(string.punctuation) for w in tokens if w.strip(string.punctuation)]
    return np.mean([len(w) for w in words]) if words else 0.0


def punctuation_density(text: str) -> float:
    n_punct = sum(1 for c in text if c in string.punctuation)
    return n_punct / len(text) if text else 0.0


def hapax_ratio(tokens: list[str]) -> float:
    freq = Counter(tokens)
    hapax = sum(1 for v in freq.values() if v == 1)
    return hapax / len(freq) if freq else 0.0


# Apply to every row
df["tokens"]        = df["text"].apply(tokenise)
df["word_count"]    = df["tokens"].apply(len)
df["char_count"]    = df["text"].apply(len)
df["sent_count"]    = df["text"].apply(lambda t: len(sentences(t)))
df["avg_sent_len"]  = df.apply(
    lambda r: r["word_count"] / r["sent_count"] if r["sent_count"] > 0 else 0, axis=1
)
df["avg_word_len"]  = df["tokens"].apply(avg_word_len)
df["ttr"]           = df["tokens"].apply(type_token_ratio)
df["punct_density"] = df["text"].apply(punctuation_density)
df["hapax_ratio"]   = df["tokens"].apply(hapax_ratio)

# Vocabulary size per article
df["vocab_size"] = df["tokens"].apply(lambda t: len(set(t)))

# Paragraph count (blank-line separated)
df["para_count"] = df["text"].apply(lambda t: len([p for p in t.split("\n\n") if p.strip()]))

print("      Done.\n")

# ══════════════════════════════════════════════════════════════════════════════
# 3. SUMMARY STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

print("[3/6] Summary statistics per language …\n")

FEATURES = [
    "word_count", "char_count", "sent_count",
    "avg_sent_len", "avg_word_len", "ttr",
    "punct_density", "hapax_ratio", "vocab_size", "para_count",
]

overall_stats = df[FEATURES].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
print("── OVERALL ──")
print(overall_stats.to_string())
print()

for lang_code, lang_name in LANG_NAMES.items():
    sub = df[df["lang"] == lang_code]
    if sub.empty:
        continue
    print(f"── {lang_name} (n={len(sub)}) ──")
    print(sub[FEATURES].describe().T[["mean", "std", "min", "50%", "max"]].to_string())
    print()

# ══════════════════════════════════════════════════════════════════════════════
# 4. VOCABULARY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

print("[4/6] Vocabulary analysis …")

# Per-language top-40 content words (excluding short tokens)
TOP_N = 40
MIN_TOKEN_LEN = 4

vocab_results = {}
for lang_code in LANG_NAMES:
    sub = df[df["lang"] == lang_code]
    if sub.empty:
        continue
    all_tokens = [tok for tokens in sub["tokens"] for tok in tokens if len(tok) >= MIN_TOKEN_LEN]
    freq = Counter(all_tokens)
    vocab_results[lang_code] = freq.most_common(TOP_N)

print("      Top-20 content words (≥4 chars) per language:\n")
for lang_code, top_words in vocab_results.items():
    words_str = ", ".join(f"{w}({c})" for w, c in top_words[:20])
    print(f"  {LANG_NAMES[lang_code]}: {words_str}\n")

# ══════════════════════════════════════════════════════════════════════════════
# 5. TOPIC IDENTIFICATION (TF-IDF + SVD)
# ══════════════════════════════════════════════════════════════════════════════

print("[5/6] Topic cluster analysis (TF-IDF + SVD) …")

TOPIC_KEYWORDS = {
    "Government / Policy": [
        "hulumeni", "umbuso", "urhulumente", "umthetho", "umbuso",
        "inqolobane", "inhlangano", "uMasipala",
        "government", "policy", "minister", "department", "national",
    ],
    "Education": [
        "isikole", "abafundi", "uthisha", "imfundo",
        "school", "education", "learner", "teacher", "university",
        "umfundi", "izikole",
    ],
    "Health": [
        "impilo", "isibhedlela", "ugciwane", "ukugula",
        "health", "hospital", "clinic", "disease", "vaccine",
    ],
    "Agriculture / Rural": [
        "ukulima", "izilimo", "inkomo", "abalimi",
        "farming", "crops", "rural", "land",
    ],
    "Social Services": [
        "izidingo", "insizakalo", "abantu", "umphakathi",
        "community", "social", "service", "support",
    ],
    "Infrastructure": [
        "imigwaqo", "ugesi", "amanzi", "izindlu",
        "road", "water", "electricity", "housing", "infrastructure",
    ],
}

# Keyword-based topic tagging
def tag_topic(text_lower: str) -> str:
    scores = {topic: 0 for topic in TOPIC_KEYWORDS}
    for topic, keywords in TOPIC_KEYWORDS.items():
        scores[topic] = sum(1 for kw in keywords if kw.lower() in text_lower)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Other / News"


df["text_lower"] = df["text"].str.lower()
df["topic"]      = df["text_lower"].apply(tag_topic)

print("\n  Topic distribution across corpus:\n")
print(df.groupby(["lang", "topic"]).size().unstack(fill_value=0).to_string())
print()

# ══════════════════════════════════════════════════════════════════════════════
# 6. PLOTS
# ══════════════════════════════════════════════════════════════════════════════

print("[6/6] Generating plots …")

colours = [PALETTE.get(lang, "#888888") for lang in df["lang"]]

# ── Figure 1: Distribution plots ──────────────────────────────────────────────
fig1, axes = plt.subplots(2, 3, figsize=(16, 9))
fig1.suptitle("Vuk'uzenzele Corpus — Text Feature Distributions", fontsize=14, fontweight="bold")

dist_features = [
    ("word_count",   "Word Count"),
    ("avg_sent_len", "Avg Sentence Length (words)"),
    ("avg_word_len", "Avg Word Length (chars)"),
    ("ttr",          "Type-Token Ratio"),
    ("punct_density","Punctuation Density"),
    ("hapax_ratio",  "Hapax Legomena Ratio"),
]

for ax, (feat, label) in zip(axes.flat, dist_features):
    for lang_code, lang_name in LANG_NAMES.items():
        sub = df[df["lang"] == lang_code][feat].dropna()
        if sub.empty:
            continue
        sub.plot.kde(ax=ax, label=lang_name, color=PALETTE.get(lang_code, "#888"))
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
fig1.savefig("plot1_distributions.png", dpi=150)
print("      Saved: plot1_distributions.png")

# ── Figure 2: Box plots for word count & sentence length ──────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle("Word Count & Sentence Length by Language", fontsize=13, fontweight="bold")

df["Language"] = df["lang"].map(LANG_NAMES)
pal_named = {LANG_NAMES[k]: v for k, v in PALETTE.items()}

sns.boxplot(data=df, x="Language", y="word_count",
            palette=pal_named, ax=axes2[0], showfliers=False)
axes2[0].set_title("Word Count (outliers hidden)")
axes2[0].set_ylabel("Words per article")

sns.boxplot(data=df, x="Language", y="avg_sent_len",
            palette=pal_named, ax=axes2[1], showfliers=False)
axes2[1].set_title("Avg Sentence Length")
axes2[1].set_ylabel("Words per sentence")

plt.tight_layout()
fig2.savefig("plot2_boxplots.png", dpi=150)
print("      Saved: plot2_boxplots.png")

# ── Figure 3: Topic bar chart ──────────────────────────────────────────────────
topic_counts = df.groupby(["Language", "topic"]).size().reset_index(name="count")
fig3, ax3 = plt.subplots(figsize=(13, 6))
topics = topic_counts["topic"].unique()
x = np.arange(len(topics))
width = 0.25

for i, (lang_code, lang_name) in enumerate(LANG_NAMES.items()):
    sub = topic_counts[topic_counts["Language"] == lang_name]
    sub_indexed = sub.set_index("topic")["count"].reindex(topics, fill_value=0)
    ax3.bar(x + i * width, sub_indexed, width,
            label=lang_name, color=PALETTE.get(lang_code, "#888"))

ax3.set_xticks(x + width)
ax3.set_xticklabels(topics, rotation=25, ha="right", fontsize=9)
ax3.set_title("Keyword-Based Topic Distribution by Language", fontsize=13, fontweight="bold")
ax3.set_ylabel("Article count")
ax3.legend()
ax3.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig3.savefig("plot3_topics.png", dpi=150)
print("      Saved: plot3_topics.png")

# ── Figure 4: 2-D TF-IDF topic space (SVD) ────────────────────────────────────
try:
    tfidf = TfidfVectorizer(max_features=5000, min_df=2, ngram_range=(1, 2))
    X = tfidf.fit_transform(df["text"])
    X_norm = normalize(X)
    svd = TruncatedSVD(n_components=2, random_state=42)
    coords = svd.fit_transform(X_norm)

    fig4, ax4 = plt.subplots(figsize=(9, 7))
    for lang_code, lang_name in LANG_NAMES.items():
        mask = df["lang"] == lang_code
        ax4.scatter(coords[mask, 0], coords[mask, 1],
                    label=lang_name, alpha=0.55, s=30,
                    color=PALETTE.get(lang_code, "#888"))
    ax4.set_title("TF-IDF + SVD — Article Topic Space", fontsize=13, fontweight="bold")
    ax4.set_xlabel(f"SVD-1 ({svd.explained_variance_ratio_[0]:.1%} var)")
    ax4.set_ylabel(f"SVD-2 ({svd.explained_variance_ratio_[1]:.1%} var)")
    ax4.legend()
    ax4.grid(alpha=0.3)
    plt.tight_layout()
    fig4.savefig("plot4_topicspace.png", dpi=150)
    print("      Saved: plot4_topicspace.png")
except Exception as e:
    print(f"      ⚠  SVD plot skipped: {e}")

plt.close("all")

# ══════════════════════════════════════════════════════════════════════════════
# 7. SYNTHETIC DATASET RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  RECOMMENDATIONS FOR SYNTHETIC DATASET CONSTRUCTION")
print("=" * 70)

# Compute per-language median word count
for lang_code, lang_name in LANG_NAMES.items():
    sub = df[df["lang"] == lang_code]
    if sub.empty:
        continue
    wc_med   = sub["word_count"].median()
    wc_p25   = sub["word_count"].quantile(0.25)
    wc_p75   = sub["word_count"].quantile(0.75)
    sl_med   = sub["avg_sent_len"].median()
    ttr_med  = sub["ttr"].median()
    punc_med = sub["punct_density"].median()

    print(f"\n  {lang_name}:")
    print(f"    Word count  : median {wc_med:.0f}  (IQR {wc_p25:.0f}–{wc_p75:.0f})")
    print(f"    Avg sent len: median {sl_med:.1f} words/sentence")
    print(f"    TTR         : median {ttr_med:.3f}  (1.0 = all unique)")
    print(f"    Punct dens. : median {punc_med:.4f}")

print("""
  ┌─────────────────────────────────────────────────────────────────┐
  │  LIKELY CAUSES OF PERFECT BASELINE SCORES                      │
  ├─────────────────────────────────────────────────────────────────┤
  │  1. LENGTH MISMATCH                                             │
  │     LLMs default to ~200-400 words. If real articles are        │
  │     shorter or longer, TF-IDF picks up length trivially.        │
  │     → Hard-constrain output length in your prompts.             │
  │       e.g. "Write a 150-word news article (exactly)"            │
  │                                                                 │
  │  2. DOMAIN / REGISTER MISMATCH                                  │
  │     Real Vuk'uzenzele text is formal government news.           │
  │     LLMs may produce generic or Wikipedia-style text.           │
  │     → Use few-shot prompting with a real article as an example. │
  │     → Specify the exact domain: government, education,          │
  │       health, infrastructure (see topic chart above).           │
  │                                                                 │
  │  3. STRUCTURAL ARTEFACTS                                        │
  │     LLMs add headers, bullet points, markdown, or intro         │
  │     phrases like "In today's news…" that don't appear in        │
  │     the corpus.                                                 │
  │     → Add: "Do NOT include headings, lists, or intro phrases."  │
  │                                                                 │
  │  4. VOCABULARY LEAKAGE                                          │
  │     LLMs may over-use English loanwords or produce unnatural    │
  │     Bantu text. Check the top-word lists above against your     │
  │     generated samples to see if word distributions diverge.     │
  │                                                                 │
  │  5. SENTENCE LENGTH UNIFORMITY                                  │
  │     LLMs tend to produce uniform sentence lengths; real         │
  │     journalistic text is more varied. Compare avg_sent_len      │
  │     distributions between your generated and real splits.       │
  │                                                                 │
  │  PROMPT TEMPLATE CHECKLIST                                      │
  │  ─────────────────────────────────────────────────────────────  │
  │  ✓ Set target language explicitly (e.g., "Write in isiZulu")    │
  │  ✓ Set a specific word-count range matching the corpus IQR      │
  │  ✓ Specify a topic from the distribution found above            │
  │  ✓ Provide one real article as a few-shot style example         │
  │  ✓ Forbid markdown, headers, English mixing, intro clichés      │
  │  ✓ Request formal news register (Vuk'uzenzele style)            │
  │  ✓ After generation: verify word count falls within real IQR    │
  └─────────────────────────────────────────────────────────────────┘
""")

# ── Save a CSV summary for quick reference ─────────────────────────────────
summary_df = df.groupby("lang")[FEATURES].agg(["mean", "median", "std"]).round(3)
summary_df.index = summary_df.index.map(LANG_NAMES)
summary_df.to_csv("corpus_summary_stats.csv")
print("  Saved summary stats to: corpus_summary_stats.csv")

print("\n  All done! Run this script, study the plots and stats, then")
print("  adjust your LLM prompts before re-generating the synthetic set.\n")