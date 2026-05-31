# MGT Detection for Bantu Languages - COS760 Group 45

> **Detecting Machine-Generated Text in low-resource Bantu languages (isiZulu, isiXhosa, Siswati) using a three-phase NLP pipeline: TF-IDF baseline → AfroXLM-R fine-tuning → SADiLaR morphological feature fusion.**

---

## Table of Contents

1. [Project Description](#project-description)
2. [Core Features](#core-features)
3. [Tech Stack](#tech-stack)
4. [Prerequisites](#prerequisites)
5. [Google Drive Access](#google-drive-access)
6. [Installation](#installation)
7. [Environment Variables](#environment-variables)
8. [Usage](#usage)
9. [Project Structure](#project-structure)

---

## Project Description

This research project tackles the problem of **Machine-Generated Text (MGT) detection** in under-resourced Bantu languages, specifically **isiZulu**, **isiXhosa**, and **Siswati**. Given that modern large language models (LLMs) - including ChatGPT-4o, Claude, and Gemini 2.5 Pro - can now generate fluent text in these languages, the need for reliable automated detection is growing, particularly for academic integrity and media verification contexts.

The system is structured as a **three-phase pipeline**:

- **Phase 1 - Baseline:** A character n-gram TF-IDF + Logistic Regression classifier trained on isiZulu and isiXhosa, then evaluated zero-shot on Siswati.
- **Phase 2 - Transfer Learning:** Fine-tuning of `Davlan/afro-xlmr-base` (an XLM-RoBERTa variant pre-trained on African languages) on the same multilingual corpus, with Optuna hyperparameter search.
- **Phase 3 - Feature Fusion:** A Random Forest ensemble that combines the AfroXLM-R predicted probabilities with morphological features derived from the **SADiLaR** morphological resource (lexical diversity, morphological coverage, bigram repetition, etc.).

A **Streamlit web application** (`app.py`) exposes all three phases as an interactive demo, including SHAP-based explainability visualisations.

Human text is sourced from the **Vukuzenzele** government newsletter corpus (via Hugging Face's `dsfsi/vukuzenzele-monolingual` dataset). Machine-generated text was produced by prompting ChatGPT-4o, Claude, and Gemini 2.5 Pro, and is stored under `data/raw/`.

---

## Core Features

- **Three-phase detection pipeline** progressing from a lightweight classical baseline to a morphology-aware neural ensemble.
- **Cross-lingual zero-shot evaluation** - models trained on isiZulu and isiXhosa are tested on unseen Siswati data with no Siswati training examples.
- **Cross-LLM generalisation analysis** - per-model breakdown of detection performance across ChatGPT, Claude, and Gemini outputs.
- **SADiLaR morphological feature extraction** - computes lexical diversity, SADiLaR dictionary coverage, morphological diversity ratio, word/bigram repetition rates, and more from language-specific morphological lexica.
- **AfroXLM-R fine-tuning** (`Davlan/afro-xlmr-base`) with Optuna-guided hyperparameter optimisation, cosine learning-rate scheduling, and custom confidence calibration.
- **Artefact and leakage cleaning** - automated removal of ChatGPT numeric tracking tokens and formulaic MGT sentence openers before training.
- **Interactive demo** - paste arbitrary Bantu-language text and receive phase-by-phase detection verdicts with confidence scores and SHAP explanations.
- **Comprehensive results artefacts** - confusion matrices, calibration curves, training curves, SHAP summary and bar plots, and phase-comparison spider graphs saved to `results/`.
- **Reproducible data preparation** script that downloads and merges the Vukuzenzele corpus with the raw MGT CSV files into a single `merged_dataset.csv`.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.10+ |
| **Notebooks** | Jupyter Notebook / Google Colab |
| **Classical ML** | scikit-learn (`TfidfVectorizer`, `LogisticRegression`, `RandomForestClassifier`, `StratifiedKFold`) |
| **Deep Learning** | PyTorch, Hugging Face `transformers` (`AutoModelForSequenceClassification`, `Trainer`, `TrainingArguments`) |
| **Pre-trained Model** | `Davlan/afro-xlmr-base` (XLM-RoBERTa architecture, African-language pre-training) |
| **Tokenizer** | `XLMRobertaTokenizer` (max length 512) |
| **Hyperparameter Search** | Optuna |
| **Dataset Loading** | Hugging Face `datasets` (`dsfsi/vukuzenzele-monolingual`) |
| **Data Manipulation** | pandas, NumPy |
| **Visualisation** | matplotlib, SHAP |
| **Web App** | Streamlit |
| **Morphological Resource** | SADiLaR lexica (isiZulu, isiXhosa, Siswati `.txt` morphological dictionaries) |
| **Optimiser** | AdamW with cosine schedule (`get_cosine_schedule_with_warmup`) |

---

## Prerequisites

Ensure the following are installed before proceeding:

| Software | Minimum Version | Notes |
|---|---|---|
| Python | 3.10 | 3.11+ recommended for `float \| None` union hints |
| pip | 22.0+ | or use conda / mamba |
| Git | 2.x | for cloning |
| CUDA (optional) | 11.8+ | GPU acceleration for Phase 2 fine-tuning |
| Google Drive (optional) | - | Phase 2 fine-tuning notebook is designed for Google Colab with Drive mount |

---
## Google Drive Access

The easiest way to run this project is through the shared Google Drive folder, which contains the complete repository, datasets, trained models, notebooks, and results:

https://drive.google.com/drive/folders/12Sexw9HhlMKX4YtGL5tTrRSwujUxUoSM?usp=drive_link

To run the project:

1. Open the shared Google Drive folder, and create shortcut to your drive.
2. Open `code/main.ipynb` in Google Colab.
3. Mount Google Drive when prompted.
4. Run all notebook cells from top to bottom.

OR
1. Unzip folder
2. Add folder to your drive
3. Open `code/main.ipynb` in Google Colab.
3. Mount Google Drive when prompted.
4. Run all notebook cells from top to bottom.
   
The notebook automatically locates the project directory within the shared Drive folder and configures all file paths dynamically. No manual path modifications are required.

Running `main.ipynb` reproduces the complete three-phase machine-generated text detection pipeline, including:

- Phase 1: TF-IDF + Logistic Regression baseline
- Phase 2: AfroXLMR fine-tuning and cross-lingual evaluation
- Phase 3: SADiLaR feature augmentation, SHAP analysis, and visualisations
- Bonus: Last two code blocks, allow you to insert human or machine generated text and it runs it against all 3 phases

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/CaitlinCalder/COS760-Group-45.git
cd COS760-Group-45
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate       # Linux / macOS
.venv\Scripts\activate          # Windows
```

### 3. Install Python dependencies

pip install -r requirements.txt

### 4. Prepare the data

Download and merge the Vukuzenzele human-text corpus with the raw MGT CSV files. This step requires internet access to pull the Hugging Face dataset.

```bash
python code/dataset_prep.py
```

This will:
- Download `dsfsi/vukuzenzele-monolingual` for isiZulu (`zul`), isiXhosa (`xho`), and Siswati (`ssw`)
- Load the three raw MGT CSV files from `data/raw/`
- Clean artefacts and formulaic sentence openers
- Truncate texts to ~800 characters at sentence boundaries
- Write `data/processed/merged_dataset.csv`, `data/processed/human_text.csv`, and `data/processed/mgt_text.csv`

### 5. Extract SADiLaR morphological features (Phase 3)

```bash
python code/sadilar_morph_features.py
```

This reads `data/sadilar/*.txt` morphological lexica and writes `data/processed/sadilar_morph_features.csv`.

---

## Environment Variables

Create a `.env` file (or export variables in your shell) before running Phase 2 or Phase 3 inference scripts:

```dotenv
# Path to the fine-tuned AfroXLM-R model directory (Phase 2 output)
# Defaults to <project_root>/models/best_model if not set
AFROXLMR_MODEL_PATH=
```

> **Security note:** Never commit actual model paths, API keys, or Google Drive credentials to version control.

---

## Usage

### Run the Phase 1 baseline

Trains a TF-IDF + Logistic Regression classifier and evaluates in-language and zero-shot cross-lingual performance. Saves metrics to `results/metrics/baseline_metrics.json` and plots to `results/plots/`.

```bash
python code/baseline.py
```

### Fine-tune AfroXLM-R (Phase 2) - Google Colab

The Phase 2 fine-tuning notebook is designed to run on Google Colab with a GPU runtime. Open it directly:

1. Upload `code/finetuning_afrolxm.ipynb` (or the equivalent `.py` script `code/phase2_finetuning_afrolxm.py`) to Colab.
2. Mount Google Drive and set `BASE_DIR` to your Drive path.
3. Upload `data/processed/merged_dataset.csv` to the Drive directory.
4. Run all cells. The best model checkpoint will be saved to your Drive under `afroxlmr_detector/`.

Alternatively, run the script locally (GPU strongly recommended):

```bash
python code/phase2_finetuning_afrolxm.py
```

### Run the Phase 3 SADiLaR + AfroXLM-R ensemble

```bash
export AFROXLMR_MODEL_PATH=/path/to/your/best_model
python code/sadilar_classifier.py
```

### Run the full three-phase experiment pipeline

```bash
python code/experiment.py
```

This retrains Phase 1 and runs all three detectors on arbitrary input text, printing verdicts and confidence scores to stdout.

### Launch the Streamlit web application

```bash
export AFROXLMR_MODEL_PATH=/path/to/your/best_model
streamlit run code/app.py
```

Open your browser at `http://localhost:8501`. Paste any isiZulu, isiXhosa, or Siswati text to receive a human/machine verdict from all three phases, with SHAP feature-importance explanations.

### Explore the main analysis notebook

```bash
jupyter notebook code/main.ipynb
```

### Run the topic finder (data pre-processing utility)

```bash
jupyter notebook "code/data pre-processing/Topic_Finder.ipynb"
# or
python "code/data pre-processing/topic_finder.py"
```

---

## Project Structure

```
COS760-Group-45/
│
├── code/                              # All source code
│   ├── app.py                         # Streamlit interactive demo (all 3 phases + SHAP)
│   ├── baseline.py                    # Phase 1: TF-IDF + Logistic Regression classifier
│   ├── dataset_prep.py                # Data ingestion, cleaning, and merging script
│   ├── experiment.py                  # Full three-phase inference pipeline
│   ├── finetuning_afrolxm.ipynb       # Phase 2: AfroXLM-R fine-tuning (Colab notebook)
│   ├── phase2_finetuning_afrolxm.py   # Phase 2: standalone Python script equivalent
│   ├── sadilar_classifier.py          # Phase 3: RF ensemble (AfroXLMR probs + morph features)
│   ├── sadilar_morph_features.py      # Phase 3: SADiLaR feature extraction
│   ├── sadilar_plots_shap.py          # SHAP and results visualisation utilities
│   ├── main.ipynb                     # Main analysis and results notebook
│   ├── best from phase 2/             # Saved Phase 2 model artefacts
│   │   ├── config.json                # XLM-RoBERTa model config
│   │   ├── tokenizer.json             # Tokenizer vocabulary
│   │   ├── tokenizer_config.json      # Tokenizer settings (max_length 512)
│   │   ├── training_args.bin          # Serialised TrainingArguments
│   │   └── best_model_notes.pdf       # Phase 2 experiment notes
│   └── data pre-processing/           # Data exploration utilities
│       ├── Topic_Finder.ipynb         # Topic modelling notebook
│       ├── topic_finder.py            # Topic finder script
│       └── preprocessng.py            # Preprocessing helpers
│
├── data/
│   ├── raw/                           # Raw MGT CSV files (one per LLM)
│   │   ├── machine_generated_chatgpt-4o.csv
│   │   ├── machine_generated_claude.csv
│   │   └── machine_generated_gemini-2.5-pro.csv
│   ├── processed/                     # Generated by dataset_prep.py and sadilar_morph_features.py
│   │   ├── merged_dataset.csv         # Combined human + MGT dataset (primary input)
│   │   ├── human_text.csv
│   │   ├── mgt_text.csv
│   │   └── sadilar_morph_features.csv # Morphological features for Phase 3
│   └── sadilar/                       # SADiLaR morphological lexica
│       ├── zulu_morph.txt
│       ├── xhosa_morph.txt
│       └── siswati_morph.txt
│
├── results/
│   ├── metrics/                       # JSON evaluation metrics per phase
│   │   ├── baseline_metrics.json
│   │   └── sadilar_results (1).json
│   └── plots/                         # All generated figures
│       ├── phase2/                    # Phase 2 training curves, confusion matrices
│       └── Phase3/                    # Phase 3 SHAP plots, feature importance, comparisons
│
└── README.md
```

---

## Key Design Decisions

**Leakage prevention.** During Phase 1, both punctuation and capitalisation are stripped before TF-IDF vectorisation because early analysis showed MGT texts contained ~3× more capital letters and ~2× more commas than human text - trivial surface features that would inflate accuracy without generalising.

**Artefact cleaning.** ChatGPT-4o outputs contained numeric tracking tokens matching the pattern `\d+-\d+-\d+` in nearly every row. These were automatically stripped in `dataset_prep.py` before training to prevent the model from exploiting a data-collection artefact.

**Text truncation.** All texts are capped at ~800 characters (cut at the nearest sentence boundary) to normalise document lengths across human and MGT sources and to keep Phase 1 feature distributions comparable.

**Zero-shot cross-lingual evaluation.** Siswati is held out entirely from training in all three phases. Its inclusion only at test time measures true cross-lingual transfer, which is particularly relevant given the limited Siswati NLP resources available.

---

## Contributors

COS760 (Natural Language Processing) - Group 45 [Caitlin Calder (u23678748) Joy Bengu (u25000307) Kelita Naidoo (u20534575)], University of Pretoria.
