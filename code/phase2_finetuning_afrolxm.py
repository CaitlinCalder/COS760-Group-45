import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    get_cosine_schedule_with_warmup
)
from torch.optim import AdamW
from datasets import Dataset
import optuna
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef
from scipy.special import softmax as scipy_softmax
import pickle
import json

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_PATH   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR    = os.path.join(BASE_PATH, "results")
FIG_DIR     = os.path.join(BASE_DIR, "figures")
METRICS_DIR = os.path.join(BASE_DIR, "metrics")

os.makedirs(BASE_DIR,    exist_ok=True)
os.makedirs(FIG_DIR,     exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

MERGED_CSV          = os.path.join(BASE_PATH, "data", "processed", "merged_dataset.csv")
PHASE1_METRICS_JSON = os.path.join(METRICS_DIR, "baseline_metrics.json")

# ── Config ─────────────────────────────────────────────────────────────────────
AFROXLMR_MODEL = "Davlan/afro-xlmr-base"

LABEL2ID   = {"human": 0, "machine": 1}
ID2LABEL   = {0: "human", 1: "machine"}
NUM_LABELS = 2

MAX_LENGTH    = 512
BATCH_SIZE    = 8
EPOCHS        = 5
LEARNING_RATE = 2e-5
SEED          = 42

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(MERGED_CSV)

df.columns = df.columns.str.strip().str.lower()
df = df.rename(columns={
    "text_generated":   "text",
    "language_code":    "language",
    "model_identifier": "model"
})

if df["label"].dtype == object:
    df["label"] = df["label"].str.strip().str.lower().map({"human": 0, "machine": 1})
df["label"]    = df["label"].astype(int)
df["language"] = df["language"].str.strip().str.lower()

assert set(df["label"].unique()).issubset({0, 1}), \
    "label column must contain only 0/1 or 'human'/'machine'"
assert {"zu", "xh", "ss"}.issubset(set(df["language"].unique())), \
    "language column must contain 'zu', 'xh', and 'ss' values"

train_val_df = df[df["language"].isin(["zu", "xh"])].copy()
siswati_df   = df[df["language"] == "ss"].copy()

siswati_val, siswati_test = train_test_split(
    siswati_df, test_size=0.7, random_state=SEED, stratify=siswati_df["label"]
)

train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.2,
    random_state=SEED,
    stratify=train_val_df[["label", "model"]]
)

print(f"Loaded {len(df)} total samples from CSV\n")
print("Language breakdown:")
print(df["language"].value_counts().to_string())
print(f"\nSplit summary:")
print(f"Train (Zulu+Xhosa) : {len(train_df):>5} samples")
print(f"Val   (Zulu+Xhosa) : {len(val_df):>5} samples")
print(f"Test  (Siswati)    : {len(siswati_df):>5} samples")
print(f"\nTrain label distribution:")
print(train_df["label"].map({0: "human", 1: "machine"}).value_counts().to_string())
print(f"\nSiswati label distribution:")
print(siswati_df["label"].map({0: "human", 1: "machine"}).value_counts().to_string())

# ── Tokenizer & datasets ───────────────────────────────────────────────────────
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(AFROXLMR_MODEL)
print(f"\nLoaded tokenizer: {AFROXLMR_MODEL}")


def tokenize(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )


def make_dataset(df):
    ds = Dataset.from_pandas(df[["text", "label"]].reset_index(drop=True))
    return ds.map(tokenize, batched=True)


train_ds   = make_dataset(train_df)
val_ds     = make_dataset(val_df)
siswati_ds = make_dataset(siswati_df)

print("Datasets have been tokenized")

# ── Metrics ────────────────────────────────────────────────────────────────────
import evaluate
from sklearn.metrics import classification_report, confusion_matrix

f1_metric        = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric    = evaluate.load("recall")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "macro_f1":  f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"],
        "precision": precision_metric.compute(predictions=preds, references=labels, average="macro")["precision"],
        "recall":    recall_metric.compute(predictions=preds, references=labels, average="macro")["recall"],
    }


# ── Custom Trainer & helpers ───────────────────────────────────────────────────
from transformers import TrainerCallback, EarlyStoppingCallback
torch.cuda.empty_cache()


class OptunaPruningCallback(TrainerCallback):
    def __init__(self, trial):
        self.trial = trial

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None and "eval_macro_f1" in metrics:
            current_value = metrics["eval_macro_f1"]
            self.trial.report(current_value, step=int(state.epoch))
            if self.trial.should_prune():
                control.should_training_stop = True
                raise optuna.TrialPruned()


def get_layerwise_optimizer(model, base_lr=2e-5, decay_factor=0.9, weight_decay=0.01):
    no_decay   = ["bias", "LayerNorm.weight"]
    num_layers = model.config.num_hidden_layers

    param_groups = []

    for n, p in model.named_parameters():
        if "classifier" in n or "pooler" in n:
            wd = 0.0 if any(nd in n for nd in no_decay) else weight_decay
            param_groups.append({"params": [p], "lr": base_lr, "weight_decay": wd})

    for i in range(num_layers - 1, -1, -1):
        layer_lr = base_lr * (decay_factor ** (num_layers - i))
        for n, p in model.named_parameters():
            if f"encoder.layer.{i}." in n:
                wd = 0.0 if any(nd in n for nd in no_decay) else weight_decay
                param_groups.append({"params": [p], "lr": layer_lr, "weight_decay": wd})

    emb_lr = base_lr * (decay_factor ** (num_layers + 1))
    for n, p in model.named_parameters():
        if "embeddings" in n:
            wd = 0.0 if any(nd in n for nd in no_decay) else weight_decay
            param_groups.append({"params": [p], "lr": emb_lr, "weight_decay": wd})

    return AdamW(param_groups)


def freeze_bottom_layers(model, freeze_n=4):
    for param in model.base_model.embeddings.parameters():
        param.requires_grad = False
    for i in range(freeze_n):
        for param in model.base_model.encoder.layer[i].parameters():
            param.requires_grad = False
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Frozen bottom {freeze_n} layers | "
          f"Trainable: {n_train/1e6:.1f}M / {n_total/1e6:.1f}M params")
    return model


class RDropTrainer(Trainer):
    def __init__(self, *args, rdrop_alpha=0.5, label_smoothing=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.rdrop_alpha     = rdrop_alpha
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        out1   = model(**inputs)
        out2   = model(**inputs)

        ce      = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        ce_loss = (ce(out1.logits, labels) + ce(out2.logits, labels)) / 2

        p1 = F.log_softmax(out1.logits, dim=-1)
        p2 = F.log_softmax(out2.logits, dim=-1)
        kl = (F.kl_div(p1, p2.exp(), reduction="batchmean") +
              F.kl_div(p2, p1.exp(), reduction="batchmean")) / 2

        loss = ce_loss + self.rdrop_alpha * kl
        return (loss, out1) if return_outputs else loss


# ── Optuna hyperparameter search ───────────────────────────────────────────────
import shutil
torch.cuda.empty_cache()


def model_init():
    return AutoModelForSequenceClassification.from_pretrained(
        AFROXLMR_MODEL, num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID
    )


def objective(trial):
    lr            = trial.suggest_float("learning_rate",   1e-5, 4e-5, log=True)
    batch_size    = trial.suggest_categorical("batch_size", [8, 16])
    weight_decay  = trial.suggest_float("weight_decay",    0.0,  0.2)
    epochs        = trial.suggest_int("epochs",            3,    5)
    warmup_ratio  = trial.suggest_float("warmup_ratio",    0.05, 0.2)
    decay_factor  = trial.suggest_float("decay_factor",    0.75, 0.95)
    rdrop_alpha   = trial.suggest_float("rdrop_alpha",     0.1,  1.0)
    label_smooth  = trial.suggest_float("label_smoothing", 0.05, 0.15)
    freeze_n      = trial.suggest_int("freeze_n",          0,    6)

    args = TrainingArguments(
        output_dir=os.path.join(BASE_DIR, f"trial_{trial.number}"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
        save_total_limit=1,
        logging_steps=20,
        report_to="none",
        seed=SEED,
        dataloader_num_workers=2,
    )

    model     = model_init()
    model     = freeze_bottom_layers(model, freeze_n=freeze_n)
    optimizer = get_layerwise_optimizer(model, base_lr=lr,
                                        decay_factor=decay_factor,
                                        weight_decay=weight_decay)
    total_steps  = (len(train_ds) // batch_size) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    trainer = RDropTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler),
        rdrop_alpha=rdrop_alpha,
        label_smoothing=label_smooth,
        callbacks=[OptunaPruningCallback(trial)],
    )
    try:
        trainer.train()
        result = trainer.evaluate(siswati_ds)
    except optuna.TrialPruned:
        shutil.rmtree(args.output_dir, ignore_errors=True)
        raise optuna.TrialPruned()
    except Exception as e:
        shutil.rmtree(args.output_dir, ignore_errors=True)
        raise e

    shutil.rmtree(args.output_dir, ignore_errors=True)
    return result["eval_macro_f1"]


print("Starting Optuna hyperparameter search...")
study = optuna.create_study(
    direction="maximize",
    study_name="afro_xlmr_zeroshot_v2",
    sampler=optuna.samplers.TPESampler(seed=SEED),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1),
)
study.optimize(objective, n_trials=15, timeout=None, gc_after_trial=True)

print(f"\nBest macro_f1 : {study.best_value:.4f}")
print(f"Best params   : {study.best_params}")

# ── Final training with best params ───────────────────────────────────────────
bp = study.best_params
print("\nRetraining with best hyperparameters...")

final_args = TrainingArguments(
    output_dir=os.path.join(BASE_DIR, "finetuned"),
    num_train_epochs=bp["epochs"],
    per_device_train_batch_size=bp["batch_size"],
    per_device_eval_batch_size=bp["batch_size"],
    learning_rate=bp["learning_rate"],
    lr_scheduler_type="cosine",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    gradient_accumulation_steps=2,
    max_grad_norm=1.0,
    save_total_limit=2,
    logging_steps=20,
    report_to="none",
    seed=SEED,
    dataloader_num_workers=2,
)

ft_model  = model_init()
ft_model  = freeze_bottom_layers(ft_model, freeze_n=bp["freeze_n"])
optimizer = get_layerwise_optimizer(ft_model, base_lr=bp["learning_rate"],
                                    decay_factor=bp["decay_factor"],
                                    weight_decay=bp["weight_decay"])
total_steps  = (len(train_ds) // bp["batch_size"]) * bp["epochs"]
warmup_steps = int(total_steps * bp["warmup_ratio"])
scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

trainer = RDropTrainer(
    model=ft_model,
    args=final_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
    optimizers=(optimizer, scheduler),
    rdrop_alpha=bp["rdrop_alpha"],
    label_smoothing=bp["label_smoothing"],
)
trainer.train()

best_model_path = os.path.join(BASE_DIR, "best_model")
trainer.save_model(best_model_path)
tokenizer.save_pretrained(best_model_path)
print(f"Best model saved to {best_model_path}")

# ── Cross-lingual evaluation on Siswati ───────────────────────────────────────
print("\nCROSS-LINGUAL EVALUATION: Siswati (zero-shot)")
print("-" * 55)

ft_results = trainer.evaluate(siswati_ds)
print("\nFine-tuned AfroXLMR results on Siswati:")
for k, v in ft_results.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.4f}")

preds_output = trainer.predict(siswati_ds)
ft_preds     = np.argmax(preds_output.predictions, axis=-1)
true_labels  = siswati_df["label"].values

print("\nClassification Report:")
print(classification_report(true_labels, ft_preds,
                             target_names=["Human", "Machine"], digits=4))

cm = confusion_matrix(true_labels, ft_preds)
print("Confusion Matrix (rows=true, cols=predicted):")
print(f"                Pred:Human  Pred:Machine")
print(f"  True:Human       {cm[0,0]:>6}        {cm[0,1]:>6}")
print(f"  True:Machine     {cm[1,0]:>6}        {cm[1,1]:>6}")

# ── Phase 1 vs Phase 2 comparison ─────────────────────────────────────────────
print("\nComparing TF-IDF Baseline (Phase 1) vs AfroXLMR (Phase 2)")
print("On Siswati zero-shot / cross-lingual test set")
print("-" * 55)

if os.path.exists(PHASE1_METRICS_JSON):
    with open(PHASE1_METRICS_JSON) as f:
        phase1_all = json.load(f)
    phase1 = phase1_all.get("siswati_zeroshot", {})
    print(f"Loaded Phase 1 metrics from {PHASE1_METRICS_JSON}\n")
else:
    print(f"Phase 1 metrics not found at {PHASE1_METRICS_JSON}\n")
    phase1 = {}

phase2 = {
    "precision": ft_results.get("eval_precision", float("nan")),
    "recall":    ft_results.get("eval_recall",    float("nan")),
    "macro_f1":  ft_results.get("eval_macro_f1",  float("nan")),
    "mcc":       round(matthews_corrcoef(true_labels, ft_preds), 4),
}

metrics_to_compare = ["precision", "recall", "macro_f1", "mcc"]
header = f"  {'Metric':<14} {'TF-IDF + LR':>14} {'AfroXLMR FT':>14} {'Δ Change':>12}"
print(header)
print("  " + "-" * (len(header) - 2))

for m in metrics_to_compare:
    p1_val    = phase1.get(m, float("nan"))
    p2_val    = phase2.get(m, float("nan"))
    delta     = p2_val - p1_val if not (np.isnan(p1_val) or np.isnan(p2_val)) else float("nan")
    sign      = "+" if delta >= 0 else ""
    delta_str = f"{sign}{delta:.4f}" if not np.isnan(delta) else "N/A"
    print(f"  {m:<14} {p1_val:>14.4f} {p2_val:>14.4f} {delta_str:>12}")

f1_delta = phase2["macro_f1"] - phase1.get("macro_f1", float("nan"))
if not np.isnan(f1_delta):
    improved = f1_delta > 0
    print(f"\n{'Improved' if improved else 'Outperformed'} AfroXLMR "
          f"{'outperforms' if improved else 'underperforms'} "
          f"TF-IDF baseline on Siswati by {f1_delta:+.4f} Macro-F1")

phase2_metrics = {
    "model":            AFROXLMR_MODEL,
    "train_languages":  ["zu", "xh"],
    "test_language":    "ss",
    "siswati_crosslingual": phase2,
}
out_path = os.path.join(METRICS_DIR, "phase2_metrics.json")
with open(out_path, "w") as f:
    json.dump(phase2_metrics, f, indent=2)
print(f"\nPhase 2 metrics saved to {out_path}")

# ── Sample predictions ─────────────────────────────────────────────────────────
def predict(text: str, verbose=True):
    ft_model.eval()
    inputs = tokenizer(
        text, return_tensors="pt",
        truncation=True, max_length=MAX_LENGTH
    ).to(ft_model.device)
    with torch.no_grad():
        logits = ft_model(**inputs).logits
    probs   = torch.softmax(logits, dim=-1).squeeze()
    pred_id = probs.argmax().item()
    if verbose:
        print(f"Text       : {text[:80]}{'...' if len(text) > 80 else ''}")
        print(f"Prediction : {ID2LABEL[pred_id].upper()}")
        print(f"Confidence : Human={probs[0]:.3f}  Machine={probs[1]:.3f}\n")
    return ID2LABEL[pred_id], probs.tolist()


print("\n" + "=" * 55)
print("Sample Siswati Predictions")
print("=" * 55 + "\n")
for row in siswati_df.head(8).itertuples():
    predict(row.text, verbose=False)

# ── Visualisations ─────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.calibration import calibration_curve

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


def save(fig, name):
    path = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved → {path}")


# Training curves
log_history = trainer.state.log_history
train_loss, eval_loss, eval_f1 = [], [], []

for entry in log_history:
    if "loss" in entry and "epoch" in entry:
        train_loss.append((entry["epoch"], entry["loss"]))
    if "eval_loss" in entry:
        eval_loss.append((entry["epoch"], entry["eval_loss"]))
        eval_f1.append((entry["epoch"],   entry.get("eval_macro_f1", float("nan"))))

if train_loss:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Training Curves-AfroXLMR Fine-tuning", fontweight="bold")

    tl_x, tl_y = zip(*train_loss)
    el_x, el_y = zip(*eval_loss)
    ef_x, ef_y = zip(*eval_f1)

    axes[0].plot(tl_x, tl_y, label="Train loss", marker="o", markersize=3)
    axes[0].plot(el_x, el_y, label="Val loss",   marker="s", markersize=3)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Cross-Entropy Loss"); axes[0].legend()

    axes[1].plot(ef_x, ef_y, color="seagreen", marker="D", markersize=4)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Macro F1")
    axes[1].set_title("Validation Macro-F1"); axes[1].set_ylim(0, 1)

    plt.tight_layout()
    save(fig, "01_training_curves")
    plt.show()
else:
    print("  (no training log found, skipping training curves)")

# Confusion matrix
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Pred: Human", "Pred: Machine"],
    yticklabels=["True: Human", "True: Machine"],
    linewidths=0.5, ax=ax,
)
ax.set_title("Confusion Matrix-Siswati Zero-Shot", fontweight="bold")
ax.set_ylabel("True label"); ax.set_xlabel("Predicted label")
plt.tight_layout()
save(fig, "02_confusion_matrix")
plt.show()

# Phase 1 vs Phase 2 grouped bar
metrics_labels = ["Precision", "Recall", "Macro F1", "MCC"]
p1_vals = [phase1.get(k, float("nan")) for k in ["precision", "recall", "macro_f1", "mcc"]]
p2_vals = [phase2.get(k, float("nan")) for k in ["precision", "recall", "macro_f1", "mcc"]]

x     = np.arange(len(metrics_labels))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
bars1 = ax.bar(x - width/2, p1_vals, width, label="TF-IDF + LR (Phase 1)",
               color="#4C72B0", edgecolor="white")
bars2 = ax.bar(x + width/2, p2_vals, width, label="AfroXLMR FT (Phase 2)",
               color="#55A868", edgecolor="white")

for bar in bars1 + bars2:
    h = bar.get_height()
    if not np.isnan(h):
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x); ax.set_xticklabels(metrics_labels)
ax.set_ylim(0, 1.08)
ax.set_ylabel("Score"); ax.set_title("Phase 1 vs Phase 2-Siswati Test Set", fontweight="bold")
ax.legend()
plt.tight_layout()
save(fig, "03_phase_comparison")
plt.show()

# Per-class F1
from sklearn.metrics import f1_score
f1_human   = f1_score(true_labels, ft_preds, pos_label=0)
f1_machine = f1_score(true_labels, ft_preds, pos_label=1)

fig, ax = plt.subplots(figsize=(5, 4))
bars = ax.bar(["Human", "Machine"], [f1_human, f1_machine],
              color=["#4C72B0", "#C44E52"], edgecolor="white", width=0.4)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", fontsize=10)
ax.set_ylim(0, 1.1)
ax.set_ylabel("F1 Score")
ax.set_title("Per-Class F1-Siswati Zero-Shot", fontweight="bold")
plt.tight_layout()
save(fig, "04_perclass_f1")
plt.show()

# Confidence distribution
xlmr_probs   = scipy_softmax(preds_output.predictions, axis=-1)
machine_conf = xlmr_probs[:, 1]

fig, ax = plt.subplots(figsize=(8, 4))
for lbl, colour in [(0, "#4C72B0"), (1, "#C44E52")]:
    mask = true_labels == lbl
    ax.hist(machine_conf[mask], bins=30, alpha=0.6,
            color=colour, label=f"True: {'Human' if lbl==0 else 'Machine'}", density=True)
ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="Decision boundary")
ax.set_xlabel("P(Machine)"); ax.set_ylabel("Density")
ax.set_title("Confidence Distribution-Siswati Predictions", fontweight="bold")
ax.legend()
plt.tight_layout()
save(fig, "05_confidence_distribution")
plt.show()

# Calibration curve
prob_true, prob_pred = calibration_curve(true_labels, machine_conf, n_bins=10)

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(prob_pred, prob_true, marker="o", label="AfroXLMR", color="#55A868")
ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect calibration")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Fraction of positives")
ax.set_title("Calibration Curve-Siswati", fontweight="bold")
ax.legend()
plt.tight_layout()
save(fig, "06_calibration_curve")
plt.show()

# Optuna trial history
trial_nums  = [t.number for t in study.trials if t.value is not None]
trial_vals  = [t.value  for t in study.trials if t.value is not None]
best_so_far = [max(trial_vals[:i+1]) for i in range(len(trial_vals))]

fig, ax = plt.subplots(figsize=(9, 4))
ax.scatter(trial_nums, trial_vals, zorder=3, label="Trial F1", color="#4C72B0")
ax.plot(trial_nums, best_so_far, color="#C44E52", linewidth=2, label="Best so far")
ax.set_xlabel("Trial number"); ax.set_ylabel("Validation Macro F1")
ax.set_title("Optuna Trial History", fontweight="bold")
ax.legend()
plt.tight_layout()
save(fig, "07_optuna_trials")
plt.show()

# Dataset split summary
split_names   = ["Train\n(Zu+Xh)", "Val\n(Zu+Xh)", "Test\n(Siswati)"]
split_human   = [(train_df["label"] == 0).sum(), (val_df["label"] == 0).sum(),   (siswati_df["label"] == 0).sum()]
split_machine = [(train_df["label"] == 1).sum(), (val_df["label"] == 1).sum(),   (siswati_df["label"] == 1).sum()]

x = np.arange(len(split_names))
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(x, split_human,   label="Human",   color="#4C72B0", edgecolor="white")
ax.bar(x, split_machine, bottom=split_human, label="Machine", color="#C44E52", edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(split_names)
ax.set_ylabel("Sample count")
ax.set_title("Dataset Split & Class Distribution", fontweight="bold")
ax.legend()
plt.tight_layout()
save(fig, "08_dataset_split")
plt.show()

# ROC curve
from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(true_labels, machine_conf)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(fpr, tpr, color="#55A868", lw=2, label=f"AfroXLMR (AUC = {roc_auc:.4f})")
ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random classifier")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve-Siswati Zero-Shot", fontweight="bold")
ax.legend(loc="lower right")
plt.tight_layout()
save(fig, "09_roc_curve")
plt.show()

# Precision-Recall curve
from sklearn.metrics import precision_recall_curve, average_precision_score

precision_vals, recall_vals, _ = precision_recall_curve(true_labels, machine_conf)
ap = average_precision_score(true_labels, machine_conf)

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(recall_vals, precision_vals, color="#4C72B0", lw=2,
        label=f"AfroXLMR (AP = {ap:.4f})")
ax.axhline(y=true_labels.mean(), color="grey", linestyle="--",
           label=f"Baseline (prevalence = {true_labels.mean():.2f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve-Siswati Zero-Shot", fontweight="bold")
ax.legend(); ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
plt.tight_layout()
save(fig, "10_precision_recall_curve")
plt.show()

# Error analysis
siswati_conf    = machine_conf
siswati_correct = (ft_preds == true_labels).astype(int)

fig, ax = plt.subplots(figsize=(8, 4))
colors     = {0: "#C44E52", 1: "#55A868"}
labels_map = {0: "Incorrect", 1: "Correct"}

for outcome in [0, 1]:
    mask = siswati_correct == outcome
    ax.scatter(siswati_conf[mask],
               np.random.uniform(-0.3, 0.3, mask.sum()),
               alpha=0.4, s=12, color=colors[outcome], label=labels_map[outcome])

ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="Decision boundary")
ax.set_xlabel("P(Machine)-model confidence")
ax.set_yticks([]); ax.set_ylabel("")
ax.set_title("Error Analysis: Confidence vs Correctness-Siswati", fontweight="bold")
ax.legend(loc="upper left")
plt.tight_layout()
save(fig, "11_error_analysis")
plt.show()

print(f"\nAll figures saved to {FIG_DIR}")

# ── Cross-LLM generalisation ───────────────────────────────────────────────────
full_ds           = make_dataset(df)
full_preds_output = trainer.predict(full_ds)
full_preds        = np.argmax(full_preds_output.predictions, axis=-1)
full_probs        = scipy_softmax(full_preds_output.predictions, axis=-1)

results_df = df.copy().reset_index(drop=True)
results_df["pred_label"]  = full_preds
results_df["prob_human"]  = full_probs[:, 0]
results_df["prob_machine"] = full_probs[:, 1]
results_df["correct"]     = results_df["pred_label"] == results_df["label"]

print(results_df[["language", "model", "label", "pred_label", "prob_machine"]].head(10))

from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef


def compute_group_metrics(group_df):
    y_true = group_df["label"].values
    y_pred = group_df["pred_label"].values
    if len(set(y_true)) < 2:
        return None
    return {
        "n_samples": len(group_df),
        "precision": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "macro_f1":  round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "mcc":       round(matthews_corrcoef(y_true, y_pred), 4),
        "accuracy":  round((y_true == y_pred).mean(), 4),
    }


rows = []
for (lang, model_name), group in results_df[results_df["label"] == 1].groupby(["language", "model"]):
    human_rows = results_df[(results_df["language"] == lang) & (results_df["label"] == 0)]
    combined   = pd.concat([group, human_rows])
    metrics    = compute_group_metrics(combined)
    if metrics:
        metrics["language"] = lang
        metrics["model"]    = model_name
        rows.append(metrics)

metrics_df = pd.DataFrame(rows)[["language", "model", "n_samples",
                                  "precision", "recall", "macro_f1", "mcc", "accuracy"]]
metrics_df = metrics_df.sort_values(["language", "macro_f1"], ascending=[True, False])
print(metrics_df.to_string(index=False))

rows_by_model = []
for model_name, group in results_df[results_df["label"] == 1].groupby("model"):
    human_rows = results_df[results_df["label"] == 0]
    combined   = pd.concat([group, human_rows])
    metrics    = compute_group_metrics(combined)
    if metrics:
        metrics["model"] = model_name
        rows_by_model.append(metrics)

model_metrics_df = pd.DataFrame(rows_by_model)[["model", "n_samples",
                                                  "precision", "recall",
                                                  "macro_f1", "mcc", "accuracy"]]
model_metrics_df = model_metrics_df.sort_values("macro_f1", ascending=False)
print(model_metrics_df.to_string(index=False))

# Cross-LLM per-language bar charts
languages = metrics_df["language"].unique()
fig, axes = plt.subplots(1, len(languages), figsize=(6 * len(languages), 5), sharey=True)
if len(languages) == 1:
    axes = [axes]

for ax, lang in zip(axes, languages):
    subset      = metrics_df[metrics_df["language"] == lang].sort_values("macro_f1")
    model_names = subset["model"].tolist()
    f1_scores   = subset["macro_f1"].tolist()

    bars = ax.barh(model_names, f1_scores, color="#55A868", edgecolor="white")
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{w:.3f}", va="center", fontsize=8)

    ax.set_xlim(0, 1.1)
    ax.set_title(f"Language: {lang.upper()}", fontweight="bold")
    ax.set_xlabel("Macro F1")
    if ax == axes[0]:
        ax.set_ylabel("Generating Model")

fig.suptitle("AfroXLMR Detection Performance by Generating Model & Language",
             fontweight="bold", y=1.02)
plt.tight_layout()
save(fig, "15_cross_llm_per_language")
plt.show()

# Heatmap model × language
pivot = metrics_df.pivot(index="model", columns="language", values="macro_f1")

fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu",
            linewidths=0.5, vmin=0, vmax=1, ax=ax)
ax.set_title("Macro F1 - Generating Model × Language", fontweight="bold")
ax.set_xlabel("Language"); ax.set_ylabel("Generating Model")
plt.tight_layout()
save(fig, "16_heatmap_model_language")
plt.show()

# Radar chart
import matplotlib.patches as mpatches

radar_metrics = ["precision", "recall", "macro_f1", "mcc", "accuracy"]
radar_labels  = ["Precision", "Recall", "Macro F1", "MCC", "Accuracy"]
N      = len(radar_metrics)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
palette = plt.cm.tab10.colors

for idx, row in model_metrics_df.iterrows():
    values_r = [row[m] for m in radar_metrics] + [row[radar_metrics[0]]]
    ax.plot(angles, values_r, lw=1.5, label=row["model"], color=palette[idx % 10])
    ax.fill(angles, values_r, alpha=0.08, color=palette[idx % 10])

ax.set_xticks(angles[:-1]); ax.set_xticklabels(radar_labels, fontsize=10)
ax.set_ylim(0, 1)
ax.set_title("Detection Performance by Generating Model", fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
plt.tight_layout()
save(fig, "17_radar_chart")
plt.show()

# Grouped bar model × language
languages_list = sorted(metrics_df["language"].unique())
models_list    = sorted(metrics_df["model"].unique())
x              = np.arange(len(models_list))
bar_width      = 0.8 / len(languages_list)
palette_lang   = ["#4C72B0", "#55A868", "#C44E52", "#DD8452"]

fig, ax = plt.subplots(figsize=(max(10, len(models_list) * 2), 5))

for i, lang in enumerate(languages_list):
    subset = metrics_df[metrics_df["language"] == lang].set_index("model")
    vals   = [subset.loc[m, "macro_f1"] if m in subset.index else 0 for m in models_list]
    offset = (i - len(languages_list) / 2 + 0.5) * bar_width
    bars   = ax.bar(x + offset, vals, bar_width * 0.9,
                    label=lang.upper(), color=palette_lang[i % 4], edgecolor="white")
    for bar in bars:
        h = bar.get_height()
        if h > 0.05:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=7)

ax.set_xticks(x); ax.set_xticklabels(models_list, rotation=30, ha="right", fontsize=9)
ax.set_ylim(0, 1.1); ax.set_ylabel("Macro F1")
ax.set_title("Detection Macro F1 by Model & Language", fontweight="bold")
ax.legend(title="Language")
plt.tight_layout()
save(fig, "18_grouped_bar_model_language")
plt.show()

# Confidence boxplots by model
machine_rows   = results_df[results_df["label"] == 1].copy()
models_ordered = (machine_rows.groupby("model")["prob_machine"]
                  .median().sort_values(ascending=False).index.tolist())

fig, ax = plt.subplots(figsize=(max(8, len(models_ordered) * 1.4), 5))
data_to_plot = [machine_rows[machine_rows["model"] == m]["prob_machine"].values
                for m in models_ordered]

bp_plot = ax.boxplot(data_to_plot, patch_artist=True, notch=False,
                     medianprops=dict(color="black", linewidth=1.5))

for patch, color in zip(bp_plot["boxes"], plt.cm.tab10.colors):
    patch.set_facecolor(color); patch.set_alpha(0.7)

ax.axhline(0.5, color="grey", linestyle="--", linewidth=1, label="Decision boundary")
ax.set_xticks(range(1, len(models_ordered) + 1))
ax.set_xticklabels(models_ordered, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("P(Machine)")
ax.set_title("Model Confidence by Generating LLM (machine-generated texts only)",
             fontweight="bold")
ax.legend()
plt.tight_layout()
save(fig, "19_confidence_boxplot_by_model")
plt.show()

# Error rate heatmap
error_pivot = metrics_df.pivot(index="model", columns="language", values="accuracy")
error_pivot = 1 - error_pivot

fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(error_pivot, annot=True, fmt=".3f", cmap="OrRd",
            linewidths=0.5, vmin=0, vmax=1, ax=ax, annot_kws={"size": 10})
ax.set_title("Error Rate (1 − Accuracy) - Model × Language", fontweight="bold")
ax.set_xlabel("Language"); ax.set_ylabel("Generating Model")
plt.tight_layout()
save(fig, "20_error_rate_heatmap")
plt.show()

# Per-class F1 by model
from sklearn.metrics import f1_score as sk_f1

per_model_class_f1 = []
for model_name, group in results_df[results_df["label"] == 1].groupby("model"):
    human_rows = results_df[results_df["label"] == 0]
    combined   = pd.concat([group, human_rows]).reset_index(drop=True)
    y_true = combined["label"].values
    y_pred = combined["pred_label"].values
    if len(set(y_true)) < 2:
        continue
    per_model_class_f1.append({
        "model":      model_name,
        "f1_human":   round(sk_f1(y_true, y_pred, pos_label=0), 4),
        "f1_machine": round(sk_f1(y_true, y_pred, pos_label=1), 4),
    })

class_f1_df = pd.DataFrame(per_model_class_f1).sort_values("f1_machine", ascending=False)
x = np.arange(len(class_f1_df))
w = 0.35

fig, ax = plt.subplots(figsize=(max(9, len(class_f1_df) * 1.5), 5))
ax.bar(x - w/2, class_f1_df["f1_human"],   w, label="Human F1",   color="#4C72B0", edgecolor="white")
ax.bar(x + w/2, class_f1_df["f1_machine"], w, label="Machine F1", color="#C44E52", edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(class_f1_df["model"].tolist(), rotation=30, ha="right", fontsize=9)
ax.set_ylim(0, 1.12); ax.set_ylabel("F1 Score")
ax.set_title("Per-Class F1 (Human vs Machine) by Generating Model", fontweight="bold")
ax.legend()
plt.tight_layout()
save(fig, "21_perclass_f1_by_model")
plt.show()

# Save cross-LLM metrics
metrics_df.to_csv(os.path.join(BASE_DIR, "metrics_by_model_language.csv"), index=False)
model_metrics_df.to_csv(os.path.join(BASE_DIR, "metrics_by_model.csv"), index=False)

with open(os.path.join(BASE_DIR, "metrics_by_model_language.json"), "w") as f:
    json.dump(metrics_df.to_dict(orient="records"), f, indent=2)

print("All per-model metrics saved.")