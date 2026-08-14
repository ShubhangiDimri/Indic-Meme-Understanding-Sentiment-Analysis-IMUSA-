"""
train_text_muril.py
====================
Text-only baseline: google/muril-base-cased -> 4-class classification.

Architecture:
    Text -> MuRIL tokenizer -> MuRIL encoder -> [CLS] pooling
         -> dropout -> Linear(768, 4) -> softmax

Training details:
    - Splits   : results/split_train.csv  /  results/split_val.csv
    - Optimiser: AdamW  lr=2e-5, weight_decay=0.01
    - Epochs   : up to 5 (early stop on val macro-F1, patience=2)
    - Batch    : 32 (GPU/Colab) / 8 (CPU-only)
    - Mixed-prec: enabled when CUDA present, disabled on CPU
    - Metric   : macro-F1 (primary)

Outputs (all under OUT_DIR):
    best_model/         <- HuggingFace model dir (tokenizer + weights)
    val_predictions.csv <- Id, TrueLabel, PredLabel, + 4 prob columns
    metrics.json        <- accuracy, precision, recall, F1, confusion matrix
    training_log.csv    <- epoch-by-epoch loss & metrics

Usage:
    Local  : set COLAB = False  (paths relative to this script)
    Colab  : set COLAB = True   (paths under /content/drive/MyDrive/IMUSA/)
"""

import sys, os, csv, json, time, random
import numpy as np
from pathlib import Path
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── dependencies ──────────────────────────────────────────────
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ══════════════════════════════════════════════════════════════
# CONFIG  ← only block you ever need to edit
# ══════════════════════════════════════════════════════════════
COLAB = False   # ← set True when running on Google Colab

SEED           = 42
MODEL_NAME     = "google/muril-base-cased"
MAX_LEN        = 128
EPOCHS         = 5
LR             = 2e-5
WEIGHT_DECAY   = 0.01
PATIENCE       = 2          # early-stop patience (epochs)
DROPOUT        = 0.1

# ── Paths ─────────────────────────────────────────────────────
if COLAB:
    # Google Drive must be mounted at /content/drive before running
    DRIVE_ROOT = Path("/content/drive/MyDrive/IMUSA")
    TRAIN_CSV  = DRIVE_ROOT / "split_train.csv"
    VAL_CSV    = DRIVE_ROOT / "split_val.csv"
    OUT_DIR    = DRIVE_ROOT / "results" / "text_muril"
else:
    BASE       = Path(__file__).parent
    TRAIN_CSV  = BASE / "results" / "split_train.csv"
    VAL_CSV    = BASE / "results" / "split_val.csv"
    OUT_DIR    = BASE / "results" / "text_muril"

OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32 if torch.cuda.is_available() else 8
NUM_WORKERS = 2 if torch.cuda.is_available() else 0
USE_AMP    = torch.cuda.is_available()          # fp16 only on GPU

LABEL2ID = {"Motivational": 0, "Neutral": 1, "Offensive": 2, "Sarcasm": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_CLASSES = len(LABEL2ID)

print(f"\n{'='*60}")
print(f"  MuRIL Text Classifier")
print(f"  Device : {DEVICE}")
print(f"  Batch  : {BATCH_SIZE}  AMP: {USE_AMP}")
print(f"  Output : {OUT_DIR}")
print(f"{'='*60}\n")


# ══════════════════════════════════════════════════════════════
# REPRODUCIBILITY
# ══════════════════════════════════════════════════════════════
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(SEED)


# ══════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════
def load_split(path: Path):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    return rows


class MemeTextDataset(Dataset):
    def __init__(self, rows, tokenizer, max_len: int):
        self.rows      = rows
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row  = self.rows[idx]
        text = row.get("Text", "") or ""
        label_str = row.get("Category", "")

        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "image_id":       row["Id"],
        }
        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"].squeeze(0)

        if label_str:
            item["labels"] = torch.tensor(LABEL2ID[label_str], dtype=torch.long)
        return item


# ══════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════
class MuRILClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int, dropout: float):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden       = self.encoder.config.hidden_size   # 768
        self.drop    = nn.Dropout(dropout)
        self.head    = nn.Linear(hidden, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # [CLS] pooling
        cls_vec = out.last_hidden_state[:, 0, :]
        cls_vec = self.drop(cls_vec)
        logits  = self.head(cls_vec)
        return logits


# ══════════════════════════════════════════════════════════════
# METRICS HELPER
# ══════════════════════════════════════════════════════════════
def compute_metrics(preds, labels):
    return {
        "accuracy"  : round(accuracy_score(labels, preds), 6),
        "macro_f1"  : round(f1_score(labels, preds, average="macro",  zero_division=0), 6),
        "macro_prec": round(precision_score(labels, preds, average="macro", zero_division=0), 6),
        "macro_rec" : round(recall_score(labels, preds, average="macro",  zero_division=0), 6),
        "per_class_f1": {
            ID2LABEL[i]: round(f1_score(labels, preds, labels=[i], average="micro", zero_division=0), 6)
            for i in range(NUM_CLASSES)
        },
        "confusion_matrix": confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES))).tolist(),
        "classification_report": classification_report(
            labels, preds,
            target_names=[ID2LABEL[i] for i in range(NUM_CLASSES)],
            zero_division=0,
        ),
    }


# ══════════════════════════════════════════════════════════════
# TRAIN / EVAL LOOPS
# ══════════════════════════════════════════════════════════════
def run_epoch(model, loader, optimiser, scheduler, scaler, criterion, train: bool):
    model.train(train)
    total_loss, all_preds, all_labels = 0.0, [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        with torch.set_grad_enabled(train):
            if USE_AMP:
                with torch.cuda.amp.autocast():
                    logits = model(input_ids, attention_mask, token_type_ids)
                    loss   = criterion(logits, labels)
            else:
                logits = model(input_ids, attention_mask, token_type_ids)
                loss   = criterion(logits, labels)

        if train:
            optimiser.zero_grad()
            if USE_AMP:
                scaler.scale(loss).backward()
                scaler.unscale_(optimiser)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimiser)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
            scheduler.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader)
    metrics  = compute_metrics(all_preds, all_labels)
    return avg_loss, metrics, all_preds, all_labels


@torch.no_grad()
def run_eval_with_probs(model, loader, criterion):
    """Evaluation pass that also returns softmax probabilities."""
    model.eval()
    total_loss, all_preds, all_labels, all_probs, all_ids = 0.0, [], [], [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        logits = model(input_ids, attention_mask, token_type_ids)
        loss   = criterion(logits, labels)
        total_loss += loss.item()

        probs  = torch.softmax(logits, dim=-1).cpu().numpy()
        preds  = probs.argmax(axis=-1)

        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        all_ids.extend(batch["image_id"])

    avg_loss = total_loss / len(loader)
    metrics  = compute_metrics(all_preds, all_labels)
    return avg_loss, metrics, all_preds, all_labels, all_probs, all_ids


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    # ── Load tokenizer ────────────────────────────────────────
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ── Datasets & loaders ────────────────────────────────────
    train_rows = load_split(TRAIN_CSV)
    val_rows   = load_split(VAL_CSV)
    print(f"Train rows: {len(train_rows)}  |  Val rows: {len(val_rows)}")

    train_ds = MemeTextDataset(train_rows, tokenizer, MAX_LEN)
    val_ds   = MemeTextDataset(val_rows,   tokenizer, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=USE_AMP)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=USE_AMP)

    # ── Model ─────────────────────────────────────────────────
    print(f"Loading model: {MODEL_NAME}")
    model = MuRILClassifier(MODEL_NAME, NUM_CLASSES, DROPOUT).to(DEVICE)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {param_count:,}")

    # ── Loss (handle class imbalance: Offensive only 52 samples) ─
    # Compute inverse-frequency weights from the training split
    label_counts = [0] * NUM_CLASSES
    for r in train_rows:
        label_counts[LABEL2ID[r["Category"]]] += 1
    total = sum(label_counts)
    weights = torch.tensor(
        [total / (NUM_CLASSES * c) for c in label_counts],
        dtype=torch.float32
    ).to(DEVICE)
    print(f"Class weights: { {ID2LABEL[i]: round(weights[i].item(),3) for i in range(NUM_CLASSES)} }")
    criterion = nn.CrossEntropyLoss(weight=weights)

    # ── Optimiser & scheduler ─────────────────────────────────
    # Apply weight decay to all params except bias and LayerNorm
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    param_groups = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters() if     any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimiser = AdamW(param_groups, lr=LR)

    total_steps   = len(train_loader) * EPOCHS
    warmup_steps  = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimiser, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    scaler = torch.cuda.amp.GradScaler() if USE_AMP else None

    # ── Training loop ─────────────────────────────────────────
    best_f1      = -1.0
    patience_ctr = 0
    log_rows     = []
    best_ckpt    = OUT_DIR / "best_model"

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        print(f"\n--- Epoch {epoch}/{EPOCHS} ---")

        train_loss, train_m, _, _ = run_epoch(
            model, train_loader, optimiser, scheduler, scaler, criterion, train=True
        )
        val_loss, val_m, val_preds, val_labels, val_probs, val_ids = run_eval_with_probs(
            model, val_loader, criterion
        )

        elapsed = time.time() - t0
        print(f"  train_loss={train_loss:.4f}  train_macro_f1={train_m['macro_f1']:.4f}")
        print(f"  val_loss  ={val_loss:.4f}  val_macro_f1  ={val_m['macro_f1']:.4f}  ({elapsed:.0f}s)")
        print(f"  val_acc   ={val_m['accuracy']:.4f}")
        print(f"  per-class F1: {val_m['per_class_f1']}")

        log_rows.append({
            "epoch"          : epoch,
            "train_loss"     : round(train_loss, 6),
            "train_macro_f1" : train_m["macro_f1"],
            "train_acc"      : train_m["accuracy"],
            "val_loss"       : round(val_loss, 6),
            "val_macro_f1"   : val_m["macro_f1"],
            "val_acc"        : val_m["accuracy"],
            "val_macro_prec" : val_m["macro_prec"],
            "val_macro_rec"  : val_m["macro_rec"],
            "elapsed_s"      : round(elapsed, 1),
        })

        # ── checkpoint if improved ─────────────────────────────
        if val_m["macro_f1"] > best_f1:
            best_f1       = val_m["macro_f1"]
            patience_ctr  = 0
            best_val_preds  = val_preds
            best_val_labels = val_labels
            best_val_probs  = val_probs
            best_val_ids    = val_ids
            best_metrics    = val_m

            # Save full model + tokenizer
            best_ckpt.mkdir(parents=True, exist_ok=True)
            model.encoder.save_pretrained(best_ckpt)
            tokenizer.save_pretrained(best_ckpt)
            # Save classifier head separately
            torch.save(model.head.state_dict(), best_ckpt / "classifier_head.pt")
            print(f"  *** New best val macro-F1 = {best_f1:.4f} — checkpoint saved ***")
        else:
            patience_ctr += 1
            print(f"  No improvement. Patience {patience_ctr}/{PATIENCE}")
            if patience_ctr >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    # ── Save training log ─────────────────────────────────────
    log_path = OUT_DIR / "training_log.csv"
    log_cols  = list(log_rows[0].keys())
    with open(log_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=log_cols)
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"\nTraining log saved: {log_path}")

    # ── Save val predictions + probabilities ──────────────────
    pred_path = OUT_DIR / "val_predictions.csv"
    prob_cols = [f"prob_{ID2LABEL[i]}" for i in range(NUM_CLASSES)]
    with open(pred_path, "w", newline="", encoding="utf-8-sig") as fh:
        cols = ["Id", "TrueLabel", "PredLabel", "Correct"] + prob_cols
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for i, img_id in enumerate(best_val_ids):
            true_lbl = ID2LABEL[best_val_labels[i]]
            pred_lbl = ID2LABEL[best_val_preds[i]]
            row_out  = {
                "Id"        : img_id,
                "TrueLabel" : true_lbl,
                "PredLabel" : pred_lbl,
                "Correct"   : int(true_lbl == pred_lbl),
            }
            for j, pcol in enumerate(prob_cols):
                row_out[pcol] = round(float(best_val_probs[i][j]), 6)
            writer.writerow(row_out)
    print(f"Predictions saved : {pred_path}")

    # ── Save metrics JSON ─────────────────────────────────────
    metrics_out = {
        "model"              : MODEL_NAME,
        "seed"               : SEED,
        "best_epoch"         : next(r["epoch"] for r in log_rows if r["val_macro_f1"] == best_f1),
        "best_val_macro_f1"  : best_metrics["macro_f1"],
        "accuracy"           : best_metrics["accuracy"],
        "macro_precision"    : best_metrics["macro_prec"],
        "macro_recall"       : best_metrics["macro_rec"],
        "per_class_f1"       : best_metrics["per_class_f1"],
        "confusion_matrix"   : best_metrics["confusion_matrix"],
        "label_order"        : [ID2LABEL[i] for i in range(NUM_CLASSES)],
        "classification_report": best_metrics["classification_report"],
        "training_log"       : log_rows,
        "generated_at"       : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    metrics_path = OUT_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics_out, fh, ensure_ascii=False, indent=2)
    print(f"Metrics saved     : {metrics_path}")

    # ── Final summary ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  BEST RESULTS (epoch {metrics_out['best_epoch']})")
    print(f"{'='*60}")
    print(f"  Accuracy      : {metrics_out['accuracy']:.4f}")
    print(f"  Macro-F1      : {metrics_out['best_val_macro_f1']:.4f}")
    print(f"  Macro-Prec    : {metrics_out['macro_precision']:.4f}")
    print(f"  Macro-Recall  : {metrics_out['macro_recall']:.4f}")
    print(f"\n  Per-class F1:")
    for cls, f1 in metrics_out["per_class_f1"].items():
        print(f"    {cls:<15} {f1:.4f}")
    print(f"\n  Confusion matrix (rows=true, cols=pred):")
    labels_order = metrics_out["label_order"]
    print(f"    {'':>15} " + "  ".join(f"{l[:6]:>6}" for l in labels_order))
    for i, row in enumerate(metrics_out["confusion_matrix"]):
        print(f"    {labels_order[i]:<15} " + "  ".join(f"{v:>6}" for v in row))
    print(f"\n  Best checkpoint : {best_ckpt}")


if __name__ == "__main__":
    main()
