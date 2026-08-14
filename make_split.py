"""
make_split.py
=============
Creates a reproducible stratified 80/20 train/validation split
from the IMUSA training CSV, using random seed 42.

Outputs (saved to results/):
  split_train.csv      – 80% of labelled rows, with Id + Category + Text
  split_val.csv        – 20% of labelled rows, with Id + Category + Text
  split_summary.txt    – human-readable split report
"""

import sys
import csv
import json
import random
import collections
from pathlib import Path
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
SEED        = 42
VAL_RATIO   = 0.20

BASE        = Path(__file__).parent
TRAIN_CSV   = BASE / "Training_Dataset" / "train_punjabi_dataset.csv"
RESULTS_DIR = BASE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUT_TRAIN   = RESULTS_DIR / "split_train.csv"
OUT_VAL     = RESULTS_DIR / "split_val.csv"
OUT_SUMMARY = RESULTS_DIR / "split_summary.txt"
OUT_JSON    = RESULTS_DIR / "split_info.json"

# ──────────────────────────────────────────────
# LOAD
# ──────────────────────────────────────────────
print(f"Loading: {TRAIN_CSV}")
with open(TRAIN_CSV, newline="", encoding="utf-8-sig") as fh:
    reader = csv.DictReader(fh)
    cols   = reader.fieldnames or []
    rows   = list(reader)

print(f"Total rows loaded: {len(rows)}")
print(f"Columns          : {cols}")

# ──────────────────────────────────────────────
# GROUP BY CATEGORY (stratify key)
# ──────────────────────────────────────────────
buckets: dict[str, list[dict]] = collections.defaultdict(list)
for row in rows:
    cat = row.get("Category", "").strip()
    if not cat:
        print(f"  WARNING: row '{row.get('Id')}' has no Category — skipped from split")
        continue
    buckets[cat].append(row)

print(f"\nClasses found: {sorted(buckets)}")

# ──────────────────────────────────────────────
# STRATIFIED SPLIT
# ──────────────────────────────────────────────
rng = random.Random(SEED)

train_rows: list[dict] = []
val_rows:   list[dict] = []

per_class_stats: dict[str, dict] = {}

for cat in sorted(buckets):
    class_rows = buckets[cat][:]
    rng.shuffle(class_rows)                         # shuffle within class

    n_total = len(class_rows)
    n_val   = max(1, round(n_total * VAL_RATIO))    # at least 1 val sample
    n_train = n_total - n_val

    class_val   = class_rows[:n_val]
    class_train = class_rows[n_val:]

    train_rows.extend(class_train)
    val_rows.extend(class_val)

    per_class_stats[cat] = {
        "total"  : n_total,
        "train"  : n_train,
        "val"    : n_val,
        "val_pct": round(n_val / n_total * 100, 2),
    }

    print(f"  {cat:<15} total={n_total:>4}  train={n_train:>4}  val={n_val:>3}  "
          f"({per_class_stats[cat]['val_pct']:.1f}% val)")

# Shuffle the combined train/val lists (don't leave them class-sorted)
rng.shuffle(train_rows)
rng.shuffle(val_rows)

print(f"\nFinal split  -> train: {len(train_rows)}  val: {len(val_rows)}")
print(f"Val ratio    -> {len(val_rows) / (len(train_rows) + len(val_rows)) * 100:.2f}%")

# ──────────────────────────────────────────────
# VERIFY NO OVERLAP
# ──────────────────────────────────────────────
train_ids = {r["Id"] for r in train_rows}
val_ids   = {r["Id"] for r in val_rows}
overlap   = train_ids & val_ids
assert len(overlap) == 0, f"OVERLAP DETECTED: {overlap}"
print(f"ID overlap check: PASSED (0 overlapping IDs)")

# ──────────────────────────────────────────────
# SAVE SPLIT CSVs
# ──────────────────────────────────────────────
out_cols = ["Id", "Category", "Text"]

def write_split_csv(path: Path, data: list[dict]):
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=out_cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data)
    print(f"  Saved: {path}  ({len(data)} rows)")

print("\n[Saving CSVs]")
write_split_csv(OUT_TRAIN, train_rows)
write_split_csv(OUT_VAL,   val_rows)

# ──────────────────────────────────────────────
# SAVE JSON METADATA
# ──────────────────────────────────────────────
split_info = {
    "seed"           : SEED,
    "val_ratio"      : VAL_RATIO,
    "generated_at"   : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "source_csv"     : str(TRAIN_CSV),
    "total_labelled" : len(train_rows) + len(val_rows),
    "train_count"    : len(train_rows),
    "val_count"      : len(val_rows),
    "actual_val_pct" : round(len(val_rows) / (len(train_rows) + len(val_rows)) * 100, 4),
    "id_overlap"     : 0,
    "per_class"      : per_class_stats,
    "train_ids"      : [r["Id"] for r in train_rows],
    "val_ids"        : [r["Id"] for r in val_rows],
}
with open(OUT_JSON, "w", encoding="utf-8") as fh:
    json.dump(split_info, fh, ensure_ascii=False, indent=2)
print(f"  Saved: {OUT_JSON}")

# ──────────────────────────────────────────────
# SAVE HUMAN-READABLE SUMMARY
# ──────────────────────────────────────────────
lines = [
    "=" * 60,
    "  TRAIN / VALIDATION SPLIT SUMMARY",
    f"  Generated: {split_info['generated_at']}",
    "=" * 60,
    f"\nSeed         : {SEED}",
    f"Val ratio    : {VAL_RATIO}",
    f"Source CSV   : {TRAIN_CSV}",
    f"Train output : {OUT_TRAIN}",
    f"Val output   : {OUT_VAL}",
    "",
    f"Total labelled rows : {split_info['total_labelled']}",
    f"Train rows          : {split_info['train_count']}",
    f"Val rows            : {split_info['val_count']}",
    f"Actual val ratio    : {split_info['actual_val_pct']:.2f}%",
    f"ID overlap          : {split_info['id_overlap']}  (NONE - clean split)",
    "",
    "-- Per-class breakdown ------------------------------------------",
    f"  {'Class':<15}  {'Total':>6}  {'Train':>6}  {'Val':>5}  {'Val%':>6}",
    "-" * 55,
]
for cat, s in per_class_stats.items():
    lines.append(
        f"  {cat:<15}  {s['total']:>6}  {s['train']:>6}  {s['val']:>5}  {s['val_pct']:>5.1f}%"
    )
lines += [
    "-" * 55,
    f"  {'TOTAL':<15}  {split_info['total_labelled']:>6}  {split_info['train_count']:>6}  "
    f"{split_info['val_count']:>5}  {split_info['actual_val_pct']:>5.1f}%",
    "",
    "Class distribution in TRAIN set:",
]
train_cat_counts = collections.Counter(r["Category"] for r in train_rows)
for cat in sorted(train_cat_counts):
    pct = train_cat_counts[cat] / len(train_rows) * 100
    lines.append(f"  {cat:<15}  {train_cat_counts[cat]:>5}  ({pct:.2f}%)")

lines += ["", "Class distribution in VAL set:"]
val_cat_counts = collections.Counter(r["Category"] for r in val_rows)
for cat in sorted(val_cat_counts):
    pct = val_cat_counts[cat] / len(val_rows) * 100
    lines.append(f"  {cat:<15}  {val_cat_counts[cat]:>5}  ({pct:.2f}%)")

with open(OUT_SUMMARY, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))
print(f"  Saved: {OUT_SUMMARY}")

print("\nDone. Split is fully reproducible with seed=42.")
