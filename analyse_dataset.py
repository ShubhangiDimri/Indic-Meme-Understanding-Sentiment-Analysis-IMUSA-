"""
IMUSA Dataset Analysis Script
==============================
Analyses the Punjabi meme training and test CSVs and saves results to results/.

Checks:
  - Class distribution & percentages
  - Missing Text values
  - Duplicate IDs
  - Duplicate Text values
  - Missing image files
"""

import sys
import csv
import os
import json
import collections
from pathlib import Path
from datetime import datetime

# Force UTF-8 output on Windows terminals that default to cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ──────────────────────────────────────────────
# PATHS  (adjust here if files move)
# ──────────────────────────────────────────────
BASE = Path(__file__).parent

TRAIN_CSV       = BASE / "Training_Dataset" / "train_punjabi_dataset.csv"
TRAIN_IMG_DIR   = BASE / "Training_Dataset" / "Training_Dataset" / "Training_images"

TEST_CSV        = BASE / "Testing_Dataset" / "Testing_Dataset" / "Test.csv"
TEST_IMG_DIR    = BASE / "Testing_Dataset" / "Testing_Dataset" / "Testing_images"

RESULTS_DIR     = BASE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def load_csv(path: Path):
    """Return (column_names, rows) from a UTF-8-BOM csv."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        cols = reader.fieldnames or []
        rows = list(reader)
    return cols, rows


def image_exists(img_stem: str, img_set: set) -> bool:
    """True if stem + any common image extension is present (case-insensitive)."""
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        if (img_stem.lower() + ext) in img_set:
            return True
    return False


def build_image_set(img_dir: Path) -> set:
    """Lower-cased set of all filenames in the image directory."""
    if not img_dir.is_dir():
        return set()
    return {f.name.lower() for f in img_dir.iterdir() if f.is_file()}


def sep(title: str, width: int = 60) -> str:
    return f"\n{'=' * width}\n  {title}\n{'=' * width}"


# ──────────────────────────────────────────────
# ANALYSIS FUNCTION
# ──────────────────────────────────────────────

def analyse(split: str, csv_path: Path, img_dir: Path) -> dict:
    print(sep(f"{split.upper()} SPLIT"))

    cols, rows = load_csv(csv_path)
    n = len(rows)
    print(f"\nCSV path  : {csv_path}")
    print(f"Img dir   : {img_dir}")
    print(f"Columns   : {cols}")
    print(f"Total rows: {n}")

    # ── 1. Class distribution ─────────────────
    cats   = [r.get("Category", "").strip() for r in rows]
    filled = [c for c in cats if c]
    empty  = n - len(filled)

    dist   = collections.Counter(filled)
    total  = len(filled) if filled else 1          # avoid /0
    pct    = {k: round(v / total * 100, 2) for k, v in dist.items()}

    print("\n[Class Distribution]")
    for cls in sorted(dist):
        print(f"  {cls:<15} {dist[cls]:>5}  ({pct[cls]:>6.2f} %)")
    if empty:
        print(f"  {'<blank>':<15} {empty:>5}  ({round(empty/n*100,2):>6.2f} %) <- unlabelled (test)")

    # ── 2. Missing Text ───────────────────────
    missing_text_ids = [r["Id"] for r in rows if not r.get("Text", "").strip()]
    print(f"\n[Missing Text]  {len(missing_text_ids)} row(s)")
    for mid in missing_text_ids[:20]:
        print(f"  {mid}")
    if len(missing_text_ids) > 20:
        print(f"  ... and {len(missing_text_ids) - 20} more")

    # ── 3. Duplicate IDs ─────────────────────
    ids = [r["Id"] for r in rows]
    id_counts   = collections.Counter(ids)
    dup_id_map  = {k: v for k, v in id_counts.items() if v > 1}
    print(f"\n[Duplicate IDs]  {len(dup_id_map)} unique ID(s) duplicated")
    for did, cnt in list(dup_id_map.items())[:20]:
        print(f"  {did}  (appears {cnt}x)")

    # ── 4. Duplicate Text ────────────────────
    texts = [r.get("Text", "").strip() for r in rows]
    non_blank_texts = [t for t in texts if t]
    text_counts     = collections.Counter(non_blank_texts)
    dup_text_map    = {k: v for k, v in text_counts.items() if v > 1}
    print(f"\n[Duplicate Text]  {len(dup_text_map)} unique text(s) appear more than once")
    for txt, cnt in list(dup_text_map.items())[:10]:
        preview = (txt[:80] + "...") if len(txt) > 80 else txt
        print(f"  [{cnt}x] {preview}")

    # Which IDs carry duplicate text?
    dup_text_id_pairs = [
        (r["Id"], r.get("Text", "").strip())
        for r in rows
        if r.get("Text", "").strip() in dup_text_map
    ]

    # ── 5. Missing image files ────────────────
    img_set  = build_image_set(img_dir)
    n_images = len(img_set)
    print(f"\n[Image Folder]   {n_images} file(s) found in {img_dir.name}/")

    ids_no_img = [
        r["Id"]
        for r in rows
        if not image_exists(Path(r["Id"]).stem, img_set)
    ]
    print(f"[Missing Images] {len(ids_no_img)} ID(s) have no matching image")
    for mid in ids_no_img[:20]:
        print(f"  {mid}")

    csv_stems    = {os.path.splitext(r["Id"])[0].lower() for r in rows}
    imgs_no_csv  = [f for f in img_set if os.path.splitext(f)[0].lower() not in csv_stems]
    print(f"[Orphan Images]  {len(imgs_no_csv)} image(s) not referenced in CSV")
    for img in sorted(imgs_no_csv)[:20]:
        print(f"  {img}")

    # ── Collect result dict ───────────────────
    return {
        "split"                    : split,
        "csv_path"                 : str(csv_path),
        "image_dir"                : str(img_dir),
        "columns"                  : cols,
        "total_rows"               : n,
        "class_distribution"       : dict(dist),
        "class_percentages"        : pct,
        "unlabelled_rows"          : empty,
        "missing_text_count"       : len(missing_text_ids),
        "missing_text_ids"         : missing_text_ids,
        "duplicate_id_count"       : len(dup_id_map),
        "duplicate_ids"            : dup_id_map,
        "duplicate_text_count"     : len(dup_text_map),
        "duplicate_text_ids"       : [pair[0] for pair in dup_text_id_pairs],
        "duplicate_texts"          : {k: v for k, v in list(dup_text_map.items())[:100]},
        "total_images_on_disk"     : n_images,
        "ids_missing_image_count"  : len(ids_no_img),
        "ids_missing_image"        : ids_no_img,
        "orphan_images_count"      : len(imgs_no_csv),
        "orphan_images"            : sorted(imgs_no_csv)[:100],
    }


# ──────────────────────────────────────────────
# SAVE HELPERS
# ──────────────────────────────────────────────

def save_json(data: dict, path: Path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    print(f"  -> Saved: {path}")


def save_txt(lines, path: Path):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"  -> Saved: {path}")


def results_to_text(r: dict):
    lines = []
    lines.append("=" * 60)
    lines.append(f"  {r['split'].upper()} SPLIT ANALYSIS")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 60)
    lines.append(f"\nCSV        : {r['csv_path']}")
    lines.append(f"Image dir  : {r['image_dir']}")
    lines.append(f"Columns    : {r['columns']}")
    lines.append(f"Total rows : {r['total_rows']}")

    lines.append("\n-- Class Distribution ------------------------------------------")
    for cls in sorted(r["class_distribution"]):
        cnt = r["class_distribution"][cls]
        p   = r["class_percentages"].get(cls, 0)
        lines.append(f"  {cls:<15} {cnt:>5}  ({p:>6.2f} %)")
    if r["unlabelled_rows"]:
        pct = round(r["unlabelled_rows"] / r["total_rows"] * 100, 2)
        lines.append(f"  {'<unlabelled>':<15} {r['unlabelled_rows']:>5}  ({pct:>6.2f} %)")

    lines.append("\n-- Missing Text ------------------------------------------------")
    lines.append(f"  Count : {r['missing_text_count']}")
    for mid in r["missing_text_ids"][:20]:
        lines.append(f"  {mid}")

    lines.append("\n-- Duplicate IDs -----------------------------------------------")
    lines.append(f"  Count : {r['duplicate_id_count']}")
    for did, cnt in list(r["duplicate_ids"].items())[:20]:
        lines.append(f"  {did}  ({cnt}x)")

    lines.append("\n-- Duplicate Text ----------------------------------------------")
    lines.append(f"  Unique texts appearing >1x : {r['duplicate_text_count']}")
    lines.append(f"  Total rows affected        : {len(r['duplicate_text_ids'])}")
    for txt, cnt in list(r["duplicate_texts"].items())[:10]:
        preview = (txt[:75] + "...") if len(txt) > 75 else txt
        lines.append(f"  [{cnt}x] {preview}")

    lines.append("\n-- Image Coverage ----------------------------------------------")
    lines.append(f"  Images on disk             : {r['total_images_on_disk']}")
    lines.append(f"  IDs with no image file     : {r['ids_missing_image_count']}")
    for mid in r["ids_missing_image"][:20]:
        lines.append(f"    {mid}")
    lines.append(f"  Orphan images (no CSV row) : {r['orphan_images_count']}")
    for img in r["orphan_images"][:20]:
        lines.append(f"    {img}")

    return lines


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    print("\nIMUSA Dataset Analyser")
    print(f"Output folder: {RESULTS_DIR}\n")

    train_results = analyse("training", TRAIN_CSV, TRAIN_IMG_DIR)
    test_results  = analyse("test",     TEST_CSV,  TEST_IMG_DIR)

    # Save individual JSON results
    print("\n[Saving results]")
    save_json(train_results, RESULTS_DIR / "train_analysis.json")
    save_json(test_results,  RESULTS_DIR / "test_analysis.json")

    # Save human-readable summary
    summary_lines = []
    summary_lines += results_to_text(train_results)
    summary_lines += ["", ""]
    summary_lines += results_to_text(test_results)
    summary_lines += [
        "",
        "=" * 60,
        "  COMBINED QUICK SUMMARY",
        "=" * 60,
        f"  Training rows          : {train_results['total_rows']}",
        f"  Test rows              : {test_results['total_rows']}",
        f"  Train class labels     : {sorted(train_results['class_distribution'])}",
        f"  Train missing text     : {train_results['missing_text_count']}",
        f"  Train duplicate IDs    : {train_results['duplicate_id_count']}",
        f"  Train duplicate texts  : {train_results['duplicate_text_count']} unique "
        f"({len(train_results['duplicate_text_ids'])} rows affected)",
        f"  Train missing images   : {train_results['ids_missing_image_count']}",
        f"  Test missing Category  : {test_results['unlabelled_rows']} (unlabelled -- expected)",
        f"  Test missing text      : {test_results['missing_text_count']}",
        f"  Test duplicate IDs     : {test_results['duplicate_id_count']}",
        f"  Test duplicate texts   : {test_results['duplicate_text_count']} unique "
        f"({len(test_results['duplicate_text_ids'])} rows affected)",
        f"  Test missing images    : {test_results['ids_missing_image_count']}",
    ]
    save_txt(summary_lines, RESULTS_DIR / "analysis_summary.txt")

    # Save lists of affected IDs if any duplicates found
    if train_results["duplicate_text_ids"]:
        save_txt(
            train_results["duplicate_text_ids"],
            RESULTS_DIR / "train_duplicate_text_ids.txt"
        )
    if test_results["duplicate_text_ids"]:
        save_txt(
            test_results["duplicate_text_ids"],
            RESULTS_DIR / "test_duplicate_text_ids.txt"
        )

    print("\nAnalysis complete. All results saved to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
