"""
Accuracy evaluation for the Tesseract + OCR-B pipeline against hand-made ground truth.

Reads GroundTruth/ground_truth.json (172 verified TD3 MRZ records), runs each
image through the Tesseract pipeline N times, and reports:

  - CER  (Character Error Rate, Levenshtein over the two MRZ lines)
  - Field accuracy (per-field exact match after ICAO parse)

Each image is run multiple times (default 3) and metrics are averaged to smooth
GPU/OCR non-determinism. Results are written to GroundTruth/accuracy_report.csv
and a summary is printed.

Usage:
    python GroundTruth/evaluate.py [--runs 3] [--limit N] [--quiet]
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

# project root on path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np

from Scripts.parsing.mrz_parse import parse_mrz
from Scripts.ocr.pipeline import _process_frame

_GT_PATH = _ROOT / "GroundTruth" / "ground_truth.json"
_IMG_DIR = _ROOT / "Images" / "MRZ_Data" / "Processed_data" / "images" / "test"
_REPORT_PATH = _ROOT / "GroundTruth" / "accuracy_report.csv"

# Fields compared for field accuracy (derived from the ICAO parse of each MRZ).
# issuing_country is intentionally excluded: it is read from line 1's country
# code (positions 2-4), which corrupts whenever the name line is noisy. The
# nationality field (read from line 2) carries the same information far more
# reliably, so issuing_country only added noise to the accuracy measurement.
_FIELDS = [
    "document_number",
    "nationality",
    "surname",
    "given_names",
    "birth_date_raw",
    "expiry_date_raw",
    "sex",
]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def levenshtein(a: str, b: str) -> int:
    """Classic edit distance (insertions/deletions/substitutions)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def cer(gt_lines: list[str], pred_lines: list[str]) -> float:
    """Character error rate over the concatenated MRZ lines.

    Aligns line-by-line over the full fixed-width lines; pads the shorter side so
    missing lines count as errors. The full 44-char width is kept on both sides
    so the denominator is consistent — a line that drops real characters
    (including a trailing check digit) is correctly penalised.
    """
    n_lines = max(len(gt_lines), len(pred_lines))
    gt = list(gt_lines) + [""] * (n_lines - len(gt_lines))
    pred = list(pred_lines) + [""] * (n_lines - len(pred_lines))

    total_dist = 0
    total_chars = 0
    for g, p in zip(gt, pred):
        total_dist += levenshtein(g, p)
        total_chars += max(len(g), 1)
    return total_dist / total_chars if total_chars else 0.0


def _fields_from_lines(lines: list[str]) -> dict[str, str]:
    """Parse MRZ lines into the comparison field set. Returns empty strings on failure."""
    result = parse_mrz(lines)
    if result is None:
        return {f: "" for f in _FIELDS}
    return {
        "document_number": result.document_number,
        "nationality": result.nationality,
        "surname": result.surname,
        "given_names": result.given_names,
        "birth_date_raw": result.birth_date_raw,
        "expiry_date_raw": result.expiry_date_raw,
        "sex": result.sex,
    }


def field_accuracy(gt_fields: dict[str, str], pred_fields: dict[str, str]) -> tuple[float, list[str]]:
    """Fraction of fields that match exactly. Returns (accuracy, list_of_failed_fields)."""
    failed = [f for f in _FIELDS if gt_fields.get(f, "") != pred_fields.get(f, "")]
    acc = 1.0 - len(failed) / len(_FIELDS)
    return acc, failed


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def _load_image(path: Path) -> np.ndarray | None:
    raw = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def _pred_lines(image: np.ndarray) -> list[str]:
    """Run the Tesseract pipeline and return the raw MRZ lines it produced."""
    output, _ = _process_frame(image)
    if output is None:
        return []
    return output.get("raw_mrz", []) or []


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(runs: int = 3, limit: int | None = None, quiet: bool = False) -> list[dict]:
    with open(_GT_PATH, encoding="utf-8") as f:
        gt = json.load(f)
    stems = [k for k in gt if not k.startswith("_")]
    if limit:
        stems = stems[:limit]

    rows: list[dict] = []
    for idx, stem in enumerate(stems, 1):
        img_path = _IMG_DIR / f"{stem}.jpg"
        country = stem.split("-")[1] if "-" in stem else "???"

        if not img_path.exists():
            rows.append({"stem": stem, "country": country, "status": "image_missing",
                         "cer": "", "field_acc": "", "failed_fields": ""})
            if not quiet:
                print(f"[{idx}/{len(stems)}] {stem}: IMAGE MISSING")
            continue

        gt_lines = gt[stem]["lines"]
        gt_fields = _fields_from_lines(gt_lines)

        image = _load_image(img_path)
        if image is None:
            rows.append({"stem": stem, "country": country, "status": "load_failed",
                         "cer": "", "field_acc": "", "failed_fields": ""})
            continue

        cers: list[float] = []
        accs: list[float] = []
        last_failed: list[str] = []
        last_pred: list[str] = []
        for _ in range(runs):
            pred_lines = _pred_lines(image)
            last_pred = pred_lines
            cers.append(cer(gt_lines, pred_lines))
            pred_fields = _fields_from_lines(pred_lines)
            acc, failed = field_accuracy(gt_fields, pred_fields)
            accs.append(acc)
            last_failed = failed

        cer_avg = sum(cers) / len(cers)
        acc_avg = sum(accs) / len(accs)
        rows.append({
            "stem": stem,
            "country": country,
            "status": "ok" if last_pred else "no_mrz",
            "cer": round(cer_avg, 4),
            "field_acc": round(acc_avg, 4),
            "failed_fields": ";".join(last_failed),
        })
        if not quiet:
            print(f"[{idx}/{len(stems)}] {country} {stem}: "
                  f"CER={cer_avg:.3f} field_acc={acc_avg:.3f} "
                  f"failed=[{','.join(last_failed)}]")

    return rows


def write_report(rows: list[dict]) -> None:
    with open(_REPORT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["stem", "country", "status", "cer", "field_acc", "failed_fields"])
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict]) -> None:
    scored = [r for r in rows if r["status"] == "ok" and r["cer"] != ""]
    if not scored:
        print("\nNo scored rows.")
        return

    n = len(scored)
    mean_cer = sum(r["cer"] for r in scored) / n
    mean_acc = sum(r["field_acc"] for r in scored) / n
    exact = sum(1 for r in scored if r["cer"] == 0.0)
    perfect_fields = sum(1 for r in scored if r["field_acc"] == 1.0)
    no_mrz = sum(1 for r in rows if r["status"] == "no_mrz")
    missing = sum(1 for r in rows if r["status"] in ("image_missing", "load_failed"))

    print("\n" + "=" * 56)
    print("ACCURACY SUMMARY")
    print("=" * 56)
    print(f"Scored images:        {n}")
    print(f"Mean CER:             {mean_cer:.4f}  ({(1-mean_cer)*100:.2f}% char accuracy)")
    print(f"Mean field accuracy:  {mean_acc:.4f}  ({mean_acc*100:.2f}%)")
    print(f"Exact MRZ match:      {exact}/{n}  ({exact/n*100:.1f}%)")
    print(f"All fields correct:   {perfect_fields}/{n}  ({perfect_fields/n*100:.1f}%)")
    print(f"No MRZ detected:      {no_mrz}")
    print(f"Missing/load-failed:  {missing}")

    # per-field failure counts
    field_fail: dict[str, int] = {}
    for r in scored:
        for fld in (r["failed_fields"].split(";") if r["failed_fields"] else []):
            field_fail[fld] = field_fail.get(fld, 0) + 1
    if field_fail:
        print("\nPer-field failures (last run):")
        for fld, c in sorted(field_fail.items(), key=lambda x: -x[1]):
            print(f"  {fld:18s} {c}")

    # worst performers
    worst = sorted(scored, key=lambda r: r["cer"], reverse=True)[:10]
    print("\nWorst 10 by CER:")
    for r in worst:
        print(f"  {r['country']} {r['stem']}: CER={r['cer']:.3f} "
              f"field_acc={r['field_acc']:.3f} failed=[{r['failed_fields']}]")

    print(f"\nReport written to: {_REPORT_PATH}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3, help="passes per image (averaged)")
    ap.add_argument("--limit", type=int, default=None, help="only first N images (debug)")
    ap.add_argument("--quiet", action="store_true", help="suppress per-image lines")
    args = ap.parse_args()

    rows = evaluate(runs=args.runs, limit=args.limit, quiet=args.quiet)
    write_report(rows)
    print_summary(rows)


if __name__ == "__main__":
    main()