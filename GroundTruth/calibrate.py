"""
Reliability-score weight calibration against ground truth.

Goal: replace the hand-picked weights in Scripts/OCR/schema.py `_reliability_score`
(0.45 cd / 0.25 struct / 0.20 ocr / 0.10 det + penalties) with weights *learned
from data*, so that reliability_score genuinely approximates P(output is correct).

This module has two stages:

  1. collect_signals(): run every ground-truth image through the Tesseract
     pipeline, recompute the same signals schema.py feeds into _reliability_score
     (cd_fraction, structural_fraction, mean_ocr/field_conf, detection_conf,
     is_specimen, is_expired, zero_docnum), and pair them with the ground-truth
     label is_correct = (field accuracy == 1.0). Writes calibration_data.csv.

  2. analyse(): correlation of each signal with correctness, then a logistic
     regression (class_weight='balanced') under 5-fold stratified CV. Reports
     learned weights, calibration curve, and Brier score — so we can see whether
     the score is honestly calibrated rather than inflated.

Usage:
    python GroundTruth/calibrate.py collect [--runs 3]
    python GroundTruth/calibrate.py analyse
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np

from Scripts.parsing.mrz_parse import parse_mrz, MRZResult, check_stop_words
from Scripts.parsing.schema import _field_reliability

_GT_PATH = _ROOT / "GroundTruth" / "ground_truth.json"
_IMG_DIR = _ROOT / "Images" / "MRZ_Data" / "Processed_data" / "images" / "test"
_DATA_PATH = _ROOT / "GroundTruth" / "calibration_data.csv"

# Same field set the accuracy evaluation uses (issuing_country excluded).
_FIELDS = [
    "document_number", "nationality", "surname", "given_names",
    "birth_date_raw", "expiry_date_raw", "sex",
]

_CHECKDIGIT_KEYS = {
    "document_number_valid", "date_of_birth_valid",
    "date_of_expiry_valid", "personal_number_valid", "composite_valid",
}
_STRUCTURAL_KEYS = {
    "line_length_valid", "dates_well_formed", "expiry_after_birth",
    "country_codes_known", "sex_value_valid", "document_type_known",
}

_SIGNAL_COLS = [
    "cd_fraction", "structural_fraction", "mean_field_conf",
    "detection_conf", "is_specimen", "is_expired", "zero_docnum",
]


# ---------------------------------------------------------------------------
# Stage 1 — signal collection
# ---------------------------------------------------------------------------

def _fields_from_lines(lines: list[str]) -> dict[str, str]:
    r = parse_mrz(lines)
    if r is None:
        return {f: "" for f in _FIELDS}
    return {
        "document_number": r.document_number, "nationality": r.nationality,
        "surname": r.surname, "given_names": r.given_names,
        "birth_date_raw": r.birth_date_raw, "expiry_date_raw": r.expiry_date_raw,
        "sex": r.sex,
    }


def _signals_from_result(result: MRZResult, det_conf: float, ocr_conf: float) -> dict[str, float]:
    """Recompute the exact signals schema.py feeds into _reliability_score."""
    checks = result.validation
    repaired = set(result.auto_repaired_fields)

    passed_cd = sum(
        1 for k, v in checks.items()
        if k in _CHECKDIGIT_KEYS and v is True
        and k.replace("_valid", "") not in repaired
    )
    cd_fraction = passed_cd / len(_CHECKDIGIT_KEYS)

    passed_struct = sum(1 for k in _STRUCTURAL_KEYS if checks.get(k) is True)
    structural_fraction = passed_struct / len(_STRUCTURAL_KEYS)

    # mean_field_conf as schema.py computes it (now per-field reliability)
    fc = {
        "document_number": _field_reliability("document_number", ocr_conf, checks.get("document_number_valid"), "document_number" in repaired),
        "date_of_birth": _field_reliability("date_of_birth", ocr_conf, checks.get("date_of_birth_valid"), "date_of_birth" in repaired),
        "date_of_expiry": _field_reliability("date_of_expiry", ocr_conf, checks.get("date_of_expiry_valid"), "date_of_expiry" in repaired),
        "personal_number": _field_reliability("personal_number", ocr_conf, checks.get("personal_number_valid"), "personal_number" in repaired),
        "nationality": _field_reliability("nationality", ocr_conf, None, "nationality" in repaired),
        "name": _field_reliability("name", ocr_conf, None, "name" in repaired),
    }
    mean_field_conf = sum(fc.values()) / len(fc)

    import datetime as _dt
    is_expired = False
    if result.expiry_date_iso:
        try:
            is_expired = _dt.date.fromisoformat(result.expiry_date_iso) < _dt.date.today()
        except ValueError:
            pass
    is_specimen = check_stop_words(result.surname, result.given_names)
    doc_clean = result.document_number.replace("<", "")
    zero_docnum = bool(doc_clean) and all(c == "0" for c in doc_clean)

    return {
        "cd_fraction": round(cd_fraction, 4),
        "structural_fraction": round(structural_fraction, 4),
        "mean_field_conf": round(mean_field_conf, 4),
        "detection_conf": round(float(det_conf), 4),
        "is_specimen": int(is_specimen),
        "is_expired": int(is_expired),
        "zero_docnum": int(zero_docnum),
    }


def collect_signals(runs: int = 3) -> None:
    from Scripts.ocr.pipeline import _process_frame
    from Scripts.detection.detect import detect_mrz

    with open(_GT_PATH, encoding="utf-8") as f:
        gt = json.load(f)
    stems = [k for k in gt if not k.startswith("_")]

    rows: list[dict] = []
    for idx, stem in enumerate(stems, 1):
        img_path = _IMG_DIR / f"{stem}.jpg"
        if not img_path.exists():
            continue
        image = cv2.imdecode(np.frombuffer(img_path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        gt_fields = _fields_from_lines(gt[stem]["lines"])

        # Average signals + correctness across runs (OCR is mildly non-deterministic).
        sig_acc: dict[str, list[float]] = {c: [] for c in _SIGNAL_COLS}
        correct_runs: list[int] = []
        for _ in range(runs):
            out, detection, raw_signals = _process_frame(image, return_signals=True)
            if out is None or detection is None or raw_signals is None:
                # No MRZ / parse fail → all signals zero, definitely incorrect.
                for c in _SIGNAL_COLS:
                    sig_acc[c].append(0.0)
                correct_runs.append(0)
                continue
            pred_lines = out.get("raw_mrz", []) or []
            parsed = parse_mrz(pred_lines)
            ocr_conf = raw_signals["ocr_confidence"]
            det_conf = raw_signals["detection_confidence"]
            if parsed is None:
                for c in _SIGNAL_COLS:
                    sig_acc[c].append(0.0)
                correct_runs.append(0)
                continue
            sig = _signals_from_result(parsed, det_conf, ocr_conf)
            for c in _SIGNAL_COLS:
                sig_acc[c].append(sig[c])
            pred_fields = _fields_from_lines(pred_lines)
            is_correct = int(all(gt_fields[f] == pred_fields[f] for f in _FIELDS))
            correct_runs.append(is_correct)

        row = {"stem": stem, "country": stem.split("-")[1] if "-" in stem else "???"}
        for c in _SIGNAL_COLS:
            row[c] = round(sum(sig_acc[c]) / len(sig_acc[c]), 4)
        # Label: correct only if every run was fully correct (strict, deterministic-leaning).
        row["is_correct"] = int(all(correct_runs) and len(correct_runs) > 0)
        row["correct_fraction"] = round(sum(correct_runs) / len(correct_runs), 3)
        rows.append(row)
        print(f"[{idx}/{len(stems)}] {row['country']} {stem}: "
              f"is_correct={row['is_correct']} cd={row['cd_fraction']} "
              f"struct={row['structural_fraction']} fconf={row['mean_field_conf']}")

    cols = ["stem", "country"] + _SIGNAL_COLS + ["is_correct", "correct_fraction"]
    with open(_DATA_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    pos = sum(r["is_correct"] for r in rows)
    print(f"\nWrote {_DATA_PATH}: {len(rows)} rows, {pos} correct / {len(rows)-pos} incorrect")


# ---------------------------------------------------------------------------
# Stage 2 — analysis (correlation only; modelling added after we see the data)
# ---------------------------------------------------------------------------

def analyse() -> None:
    import numpy as np
    with open(_DATA_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    y = np.array([int(r["is_correct"]) for r in rows])
    print(f"Rows: {len(rows)} | correct: {y.sum()} | incorrect: {(1-y).sum()}\n")

    print("Signal correlation with is_correct (Pearson):")
    for c in _SIGNAL_COLS:
        x = np.array([float(r[c]) for r in rows])
        if x.std() == 0:
            print(f"  {c:22s}  constant (no variance)")
            continue
        corr = np.corrcoef(x, y)[0, 1]
        print(f"  {c:22s}  r = {corr:+.3f}   mean_correct={x[y==1].mean():.3f}  mean_incorrect={x[y==0].mean():.3f}")

    _fit_logreg(rows, y)


def _fit_logreg(rows: list[dict], y) -> None:
    """Learn reliability weights from data via logistic regression under 5-fold
    stratified CV, then check whether the resulting score is honestly calibrated
    (not inflated) using out-of-fold predictions, a reliability curve, and Brier."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import brier_score_loss, roc_auc_score

    # Keep only signals that *causally* reflect read quality and therefore
    # generalise beyond this dataset:
    #   - mean_field_conf:     direct OCR confidence over the parsed fields
    #   - structural_fraction: how internally consistent the parse is
    # Deliberately excluded after inspecting the data:
    #   - cd_fraction / detection_conf: ~zero (or negative) correlation here;
    #     check digits pass even on images whose name line is wrong.
    #   - is_specimen / zero_docnum: correlate only because this dataset's
    #     specimens happen to be clean scans — a spurious, non-causal signal
    #     that would misfire on real-world (non-specimen) documents.
    feat_cols = ["mean_field_conf", "structural_fraction"]
    X = np.array([[float(r[c]) for c in feat_cols] for r in rows])

    # Standardise so learned coefficients are comparable as "importances".
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)

    # Out-of-fold probabilities — the honest estimate (each row predicted by a
    # model that never saw it). This is what guards against score inflation.
    oof = cross_val_predict(clf, Xs, y, cv=cv, method="predict_proba")[:, 1]

    auc = roc_auc_score(y, oof)
    brier = brier_score_loss(y, oof)
    print("\n" + "=" * 56)
    print("LOGISTIC REGRESSION (5-fold stratified, class_weight=balanced)")
    print("=" * 56)
    print(f"Out-of-fold AUC:   {auc:.3f}  (0.5=random, 1.0=perfect)")
    print(f"Out-of-fold Brier: {brier:.4f}  (lower=better calibrated)")

    # Reliability curve: bucket OOF scores, compare predicted vs actual accuracy.
    print("\nCalibration curve (OOF score bucket -> actual correct rate):")
    edges = [0.0, 0.5, 0.7, 0.85, 0.95, 1.01]
    for lo, hi in zip(edges, edges[1:]):
        mask = (oof >= lo) & (oof < hi)
        if mask.sum() == 0:
            continue
        print(f"  score [{lo:.2f},{hi:.2f}): n={mask.sum():3d}  "
              f"mean_score={oof[mask].mean():.3f}  actual_correct={y[mask].mean():.3f}")

    # Fit once on all data to read final coefficients (the learned weights).
    clf.fit(Xs, y)
    print("\nLearned weights (standardised coefficients, |w| = importance):")
    order = np.argsort(-np.abs(clf.coef_[0]))
    for i in order:
        print(f"  {feat_cols[i]:22s}  coef = {clf.coef_[0][i]:+.3f}")
    print(f"  intercept = {clf.intercept_[0]:+.3f}")

    # Persist the model parameters in raw (un-standardised) form so they can be
    # written into schema.py without needing the scaler at inference time.
    raw_coef = clf.coef_[0] / sd
    raw_intercept = clf.intercept_[0] - float((clf.coef_[0] * mu / sd).sum())
    model = {
        "features": feat_cols,
        "raw_coef": [float(c) for c in raw_coef],
        "raw_intercept": float(raw_intercept),
        "oof_auc": float(auc),
        "oof_brier": float(brier),
    }
    out_path = _ROOT / "GroundTruth" / "calibration_model.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2)
    print(f"\nModel written to: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--runs", type=int, default=3)
    sub.add_parser("analyse")
    args = ap.parse_args()

    if args.cmd == "collect":
        collect_signals(runs=args.runs)
    elif args.cmd == "analyse":
        analyse()


if __name__ == "__main__":
    main()