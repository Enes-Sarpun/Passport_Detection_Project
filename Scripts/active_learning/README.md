# Active Learning Loop

Human-in-the-loop pipeline that turns confirmed scans from the Trust Console into
new ground-truth data and periodically re-calibrates the reliability model.

## How it works

Low-confidence reads (reliability < 0.75) require a human to correct them in the UI.
On **Verify and Save**, the image + model output + corrected fields + confirmed raw
MRZ lines are written to PostgreSQL (`scan_records`).

Only useful records are stored (see the record policy):
- **High-confidence, no mandatory fix** → saved (positive example).
- **Low-confidence, left uncorrected** → rejected (422), keeps the dataset clean.
- **Low-confidence, corrected** → saved (negative example).

Calibration needs *both* correct and corrected examples, so both are kept.

## Why not retrain the OCR/YOLO models directly?

The OCR is a fixed third-party Tesseract OCR-B model and the parser is rule-based
ICAO 9303 code — neither is "trained" here. The one learned model is the
**reliability score** (logistic regression, `GroundTruth/calibrate.py`). So the
realistic, high-value use of collected data is: **grow the ground truth** with real
confirmed examples and **re-calibrate** the reliability model on the larger set.

## Scripts

| Script | Purpose |
|--------|---------|
| `export_dataset.py` | Dump human-corrected records (image + labels) to `export/` |
| `merge_to_gt.py` | Build `GroundTruth/ground_truth_active.json` + `images/active/` from confirmed `corrected_mrz` |
| `maybe_recalibrate.py` | Fire the full re-calibration loop once ≥N (default 10) new confirmed records exist |

## Running the loop

```bash
export DATABASE_URL=postgres://...   # Render Postgres external URL

# One-shot, threshold-gated (recommended): merge → collect → analyse
python Scripts/active_learning/maybe_recalibrate.py            # fires at 10 new records
python Scripts/active_learning/maybe_recalibrate.py --force    # ignore the threshold

# Or run steps manually:
python Scripts/active_learning/merge_to_gt.py
python GroundTruth/calibrate.py collect --runs 3
python GroundTruth/calibrate.py analyse                        # writes calibration_model.json
```

After calibration, review the printed AUC / Brier / calibration curve. Only if the
curve is honest (not inflated), update the `_reliability_score` weights in
`Scripts/parsing/schema.py` from `GroundTruth/calibration_model.json`.

## Safety guarantees

- **No leakage:** records whose image hash matches a ground-truth/test image are
  excluded from the active set (`_gt_image_hashes`).
- **Golden GT untouched:** the hand-verified `ground_truth.json` and `images/test/`
  are never modified; active data lives in `ground_truth_active.json` / `images/active/`.
- **No model collapse:** the reliability model learns from signals + human-confirmed
  labels (`corrected_mrz`), never from the model's own unverified output.

Generated files (`ground_truth_active.json`, `images/active/`, `.calibration_state.json`,
`export/`) are gitignored — they are derived from the database.
