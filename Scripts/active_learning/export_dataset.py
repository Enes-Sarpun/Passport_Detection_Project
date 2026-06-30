"""Export human-corrected scan records into a retraining dataset.

The Trust Console saves every confirmed scan (image + model output + the
human-corrected fields) to PostgreSQL. This script pulls the records that a
human actually corrected — the high-value "gold labels" for active learning —
and writes them to disk as images + labels ready for retraining.

Active-learning safety (see GroundTruth/ground_truth.json and the plan):
  * Only `human_corrected = true` records are exported. Records the model got
    right (no human edit) are excluded so the model is never retrained on its
    own output (avoids feedback-loop / model collapse).
  * Records whose image SHA-256 matches a ground-truth/test image are EXCLUDED,
    so the evaluation set never leaks into training (data leakage).

Usage:
    DATABASE_URL=postgres://... python Scripts/active_learning/export_dataset.py
    # optional: --out <dir>  --since <ISO timestamp>
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_GT_PATH = _ROOT / "GroundTruth" / "ground_truth.json"
_GT_IMG_DIR = _ROOT / "Images" / "MRZ_Data" / "Processed_data" / "images" / "test"
_DEFAULT_OUT = _ROOT / "export" / "active_learning"


def _gt_image_hashes() -> set[str]:
    """SHA-256 of every ground-truth/test image, for leakage exclusion."""
    hashes: set[str] = set()
    if not _GT_PATH.exists():
        return hashes
    gt = json.loads(_GT_PATH.read_text(encoding="utf-8"))
    for stem in (k for k in gt if not k.startswith("_")):
        img = _GT_IMG_DIR / f"{stem}.jpg"
        if img.exists():
            hashes.add(hashlib.sha256(img.read_bytes()).hexdigest())
    return hashes


def _connect():
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        print("ERROR: DATABASE_URL is not set.", file=sys.stderr)
        sys.exit(1)
    try:
        import psycopg
    except ImportError:
        print("ERROR: psycopg not installed (pip install 'psycopg[binary]').", file=sys.stderr)
        sys.exit(1)
    return psycopg.connect(dsn)


def export(out_dir: Path, since: str | None) -> None:
    gt_hashes = _gt_image_hashes()
    print(f"Ground-truth image hashes loaded: {len(gt_hashes)} (these are excluded)")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / stamp
    img_dir = run_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    labels_path = run_dir / "labels.jsonl"

    query = (
        "SELECT id, filename, image, image_sha256, corrected_fields, mrz_format, created_at "
        "FROM scan_records WHERE human_corrected = TRUE"
    )
    params: list = []
    if since:
        query += " AND created_at >= %s"
        params.append(since)
    query += " ORDER BY id"

    exported = 0
    skipped_leak = 0
    with _connect() as conn, open(labels_path, "w", encoding="utf-8") as labels:
        for row in conn.execute(query, params):
            rec_id, filename, image, sha, corrected, mrz_format, created_at = row
            image = bytes(image)

            if sha in gt_hashes:
                skipped_leak += 1
                continue

            img_name = f"{rec_id}.jpg"
            (img_dir / img_name).write_bytes(image)
            labels.write(json.dumps({
                "id": rec_id,
                "image": f"images/{img_name}",
                "original_filename": filename,
                "image_sha256": sha,
                "mrz_format": mrz_format,
                "fields": corrected,
                "created_at": created_at.isoformat() if created_at else None,
            }, ensure_ascii=False) + "\n")
            exported += 1

    print(f"Exported: {exported} records → {run_dir}")
    print(f"Excluded (ground-truth leakage): {skipped_leak}")
    if exported == 0:
        print("Note: no human-corrected, non-GT records found yet.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=_DEFAULT_OUT,
                    help="output base directory (default: export/active_learning)")
    ap.add_argument("--since", default=None,
                    help="only records created on/after this ISO timestamp")
    args = ap.parse_args()
    export(args.out, args.since)


if __name__ == "__main__":
    main()
