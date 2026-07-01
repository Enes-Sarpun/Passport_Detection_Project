"""Batch-triggered reliability re-calibration (Rule B).

Re-calibrating on every new record would be noisy and wasteful. Instead this
script only fires once enough NEW confirmed records have accumulated since the
last calibration (default threshold: 10). When the threshold is met it runs the
full loop:

    merge_to_gt.py  →  calibrate.py collect  →  calibrate.py analyse

State is a tiny JSON file (GroundTruth/.calibration_state.json) holding the max
scan_records.id calibrated so far. Manual override with --force.

Usage:
    DATABASE_URL=postgres://... python Scripts/active_learning/maybe_recalibrate.py
    #   [--threshold N] [--force] [--runs R]
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from Scripts.active_learning.export_dataset import _connect

_STATE_PATH = _ROOT / "GroundTruth" / ".calibration_state.json"
_CALIBRATE = _ROOT / "GroundTruth" / "calibrate.py"
_MERGE = _ROOT / "Scripts" / "active_learning" / "merge_to_gt.py"
_DEFAULT_THRESHOLD = 10


def _load_state() -> dict:
    if _STATE_PATH.exists():
        try:
            return json.loads(_STATE_PATH.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            pass
    return {"last_calibrated_max_id": 0}


def _save_state(max_id: int) -> None:
    _STATE_PATH.write_text(
        json.dumps({"last_calibrated_max_id": max_id}, indent=2), encoding="utf-8"
    )


def _mergeable_stats(last_id: int) -> tuple[int, int]:
    """Return (new_count, current_max_id) of confirmed records past last_id.
    'Mergeable' = a human confirmed the raw MRZ (corrected_mrz IS NOT NULL)."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*), COALESCE(MAX(id), 0) FROM scan_records "
            "WHERE corrected_mrz IS NOT NULL AND id > %s",
            (last_id,),
        ).fetchone()
    return int(row[0]), int(row[1])


def _run(script: Path, *args: str) -> None:
    print(f"\n$ python {script.relative_to(_ROOT)} {' '.join(args)}")
    subprocess.run([sys.executable, str(script), *args], check=True, cwd=_ROOT)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threshold", type=int, default=_DEFAULT_THRESHOLD,
                    help=f"new confirmed records needed to fire (default {_DEFAULT_THRESHOLD})")
    ap.add_argument("--force", action="store_true", help="recalibrate regardless of threshold")
    ap.add_argument("--runs", type=int, default=3, help="calibrate.py passes per image")
    args = ap.parse_args()

    state = _load_state()
    last_id = int(state.get("last_calibrated_max_id", 0))
    new_count, current_max = _mergeable_stats(last_id)

    print(f"Confirmed records since last calibration (id > {last_id}): {new_count}")
    if not args.force and new_count < args.threshold:
        print(f"{new_count}/{args.threshold} — threshold not met, skipping. "
              f"Use --force to override.")
        return

    if new_count == 0 and not args.force:
        print("Nothing new to calibrate.")
        return

    # Full loop: rebuild active GT → collect signals → fit + write model.
    _run(_MERGE)
    _run(_CALIBRATE, "collect", "--runs", str(args.runs))
    _run(_CALIBRATE, "analyse")

    _save_state(current_max)
    print(f"\nRe-calibration complete. State advanced to max id {current_max}.")
    print("Review the new AUC/Brier above; update schema.py _reliability_score "
          "only if the calibration curve looks honest.")


if __name__ == "__main__":
    main()
