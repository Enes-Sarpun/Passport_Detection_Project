"""Shared field-comparison helpers for accuracy evaluation and calibration.

Both the accuracy evaluator and the reliability calibrator parse MRZ lines down
to the same field set and compare them against ground truth; keeping that field
list and extraction in one place avoids the two drifting apart.
"""
from __future__ import annotations
from .mrz_parse import parse_mrz

# Fields compared for field-level accuracy. issuing_country is intentionally
# excluded: it is read from line 1's country code (positions 2-4), which
# corrupts whenever the name line is noisy. The nationality field (line 2)
# carries the same information far more reliably.
COMPARISON_FIELDS = [
    "document_number",
    "nationality",
    "surname",
    "given_names",
    "birth_date_raw",
    "expiry_date_raw",
    "sex",
]


def fields_from_lines(lines: list[str]) -> dict[str, str]:
    """Parse MRZ lines into the comparison field set. Empty strings on failure."""
    result = parse_mrz(lines)
    if result is None:
        return {f: "" for f in COMPARISON_FIELDS}
    return {
        "document_number": result.document_number,
        "nationality": result.nationality,
        "surname": result.surname,
        "given_names": result.given_names,
        "birth_date_raw": result.birth_date_raw,
        "expiry_date_raw": result.expiry_date_raw,
        "sex": result.sex,
    }
