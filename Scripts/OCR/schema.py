"""Output schema for the MRZ pipeline. Converts MRZResult to the JSON contract defined in the plan."""

from __future__ import annotations

import json
from typing import Any, Optional

from .country_lookup import resolve_country
from .mrz_parse import MRZResult


def _date_field(raw: str, iso: Optional[str]) -> dict[str, str]:
    return {"raw": raw, "iso": iso or ""}


def build_output(
    result: MRZResult,
    detection_confidence: float = 0.0,
    ocr_confidence: float = 0.0,
    raw_mrz: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Assemble the final JSON-ready dict from a parsed MRZResult."""
    checks = result.validation
    n_checks = len(checks)
    passed_before_repair = sum(
        1 for k, v in checks.items()
        if k != "auto_repaired_fields" and v
    )
    repaired_count = len(result.auto_repaired_fields)
    pre_repair_fraction = (passed_before_repair - repaired_count) / max(n_checks, 1)
    pre_repair_fraction = max(0.0, pre_repair_fraction)

    overall = round(
        0.4 * pre_repair_fraction
        + 0.3 * float(detection_confidence)
        + 0.3 * float(ocr_confidence),
        4,
    )

    return {
        "status": "ok",
        "document_type": result.document_type,
        "issuing_country": resolve_country(result.issuing_country),
        "fields": {
            "surname": result.surname,
            "given_names": result.given_names,
            "document_number": result.document_number,
            "nationality": resolve_country(result.nationality),
            "date_of_birth": _date_field(result.birth_date_raw, result.birth_date_iso),
            "sex": result.sex,
            "date_of_expiry": _date_field(result.expiry_date_raw, result.expiry_date_iso),
            "personal_number": result.personal_number,
        },
        "validation": {
            **checks,
            "auto_repaired_fields": result.auto_repaired_fields,
        },
        "detection_confidence": round(float(detection_confidence), 4),
        "ocr_confidence": round(float(ocr_confidence), 4),
        "overall_confidence": overall,
        "raw_mrz": raw_mrz or [],
    }


def failure_output(status: str, raw_mrz: Optional[list[str]] = None) -> dict[str, Any]:
    """Return a minimal failure dict — never raises."""
    return {"status": status, "raw_mrz": raw_mrz or []}


def to_json(data: dict[str, Any], indent: int = 2) -> str:
    return json.dumps(data, ensure_ascii=False, indent=indent)