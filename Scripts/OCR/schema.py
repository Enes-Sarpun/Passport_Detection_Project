from __future__ import annotations
import datetime as _dt
import json
from typing import Any, Optional
from .country_lookup import resolve_country
from .mrz_parse import MRZResult

_DOC_TYPE_MAP: dict[str, str] = {
    "P": "Passport",
    "P<": "Passport",
    "PP": "Passport",
    "PD": "Diplomatic passport",
    "PS": "Service passport",
    "PL": "Special passport",
    "PO": "Official passport",
    "PT": "Travel document",
    "AC": "Crew member certificate",
    "IP": "Passport card",
    "ID": "Identity card",
    "I": "Identity card",
    "IR": "Residence permit",
    "V": "Visa",
}

_SEX_MAP: dict[str, str] = {
    "M": "Male",
    "F": "Female",
    "X": "Unspecified",
    "<": "Unspecified",
    "": "Unspecified",
}

def _date_field(raw: str, iso: Optional[str]) -> dict[str, str]:
    return {"raw": raw, "iso": iso or ""}

_DOB_AMBIGUOUS_AGE_MAX = 100  # >= 100 yaş → pivot muhtemelen yanlış yüzyıl seçti
_DOB_AMBIGUOUS_AGE_MIN = 5   # < 5 yaş  → pivot muhtemelen 1900s'i 2000s seçti

def _is_dob_century_ambiguous(iso_dob: Optional[str]) -> bool:
    if not iso_dob:
        return False
    try:
        dob = _dt.date.fromisoformat(iso_dob)
        today = _dt.date.today()
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        return age < _DOB_AMBIGUOUS_AGE_MIN or age >= _DOB_AMBIGUOUS_AGE_MAX
    except (ValueError, TypeError):
        return False

def _document_type_description(doc_type: str) -> str:
    return _DOC_TYPE_MAP.get(doc_type, "Unknown")

def _sex_description(sex: str) -> str:
    return _SEX_MAP.get(sex.upper() if sex else "", "Unspecified")

_LOW_OCR_CONF_THRESHOLD = 0.60
_LOW_DETECTION_CONF_THRESHOLD = 0.50
_LOW_OVERALL_THRESHOLD = 0.75


def build_output(
    result: MRZResult,
    detection_confidence: float = 0.0,
    ocr_confidence: float = 0.0,
    raw_mrz: Optional[list[str]] = None,
    extra_warnings: Optional[list[str]] = None,
) -> dict[str, Any]:
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

    sex_raw = result.sex or ""
    sex_desc = _sex_description(sex_raw)
    doc_type_desc = _document_type_description(result.document_type)

    # --- Build warnings ---
    warnings: list[str] = list(extra_warnings or [])

    # Detection confidence
    if 0 < float(detection_confidence) < _LOW_DETECTION_CONF_THRESHOLD:
        warnings.append("low_detection_confidence")

    # OCR confidence
    if 0 < float(ocr_confidence) < _LOW_OCR_CONF_THRESHOLD:
        warnings.append("low_ocr_confidence")

    # Check-digit failures
    for field_key, valid in checks.items():
        if field_key == "auto_repaired_fields":
            continue
        if valid is False:
            warnings.append(f"checkdigit_failed:{field_key.replace('_valid', '')}")

    # Unresolved country codes
    if resolve_country(result.issuing_country)["name"] == "Unknown":
        warnings.append("issuing_country_unresolved")
    if resolve_country(result.nationality)["name"] == "Unknown":
        warnings.append("nationality_code_unresolved")

    # Unknown document type
    if doc_type_desc == "Unknown":
        warnings.append("unknown_document_type")

    # Unrecognized sex
    if sex_raw and sex_raw not in _SEX_MAP:
        warnings.append("sex_unrecognized")

    # DOB unparseable
    if not result.birth_date_iso:
        warnings.append("dob_unparseable")

    # DOB century ambiguous — resolved age >= 100 almost always means wrong century pivot
    if _is_dob_century_ambiguous(result.birth_date_iso):
        warnings.append("dob_century_ambiguous")

    # Name low confidence — any single '<' inside the raw name segments signals
    # that OCR may have split a continuous name word (e.g. SPECIMEN → SPE<IMEN).
    # We detect this by checking if the name field in raw_mrz contains '<' within
    # a name segment (not as padding or '<<' separator).
    if raw_mrz:
        line1 = raw_mrz[0] if raw_mrz else ""
        # Name field starts at pos 5 in TD3 line 1.
        # '<<' separates surname from given names. Within each segment, single
        # '<' separates individual name words — that is normal.
        # We warn only when the SURNAME segment contains a single '<', because
        # surnames are always a single token in the MRZ. A '<' inside a surname
        # almost certainly means OCR split one word (e.g. SPE<IMEN → SPECIMEN).
        # In the given-names segment, single '<' between words is expected and
        # is NOT a low-confidence signal.
        name_field = line1[5:] if len(line1) >= 5 else ""
        parts = name_field.split("<<")
        # Single '<' between name words is a normal MRZ separator:
        #   FORTUNA<RAMIREZ  — double surname, both long tokens → normal
        #   FRED<WIREMU<JOHN — multiple given names → normal
        # OCR split signals:
        #   SPE<IMEN  — two tokens where one is suspiciously short (< 3 chars)
        #   LU<Y      — 'Y' is 1 char, almost certainly an OCR fragment
        # Rule: warn if any name segment contains a token of ≤ 2 characters
        # that results from a single-'<' split (not padding '<').
        for part in parts:
            tokens = [t for t in part.rstrip("<").split("<") if t]
            if any(len(t) <= 2 for t in tokens) and len(tokens) > 1:
                warnings.append("name_low_confidence")
                break

    # Document expired
    if result.expiry_date_iso:
        try:
            expiry = _dt.date.fromisoformat(result.expiry_date_iso)
            if expiry < _dt.date.today():
                warnings.append("document_expired")
        except ValueError:
            pass

    # Status downgrade when overall confidence is low
    status = "ok"
    if overall < _LOW_OVERALL_THRESHOLD:
        status = "low_confidence"
        if "low_confidence" not in warnings:
            warnings.append("low_confidence")

    return {
        "status": status,
        "document_type": result.document_type,
        "document_type_description": doc_type_desc,
        "issuing_country": resolve_country(result.issuing_country),
        "name": result.name_dict or {
            "surname": result.surname,
            "given_names": result.given_names,
            "given_names_list": result.given_names.split() if result.given_names else [],
            "full_name": f"{result.given_names} {result.surname}".strip(),
        },
        "fields": {
            "document_number": result.document_number,
            "nationality": resolve_country(result.nationality),
            "date_of_birth": _date_field(result.birth_date_raw, result.birth_date_iso),
            "sex": sex_raw,
            "sex_description": sex_desc,
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
        "warnings": warnings,
        "raw_mrz": raw_mrz or [],
    }


def failure_output(
    status: str,
    raw_mrz: Optional[list[str]] = None,
    warnings: Optional[list[str]] = None,
) -> dict[str, Any]:
    return {"status": status, "warnings": warnings or [], "raw_mrz": raw_mrz or []}

def to_json(data: dict[str, Any], indent: int = 2) -> str:
    return json.dumps(data, ensure_ascii=False, indent=indent)




