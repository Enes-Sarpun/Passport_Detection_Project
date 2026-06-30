from __future__ import annotations
import datetime as _dt
import json
from typing import Any, Optional
from .country_lookup import resolve_country
from .mrz_parse import MRZResult, check_stop_words
from .schema_helpers import _DOC_TYPE_MAP, _SEX_MAP

SCHEMA_VERSION = "5"

# Internal thresholds
_LOW_OCR_CONF = 0.60
_LOW_DETECT_CONF = 0.50
# Rescan / manual-review threshold, chosen from the calibration data: at 0.75 the
# score flags 22/22 truly-incorrect images (100% recall) for +2 false alarms over
# 0.70 (which let one borderline error through at exactly 0.70). 100% recall is
# worth the extra two manual reviews (GroundTruth/calibrate.py).
_RESCAN_THRESHOLD = 0.75
_FIELD_CONF_THRESHOLD = 0.85
_DOB_AGE_MIN = 5
_DOB_AGE_MAX = 100


# Small helpers

def _document_type_description(doc_type: str) -> str:
    return _DOC_TYPE_MAP.get(doc_type, "Unknown")


def _sex_description(sex: str) -> str:
    return _SEX_MAP.get(sex.upper() if sex else "", "Unspecified")


# Per-field base reliability: the empirical accuracy of each MRZ field measured
# on the 168-image ground-truth set (GroundTruth/accuracy_report.csv). These are
# NOT hand-picked — they are "given this is field X, how often is the pipeline's
# read actually correct". Name fields are hardest (stamps/guilloche corrupt the
# name line); check-digit-backed fields are nearly perfect. Re-derive from the
# accuracy report's per-field failure counts.
_FIELD_BASE_RELIABILITY = {
    "document_number": 0.982,
    "personal_number": 0.982,   # same structural class as document number
    "nationality": 0.994,
    "name": 0.917,              # mean of surname (0.923) and given_names (0.911)
    "date_of_birth": 0.994,
    "date_of_expiry": 0.988,
    "sex": 0.994,
}


def _field_reliability(
    field_key: str,
    ocr_c: float,
    checkdigit_valid: Optional[bool],
    repaired: bool,
) -> float:
    """Per-field reliability ≈ P(this field's value is correct), grounded in data.

    Starts from the field's empirical accuracy on ground truth (_FIELD_BASE_…)
    and modulates it by this document's live signals:
      - a FAILED check digit is strong evidence the read is wrong → heavy penalty;
      - an auto-repaired value is less certain than a clean pass → mild penalty;
      - low OCR confidence drags the score down proportionally.
    A passing check digit leaves the base intact (it is already the measured
    accuracy of check-digit-passing reads). The result is clamped to [0, 1].
    """
    base = _FIELD_BASE_RELIABILITY.get(field_key, 0.95)

    # Check-digit evidence (only for fields that have one).
    if checkdigit_valid is False:
        base *= 0.35           # check digit says the read is almost certainly wrong
    elif checkdigit_valid is True and repaired:
        base *= 0.90           # recovered via confusion repair — slightly less sure
    elif checkdigit_valid is None and repaired:
        base *= 0.90           # name/nationality repaired (no check digit to confirm)

    # OCR confidence modulation: scale toward the base as confidence rises.
    # At ocr_c=1.0 the base is untouched; lower confidence pulls it down.
    conf_factor = 0.7 + 0.3 * float(ocr_c)   # maps ocr_c∈[0,1] → [0.7, 1.0]
    score = base * conf_factor
    return round(max(0.0, min(1.0, score)), 2)


def _is_dob_century_ambiguous(iso_dob: Optional[str]) -> bool:
    if not iso_dob:
        return False
    try:
        dob = _dt.date.fromisoformat(iso_dob)
        today = _dt.date.today()
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        return age < _DOB_AGE_MIN or age >= _DOB_AGE_MAX
    except (ValueError, TypeError):
        return False


import math as _math

# Overall reliability_score weights, fitted by logistic regression on 168 GT
# images (GroundTruth/calibrate.py) predicting P(all fields correct) from
# mean_field_reliability + structural_fraction. Out-of-fold AUC 0.915, Brier
# 0.119. Re-derive after any change to _field_reliability:
#     python GroundTruth/calibrate.py collect && ... analyse
_REL_COEF_FIELD_CONF = 29.5137
_REL_COEF_STRUCTURAL = 10.4693
_REL_INTERCEPT = -35.9797


def _reliability_score(
    cd_fraction: float,
    structural_fraction: float,
    mean_ocr_conf: float,
    detection_conf: float,
    is_specimen: bool,
    zero_docnum: bool,
    is_expired: bool,
) -> float:
    z = (
        _REL_COEF_FIELD_CONF * float(mean_ocr_conf)
        + _REL_COEF_STRUCTURAL * float(structural_fraction)
        + _REL_INTERCEPT
    )
    prob = 1.0 / (1.0 + _math.exp(-z))
    return round(max(0.0, min(1.0, prob)), 2)


# Main builder

def build_output(
    result: MRZResult,
    detection_confidence: float = 0.0,
    ocr_confidence: float = 0.0,
    raw_mrz: Optional[list[str]] = None,
    extra_warnings: Optional[list[str]] = None,
    mrz_format: str = "TD3",
) -> dict[str, Any]:

    checks = result.validation
    repaired_set = set(result.auto_repaired_fields)
    ocr_c = float(ocr_confidence)
    det_c = float(detection_confidence)

    field_confs = {
        "document_number": _field_reliability(
            "document_number", ocr_c, checks.get("document_number_valid"),
            "document_number" in repaired_set),
        "date_of_birth": _field_reliability(
            "date_of_birth", ocr_c, checks.get("date_of_birth_valid"),
            "date_of_birth" in repaired_set),
        "date_of_expiry": _field_reliability(
            "date_of_expiry", ocr_c, checks.get("date_of_expiry_valid"),
            "date_of_expiry" in repaired_set),
        "personal_number": _field_reliability(
            "personal_number", ocr_c, checks.get("personal_number_valid"),
            "personal_number" in repaired_set),
        "nationality": _field_reliability(
            "nationality", ocr_c, None, "nationality" in repaired_set),
        "name": _field_reliability("name", ocr_c, None, "name" in repaired_set),
    }

    checkdigit_keys = {
        "document_number_valid", "date_of_birth_valid",
        "date_of_expiry_valid", "personal_number_valid", "composite_valid",
    }
    passed_cd = sum(
        1 for k, v in checks.items()
        if k in checkdigit_keys and v is True
        and k.replace("_valid", "") not in repaired_set
    )
    total_cd = len(checkdigit_keys)
    cd_fraction = passed_cd / total_cd

    structural_keys = {
        "line_length_valid", "dates_well_formed", "expiry_after_birth",
        "country_codes_known", "sex_value_valid", "document_type_known",
    }
    passed_struct = sum(1 for k in structural_keys if checks.get(k) is True)
    structural_fraction = passed_struct / len(structural_keys)

    mean_field_conf = sum(field_confs.values()) / len(field_confs)

    today = _dt.date.today()
    is_expired = False
    if result.expiry_date_iso:
        try:
            is_expired = _dt.date.fromisoformat(result.expiry_date_iso) < today
        except ValueError:
            pass

    is_specimen = check_stop_words(result.surname, result.given_names)
    doc_number_clean = result.document_number.replace("<", "")
    zero_docnum = bool(doc_number_clean) and all(c == "0" for c in doc_number_clean)

    rel_score = _reliability_score(
        cd_fraction, structural_fraction, mean_field_conf, det_c,
        is_specimen, zero_docnum, is_expired,
    )
    rescan_recommended = (
        rel_score < _RESCAN_THRESHOLD
        or any(checks.get(k) is False for k in checkdigit_keys)
        or checks.get("line_length_valid") is False
    )

    warnings: list[str] = list(extra_warnings or [])

    if 0 < det_c < _LOW_DETECT_CONF:
        warnings.append("low_detection_confidence")
    if 0 < ocr_c < _LOW_OCR_CONF:
        warnings.append("low_ocr_confidence")

    # name separator
    name_separator_missing = False
    if raw_mrz:
        line1 = raw_mrz[0] if raw_mrz else ""
        name_field = line1[5:] if len(line1) >= 5 else ""
        if "<<" not in name_field.rstrip("<"):
            name_separator_missing = True
    elif not result.given_names and result.surname:
        name_separator_missing = True
    if name_separator_missing:
        warnings.append("name_separator_missing")

    # check digit failures
    for k in checkdigit_keys:
        if checks.get(k) is False:
            warnings.append(f"checkdigit_failed:{k.replace('_valid', '')}")

    if not checks.get("line_length_valid", True):
        warnings.append("line_length_invalid")
    if not checks.get("dates_well_formed", True):
        warnings.append("dates_malformed")
    if not checks.get("expiry_after_birth", True):
        warnings.append("expiry_before_birth")

    issuing = resolve_country(result.issuing_country)
    nat = resolve_country(result.nationality)

    if issuing["name"] == "Unknown":
        warnings.append("issuing_country_unresolved")
    if nat["name"] == "Unknown":
        warnings.append("nationality_code_unresolved")
    if _document_type_description(result.document_type) == "Unknown":
        warnings.append("unknown_document_type")

    sex_raw = result.sex or ""
    if sex_raw and sex_raw not in _SEX_MAP:
        warnings.append("sex_unrecognized")
    if not result.birth_date_iso:
        warnings.append("dob_unparseable")
    if _is_dob_century_ambiguous(result.birth_date_iso):
        warnings.append("dob_century_ambiguous")

    # name token quality
    if raw_mrz and not name_separator_missing:
        line1 = raw_mrz[0]
        name_field = line1[5:] if len(line1) >= 5 else ""
        for part in name_field.split("<<"):
            tokens = [t for t in part.rstrip("<").split("<") if t]
            if any(len(t) <= 2 for t in tokens) and len(tokens) > 1:
                warnings.append("name_low_confidence")
                break

    if is_expired:
        warnings.append("document_expired")
    if is_specimen:
        warnings.append("specimen_or_test_document")
    if zero_docnum:
        warnings.append("zero_document_number")

    # nationality ≠ issuer (soft flag, J2)
    if (result.nationality and result.issuing_country
            and result.nationality != result.issuing_country
            and nat["name"] != "Unknown" and issuing["name"] != "Unknown"):
        warnings.append("nationality_differs_from_issuer")

    for fname, fc in field_confs.items():
        if fc < _FIELD_CONF_THRESHOLD and fname != "name":
            warnings.append(f"{fname}_low_confidence")

    # validation summary
    all_checks = {
        "document_number_checkdigit": checks.get("document_number_valid"),
        "date_of_birth_checkdigit":   checks.get("date_of_birth_valid"),
        "date_of_expiry_checkdigit":  checks.get("date_of_expiry_valid"),
        "personal_number_checkdigit": checks.get("personal_number_valid"),
        "composite_checkdigit":       checks.get("composite_valid"),
        "line_length":                checks.get("line_length_valid"),
        "dates_well_formed":          checks.get("dates_well_formed"),
        "expiry_after_birth":         checks.get("expiry_after_birth"),
        "country_codes_known":        checks.get("country_codes_known"),
        "sex_value_valid":            checks.get("sex_value_valid"),
        "document_type_known":        checks.get("document_type_known"),
    }
    failed_checks = [k for k, v in all_checks.items() if v is False]
    mrz_overall_valid = len(failed_checks) == 0

    # clean auto_repaired list (exclude phantom document_type repairs)
    clean_repaired = [f for f in result.auto_repaired_fields if f != "document_type"]
    if "document_type" in result.auto_repaired_fields:
        raw_doc_type_char = (raw_mrz[0][0:2] if raw_mrz and raw_mrz[0] else "")
        if raw_doc_type_char and raw_doc_type_char[1] != "<":
            clean_repaired.append("document_type")

    status = "ok"
    if rel_score < _RESCAN_THRESHOLD:
        status = "low_confidence"
        if "low_confidence" not in warnings:
            warnings.append("low_confidence")

    doc_type_desc = _document_type_description(result.document_type)
    sex_desc = _sex_description(sex_raw)

    out: dict[str, Any] = {}
    if status != "ok":
        out["status"] = status

    out.update({
        "document": {
            "type": {
                "code": result.document_type,
                "description": doc_type_desc,
            },
            "number": {
                "value": result.document_number,
                "reliability": field_confs["document_number"],
            },
            "personal_number": {
                "value": result.personal_number if result.personal_number else "00000000000",
                "reliability": field_confs["personal_number"],
            },
            "mrz_format": mrz_format,
        },

        "holder": {
            "surname": {
                "value": result.surname,
                "reliability": round(field_confs["name"], 2),
            },
            "given_names": {
                "value": result.given_names,
                "reliability": round(field_confs["name"], 2),
            },
            "given_names_list": (
                result.name_dict.get("given_names_list", []) if result.name_dict else []
            ),
            "full_name": (
                result.name_dict.get("full_name") if result.name_dict
                else f"{result.given_names} {result.surname}".strip()
            ),
            "nationality": {
                "code": result.nationality,
                "name": nat["name"],
                "reliability": round(field_confs["nationality"], 2),
            },
            "sex": {
                "code": sex_raw,
                "description": sex_desc,
                "reliability": round(field_confs["document_number"], 2),
            },
        },

        "dates": {
            "date_of_birth": {
                "raw": result.birth_date_raw,
                "iso": result.birth_date_iso or "",
                "reliability": field_confs["date_of_birth"],
            },
            "date_of_expiry": {
                "raw": result.expiry_date_raw,
                "iso": result.expiry_date_iso or "",
                "reliability": field_confs["date_of_expiry"],
            },
            "is_expired": is_expired,
        },

        "validation": {
            "mrz_overall_valid": mrz_overall_valid,
            "failed_checks": failed_checks,
            "auto_repaired_fields": clean_repaired,
        },

        "quality": {
            "reliability_score": rel_score,
            "rescan_recommended": rescan_recommended,
        },

        "warnings": warnings,
        "raw_mrz": raw_mrz or [],
    })

    if is_specimen:
        out["quality"]["is_specimen"] = True

    return out


def failure_output(
    status: str,
    raw_mrz: Optional[list[str]] = None,
    warnings: Optional[list[str]] = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "warnings": warnings or [],
        "raw_mrz": raw_mrz or [],
    }


def to_json(data: dict[str, Any], indent: int = 2) -> str:
    return json.dumps(data, ensure_ascii=False, indent=indent)