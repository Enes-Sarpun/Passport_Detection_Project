from __future__ import annotations
import datetime as _dt
import json
from typing import Any, Optional
from .country_lookup import resolve_country
from .mrz_parse import MRZResult
from .schema_helpers import _DOC_TYPE_MAP, _SEX_MAP

def _date_field(raw: str, iso: Optional[str]) -> dict[str, str]:
    return {"raw": raw, "iso": iso or ""}

_DOB_AMBIGUOUS_AGE_MAX = 100
_DOB_AMBIGUOUS_AGE_MIN = 5

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
_FIELD_CONF_THRESHOLD = 0.85


def _field_confidence(
    ocr_c: float,
    checkdigit_valid: Optional[bool],
    repaired: bool,
) -> float:
    if checkdigit_valid is None:
        return round(float(ocr_c), 2)
    if checkdigit_valid and not repaired:
        return round(min(0.90 + 0.10 * float(ocr_c), 1.0), 2)
    if checkdigit_valid and repaired:
        return round(min(0.80 + 0.10 * float(ocr_c), 0.90), 2)
    return round(min(float(ocr_c), 0.40), 2)


def build_output(
    result: MRZResult,
    detection_confidence: float = 0.0,
    ocr_confidence: float = 0.0,
    raw_mrz: Optional[list[str]] = None,
    extra_warnings: Optional[list[str]] = None,
) -> dict[str, Any]:
    checks = result.validation
    repaired_set = set(result.auto_repaired_fields)
    ocr_c = float(ocr_confidence)

    field_confs = {
        "document_number": _field_confidence(
            ocr_c, checks.get("document_number_valid"), "document_number" in repaired_set,
        ),
        "date_of_birth": _field_confidence(
            ocr_c, checks.get("date_of_birth_valid"), "date_of_birth" in repaired_set,
        ),
        "date_of_expiry": _field_confidence(
            ocr_c, checks.get("date_of_expiry_valid"), "date_of_expiry" in repaired_set,
        ),
        "personal_number": _field_confidence(
            ocr_c, checks.get("personal_number_valid"), "personal_number" in repaired_set,
        ),
        "nationality": _field_confidence(ocr_c, None, "nationality" in repaired_set),
        "name": _field_confidence(ocr_c, None, "name" in repaired_set),
    }

    checkdigit_keys = {
        "document_number_valid", "date_of_birth_valid",
        "date_of_expiry_valid", "personal_number_valid", "composite_valid",
    }
    passed_cd = sum(
        1 for k, v in checks.items()
        if k in checkdigit_keys and v is True
        and k.replace("_valid", "").replace("_checkdigit", "") not in repaired_set
    )
    total_cd = len(checkdigit_keys)
    cd_fraction = passed_cd / total_cd

    mean_field_conf = sum(field_confs.values()) / len(field_confs)

    overall = round(
        0.2 * float(detection_confidence)
        + 0.4 * mean_field_conf
        + 0.4 * cd_fraction,
        2,
    )

    sex_raw = result.sex or ""
    sex_desc = _sex_description(sex_raw)
    doc_type_desc = _document_type_description(result.document_type)

    warnings: list[str] = list(extra_warnings or [])

    if 0 < float(detection_confidence) < _LOW_DETECTION_CONF_THRESHOLD:
        warnings.append("low_detection_confidence")

    if 0 < ocr_c < _LOW_OCR_CONF_THRESHOLD:
        warnings.append("low_ocr_confidence")

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

    for field_key, valid in checks.items():
        if field_key in ("auto_repaired_fields", "mrz_overall_valid", "failed_checks"):
            continue
        if field_key not in checkdigit_keys:
            continue
        if valid is False:
            warnings.append(f"checkdigit_failed:{field_key.replace('_valid', '')}")

    if not checks.get("line_length_valid", True):
        warnings.append("line_length_invalid")
    if not checks.get("dates_well_formed", True):
        warnings.append("dates_malformed")
    if not checks.get("expiry_after_birth", True):
        warnings.append("expiry_before_birth")

    if resolve_country(result.issuing_country)["name"] == "Unknown":
        warnings.append("issuing_country_unresolved")
    if resolve_country(result.nationality)["name"] == "Unknown":
        warnings.append("nationality_code_unresolved")

    if doc_type_desc == "Unknown":
        warnings.append("unknown_document_type")

    if sex_raw and sex_raw not in _SEX_MAP:
        warnings.append("sex_unrecognized")

    if not result.birth_date_iso:
        warnings.append("dob_unparseable")

    if _is_dob_century_ambiguous(result.birth_date_iso):
        warnings.append("dob_century_ambiguous")

    if raw_mrz:
        line1 = raw_mrz[0] if raw_mrz else ""
        name_field = line1[5:] if len(line1) >= 5 else ""
        if not name_separator_missing:
            parts = name_field.split("<<")
            for part in parts:
                tokens = [t for t in part.rstrip("<").split("<") if t]
                if any(len(t) <= 2 for t in tokens) and len(tokens) > 1:
                    warnings.append("name_low_confidence")
                    break

    if result.expiry_date_iso:
        try:
            expiry = _dt.date.fromisoformat(result.expiry_date_iso)
            if expiry < _dt.date.today():
                warnings.append("document_expired")
        except ValueError:
            pass

    for fname, fc in field_confs.items():
        if fc < _FIELD_CONF_THRESHOLD and fname != "name":
            warnings.append(f"{fname}_low_confidence")

    status = "ok"
    if overall < _LOW_OVERALL_THRESHOLD:
        status = "low_confidence"
        if "low_confidence" not in warnings:
            warnings.append("low_confidence")

    clean_repaired = [f for f in result.auto_repaired_fields if f != "document_type"]
    if "document_type" in result.auto_repaired_fields:
        raw_doc_type_char = (raw_mrz[0][0:2] if raw_mrz and raw_mrz[0] else "")
        if raw_doc_type_char and raw_doc_type_char[1] != "<":
            clean_repaired.append("document_type")

    # Validation checks
    validation_checks = {
        "document_number_checkdigit": checks.get("document_number_valid"),
        "date_of_birth_checkdigit": checks.get("date_of_birth_valid"),
        "date_of_expiry_checkdigit": checks.get("date_of_expiry_valid"),
        "personal_number_checkdigit": checks.get("personal_number_valid"),
        "composite_checkdigit": checks.get("composite_valid"),
        "line_length": checks.get("line_length_valid"),
        "dates_well_formed": checks.get("dates_well_formed"),
        "expiry_after_birth": checks.get("expiry_after_birth"),
        "country_codes_known": checks.get("country_codes_known"),
        "sex_value_valid": checks.get("sex_value_valid"),
        "document_type_known": checks.get("document_type_known"),
    }

    issuing = resolve_country(result.issuing_country)
    nat = resolve_country(result.nationality)

    out: dict[str, Any] = {}
    if status != "ok":
        out["status"] = status

    out.update({
        "document": {
            "type_code": result.document_type,
            "type_description": doc_type_desc,
            "number": {
                "value": result.document_number,
                "confidence": field_confs["document_number"],
            },
            "issuing_country": issuing["name"],
            "personal_number": {
                "value": result.personal_number if result.personal_number else "00000000000",
                "confidence": field_confs["personal_number"],
            },
        },

        "holder": {
            "surname": {
                "value": result.surname,
                "confidence": round(field_confs["name"], 2),
            },
            "given_names": {
                "value": result.given_names,
                "confidence": round(field_confs["name"], 2),
            },
            "full_name": (
                result.name_dict.get("full_name") if result.name_dict
                else f"{result.given_names} {result.surname}".strip()
            ),
            "nationality": {
                "name": nat["name"],
                "confidence": field_confs["nationality"],
            },
            "sex": {
                "code": sex_raw,
                "description": sex_desc,
                "confidence": round(field_confs["document_number"], 2),
            },
        },

        "dates": {
            "date_of_birth": {
                **_date_field(result.birth_date_raw, result.birth_date_iso),
                "confidence": field_confs["date_of_birth"],
            },
            "date_of_expiry": {
                **_date_field(result.expiry_date_raw, result.expiry_date_iso),
                "confidence": field_confs["date_of_expiry"],
            },
        },

        "confidence": {
            "detection": round(float(detection_confidence), 2),
            "ocr_mean": round(ocr_c, 2),
            "overall": overall,
        },

        "warnings": warnings,
        "raw_mrz": raw_mrz or [],
    })
    return out


def failure_output(
    status: str,
    raw_mrz: Optional[list[str]] = None,
    warnings: Optional[list[str]] = None,
) -> dict[str, Any]:
    return {"status": status, "warnings": warnings or [], "raw_mrz": raw_mrz or []}

def to_json(data: dict[str, Any], indent: int = 2) -> str:
    return json.dumps(data, ensure_ascii=False, indent=indent)
