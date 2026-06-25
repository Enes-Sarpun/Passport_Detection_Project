from __future__ import annotations
import datetime as _dt
import itertools
from dataclasses import dataclass, field
from typing import Optional
from .country_lookup import resolve_country as _resolve_country, _repair_country_digits

MRZ_CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"

# Standard OCR confusion swaps used during check-digit self-repair.
CONFUSION_PAIRS = [
    ("0", "O"),
    ("1", "I"),
    ("5", "S"),
    ("8", "B"),
    ("2", "Z"),
    ("O", "Q"),
    ("0", "D"),
    ("0", "Q"),
    ("1", "L"),
    ("6", "G"),
]

def char_value(c: str) -> int:
    if c == "<":
        return 0
    if c.isdigit():
        return int(c)
    if "A" <= c <= "Z":
        return ord(c) - ord("A") + 10
    raise ValueError(f"Invalid MRZ character: {c!r}")

def check_digit(data: str) -> str:
    weights = (7, 3, 1)
    total = 0
    for i, c in enumerate(data):
        total += char_value(c) * weights[i % 3]
    return str(total % 10)

def check_digit_valid(data: str, expected: str) -> bool:
    if expected == "<":
        # A '<' in a check-digit slot is treated as 0 per common practice.
        expected = "0"
    if not expected.isdigit():
        return False
    return check_digit(data) == expected

def _repair_field(data: str, expected_cd: str) -> Optional[str]:
    if check_digit_valid(data, expected_cd):
        return data

    chars = list(data)
    positions = range(len(chars))

    # Build per-position candidate replacements from the confusion table.
    def candidates_for(c: str) -> list[str]:
        out = []
        for a, b in CONFUSION_PAIRS:
            if c == a:
                out.append(b)
            elif c == b:
                out.append(a)
        return out

    # Single-character swaps.
    for i in positions:
        for repl in candidates_for(chars[i]):
            trial = chars.copy()
            trial[i] = repl
            cand = "".join(trial)
            if check_digit_valid(cand, expected_cd):
                return cand

    # Two-character swaps (only for short fields to bound the search).
    if len(chars) <= 14:
        for i, j in itertools.combinations(positions, 2):
            ci = candidates_for(chars[i])
            cj = candidates_for(chars[j])
            for ri, rj in itertools.product(ci, cj):
                trial = chars.copy()
                trial[i] = ri
                trial[j] = rj
                cand = "".join(trial)
                if check_digit_valid(cand, expected_cd):
                    return cand

    return None

def parse_date(yymmdd: str, *, is_birth: bool) -> Optional[str]:
    if len(yymmdd) != 6 or not yymmdd.isdigit():
        return None
    yy, mm, dd = int(yymmdd[:2]), int(yymmdd[2:4]), int(yymmdd[4:6])
    current_yy = _dt.date.today().year % 100
    if is_birth:
        century = 1900 if yy > current_yy else 2000
    else:
        century = 2000 if yy <= current_yy + 50 else 1900
    year = century + yy
    try:
        return _dt.date(year, mm, dd).isoformat()
    except ValueError:
        return None

# Digit→letter map for name fields (letters + '<' only, digits are OCR confusions).
_NAME_DIGIT_TO_LETTER: dict[str, str] = {
    "0": "O",
    "1": "I",
    "5": "S",
    "8": "B",
    "2": "Z",
    "6": "G",
}

def _repair_name_digits(segment: str) -> str:
    return "".join(_NAME_DIGIT_TO_LETTER.get(c, c) for c in segment)

def parse_name(name_field: str, repaired: list[str] | None = None) -> tuple[str, str, dict]:
    # Split on '<<' to separate surname from given names.
    parts = name_field.split("<<", 1)
    raw_surname = parts[0]
    raw_given = parts[1] if len(parts) > 1 else ""

    # Detect and repair digits in name segments (Class 2 fix).
    had_digit = any(c.isdigit() for c in raw_surname + raw_given)
    raw_surname = _repair_name_digits(raw_surname)
    raw_given = _repair_name_digits(raw_given)
    if had_digit and repaired is not None:
        repaired.append("name")

    # Convert single '<' to spaces, strip trailing padding.
    surname = " ".join(raw_surname.replace("<", " ").split())
    given_names = " ".join(raw_given.replace("<", " ").split())

    # Build given_names_list from individual name components (split by space).
    given_names_list = [g for g in given_names.split(" ") if g] if given_names else []

    full_name = f"{given_names} {surname}".strip() if given_names else surname

    name_dict = {
        "surname": surname,
        "given_names": given_names,
        "given_names_list": given_names_list,
        "full_name": full_name,
    }
    return surname, given_names, name_dict

def _structural_checks(
    fmt: str,
    lines: list[str],
    doc_type: str,
    issuing_country: str,
    nationality: str,
    sex: str,
    birth_date_iso: Optional[str],
    expiry_date_iso: Optional[str],
) -> dict[str, bool]:
    from .country_lookup import resolve_country as _rc

    lengths = {"TD3": [44, 44], "TD2": [36, 36], "TD1": [30, 30, 30]}
    expected = lengths.get(fmt, [])
    line_length_valid = len(lines) == len(expected) and all(
        len(l) == e for l, e in zip(lines, expected)
    )

    dates_well_formed = True
    for raw_date in (
        lines[1][13:19] if fmt == "TD3" and len(lines) > 1 else "",
        lines[1][21:27] if fmt == "TD3" and len(lines) > 1 else "",
        lines[1][0:6] if fmt in ("TD2", "TD1") and len(lines) > 1 else "",
        lines[1][8:14] if fmt in ("TD2", "TD1") and len(lines) > 1 else "",
    ):
        if not raw_date or not raw_date.isdigit():
            continue
        mm, dd = int(raw_date[2:4]), int(raw_date[4:6])
        if not (1 <= mm <= 12 and 1 <= dd <= 31):
            dates_well_formed = False
            break

    expiry_after_birth = True
    if birth_date_iso and expiry_date_iso:
        try:
            import datetime as _dt2
            expiry_after_birth = (
                _dt2.date.fromisoformat(expiry_date_iso)
                > _dt2.date.fromisoformat(birth_date_iso)
            )
        except ValueError:
            expiry_after_birth = False

    country_codes_known = (
        _rc(issuing_country)["name"] != "Unknown"
        and _rc(nationality)["name"] != "Unknown"
    )

    sex_value_valid = sex in {"M", "F", "X", "<", ""}

    from .schema_helpers import _DOC_TYPE_MAP
    document_type_known = doc_type in _DOC_TYPE_MAP or doc_type.rstrip("<") in _DOC_TYPE_MAP

    return {
        "line_length_valid": line_length_valid,
        "dates_well_formed": dates_well_formed,
        "expiry_after_birth": expiry_after_birth,
        "country_codes_known": country_codes_known,
        "sex_value_valid": sex_value_valid,
        "document_type_known": document_type_known,
    }


def _build_validation(
    fmt: str,
    lines: list[str],
    doc_type: str,
    issuing_country: str,
    nationality: str,
    sex: str,
    birth_date_iso: Optional[str],
    expiry_date_iso: Optional[str],
    doc_valid: bool,
    birth_valid: bool,
    expiry_valid: bool,
    personal_valid: bool,
    composite_valid: bool,
    repaired: list[str],
) -> dict:
    structural = _structural_checks(
        fmt, lines, doc_type, issuing_country, nationality,
        sex, birth_date_iso, expiry_date_iso,
    )

    checkdigit_results = {
        "document_number_valid": doc_valid,
        "date_of_birth_valid": birth_valid,
        "date_of_expiry_valid": expiry_valid,
        "personal_number_valid": personal_valid,
        "composite_valid": composite_valid,
    }

    all_checks = {**checkdigit_results, **structural}
    failed = [k for k, v in all_checks.items() if v is False]
    mrz_overall_valid = len(failed) == 0

    return {
        **checkdigit_results,
        **structural,
        "mrz_overall_valid": mrz_overall_valid,
        "failed_checks": failed,
    }


@dataclass
class MRZResult:
    document_type: str
    issuing_country: str
    document_number: str
    nationality: str
    birth_date_raw: str
    birth_date_iso: Optional[str]
    sex: str
    expiry_date_raw: str
    expiry_date_iso: Optional[str]
    personal_number: str
    surname: str
    given_names: str
    name_dict: dict = field(default_factory=dict)
    validation: dict = field(default_factory=dict)
    auto_repaired_fields: list[str] = field(default_factory=list)

def _validate_and_repair(value: str, expected_cd: str, field_name: str, repaired: list[str]) -> tuple[str, bool]:
    if check_digit_valid(value, expected_cd):
        return value, True
    fixed = _repair_field(value, expected_cd)
    if fixed is not None:
        repaired.append(field_name)
        return fixed, True
    return value, False

def _repair_alpha3(code: str, field_name: str, repaired: list[str]) -> str:
    if not any(c.isdigit() for c in code):
        return code
    fixed = _repair_country_digits(code)
    if fixed != code:
        repaired.append(field_name)
    return fixed

def parse_td3(line1: str, line2: str) -> MRZResult:
    line1 = line1.ljust(44, "<")[:44]
    line2 = line2.ljust(44, "<")[:44]
    repaired: list[str] = []

    _FILLER_LOOKALIKES = {"V", "U", "W", "N", "M", "T"}
    line1_chars = list(line1)
    if line1_chars[0] in "PICVA" and line1_chars[1] != "<":
        from .country_lookup import resolve_country as _rc
        cand_at_1 = _repair_country_digits("".join(line1_chars[1:4]))
        cand_at_2 = _repair_country_digits("".join(line1_chars[2:5]))
        if _rc(cand_at_1)["name"] != "Unknown":
            line1_chars.insert(1, "<")
            line1 = "".join(line1_chars[:44]).ljust(44, "<")[:44]
            repaired.append("document_type")
        elif _rc(cand_at_2)["name"] != "Unknown":
            line1_chars[1] = "<"
            line1 = "".join(line1_chars)
            repaired.append("document_type")
        elif line1_chars[1] in _FILLER_LOOKALIKES:
            line1_chars[1] = "<"
            line1 = "".join(line1_chars)
            repaired.append("document_type")

    doc_type = line1[0:2].replace("<", "").strip()
    issuing = _repair_alpha3(line1[2:5], "issuing_country", repaired)
    surname, given, name_dict = parse_name(line1[5:44], repaired)

    doc_number = line2[0:9]
    doc_number_cd = line2[9]
    nationality = _repair_alpha3(line2[10:13], "nationality", repaired)
    birth = line2[13:19]
    birth_cd = line2[19]
    sex = line2[20].replace("<", "")
    expiry = line2[21:27]
    expiry_cd = line2[27]
    personal = line2[28:42]
    personal_cd = line2[42]
    composite_cd = line2[43]

    doc_number, doc_valid = _validate_and_repair(doc_number, doc_number_cd, "document_number", repaired)
    birth, birth_valid = _validate_and_repair(birth, birth_cd, "date_of_birth", repaired)
    expiry, expiry_valid = _validate_and_repair(expiry, expiry_cd, "date_of_expiry", repaired)

    if personal.replace("<", "") == "":
        personal_valid = personal_cd in ("<", "0")
    else:
        personal, personal_valid = _validate_and_repair(personal, personal_cd, "personal_number", repaired)

    composite_data = line2[0:10] + line2[13:20] + line2[21:43]
    composite_valid = check_digit_valid(composite_data, composite_cd)

    birth_iso = parse_date(birth, is_birth=True)
    expiry_iso = parse_date(expiry, is_birth=False)
    doc_type_final = doc_type or "P"

    validation = _build_validation(
        "TD3", [line1, line2], doc_type_final, issuing, nationality, sex,
        birth_iso, expiry_iso, doc_valid, birth_valid, expiry_valid,
        personal_valid, composite_valid, repaired,
    )

    return MRZResult(
        document_type=doc_type_final,
        issuing_country=issuing,
        document_number=doc_number.replace("<", ""),
        nationality=nationality,
        birth_date_raw=birth,
        birth_date_iso=birth_iso,
        sex=sex,
        expiry_date_raw=expiry,
        expiry_date_iso=expiry_iso,
        personal_number=personal.replace("<", ""),
        surname=surname,
        given_names=given,
        name_dict=name_dict,
        validation=validation,
        auto_repaired_fields=repaired,
    )

def parse_td2(line1: str, line2: str) -> MRZResult:
    line1 = line1.ljust(36, "<")[:36]
    line2 = line2.ljust(36, "<")[:36]
    repaired: list[str] = []

    doc_type = line1[0:2].replace("<", "").strip()
    issuing = _repair_alpha3(line1[2:5], "issuing_country", repaired)
    surname, given, name_dict = parse_name(line1[5:36], repaired)

    doc_number = line2[0:9]
    doc_number_cd = line2[9]
    nationality = _repair_alpha3(line2[10:13], "nationality", repaired)
    birth = line2[13:19]
    birth_cd = line2[19]
    sex = line2[20].replace("<", "")
    expiry = line2[21:27]
    expiry_cd = line2[27]
    optional = line2[28:35]
    composite_cd = line2[35]

    doc_number, doc_valid = _validate_and_repair(doc_number, doc_number_cd, "document_number", repaired)
    birth, birth_valid = _validate_and_repair(birth, birth_cd, "date_of_birth", repaired)
    expiry, expiry_valid = _validate_and_repair(expiry, expiry_cd, "date_of_expiry", repaired)

    composite_data = line2[0:10] + line2[13:20] + line2[21:35]
    composite_valid = check_digit_valid(composite_data, composite_cd)

    birth_iso = parse_date(birth, is_birth=True)
    expiry_iso = parse_date(expiry, is_birth=False)
    doc_type_final = doc_type or "I"

    validation = _build_validation(
        "TD2", [line1, line2], doc_type_final, issuing, nationality, sex,
        birth_iso, expiry_iso, doc_valid, birth_valid, expiry_valid,
        True, composite_valid, repaired,
    )

    return MRZResult(
        document_type=doc_type_final,
        issuing_country=issuing,
        document_number=doc_number.replace("<", ""),
        nationality=nationality,
        birth_date_raw=birth,
        birth_date_iso=birth_iso,
        sex=sex,
        expiry_date_raw=expiry,
        expiry_date_iso=expiry_iso,
        personal_number=optional.replace("<", ""),
        surname=surname,
        given_names=given,
        name_dict=name_dict,
        validation=validation,
        auto_repaired_fields=repaired,
    )

def parse_td1(line1: str, line2: str, line3: str) -> MRZResult:
    line1 = line1.ljust(30, "<")[:30]
    line2 = line2.ljust(30, "<")[:30]
    line3 = line3.ljust(30, "<")[:30]
    repaired: list[str] = []

    doc_type = line1[0:2].replace("<", "").strip()
    issuing = _repair_alpha3(line1[2:5], "issuing_country", repaired)
    doc_number = line1[5:14]
    doc_number_cd = line1[14]
    optional1 = line1[15:30]

    birth = line2[0:6]
    birth_cd = line2[6]
    sex = line2[7].replace("<", "")
    expiry = line2[8:14]
    expiry_cd = line2[14]
    nationality = _repair_alpha3(line2[15:18], "nationality", repaired)
    optional2 = line2[18:29]
    composite_cd = line2[29]

    surname, given, name_dict = parse_name(line3, repaired)

    doc_number, doc_valid = _validate_and_repair(doc_number, doc_number_cd, "document_number", repaired)
    birth, birth_valid = _validate_and_repair(birth, birth_cd, "date_of_birth", repaired)
    expiry, expiry_valid = _validate_and_repair(expiry, expiry_cd, "date_of_expiry", repaired)

    composite_data = line1[5:30] + line2[0:7] + line2[8:15] + line2[18:29]
    composite_valid = check_digit_valid(composite_data, composite_cd)

    birth_iso = parse_date(birth, is_birth=True)
    expiry_iso = parse_date(expiry, is_birth=False)
    doc_type_final = doc_type or "I"

    validation = _build_validation(
        "TD1", [line1, line2, line3], doc_type_final, issuing, nationality, sex,
        birth_iso, expiry_iso, doc_valid, birth_valid, expiry_valid,
        True, composite_valid, repaired,
    )

    return MRZResult(
        document_type=doc_type_final,
        issuing_country=issuing,
        document_number=doc_number.replace("<", ""),
        nationality=nationality,
        birth_date_raw=birth,
        birth_date_iso=birth_iso,
        sex=sex,
        expiry_date_raw=expiry,
        expiry_date_iso=expiry_iso,
        personal_number=(optional1 + optional2).replace("<", ""),
        surname=surname,
        given_names=given,
        name_dict=name_dict,
        validation=validation,
        auto_repaired_fields=repaired,
    )

def parse_mrz(lines: list[str]) -> Optional[MRZResult]:
    lines = [l.strip().upper() for l in lines if l.strip()]
    if len(lines) == 2:
        avg = (len(lines[0]) + len(lines[1])) / 2
        if avg >= 40:
            return parse_td3(lines[0], lines[1])
        return parse_td2(lines[0], lines[1])
    if len(lines) == 3:
        return parse_td1(lines[0], lines[1], lines[2])
    return None

