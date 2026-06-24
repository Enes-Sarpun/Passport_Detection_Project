from __future__ import annotations
import datetime as _dt
import itertools
from dataclasses import dataclass, field
from typing import Optional

MRZ_CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"

# Standard OCR confusion swaps used during check-digit self-repair.
# Each tuple is a bidirectional pair; repair tries replacing one with the other.
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
        # Expiry: assume 2000s unless clearly far past.
        century = 2000 if yy <= current_yy + 50 else 1900
    year = century + yy
    try:
        return _dt.date(year, mm, dd).isoformat()
    except ValueError:
        return None

def parse_name(name_field: str) -> tuple[str, str]:
    parts = name_field.split("<<", 1)
    surname = parts[0].replace("<", " ").strip()
    given = ""
    if len(parts) > 1:
        given = parts[1].replace("<", " ").strip()
    given = " ".join(given.split())
    surname = " ".join(surname.split())
    return surname, given

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

def parse_td3(line1: str, line2: str) -> MRZResult:
    line1 = line1.ljust(44, "<")[:44]
    line2 = line2.ljust(44, "<")[:44]
    repaired: list[str] = []

    doc_type = line1[0:2].replace("<", "").strip()
    issuing = line1[2:5]
    surname, given = parse_name(line1[5:44])

    doc_number = line2[0:9]
    doc_number_cd = line2[9]
    nationality = line2[10:13]
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

    # Personal number may be all-filler; that's valid with check digit 0/<.
    if personal.replace("<", "") == "":
        personal_valid = personal_cd in ("<", "0")
    else:
        personal, personal_valid = _validate_and_repair(personal, personal_cd, "personal_number", repaired)

    composite_data = line2[0:10] + line2[13:20] + line2[21:43]
    composite_valid = check_digit_valid(composite_data, composite_cd)

    return MRZResult(
        document_type=doc_type or "P",
        issuing_country=issuing,
        document_number=doc_number.replace("<", ""),
        nationality=nationality,
        birth_date_raw=birth,
        birth_date_iso=parse_date(birth, is_birth=True),
        sex=sex,
        expiry_date_raw=expiry,
        expiry_date_iso=parse_date(expiry, is_birth=False),
        personal_number=personal.replace("<", ""),
        surname=surname,
        given_names=given,
        validation={
            "document_number_valid": doc_valid,
            "date_of_birth_valid": birth_valid,
            "date_of_expiry_valid": expiry_valid,
            "personal_number_valid": personal_valid,
            "composite_valid": composite_valid,
        },
        auto_repaired_fields=repaired,
    )

def parse_td2(line1: str, line2: str) -> MRZResult:
    line1 = line1.ljust(36, "<")[:36]
    line2 = line2.ljust(36, "<")[:36]
    repaired: list[str] = []

    doc_type = line1[0:2].replace("<", "").strip()
    issuing = line1[2:5]
    surname, given = parse_name(line1[5:36])

    doc_number = line2[0:9]
    doc_number_cd = line2[9]
    nationality = line2[10:13]
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

    return MRZResult(
        document_type=doc_type or "I",
        issuing_country=issuing,
        document_number=doc_number.replace("<", ""),
        nationality=nationality,
        birth_date_raw=birth,
        birth_date_iso=parse_date(birth, is_birth=True),
        sex=sex,
        expiry_date_raw=expiry,
        expiry_date_iso=parse_date(expiry, is_birth=False),
        personal_number=optional.replace("<", ""),
        surname=surname,
        given_names=given,
        validation={
            "document_number_valid": doc_valid,
            "date_of_birth_valid": birth_valid,
            "date_of_expiry_valid": expiry_valid,
            "composite_valid": composite_valid,
        },
        auto_repaired_fields=repaired,
    )

def parse_td1(line1: str, line2: str, line3: str) -> MRZResult:
    line1 = line1.ljust(30, "<")[:30]
    line2 = line2.ljust(30, "<")[:30]
    line3 = line3.ljust(30, "<")[:30]
    repaired: list[str] = []

    doc_type = line1[0:2].replace("<", "").strip()
    issuing = line1[2:5]
    doc_number = line1[5:14]
    doc_number_cd = line1[14]
    optional1 = line1[15:30]

    birth = line2[0:6]
    birth_cd = line2[6]
    sex = line2[7].replace("<", "")
    expiry = line2[8:14]
    expiry_cd = line2[14]
    nationality = line2[15:18]
    optional2 = line2[18:29]
    composite_cd = line2[29]

    surname, given = parse_name(line3)

    doc_number, doc_valid = _validate_and_repair(doc_number, doc_number_cd, "document_number", repaired)
    birth, birth_valid = _validate_and_repair(birth, birth_cd, "date_of_birth", repaired)
    expiry, expiry_valid = _validate_and_repair(expiry, expiry_cd, "date_of_expiry", repaired)

    composite_data = line1[5:30] + line2[0:7] + line2[8:15] + line2[18:29]
    composite_valid = check_digit_valid(composite_data, composite_cd)

    return MRZResult(
        document_type=doc_type or "I",
        issuing_country=issuing,
        document_number=doc_number.replace("<", ""),
        nationality=nationality,
        birth_date_raw=birth,
        birth_date_iso=parse_date(birth, is_birth=True),
        sex=sex,
        expiry_date_raw=expiry,
        expiry_date_iso=parse_date(expiry, is_birth=False),
        personal_number=(optional1 + optional2).replace("<", ""),
        surname=surname,
        given_names=given,
        validation={
            "document_number_valid": doc_valid,
            "date_of_birth_valid": birth_valid,
            "date_of_expiry_valid": expiry_valid,
            "composite_valid": composite_valid,
        },
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

