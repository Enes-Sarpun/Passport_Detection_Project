"""Shared MRZ constants and OCR-confusion helpers.

Single source of truth for values that were previously duplicated across the
detection, OCR and parsing modules: the ICAO 9303 character set, the canonical
line lengths per document format, and the digit/letter confusion maps used for
check-digit self-repair.
"""
from __future__ import annotations

# ICAO 9303 MRZ character set: A–Z, 0–9 and the '<' filler.
MRZ_CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
MRZ_VALID_CHARS = frozenset(MRZ_CHARSET)

# Canonical line lengths and line counts per MRZ document format.
TD1_LEN, TD1_LINES = 30, 3
TD2_LEN, TD2_LINES = 36, 2
TD3_LEN, TD3_LINES = 44, 2

FORMAT_LINE_LEN = {"TD1": TD1_LEN, "TD2": TD2_LEN, "TD3": TD3_LEN}
FORMAT_LINE_COUNT = {"TD1": TD1_LINES, "TD2": TD2_LINES, "TD3": TD3_LINES}


def format_for_max_len(max_len: int) -> str:
    """Infer the MRZ format from the longest genuine line length.

    Uses the pipeline-wide thresholds: >=40 → TD3, >=33 → TD2, else TD1.
    """
    if max_len >= 40:
        return "TD3"
    if max_len >= 33:
        return "TD2"
    return "TD1"


# OCR digit→letter confusions for alphabetic-only MRZ fields (country codes, names).
DIGIT_TO_LETTER: dict[str, str] = {
    "0": "O",
    "1": "I",
    "5": "S",
    "8": "B",
    "2": "Z",
    "6": "G",
}

# Extended digit→letter map for class-constraint coercion of letter-only fields
# (adds the rarer 3→E / 4→A swaps on top of DIGIT_TO_LETTER).
DIGIT_TO_LETTER_EXTENDED: dict[str, str] = {**DIGIT_TO_LETTER, "3": "E", "4": "A"}

# OCR letter→digit confusions for numeric-only MRZ fields (dates, check digits).
LETTER_TO_DIGIT: dict[str, str] = {
    "O": "0",
    "I": "1",
    "S": "5",
    "B": "8",
    "Z": "2",
    "G": "6",
    "D": "0",
    "Q": "0",
    "L": "1",
}


def repair_digits_to_letters(text: str, mapping: dict[str, str] = DIGIT_TO_LETTER) -> str:
    """Map OCR digit misreads back to letters in an alphabetic MRZ field."""
    return "".join(mapping.get(c, c) for c in text)
