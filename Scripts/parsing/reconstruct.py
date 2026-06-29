"""MRZ line alignment and scoring helpers.

Despite living under Scripts/OCR (the original multi-engine package), this module
is now a pure utility set shared by the Tesseract pipeline: line normalisation,
sliding-window alignment, format detection, and check-digit-based validation
scoring. It has no OCR-engine dependency.
"""
from __future__ import annotations

# Canonical line lengths per format.
TD1_LINES, TD1_LEN = 3, 30
TD2_LINES, TD2_LEN = 2, 36
TD3_LINES, TD3_LEN = 2, 44

# Characters OCR commonly uses instead of '<'.
_FILLER_MAP = str.maketrans("k([ ", "<<<<")

_VALID_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")


def _normalize(text: str) -> str:
    text = text.upper().translate(_FILLER_MAP)
    return "".join(c if c in _VALID_CHARS else "<" for c in text)


def _snap_line(text: str, length: int) -> str:
    text = text.ljust(length, "<")[:length]
    return text


def _align_line(text: str, length: int, line_idx: int) -> str:
    text = _normalize(text)
    if not text:
        return "<" * length

    # Strip leading '<' — OCR often adds leading fillers from strip padding.
    stripped = text.lstrip("<")
    if stripped:
        text = stripped

    # Sliding-window: find offset that maximises non-'<' in [offset, offset+length).
    best_offset = 0
    best_score = -1
    search_range = max(1, len(text) - length + 1)
    for offset in range(search_range):
        window = text[offset:offset + length]
        score = sum(1 for c in window if c != "<")
        if score > best_score:
            best_score = score
            best_offset = offset

    aligned = text[best_offset:best_offset + length]
    return _snap_line(aligned, length)


def _best_aligned_pair(line1_raw: str, line2_raw: str, length: int) -> tuple[str, str]:
    from .mrz_parse import parse_mrz

    line1 = _align_line(line1_raw, length, 0)
    line2_norm = _normalize(line2_raw).lstrip("<")

    search_range = max(1, len(line2_norm) - length + 1)
    best_offset = 0
    best_passes = -1

    for offset in range(search_range):
        candidate = _snap_line(line2_norm[offset:offset + length], length)
        try:
            result = parse_mrz([line1, candidate])
            if result is None:
                passes = 0
            else:
                passes = sum(1 for v in result.validation.values() if v is True)
        except Exception:
            passes = 0
        if passes > best_passes:
            best_passes = passes
            best_offset = offset

    return line1, _snap_line(line2_norm[best_offset:best_offset + length], length)


def detect_format(lines: list[str]) -> tuple[str, int]:
    n = len(lines)
    avg_len = sum(len(l) for l in lines) / max(n, 1)

    if n == 3:
        return "TD1", TD1_LEN
    if n == 2:
        if avg_len >= 40:
            return "TD3", TD3_LEN
        return "TD2", TD2_LEN
    # Fallback: pick by closest line length.
    dists = {
        "TD3": abs(avg_len - TD3_LEN),
        "TD2": abs(avg_len - TD2_LEN),
        "TD1": abs(avg_len - TD1_LEN),
    }
    best = min(dists, key=dists.get)
    return best, {"TD3": TD3_LEN, "TD2": TD2_LEN, "TD1": TD1_LEN}[best]


def _validation_score(lines: list[str]) -> tuple[int, float, float]:
    """Return (check_digit_passes, composite_bonus, char_density) for a line pair.

    Used by the Tesseract pipeline to rank candidate line selections: pairs that
    pass more check digits (and have a valid composite) score higher.
    """
    if not lines:
        return (0, 0.0, 0.0)
    total_chars = sum(len(l) for l in lines)
    non_filler = sum(c != "<" for l in lines for c in l)
    if total_chars == 0:
        return (0, 0.0, 0.0)
    density = non_filler / total_chars
    if density < 0.15:
        return (0, 0.0, density)
    try:
        from .mrz_parse import parse_mrz
        result = parse_mrz(lines)
        cd_passes = 0
        composite_passed = False
        if result is not None:
            repaired = set(result.auto_repaired_fields)
            meaningful_fields = {
                "document_number_valid": result.document_number,
                "date_of_birth_valid": result.birth_date_raw,
                "date_of_expiry_valid": result.expiry_date_raw,
                "personal_number_valid": result.personal_number,
                "composite_valid": None,
            }
            field_to_repair_key = {
                "document_number_valid": "document_number",
                "date_of_birth_valid": "date_of_birth",
                "date_of_expiry_valid": "date_of_expiry",
                "personal_number_valid": "personal_number",
                "composite_valid": "composite",
            }
            for key, value in meaningful_fields.items():
                val = result.validation.get(key)
                if val is not True:
                    continue
                repair_key = field_to_repair_key[key]
                if repair_key in repaired:
                    continue
                if value is not None and value.replace("<", "") == "":
                    continue
                cd_passes += 1
                if key == "composite_valid":
                    composite_passed = True

        composite_bonus = 1.0 if composite_passed else 0.0
        return (cd_passes, composite_bonus, density)
    except Exception:
        return (0, 0.0, density)