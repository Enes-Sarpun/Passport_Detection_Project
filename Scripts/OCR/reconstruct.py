from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from .ocr import OcrResult

# Canonical line lengths per format.
TD1_LINES, TD1_LEN = 3, 30
TD2_LINES, TD2_LEN = 2, 36
TD3_LINES, TD3_LEN = 2, 44

# Characters OCR commonly uses instead of '<'.
# Note: 'C' and 'c' are NOT mapped — 'C' is a valid MRZ character that appears
# in document numbers, country codes (e.g. SC1488168), and names.
_FILLER_MAP = str.maketrans("k([ ", "<<<<")

# MRZ line-start anchors: first non-< character expected at position 0.
# TD3 line 1 always starts with P (passport) or other doc-type letter.
# TD3 line 2 always starts with a digit or uppercase letter (doc number).
# We use these to strip leading junk characters introduced by OCR.
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
    from .mrz_parse import parse_mrz, check_digit_valid

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

def _extract_lines(ocr_result: OcrResult, n_lines: int) -> list[str]:
    lines = [l.text for l in ocr_result.lines[:n_lines]]
    while len(lines) < n_lines:
        lines.append("")
    return lines

def column_vote(candidates: list[str], confidences: list[float], length: int) -> str:
    snapped = [_snap_line(c, length) for c in candidates]
    result = []
    for col in range(length):
        votes: Counter = Counter()
        for line, conf in zip(snapped, confidences):
            ch = line[col]
            # Down-weight '<' so real characters win over filler ambiguity.
            weight = conf * 0.5 if ch == "<" else conf
            votes[ch] += weight
        result.append(votes.most_common(1)[0][0])
    return "".join(result)

@dataclass
class ReconstructedMRZ:
    fmt: str
    lines: list[str]
    line_length: int

# Minimum per-candidate confidence to participate in column voting.
# Candidates below this threshold produce mostly noise and drag down the vote.
_MIN_VOTE_CONFIDENCE = 0.15

def _validation_score(lines: list[str]) -> float:
    if not lines:
        return 0.0
    total_chars = sum(len(l) for l in lines)
    non_filler = sum(c != "<" for l in lines for c in l)
    if total_chars == 0:
        return 0.0
    density = non_filler / total_chars
    # Exclude near-empty results — they trivially pass check digits.
    if density < 0.15:
        return 0.0
    try:
        from .mrz_parse import parse_mrz
        result = parse_mrz(lines)
        cd_passes = 0
        composite_passed = False
        if result is not None:
            repaired = set(result.auto_repaired_fields)
            # field name → the actual value that was validated
            meaningful_fields = {
                "document_number_valid": result.document_number,
                "date_of_birth_valid": result.birth_date_raw,
                "date_of_expiry_valid": result.expiry_date_raw,
                "personal_number_valid": result.personal_number,
                "composite_valid": None,  # composite covers the whole line 2
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
                # Skip if repaired from garbage.
                if repair_key in repaired:
                    continue
                # Skip if the field value is all-filler — trivially passes but
                # carries no information (OCR just read nothing there).
                if value is not None and value.replace("<", "") == "":
                    continue
                cd_passes += 1
                if key == "composite_valid":
                    composite_passed = True

        # Composite check digit covers all of line 2 and is the strongest signal
        # of a correctly-read MRZ.  Reward it with a large bonus; candidates that
        # fail composite almost certainly have wrong alignment or garbage OCR.
        composite_bonus = 20.0 if composite_passed else 0.0
        return cd_passes * 10.0 + density * 5.0 + composite_bonus
    except Exception:
        return density * 5.0


def reconstruct(ocr_results: list[OcrResult]) -> ReconstructedMRZ:
    if not ocr_results:
        return ReconstructedMRZ(fmt="TD3", lines=[], line_length=TD3_LEN)

    # Filter out near-zero-confidence candidates — they add noise to the vote.
    usable = [r for r in ocr_results if r.lines and r.mean_confidence >= _MIN_VOTE_CONFIDENCE]
    if not usable:
        usable = [max(ocr_results, key=lambda r: r.mean_confidence if r.lines else 0.0)]

    # Use the result with the longest max line to infer format.
    def _max_line_len(r: OcrResult) -> int:
        return max((len(_normalize(l.text)) for l in r.lines), default=0)

    best_for_format = max(usable, key=lambda r: (_max_line_len(r), len(r.lines)))
    raw_lines = [_normalize(l.text) for l in best_for_format.lines]
    fmt, line_len = detect_format(raw_lines)
    n_lines = {"TD1": 3, "TD2": 2, "TD3": 2}[fmt]

    # --- Per-candidate alignment & validation scoring ---
    # For 2-line formats (TD3/TD2) we use check-digit-aware line2 alignment:
    # try all plausible offsets and pick the one that maximises validation passes.
    # This prevents OCR engines that read past the MRZ boundary from landing on
    # a wrong offset that happens to pass a check digit by coincidence.
    # Composite score = cd_passes_without_repair * 10 + density * 5 + ocr_conf * 3
    scored: list[tuple[float, float, list[str]]] = []
    for ocr_r in usable:
        extracted = _extract_lines(ocr_r, n_lines)
        if n_lines == 2:
            line1, line2 = _best_aligned_pair(extracted[0], extracted[1], line_len)
            aligned = [line1, line2]
        else:
            aligned = [_align_line(extracted[i], line_len, i) for i in range(n_lines)]
        vscore = _validation_score(aligned)
        # Composite ranking: OCR confidence (primary) + structural validation (secondary).
        # vscore uses a 0.3 multiplier so that a high-confidence clean read (paddle_det
        # at 0.86) beats a low-confidence read that happens to pass more check digits
        # by coincidence. The 10× confidence weight reflects that engine confidence
        # is calibrated and reliable, while check-digit coincidences are common in
        # garbled OCR output.
        composite = vscore * 0.3 + ocr_r.mean_confidence * 10.0
        scored.append((composite, ocr_r.mean_confidence, aligned))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best_score, best_conf, best_lines = scored[0]

    # Use the best candidate directly — voting across noisy candidates degrades quality.
    if best_score > 0.0:
        return ReconstructedMRZ(fmt=fmt, lines=best_lines, line_length=line_len)

    # All candidates scored zero — fall back to confidence-weighted column voting.
    result_lines: list[str] = []
    for line_idx in range(n_lines):
        cand_texts = [s[2][line_idx] for s in scored]
        cand_confs = [max(s[1], 0.01) for s in scored]
        voted = column_vote(cand_texts, cand_confs, line_len)
        result_lines.append(voted)

    return ReconstructedMRZ(fmt=fmt, lines=result_lines, line_length=line_len)


