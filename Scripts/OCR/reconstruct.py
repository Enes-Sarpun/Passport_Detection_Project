from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from .ocr import OcrResult

# Canonical line lengths per format.
TD1_LINES, TD1_LEN = 3, 30
TD2_LINES, TD2_LEN = 2, 36
TD3_LINES, TD3_LEN = 2, 44

# Characters OCR commonly uses instead of '<'.
_FILLER_MAP = str.maketrans("ckC([ ", "<<<<<<")

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

    # Strip leading '<' from all lines — OCR often adds 1-3 leading '<' due
    # to strip padding capturing a tiny bit of the left whitespace border.
    stripped = text.lstrip("<")
    if stripped:
        text = stripped

    # Sliding-window: find offset that maximises non-'<' in [offset, offset+length).
    # This handles cases where leading junk could not be fully stripped.
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

def reconstruct(ocr_results: list[OcrResult]) -> ReconstructedMRZ:
    if not ocr_results:
        return ReconstructedMRZ(fmt="TD3", lines=[], line_length=TD3_LEN)

    # Use the result with the longest max line to infer format.
    # Confidence alone is unreliable: high-confidence results may have eaten '<'
    # fillers and appear shorter than they actually are.
    def _max_line_len(r: OcrResult) -> int:
        return max((len(_normalize(l.text)) for l in r.lines), default=0)

    best = max(ocr_results, key=lambda r: (_max_line_len(r), len(r.lines)))
    raw_lines = [_normalize(l.text) for l in best.lines]
    fmt, line_len = detect_format(raw_lines)

    n_lines = {"TD1": 3, "TD2": 2, "TD3": 2}[fmt]
    result_lines: list[str] = []

    for line_idx in range(n_lines):
        # Collect this line's text and confidence from each OCR candidate.
        cand_texts: list[str] = []
        cand_confs: list[float] = []
        for ocr_r in ocr_results:
            extracted = _extract_lines(ocr_r, n_lines)
            # Align before voting to correct OCR-induced leading drift.
            aligned = _align_line(extracted[line_idx], line_len, line_idx)
            cand_texts.append(aligned)
            cand_confs.append(ocr_r.mean_confidence if ocr_r.lines else 0.01)

        voted = column_vote(cand_texts, cand_confs, line_len)
        result_lines.append(voted)

    return ReconstructedMRZ(fmt=fmt, lines=result_lines, line_length=line_len)


