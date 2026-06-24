"""OCR backend for MRZ recognition — EasyOCR (GPU, primary) + PaddleOCR (secondary).

Strategy:
  - Line positions are detected via CV horizontal-projection profile (no OCR needed).
  - EasyOCR runs first with a strict MRZ allowlist and GPU acceleration.
  - PaddleOCR runs as a fallback / second vote source.
  - run_ocr() returns BOTH results so reconstruct.column_vote() can merge them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import numpy as np

os.environ.setdefault("FLAGS_use_mkldnn", "0")

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

_PADDLE_INSTANCE = None
_EASY_INSTANCE = None

# ---------------------------------------------------------------------------
# MRZ character constants
# ---------------------------------------------------------------------------

_JUNK_MAP = str.maketrans(
    "OoIilBsSzZqQdD .,;:!?()-",
    "001188552200DD<<<<<<<<<<",
)
_VALID_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")
_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
_MRZ_LENGTHS = (30, 36, 44)

# Minimum strip height in pixels — critical for OCR accuracy on small crops.
# MRZ OCR-B characters need at least 64 px cap-height; 80 is a safe minimum.
_MIN_STRIP_H = 80


# ---------------------------------------------------------------------------
# Lazy engine loaders
# ---------------------------------------------------------------------------

def _get_paddle():
    global _PADDLE_INSTANCE
    if _PADDLE_INSTANCE is None:
        from paddleocr import PaddleOCR
        _PADDLE_INSTANCE = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
    return _PADDLE_INSTANCE


def _get_easy():
    """Lazy-load EasyOCR reader with GPU and English model."""
    global _EASY_INSTANCE
    if _EASY_INSTANCE is None:
        import easyocr
        _EASY_INSTANCE = easyocr.Reader(["en"], gpu=True, verbose=False)
    return _EASY_INSTANCE


# ---------------------------------------------------------------------------
# Text cleaning helpers
# ---------------------------------------------------------------------------

def _clean_char(c: str) -> str:
    c = c.upper()
    if c in _VALID_CHARS:
        return c
    mapped = c.translate(_JUNK_MAP)
    return mapped if mapped in _VALID_CHARS else "<"


def _clean_text(raw: str) -> str:
    return "".join(_clean_char(c) for c in raw.upper().strip())


def _nearest_mrz_length(n: int) -> int:
    return min(_MRZ_LENGTHS, key=lambda length: abs(length - n))


def _snap(text: str, target: int) -> str:
    """Pad with '<' or truncate to exactly `target` characters."""
    if len(text) < target:
        return text + "<" * (target - len(text))
    return text[:target]


def _upscale_strip(strip: np.ndarray, min_h: int = _MIN_STRIP_H) -> np.ndarray:
    """Upscale a line strip so its height is at least min_h pixels."""
    h = strip.shape[0]
    if h < min_h:
        scale = min_h / h
        strip = cv2.resize(strip, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_CUBIC)
    return strip


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class OcrLine:
    text: str
    confidence: float


@dataclass
class OcrResult:
    lines: list[OcrLine]
    engine: str = "unknown"

    @property
    def mean_confidence(self) -> float:
        if not self.lines:
            return 0.0
        return sum(ln.confidence for ln in self.lines) / len(self.lines)


# ---------------------------------------------------------------------------
# CV-based line position detection (replaces PaddleOCR detection pass)
# ---------------------------------------------------------------------------

def _detect_line_ys_cv(image: np.ndarray, n_lines: int) -> list[tuple[float, float]]:
    """Locate MRZ text-line y-extents using a horizontal projection profile.

    Much faster and more reliable than running PaddleOCR in detection mode:
    - Binarise with Otsu.
    - Sum ink pixels per row.
    - Find contiguous ink runs and pick the n_lines largest ones.
    Falls back to equal-height split if projection fails.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    img_h = gray.shape[0]

    # Invert so text pixels are white (high value).
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    row_sum = binary.sum(axis=1).astype(np.float32)

    # Smooth to bridge small within-line gaps.
    k = max(3, img_h // max(n_lines * 6, 1))
    kernel = np.ones(k) / k
    smoothed = np.convolve(row_sum, kernel, mode="same")

    # Threshold at 15 % of peak ink density.
    threshold = smoothed.max() * 0.15
    ink = smoothed > threshold

    # Extract contiguous ink runs.
    regions: list[tuple[int, int]] = []
    in_run = False
    run_start = 0
    for i, has_ink in enumerate(ink):
        if has_ink and not in_run:
            run_start = i
            in_run = True
        elif not has_ink and in_run:
            regions.append((run_start, i))
            in_run = False
    if in_run:
        regions.append((run_start, img_h))

    if len(regions) >= n_lines:
        # Keep the n_lines tallest regions, then re-sort top-to-bottom.
        regions.sort(key=lambda r: r[1] - r[0], reverse=True)
        regions = regions[:n_lines]
        regions.sort(key=lambda r: r[0])
        return [(float(y0), float(y1)) for y0, y1 in regions]

    # Fallback: equal split.
    step = img_h / n_lines
    return [(i * step, (i + 1) * step) for i in range(n_lines)]


def _extract_strips(
    image: np.ndarray,
    line_ys: list[tuple[float, float]],
) -> list[np.ndarray]:
    """Slice the image into one horizontal strip per detected line."""
    img_h = image.shape[0]
    strips = []
    for y_top, y_bot in line_ys:
        # Add 10 % vertical padding to avoid clipping ascenders/descenders.
        pad = max(4, int((y_bot - y_top) * 0.10))
        y0 = max(0, int(y_top) - pad)
        y1 = min(img_h, int(y_bot) + pad)
        strips.append(image[y0:y1, :])
    return strips


# ---------------------------------------------------------------------------
# EasyOCR backend — primary engine
# ---------------------------------------------------------------------------

def _easy_read_strip(strip: np.ndarray, target_len: int) -> tuple[str, float]:
    """Read one MRZ line strip with EasyOCR (GPU, MRZ allowlist)."""
    reader = _get_easy()
    strip = _upscale_strip(strip)

    results = reader.readtext(
        strip,
        allowlist=_ALLOWLIST,
        detail=1,
        paragraph=False,
        # Tune detector for wide, single-line MRZ strips.
        min_size=10,
        text_threshold=0.5,
        low_text=0.3,
        link_threshold=0.3,
        width_ths=0.9,   # merge horizontally close detections
        height_ths=0.5,
    )

    if not results:
        return "<" * target_len, 0.0

    # Sort fragments left-to-right and concatenate.
    results.sort(key=lambda r: r[0][0][0])
    combined = "".join(r[1] for r in results)
    avg_conf = sum(r[2] for r in results) / len(results)

    return _snap(_clean_text(combined), target_len), float(avg_conf)


def _run_easy(image: np.ndarray, n_lines: int = 2) -> OcrResult:
    """Run EasyOCR over all MRZ lines detected via CV projection."""
    target_len = 30 if n_lines >= 3 else 44
    line_ys = _detect_line_ys_cv(image, n_lines)
    strips = _extract_strips(image, line_ys)
    lines: list[OcrLine] = []
    for strip in strips:
        if strip.shape[0] < 4:
            continue
        text, conf = _easy_read_strip(strip, target_len)
        lines.append(OcrLine(text=text, confidence=conf))
    return OcrResult(lines=lines, engine="easyocr")


# ---------------------------------------------------------------------------
# PaddleOCR backend — secondary engine (recognition-only, det=False)
# ---------------------------------------------------------------------------

def _paddle_read_strip(strip: np.ndarray, target_len: int) -> tuple[str, float]:
    """Read one MRZ line strip with PaddleOCR recognition-only mode."""
    paddle = _get_paddle()
    strip = _upscale_strip(strip)
    raw = paddle.ocr(strip, det=False, cls=False)
    if not raw or not raw[0]:
        return "<" * target_len, 0.0
    best_text, best_conf = max(raw[0], key=lambda x: x[1])
    return _snap(_clean_text(best_text), target_len), float(best_conf)


def _run_paddle(image: np.ndarray, n_lines: int = 2) -> OcrResult:
    """Run PaddleOCR over CV-detected line strips (no Paddle detection pass)."""
    target_len = 30 if n_lines >= 3 else 44
    line_ys = _detect_line_ys_cv(image, n_lines)   # CV, not Paddle det
    strips = _extract_strips(image, line_ys)
    lines: list[OcrLine] = []
    for strip in strips:
        if strip.shape[0] < 4:
            continue
        text, conf = _paddle_read_strip(strip, target_len)
        lines.append(OcrLine(text=text, confidence=conf))
    return OcrResult(lines=lines, engine="paddleocr")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_ocr(image: np.ndarray, n_lines: int = 2) -> list[OcrResult]:
    """Run EasyOCR (primary) and PaddleOCR (secondary) on one candidate image.

    Returns a list with up to 2 OcrResult objects — one per engine.
    Both results are fed into reconstruct.column_vote() for character-level
    majority voting.
    """
    results: list[OcrResult] = []

    # --- EasyOCR (GPU, primary) ---
    try:
        results.append(_run_easy(image, n_lines))
    except Exception as exc:
        # Don't crash the pipeline if EasyOCR fails on one candidate.
        results.append(OcrResult(lines=[], engine=f"easyocr_error:{exc}"))

    # --- PaddleOCR (secondary) ---
    try:
        results.append(_run_paddle(image, n_lines))
    except Exception as exc:
        results.append(OcrResult(lines=[], engine=f"paddleocr_error:{exc}"))

    return results


def run_ocr_multi(images: list[np.ndarray]) -> list[OcrResult]:
    """Run run_ocr() on every candidate image and collect all OcrResults."""
    results: list[OcrResult] = []
    for img in images:
        results.extend(run_ocr(img))
    return results