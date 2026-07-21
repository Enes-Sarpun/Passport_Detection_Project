from __future__ import annotations
import base64
import os
from typing import Optional
import cv2
import numpy as np

# Reuse the exact MRZ normalisation Tesseract uses, so both engines emit lines
# in the same charset/length format for a fair check-digit comparison.
from .engine import _clean_text, _snap, _MRZ_CHARSET

_GOOGLE_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"
_TIMEOUT = 15  # seconds; a slow fallback must not hang the request


def is_available() -> bool:
    """True if a cloud OCR provider is configured via env."""
    return bool(os.environ.get("GOOGLE_VISION_API_KEY"))


def _looks_like_mrz(line: str) -> bool:
    """A genuine MRZ line is long and dominated by MRZ charset + '<' filler."""
    stripped = line.strip()
    if len(stripped) < 20:
        return False
    valid = sum(1 for c in stripped if c in _MRZ_CHARSET)
    return valid / len(stripped) >= 0.6


def _extract_mrz_lines(full_text: str) -> list[str]:
    """Pull MRZ-looking lines out of the raw OCR text, normalised to the same
    charset/length Tesseract emits. Keeps the 2-3 longest MRZ-like lines."""
    candidates: list[str] = []
    for raw_line in full_text.splitlines():
        collapsed = raw_line.replace(" ", "")
        if not _looks_like_mrz(collapsed):
            continue
        cleaned = _clean_text(collapsed)
        # Snap to the nearest standard MRZ width (44 TD3 / 36 TD2 / 30 TD1).
        target = min((44, 36, 30), key=lambda t: abs(len(cleaned) - t))
        candidates.append(_snap(cleaned, target))

    # Longest lines first (the real data lines), cap at 3 (TD1 max).
    candidates.sort(key=lambda s: len(s.rstrip("<")), reverse=True)
    return candidates[:3]


def _google_vision_text(image_bgr: np.ndarray) -> Optional[str]:
    """Call Google Vision TEXT_DETECTION on the given image; return raw text or
    None on any failure (missing key, encode error, network, bad response)."""
    api_key = os.environ.get("GOOGLE_VISION_API_KEY")
    if not api_key:
        return None

    ok, buf = cv2.imencode(".jpg", image_bgr)
    if not ok:
        return None
    content = base64.b64encode(buf.tobytes()).decode("ascii")

    payload = {
        "requests": [{
            "image": {"content": content},
            "features": [{"type": "TEXT_DETECTION"}],
        }]
    }

    try:
        import requests
        resp = requests.post(
            _GOOGLE_ENDPOINT,
            params={"key": api_key},
            json=payload,
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        responses = data.get("responses") or []
        if not responses:
            return None
        annotation = responses[0].get("fullTextAnnotation") or {}
        text = annotation.get("text")
        if text:
            return text
        # Fallback to the first textAnnotation description if fullText is absent.
        texts = responses[0].get("textAnnotations") or []
        return texts[0].get("description") if texts else None
    except Exception:
        # Best-effort rescue: any failure just means "no fallback result".
        return None


def read_mrz(image_bgr: np.ndarray) -> Optional[list[str]]:
    """Read MRZ lines from a (cropped MRZ) image via the cloud provider.
    Returns a list of normalised MRZ lines, or None if unavailable/failed."""
    if not is_available():
        return None
    text = _google_vision_text(image_bgr)
    if not text:
        return None
    lines = _extract_mrz_lines(text)
    return lines or None
