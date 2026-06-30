from __future__ import annotations
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np


def _resolve_tesseract_cmd() -> str:
    env = os.environ.get("TESSERACT_CMD")
    if env:
        return env
    found = shutil.which("tesseract")
    if found:
        return found
    return r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def _resolve_tessdata_dir() -> Path:
    env = os.environ.get("TESSDATA_PREFIX")
    if env:
        return Path(env)
    if os.name == "nt":
        ascii_dir = Path(r"C:\tessdata_ocrb")
        if (ascii_dir / "ocrb.traineddata").exists():
            return ascii_dir
    bundled = Path(__file__).parent / "tessdata"
    if (bundled / "ocrb.traineddata").exists():
        return bundled
    return Path(r"C:\tessdata_ocrb")


# Tesseract binary path (resolved per platform).
_TESSERACT_CMD = _resolve_tesseract_cmd()

# OCR-B tessdata — set via TESSDATA_PREFIX env variable.
_TESSDATA_DIR = _resolve_tessdata_dir()

_MRZ_CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
_MRZ_LENGTHS = (30, 36, 44)
_MIN_STRIP_H = 80

_JUNK_MAP = str.maketrans(
    "OoIilBsSzZqQdD .,;:!?()-",
    "001188552200DD<<<<<<<<<<",
)
_VALID_CHARS = set(_MRZ_CHARSET)


def _get_pytesseract():
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = _TESSERACT_CMD
    os.environ["TESSDATA_PREFIX"] = str(_TESSDATA_DIR)
    return pytesseract


def _clean_char(c: str) -> str:
    c = c.upper()
    if c in _VALID_CHARS:
        return c
    mapped = c.translate(_JUNK_MAP)
    return mapped if mapped in _VALID_CHARS else "<"


def _clean_text(raw: str) -> str:
    return "".join(_clean_char(c) for c in raw.upper().strip())


def _snap(text: str, target: int) -> str:
    return (text + "<" * target)[:target]


def _upscale(img: np.ndarray, min_h: int = _MIN_STRIP_H) -> np.ndarray:
    h = img.shape[0]
    if h < min_h:
        scale = min_h / h
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return img


@dataclass
class TesseractLine:
    text: str
    confidence: float


@dataclass
class TesseractResult:
    lines: list[TesseractLine]
    engine: str = "tesseract_ocrb"

    @property
    def mean_confidence(self) -> float:
        if not self.lines:
            return 0.0
        return sum(ln.confidence for ln in self.lines) / len(self.lines)


def _read_image_ocrb(image: np.ndarray, n_lines: int) -> tuple[list[str], float]:
    pytesseract = _get_pytesseract()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = _upscale(gray, min_h=_MIN_STRIP_H * n_lines)

    config = (
        "--oem 1 --psm 6 "
        "-l ocrb "
        f"-c tessedit_char_whitelist={_MRZ_CHARSET}"
    )

    try:
        data = pytesseract.image_to_data(
            gray,
            config=config,
            output_type=pytesseract.Output.DICT,
        )
    except Exception:
        return [], 0.0

    # Group words by line_num
    from collections import defaultdict
    line_words: dict[int, list[tuple[str, float]]] = defaultdict(list)
    for text, conf, line_num in zip(data["text"], data["conf"], data["line_num"]):
        text = str(text).strip()
        if text and int(conf) != -1:
            line_words[line_num].append((text, float(conf) / 100.0))

    if not line_words:
        return [], 0.0

    # Build one string per line, sorted by line_num
    result_lines: list[str] = []
    all_confs: list[float] = []
    target_len = 30 if n_lines >= 3 else 44

    for line_num in sorted(line_words.keys()):
        words = line_words[line_num]
        combined = "".join(w for w, _ in words)
        confs = [c for _, c in words]
        result_lines.append(_snap(_clean_text(combined), target_len))
        all_confs.extend(confs)

    avg_conf = sum(all_confs) / len(all_confs) if all_confs else 0.0

    return result_lines, avg_conf


def run_tesseract_ocrb(image: np.ndarray, n_lines: int = 2) -> TesseractResult:
    raw_lines, conf = _read_image_ocrb(image, n_lines)

    lines: list[TesseractLine] = [
        TesseractLine(text=text, confidence=conf) for text in raw_lines
    ]

    # Pad to at least n_lines so downstream code always has something to score.
    while len(lines) < n_lines:
        lines.append(TesseractLine(text="<" * (30 if n_lines >= 3 else 44), confidence=0.0))

    return TesseractResult(lines=lines)