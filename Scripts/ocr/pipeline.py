from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Union
import cv2
import numpy as np
from Scripts.detection.detect import detect_mrz, Detection
from Scripts.detection.preprocess import crop, deskew, upscale, _to_gray
from Scripts.parsing.mrz_parse import parse_mrz, MRZResult
from Scripts.parsing.country_lookup import resolve_country
from Scripts.parsing.schema import build_output, failure_output, to_json
from Scripts.parsing.reconstruct import (
    _align_line, _best_aligned_pair, _normalize,
    TD3_LEN, TD2_LEN, TD1_LEN,
    detect_format, _validation_score,
)
from .engine import run_tesseract_ocrb, TesseractResult


def _count_cd_passes(lines: list[str]) -> int:
    cd_passes, _, _ = _validation_score(lines)
    return cd_passes


def _tesseract_preprocess(image: np.ndarray, box: tuple, n_lines: int) -> list[np.ndarray]:
    import cv2 as _cv2
    cropped = crop(image, box)
    gray = _to_gray(cropped)
    gray = deskew(gray)
    gray = upscale(gray, n_lines=n_lines)

    clahe = _cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    blurred = _cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
    sharpened = _cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)

    return [
        _cv2.cvtColor(raw, _cv2.COLOR_GRAY2BGR)
        for raw in [gray, enhanced, sharpened]
    ]


def _infer_format(result: TesseractResult) -> tuple[str, int, int]:
    lengths = sorted(
        (len(_normalize(ln.text).rstrip("<")) for ln in result.lines),
        reverse=True,
    )
    # Use the longest genuine line as the format signal.
    max_len = lengths[0] if lengths else 0

    if max_len >= 40:
        return "TD3", TD3_LEN, 2
    if max_len >= 33:
        return "TD2", TD2_LEN, 2
    # Short lines: could be genuine TD1 (3x30) or just garbage. Only call it TD1
    substantial = sum(1 for ln_len in lengths if ln_len >= 25)
    if substantial >= 3:
        return "TD1", TD1_LEN, 3
    # Default to TD3 — the dominant passport format in this dataset.
    return "TD3", TD3_LEN, 2


def _mrz_likeness(text: str, line_len: int) -> float:
    norm = _normalize(text)
    stripped = norm.rstrip("<")
    if not stripped:
        return 0.0

    # Fill ratio: how much of the (rstripped) content is real, non-filler chars.
    content = stripped.replace("<", "")
    fill_ratio = len(content) / max(len(stripped), 1)

    # Length proximity to the expected line length.
    length_score = 1.0 - abs(len(stripped) - line_len) / line_len
    length_score = max(0.0, length_score)

    return 0.5 * fill_ratio + 0.5 * length_score


def _l1_structure_score(text: str) -> float:
    norm = _normalize(text)
    if len(norm) < 5:
        return 0.0
    score = 0.0
    # Valid 3-letter country code at the canonical position.
    code = norm[2:5]
    if code.isalpha() and resolve_country(code)["name"] != "Unknown":
        score += 0.5
    # Surname/given-names separator present in the name field.
    if "<<" in norm[5:]:
        score += 0.3
    # Name lines are essentially digit-free.
    if sum(1 for c in norm if c.isdigit()) <= 1:
        score += 0.2
    return score


def _select_mrz_lines(
    candidates: list[str], line_len: int
) -> tuple[list[str], bool]:
    scored = sorted(
        ((_mrz_likeness(c, line_len), c) for c in candidates if c),
        key=lambda x: x[0],
        reverse=True,
    )
    # Keep meaningful candidates (cap at 5 to bound the pair search).
    pool = [c for s, c in scored if s > 0.05][:5]
    if len(pool) < 2:
        # Not enough usable lines; pad and bail with manual-review flag.
        padded = (pool + ["", ""])[:2]
        l1, l2 = _best_aligned_pair(padded[0], padded[1], line_len)
        return [l1, l2], True

    best_pair: Optional[list[str]] = None
    best_key: tuple[int, float] = (-1, -1.0)
   
    for i in range(len(pool)):
        for j in range(len(pool)):
            if i == j:
                continue
            l1, l2 = _best_aligned_pair(pool[i], pool[j], line_len)
            cd = _count_cd_passes([l1, l2])
            key = (cd, _l1_structure_score(l1))
            if key > best_key:
                best_key = key
                best_pair = [l1, l2]

    best_cd = best_key[0]
    if best_cd <= 0 or best_pair is None:
        l1, l2 = _best_aligned_pair(pool[0], pool[1], line_len)
        return [l1, l2], True

    return best_pair, False


def _align_result(result: TesseractResult, n_lines: int, line_len: int) -> tuple[list[str], bool]:
    """Return (aligned_lines, needs_manual_review)."""
    raw_lines = [ln.text for ln in result.lines]

    if n_lines == 2:
        return _select_mrz_lines(raw_lines, line_len)

    # TD1 (3-line) path: keep positional alignment, no manual-review heuristic yet.
    raw_lines = (raw_lines + [""] * n_lines)[:n_lines]
    return [_align_line(raw_lines[i], line_len, i) for i in range(n_lines)], False


import os as _os


_OCR_MAX_WORKERS = int(_os.environ.get("OCR_MAX_WORKERS", "0"))  # 0 = auto (all)


def _run_ocr_passes(candidates: list[np.ndarray]) -> list[TesseractResult]:
    workers = _OCR_MAX_WORKERS or len(candidates)
    workers = max(1, min(workers, len(candidates)))

    if workers == 1:
        results: list[TesseractResult] = []
        for img in candidates:
            try:
                results.append(run_tesseract_ocrb(img, 2))
            except Exception:
                pass
        return results

    from concurrent.futures import ThreadPoolExecutor, as_completed
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_tesseract_ocrb, img, 2) for img in candidates]
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception:
                pass
    return results


def _process_frame(
    image: np.ndarray,
    weights: Optional[Path] = None,
    conf_threshold: float = 0.5,
    return_signals: bool = False,
):

    def _ret(out, det, signals=None):
        return (out, det, signals) if return_signals else (out, det)

    detection = detect_mrz(image, weights=weights, conf_threshold=conf_threshold)
    if detection is None:
        return _ret(None, None)

    candidates = _tesseract_preprocess(image, detection.box, n_lines=2)

    all_results: list[TesseractResult] = _run_ocr_passes(candidates)

    if not all_results:
        return _ret(None, detection)

    best_for_fmt = max(
        (r for r in all_results if r.lines),
        key=lambda r: max((len(_normalize(ln.text)) for ln in r.lines), default=0),
        default=all_results[0],
    )
    fmt, line_len, n_lines = _infer_format(best_for_fmt)

    scored: list[tuple[int, float, list[str], bool]] = []
    for res in all_results:
        if not res.lines:
            continue
        aligned, needs_review = _align_result(res, n_lines, line_len)
        cd = _count_cd_passes(aligned)
        scored.append((cd, res.mean_confidence, aligned, needs_review))

    if not scored:
        return _ret(None, detection)

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best_cd, best_conf, best_lines, best_needs_review = scored[0]
    chosen_lines = best_lines

    signals = {
        "detection_confidence": float(detection.confidence),
        "ocr_confidence": float(best_conf),
    }

    parsed = parse_mrz(chosen_lines)
    if parsed is None:
        return _ret(failure_output("parse_failed", raw_mrz=chosen_lines,
                                   warnings=["mrz_format_invalid"]), detection, signals)

    pipeline_warnings: list[str] = []
    if detection.confidence < 0.5:
        pipeline_warnings.append("low_detection_confidence")
    x1, y1, x2, y2 = detection.box
    img_h, img_w = image.shape[:2]
    if (x2 - x1) < img_w * 0.5 or (y2 - y1) < img_h * 0.05:
        pipeline_warnings.append("mrz_partially_occluded")
    # Line selection found no check-digit-valid pair → route to manual review.
    if best_needs_review:
        pipeline_warnings.append("manual_review_required")

    output = build_output(
        parsed,
        detection_confidence=detection.confidence,
        ocr_confidence=best_conf,
        raw_mrz=chosen_lines,
        extra_warnings=pipeline_warnings,
    )
    return _ret(output, detection, signals)


def process_image(
    source: Union[str, Path, np.ndarray],
    weights: Optional[Path] = None,
    conf_threshold: float = 0.5,
    output_dir: Optional[Union[str, Path]] = None,
) -> dict:
    if isinstance(source, np.ndarray):
        image = source
        stem = "frame"
    else:
        source = Path(str(source).strip()).resolve()
        stem = source.stem
        if not source.exists():
            return failure_output(f"file_not_found: {source}")
        raw = np.frombuffer(source.read_bytes(), dtype=np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if image is None:
            return failure_output("image_load_failed")

    try:
        output, detection = _process_frame(image, weights=weights, conf_threshold=conf_threshold)
        result = output if output is not None else failure_output(
            "no_mrz_detected" if detection is None else "parse_failed",
            warnings=["no_mrz_detected" if detection is None else "mrz_format_invalid"],
        )

        if output_dir is not None:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            json_path = out_path / f"{stem}_tess_ocr.json"
            json_path.write_text(to_json(result), encoding="utf-8")

            if detection is not None:
                annotated = image.copy()
                x1, y1, x2, y2 = detection.box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 255), 2)
                label = f"MRZ {detection.confidence:.2f}"
                cv2.putText(annotated, label, (x1, max(y1 - 8, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                cv2.imwrite(str(out_path / f"{stem}_tess_annotated.jpg"), annotated)

        return result

    except Exception as exc:
        return failure_output(f"error: {exc}")
    
    