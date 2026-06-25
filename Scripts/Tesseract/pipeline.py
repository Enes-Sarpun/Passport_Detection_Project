from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Union
import cv2
import numpy as np

# Shared modules from the OCR pipeline
from Scripts.OCR.detect import detect_mrz, Detection
from Scripts.OCR.preprocess import crop, deskew, upscale, _to_gray
from Scripts.OCR.mrz_parse import parse_mrz, MRZResult
from Scripts.OCR.schema import build_output, failure_output, to_json
from Scripts.OCR.reconstruct import (
    _align_line, _best_aligned_pair, _normalize,
    TD3_LEN, TD2_LEN, TD1_LEN,
    detect_format, _validation_score,
)

from .ocr import run_tesseract_ocrb, TesseractResult


def _count_cd_passes(lines: list[str]) -> int:
    cd_passes, _, _ = _validation_score(lines)
    return cd_passes


def _tesseract_preprocess(image: np.ndarray, box: tuple, n_lines: int) -> list[np.ndarray]:
    """Minimal preprocessing for Tesseract — no binarization.

    Tesseract's internal preprocessing outperforms our binary candidates.
    We provide: raw gray, CLAHE-enhanced gray, and lightly sharpened gray.
    """
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
    raw_lines = [_normalize(ln.text) for ln in result.lines]
    fmt, line_len = detect_format(raw_lines)
    n_lines = {"TD1": 3, "TD2": 2, "TD3": 2}[fmt]
    return fmt, line_len, n_lines


def _align_result(result: TesseractResult, n_lines: int, line_len: int) -> list[str]:
    raw_lines = [ln.text for ln in result.lines[:n_lines]]
    while len(raw_lines) < n_lines:
        raw_lines.append("")

    if n_lines == 2:
        # Try both line orderings and keep the one with more check-digit passes.
        # This handles PSM 6 line-order reversals (e.g. stamp noise hiding line 1).
        l1a, l2a = _best_aligned_pair(raw_lines[0], raw_lines[1], line_len)
        cd_normal = _count_cd_passes([l1a, l2a])

        l1b, l2b = _best_aligned_pair(raw_lines[1], raw_lines[0], line_len)
        cd_swapped = _count_cd_passes([l1b, l2b])

        if cd_swapped > cd_normal:
            return [l1b, l2b]
        return [l1a, l2a]

    return [_align_line(raw_lines[i], line_len, i) for i in range(n_lines)]


def _process_frame(
    image: np.ndarray,
    weights: Optional[Path] = None,
    conf_threshold: float = 0.5,
) -> tuple[Optional[dict], Optional[Detection]]:
    detection = detect_mrz(image, weights=weights, conf_threshold=conf_threshold)
    if detection is None:
        return None, None

    candidates = _tesseract_preprocess(image, detection.box, n_lines=2)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    all_results: list[TesseractResult] = []
    with ThreadPoolExecutor(max_workers=len(candidates)) as pool:
        futures = [pool.submit(run_tesseract_ocrb, img, 2) for img in candidates]
        for future in as_completed(futures):
            try:
                all_results.append(future.result())
            except Exception:
                pass

    if not all_results:
        return None, detection

    best_for_fmt = max(
        (r for r in all_results if r.lines),
        key=lambda r: max((len(_normalize(ln.text)) for ln in r.lines), default=0),
        default=all_results[0],
    )
    fmt, line_len, n_lines = _infer_format(best_for_fmt)

    scored: list[tuple[int, float, list[str]]] = []
    for res in all_results:
        if not res.lines:
            continue
        aligned = _align_result(res, n_lines, line_len)
        cd = _count_cd_passes(aligned)
        scored.append((cd, res.mean_confidence, aligned))

    if not scored:
        return None, detection

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best_cd, best_conf, best_lines = scored[0]
    chosen_lines = best_lines

    parsed = parse_mrz(chosen_lines)
    if parsed is None:
        return failure_output("parse_failed", raw_mrz=chosen_lines,
                              warnings=["mrz_format_invalid"]), detection

    pipeline_warnings: list[str] = []
    if detection.confidence < 0.5:
        pipeline_warnings.append("low_detection_confidence")
    x1, y1, x2, y2 = detection.box
    img_h, img_w = image.shape[:2]
    if (x2 - x1) < img_w * 0.5 or (y2 - y1) < img_h * 0.05:
        pipeline_warnings.append("mrz_partially_occluded")

    output = build_output(
        parsed,
        detection_confidence=detection.confidence,
        ocr_confidence=best_conf,
        raw_mrz=chosen_lines,
        extra_warnings=pipeline_warnings,
    )
    return output, detection


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