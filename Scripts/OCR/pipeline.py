from __future__ import annotations
import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional, Union
import cv2
import numpy as np
from .detect import detect_mrz, Detection
from .mrz_parse import parse_mrz, MRZResult
from .ocr import run_ocr_multi, OcrResult
from .preprocess import preprocess
from .reconstruct import reconstruct, column_vote, ReconstructedMRZ
from .schema import build_output, failure_output, to_json

def _is_validated(result: MRZResult) -> bool:
    v = result.validation
    return all([
        v.get("document_number_valid"),
        v.get("date_of_birth_valid"),
        v.get("date_of_expiry_valid"),
        v.get("composite_valid", True),
    ])

def _process_frame(
    image: np.ndarray,
    weights: Optional[Path] = None,
    conf_threshold: float = 0.5,
) -> tuple[Optional[dict], Optional[Detection], Optional[ReconstructedMRZ]]:
    detection = detect_mrz(image, weights=weights, conf_threshold=conf_threshold)
    if detection is None:
        return None, None, None

    candidates = preprocess(image, detection.box, n_lines=2)
    ocr_results: list[OcrResult] = run_ocr_multi(candidates)

    # Score each OcrResult individually by how many check digits pass after parse.
    # This lets a lower-confidence-but-accurate engine win over a high-confidence
    # engine that produces plausible-looking but wrong characters.
    # NOTE: we normalize text before parse to match what reconstruct() does.
    from .reconstruct import _align_line, TD3_LEN, TD3_LINES
    def _validation_score(ocr_r: OcrResult) -> int:
        if not ocr_r.lines:
            return -1
        lines = [_align_line(l.text, TD3_LEN, i) for i, l in enumerate(ocr_r.lines)]
        parsed_try = parse_mrz(lines)
        if parsed_try is None:
            return -1
        v = parsed_try.validation
        return sum(1 for k, val in v.items() if val is True)

    # Build column-voted reconstruction from ALL candidates (primary path).
    reconstructed = reconstruct(ocr_results)

    # Also score individual OcrResults as a secondary path.
    best_single = max(ocr_results, key=_validation_score, default=None)
    best_score = _validation_score(best_single) if best_single else -1

    # Always use the reconstructed lines for output — they have been aligned and voted.
    # Only fall back to best_single's raw text if reconstruct fails to produce lines.
    if reconstructed.lines:
        chosen_lines = reconstructed.lines
        parsed = parse_mrz(chosen_lines)
    else:
        return None, detection, None

    if not chosen_lines:
        return None, detection, reconstructed

    if parsed is None:
        # Check digits don't match but we have raw MRZ text — return it for debugging.
        result = failure_output("parse_failed", raw_mrz=chosen_lines)
        return result, detection, reconstructed

    ocr_conf = sum(r.mean_confidence for r in ocr_results if r.lines) / max(
        sum(1 for r in ocr_results if r.lines), 1
    )

    output = build_output(
        parsed,
        detection_confidence=detection.confidence,
        ocr_confidence=ocr_conf,
        raw_mrz=chosen_lines,
    )
    return output, detection, reconstructed

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
        # Strip leading/trailing whitespace — CLI users sometimes pass paths
        # with an accidental trailing space (e.g. "file .jpg").
        source = Path(str(source).strip())
        source = source.resolve()
        stem = source.stem
        if not source.exists():
            return failure_output(f"file_not_found: {source}")
        with open(source, "rb") as f:
            raw = np.frombuffer(f.read(), dtype=np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if image is None:
            return failure_output("image_load_failed")

    try:
        output, detection, _ = _process_frame(image, weights=weights, conf_threshold=conf_threshold)
        if output is None:
            result = failure_output("no_mrz_detected" if detection is None else "parse_failed")
        else:
            result = output

        if output_dir is not None:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)

            json_path = out / f"{stem}_ocr.json"
            json_path.write_text(to_json(result), encoding="utf-8")

            if detection is not None:
                annotated = image.copy()
                x1, y1, x2, y2 = detection.box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = result.get("status", "")
                cv2.putText(annotated, label, (x1, max(y1 - 8, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                img_path = out / f"{stem}_annotated.jpg"
                cv2.imwrite(str(img_path), annotated)

        return result
    except Exception as exc:
        return failure_output(f"error: {exc}")

def run_camera(
    camera_index: int = 0,
    weights: Optional[Path] = None,
    conf_threshold: float = 0.45,
    stable_frames: int = 3,
    max_vote_frames: int = 15,
    display: bool = True,
) -> dict:
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return failure_output(f"camera_open_failed (index={camera_index})")

    status_msg = "Scanning..."
    consecutive = 0
    collected_lines: list[list[str]] = []
    collected_confs: list[float] = []
    result_out: Optional[dict] = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            output, detection, reconstructed = _process_frame(
                frame, weights=weights, conf_threshold=conf_threshold
            )

            if detection is None:
                consecutive = 0
                status_msg = "Scanning..."
            else:
                consecutive += 1
                status_msg = f"MRZ found ({consecutive}) - Validating..."

                if output is not None and reconstructed and _is_validated(
                    parse_mrz(reconstructed.lines)
                ):
                    result_out = output
                    status_msg = "Done!"
                    if display:
                        _draw_overlay(frame, detection, status_msg)
                        cv2.imshow("MRZ Scanner", frame)
                        cv2.waitKey(800)
                    break

                if reconstructed and reconstructed.lines:
                    collected_lines.append(reconstructed.lines)
                    collected_confs.append(detection.confidence)

                if len(collected_lines) >= max_vote_frames:
                    result_out = _vote_and_parse(
                        collected_lines, collected_confs, detection, reconstructed
                    )
                    status_msg = "Done (voted)!"
                    if display:
                        _draw_overlay(frame, detection, status_msg)
                        cv2.imshow("MRZ Scanner", frame)
                        cv2.waitKey(800)
                    break

            if display:
                if detection:
                    _draw_overlay(frame, detection, status_msg)
                cv2.imshow("MRZ Scanner", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    if result_out is None:
        return failure_output("no_mrz_detected")
    return result_out

def _draw_overlay(frame: np.ndarray, detection: Detection, status: str) -> None:
    x1, y1, x2, y2 = detection.box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

def _vote_and_parse(
    all_lines: list[list[str]],
    confs: list[float],
    last_detection: Detection,
    last_reconstructed: ReconstructedMRZ,
) -> dict:
    fmt = last_reconstructed.fmt
    line_len = last_reconstructed.line_length
    n_lines = len(last_reconstructed.lines)

    voted_lines: list[str] = []
    for li in range(n_lines):
        cand_texts = [frame_lines[li] if li < len(frame_lines) else "" for frame_lines in all_lines]
        voted = column_vote(cand_texts, confs, line_len)
        voted_lines.append(voted)

    parsed = parse_mrz(voted_lines)
    if parsed is None:
        return failure_output("parse_failed", raw_mrz=voted_lines)

    return build_output(
        parsed,
        detection_confidence=last_detection.confidence,
        ocr_confidence=0.0,
        raw_mrz=voted_lines,
    )


