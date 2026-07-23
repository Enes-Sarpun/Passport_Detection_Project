"""Shared image I/O and annotation helpers.

Small utilities used across the pipeline, the web backend and the ground-truth
tooling so the raw OpenCV decode/annotate calls live in one place.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Union
import cv2
import numpy as np

# Default annotation colour for the MRZ detection box (BGR).
_DEFAULT_BOX_COLOR = (0, 200, 255)


def decode_image(data: bytes) -> Optional[np.ndarray]:
    """Decode raw image bytes into a BGR array, or None if undecodable."""
    buf = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def load_image(path: Union[str, Path]) -> Optional[np.ndarray]:
    """Read an image file into a BGR array, or None if it cannot be decoded."""
    return decode_image(Path(path).read_bytes())


def draw_detection_box(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    confidence: float,
    color: tuple[int, int, int] = _DEFAULT_BOX_COLOR,
) -> np.ndarray:
    """Draw the MRZ detection rectangle + confidence label onto a copy of `image`."""
    annotated = image.copy()
    x1, y1, x2, y2 = box
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
    label = f"MRZ {confidence:.2f}"
    cv2.putText(annotated, label, (x1, max(y1 - 8, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return annotated
