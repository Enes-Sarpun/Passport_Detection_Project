from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np

_DEFAULT_WEIGHTS = Path(__file__).parent.parent.parent / "Images" / "Results" / "mrz_detect2" / "weights" / "best.pt"


def _limit_torch_threads() -> None:
    try:
        import torch
        n = int(os.environ.get("TORCH_NUM_THREADS", "1"))
        torch.set_num_threads(max(1, n))
    except Exception:
        pass

@dataclass
class Detection:
    box: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float

_model_cache: dict[str, object] = {}

def _load_model(weights: Path):
    key = str(weights)
    if key not in _model_cache:
        _limit_torch_threads()
        from ultralytics import YOLO
        _model_cache[key] = YOLO(str(weights))
    return _model_cache[key]

def detect_mrz(image: np.ndarray,weights: Optional[Path] = None,conf_threshold: float = 0.5,) -> Optional[Detection]:
    weights = Path(weights) if weights else _DEFAULT_WEIGHTS
    model = _load_model(weights)

    results = model.predict(
        source=image,
        conf=conf_threshold,
        max_det=1,
        verbose=False,
    )

    if not results or not results[0].boxes or len(results[0].boxes) == 0:
        return None

    box = results[0].boxes[0]
    x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
    confidence = float(box.conf[0])
    return Detection(box=(x1, y1, x2, y2), confidence=confidence)