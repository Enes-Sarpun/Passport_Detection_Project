from __future__ import annotations
import base64
import sys
from pathlib import Path
import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Make the project root importable so `Scripts.*` resolves regardless of CWD.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Scripts.ocr.pipeline import _process_frame  # noqa: E402
from Scripts.parsing.schema import failure_output  # noqa: E402

app = FastAPI(title="Passport Detection Trust Console", version="1.0")

# The React dev server runs on a different origin (Vite default 5173); allow it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_MAX_BYTES = 15 * 1024 * 1024  # 15 MB upload cap
_ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _annotate(image: np.ndarray, detection) -> str:
    annotated = image.copy()
    if detection is not None:
        x1, y1, x2, y2 = detection.box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (212, 105, 28), 2)  # BMW blue (BGR)
        label = f"MRZ {detection.confidence:.2f}"
        cv2.putText(annotated, label, (x1, max(y1 - 8, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (212, 105, 28), 2)
    ok, buf = cv2.imencode(".jpg", annotated)
    if not ok:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/scan")
async def scan(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix and suffix not in _ALLOWED_SUFFIXES:
        raise HTTPException(415, f"Unsupported file type: {suffix}")

    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty file")
    if len(raw) > _MAX_BYTES:
        raise HTTPException(413, "File too large (max 15 MB)")

    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(422, "Could not decode image")

    try:
        output, detection = _process_frame(image)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(500, f"Processing error: {exc}")

    if output is None:
        output = failure_output(
            "no_mrz_detected" if detection is None else "parse_failed",
            warnings=["no_mrz_detected" if detection is None else "mrz_format_invalid"],
        )

    return {
        "result": output,
        "preview": _annotate(image, detection),
        "filename": file.filename,
    }
