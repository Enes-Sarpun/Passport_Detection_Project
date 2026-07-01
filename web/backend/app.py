from __future__ import annotations
import base64
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Optional
import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Make the project root importable so `Scripts.*` resolves regardless of CWD.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Scripts.ocr.pipeline import _process_frame  # noqa: E402
from Scripts.parsing.schema import failure_output  # noqa: E402
from web.backend import db  # noqa: E402

app = FastAPI(title="Passport Detection Trust Console", version="1.0")


@app.on_event("startup")
def _startup() -> None:
    # Create the scan_records table if a database is configured; no-op otherwise.
    try:
        db.init_schema()
    except Exception:  # pragma: no cover - never block startup on DB issues
        pass

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


# Map each editable field key (as the frontend emits in corrected_fields) to the
# value the model produced, pulled out of the scan's model_output JSON. Mirrors
# the field set in web/frontend/src/fields.js.
def _model_field_values(model_output: dict) -> dict[str, str]:
    if not isinstance(model_output, dict):
        return {}
    d = model_output.get("document") or {}
    h = model_output.get("holder") or {}
    dt = model_output.get("dates") or {}

    def s(v: Any) -> str:
        return "" if v is None else str(v)

    return {
        "document_type": s((d.get("type") or {}).get("code")),
        "document_number": s((d.get("number") or {}).get("value")),
        "personal_number": s((d.get("personal_number") or {}).get("value")),
        "nationality": s((h.get("nationality") or {}).get("code")),
        "surname": s((h.get("surname") or {}).get("value")),
        "given_names": s((h.get("given_names") or {}).get("value")),
        "date_of_birth": s((dt.get("date_of_birth") or {}).get("iso")),
        "date_of_expiry": s((dt.get("date_of_expiry") or {}).get("iso")),
        "sex": s((h.get("sex") or {}).get("code")),
    }


def _is_human_corrected(
    model_output: dict, corrected: dict, model_mrz: list, corrected_mrz: Optional[list]
) -> bool:
    """True if the human changed any field value OR the raw MRZ lines."""
    if isinstance(corrected, dict):
        model_vals = _model_field_values(model_output)
        for key, val in corrected.items():
            if str(val).strip() != str(model_vals.get(key, "")).strip():
                return True
    if corrected_mrz is not None:
        norm = [str(x).strip() for x in corrected_mrz]
        base = [str(x).strip() for x in (model_mrz or [])]
        if norm != base:
            return True
    return False


# Field reliability threshold below which a correction is MANDATORY.
# Mirrors THRESHOLD in web/frontend/src/fields.js.
_RELIABILITY_THRESHOLD = 0.75


def _model_field_reliabilities(model_output: dict) -> dict[str, Optional[float]]:
    """Per-field reliability from model_output. document_type has none (never
    mandatory) — mirrors fields.js hasReliability=false for that key."""
    if not isinstance(model_output, dict):
        return {}
    d = model_output.get("document") or {}
    h = model_output.get("holder") or {}
    dt = model_output.get("dates") or {}

    def r(node) -> Optional[float]:
        v = (node or {}).get("reliability")
        return float(v) if isinstance(v, (int, float)) else None

    return {
        "document_number": r(d.get("number")),
        "personal_number": r(d.get("personal_number")),
        "nationality": r(h.get("nationality")),
        "surname": r(h.get("surname")),
        "given_names": r(h.get("given_names")),
        "date_of_birth": r(dt.get("date_of_birth")),
        "date_of_expiry": r(dt.get("date_of_expiry")),
        "sex": r(h.get("sex")),
    }


def _is_present(v: Any) -> bool:
    """Mirror of isPresent() in fields.js: non-empty after stripping '<' and spaces."""
    return str(v if v is not None else "").replace("<", "").strip() != ""


def _mandatory_unresolved(model_output: dict, corrected: dict) -> bool:
    """Data-quality gate (Rule A), kept in lock-step with fields.js so the
    frontend's Save-enabled state and this server-side check never disagree.

    A field is mandatory when it's missing (!found) OR its reliability < 0.75
    (isMandatory in fields.js). If a mandatory field is left empty or unchanged
    from the model's value, the record must NOT be saved."""
    rels = _model_field_reliabilities(model_output)
    model_vals = _model_field_values(model_output)
    corrected = corrected if isinstance(corrected, dict) else {}
    for key, rel in rels.items():
        model_val = model_vals.get(key, "")
        found = _is_present(model_val)
        # fields.js: reliability defaults to 0 when not found → mandatory.
        effective_rel = rel if rel is not None else (1.0 if found else 0.0)
        mandatory = (not found) or effective_rel < _RELIABILITY_THRESHOLD
        if not mandatory:
            continue
        # The final value the user is submitting for this field.
        final_val = str(corrected.get(key, model_val)).strip()
        if final_val == "" or final_val == str(model_val).strip():
            return True
    return False


def _parse_json_field(raw: Optional[str], field_name: str) -> Any:
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        raise HTTPException(400, f"Invalid JSON in '{field_name}'")


@app.post("/api/save")
async def save(
    file: UploadFile = File(...),
    model_output: str = Form(...),
    corrected_fields: str = Form(...),
    corrected_mrz: str = Form(None),
) -> dict:
    if not db.is_available():
        raise HTTPException(503, "Database not configured; record not saved")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix and suffix not in _ALLOWED_SUFFIXES:
        raise HTTPException(415, f"Unsupported file type: {suffix}")

    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty file")
    if len(raw) > _MAX_BYTES:
        raise HTTPException(413, "File too large (max 15 MB)")

    model = _parse_json_field(model_output, "model_output")
    corrected = _parse_json_field(corrected_fields, "corrected_fields")
    corrected_lines = _parse_json_field(corrected_mrz, "corrected_mrz")

    # Rule A (data-quality gate): a low-reliability field that still needs a fix
    # but was left unchanged must NOT be saved — keeps the dataset clean.
    if _mandatory_unresolved(model or {}, corrected or {}):
        raise HTTPException(
            422, "Correction required for low-confidence fields; record not saved"
        )

    quality = (model or {}).get("quality") or {}
    document = (model or {}).get("document") or {}
    reliability = quality.get("reliability_score")
    mrz_format = document.get("mrz_format")
    model_mrz = (model or {}).get("raw_mrz") or []

    try:
        record_id = db.insert_record(
            filename=file.filename,
            image=raw,
            image_sha256=hashlib.sha256(raw).hexdigest(),
            image_mime=file.content_type,
            model_output=model,
            corrected_fields=corrected,
            corrected_mrz=corrected_lines,
            human_corrected=_is_human_corrected(
                model or {}, corrected or {}, model_mrz, corrected_lines
            ),
            reliability_score=reliability,
            mrz_format=mrz_format,
        )
    except Exception as exc:
        raise HTTPException(500, f"Save failed: {exc}")

    return {"saved": True, "id": record_id}


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
