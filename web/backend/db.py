"""PostgreSQL access layer for the Trust Console.

Stores every saved scan (image + model output + the human-corrected fields) so
the corrected "gold label" records can later feed a human-in-the-loop active
learning loop. Raw SQL over psycopg3, matching the project's existing raw-SQL
style (SQL/Data_Scrapping.py).

The connection is optional: if DATABASE_URL is unset (e.g. local dev without a
database), get_pool() returns None and the /api/save endpoint reports 503 while
the scan flow keeps working.
"""
from __future__ import annotations
import os
from typing import Any, Optional

# psycopg / psycopg_pool are only needed when a database is configured. Import
# lazily so the app still boots (and /api/scan works) without them installed.
_pool = None
_pool_initialised = False

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scan_records (
    id                SERIAL PRIMARY KEY,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    filename          TEXT,
    image             BYTEA NOT NULL,
    image_sha256      TEXT NOT NULL,
    image_mime        TEXT,
    model_output      JSONB,
    corrected_fields  JSONB,
    human_corrected   BOOLEAN NOT NULL DEFAULT FALSE,
    reliability_score REAL,
    mrz_format        TEXT
);
CREATE INDEX IF NOT EXISTS idx_scan_records_sha256 ON scan_records (image_sha256);
CREATE INDEX IF NOT EXISTS idx_scan_records_corrected ON scan_records (human_corrected);
"""


def get_pool():
    """Return a lazily-created connection pool, or None if no DB is configured."""
    global _pool, _pool_initialised
    if _pool_initialised:
        return _pool
    _pool_initialised = True

    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        return None

    try:
        from psycopg_pool import ConnectionPool
        # Small pool — Render's free tier has little RAM and few connections.
        _pool = ConnectionPool(dsn, min_size=1, max_size=2, open=True)
    except Exception:
        # Misconfigured/unreachable DB must not crash the app; save() will 503.
        _pool = None
    return _pool


def is_available() -> bool:
    return get_pool() is not None


def init_schema() -> None:
    """Create the table/indexes if a database is configured. Safe to re-run."""
    pool = get_pool()
    if pool is None:
        return
    with pool.connection() as conn:
        conn.execute(_SCHEMA)


def insert_record(
    *,
    filename: Optional[str],
    image: bytes,
    image_sha256: str,
    image_mime: Optional[str],
    model_output: Any,
    corrected_fields: Any,
    human_corrected: bool,
    reliability_score: Optional[float],
    mrz_format: Optional[str],
) -> int:
    """Insert one scan record and return its id. Raises if no DB is configured."""
    pool = get_pool()
    if pool is None:
        raise RuntimeError("No database configured (DATABASE_URL unset)")

    from psycopg.types.json import Json

    with pool.connection() as conn:
        row = conn.execute(
            """
            INSERT INTO scan_records (
                filename, image, image_sha256, image_mime,
                model_output, corrected_fields, human_corrected,
                reliability_score, mrz_format
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                filename,
                image,
                image_sha256,
                image_mime,
                Json(model_output) if model_output is not None else None,
                Json(corrected_fields) if corrected_fields is not None else None,
                human_corrected,
                reliability_score,
                mrz_format,
            ),
        ).fetchone()
    return int(row[0])
