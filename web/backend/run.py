"""Production launcher for the FastAPI backend.

Reads the port from $PORT (injected by Render/Railway) directly in Python so
the bind never relies on shell variable expansion in the Docker CMD — that
expansion failing is what produced Render's "No open ports detected".
"""
from __future__ import annotations
import os
import uvicorn


def main() -> None:
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(
        "web.backend.app:app",
        host="0.0.0.0",
        port=port,
        workers=1,
    )


if __name__ == "__main__":
    main()
