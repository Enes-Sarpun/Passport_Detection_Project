from __future__ import annotations
from pathlib import Path

_MODEL_URL = (
    "https://github.com/Shreeshrii/tessdata_ocrb/raw/master/ocrb.traineddata"
)
_TESSDATA_DIR = Path(__file__).parent / "tessdata"
_MODEL_PATH = _TESSDATA_DIR / "ocrb.traineddata"


def download_ocrb_model() -> None:
    _TESSDATA_DIR.mkdir(exist_ok=True)

    if _MODEL_PATH.exists():
        print(f"Model already exists: {_MODEL_PATH}")
        return

    print(f"Downloading OCR-B model from:\n  {_MODEL_URL}")
    print("This may take a moment (~10 MB)...")

    try:
        import requests
        response = requests.get(_MODEL_URL, stream=True, timeout=60)
        response.raise_for_status()
        with open(_MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    except ImportError:
        # fallback to urllib with redirect support
        import urllib.request
        opener = urllib.request.build_opener(urllib.request.HTTPRedirectHandler())
        with opener.open(_MODEL_URL, timeout=60) as resp, open(_MODEL_PATH, "wb") as f:
            f.write(resp.read())

    print(f"Saved: {_MODEL_PATH}")
    print("OCR-B model ready.")


if __name__ == "__main__":
    download_ocrb_model()