# Base image: Python 3.10-slim
FROM python:3.10-slim

# Install system dependencies for OpenCV and Tesseract
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-tur \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app
ENV PYTHONPATH=/app

# Tesseract config for the OCR engine (Scripts/ocr/engine.py reads these).
# The OCR-B traineddata is downloaded into the repo's tessdata dir below, so
# point TESSDATA_PREFIX there; tesseract is on PATH from the apt install above.
ENV TESSERACT_CMD=/usr/bin/tesseract
ENV TESSDATA_PREFIX=/app/Scripts/ocr/tessdata

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project to the working directory
COPY . /app/

# Download Tesseract OCR-B trained data
RUN python Scripts/ocr/setup_model.py

# Expose the port (Render uses PORT env variable, defaulting to 8000 here)
EXPOSE 8000

# Start Uvicorn, pointing to web.backend.app:app
# Render/Railway will inject $PORT, so we bind to 0.0.0.0 and $PORT
CMD uvicorn web.backend.app:app --host 0.0.0.0 --port ${PORT:-8000}