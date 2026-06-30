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

# Ultralytics writes a config/cache dir; /root/.config is not writable on Render.
# Point it at a writable location to silence the warning and avoid any stall.
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics
ENV MPLCONFIGDIR=/tmp/matplotlib

# Render injects $PORT at runtime; default to 8000 for local docker runs.
ENV PORT=8000

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project to the working directory
COPY . /app/

# Download Tesseract OCR-B trained data
RUN python Scripts/ocr/setup_model.py

# Expose the port (Render uses PORT env variable, defaulting to 8000 here)
EXPOSE 8000

# Start Uvicorn via a tiny launcher that reads $PORT itself, so binding never
# depends on shell variable expansion (the cause of Render's "No open ports").
CMD ["python", "-m", "web.backend.run"]